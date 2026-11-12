"""ЛР №3. Вариант 05 «Городская улица» (папка корпуса NOIZEUS: street).

Конвейер: проверка 150 WAV -> графики -> анализ шума -> один частотный фильтр
-> SNR до/после -> таблицы и текстовая сводка.

Запуск (из папки варианта, где уже лежит dataset/):
    python main.py
    python main.py --dataset путь/к/dataset --out results
"""

from __future__ import annotations

import argparse
import re
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

NOISE = "street"
LEVELS = (0, 5, 10, 15)
PHRASES = ("sp01", "sp12", "sp25")

FILTER_ORDER = 2
FILTER_BAND_HZ = (200.0, 2200.0)
SWEEP_LOW_HZ = (100, 150, 200, 250, 300)
SWEEP_HIGH_HZ = (1800, 2000, 2200, 2500, 3000)
BAND_TAG = f"bp{FILTER_BAND_HZ[0]:.0f}-{FILTER_BAND_HZ[1]:.0f}"

BANDS_HZ = ((0, 300), (300, 1000), (1000, 2000), (2000, 3000), (3000, 4000))
MAX_LAG = 160
EPS = 1e-12


def read_wav(path: Path) -> tuple[np.ndarray, dict]:
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.getnframes()
        raw = w.readframes(frames)
    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.int64)
    else:
        raise ValueError(f"{path.name}: неподдерживаемая разрядность {8 * width} бит")
    full_scale = float(2 ** (8 * width - 1))
    data = data.reshape(-1, channels)
    info = {
        "sample_rate_hz": rate,
        "channels": channels,
        "bit_depth": 8 * width,
        "n_samples": data.shape[0],
        "duration_s": data.shape[0] / rate,
        "min_amplitude": int(data.min()),
        "max_amplitude": int(data.max()),
        "full_scale": full_scale,
    }
    x = data.astype(np.float64).mean(axis=1) / full_scale
    return x, info


def write_wav16(path: Path, x: np.ndarray, rate: int) -> None:
    pcm = np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def db(value: float) -> float:
    return float(20.0 * np.log10(max(value, EPS)))


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    err = clean - test
    return float(10.0 * np.log10(np.sum(clean ** 2) / max(np.sum(err ** 2), EPS)))


def find_lag(clean: np.ndarray, test: np.ndarray, max_lag: int = MAX_LAG) -> int:
    n = min(len(clean), len(test))
    corr = signal.correlate(test[:n], clean[:n], mode="full", method="fft")
    lags = signal.correlation_lags(n, n, mode="full")
    keep = np.abs(lags) <= max_lag
    return int(lags[keep][np.argmax(corr[keep])])


def align(clean: np.ndarray, test: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        test = test[lag:]
    elif lag < 0:
        clean = clean[-lag:]
    n = min(len(clean), len(test))
    return clean[:n], test[:n]


def design_filter(rate: int, band_hz: tuple[float, float] = FILTER_BAND_HZ, order: int = FILTER_ORDER):
    return signal.butter(order, band_hz, btype="bandpass", fs=rate, output="sos")


def apply_filter(x: np.ndarray, sos) -> np.ndarray:
    return signal.sosfiltfilt(sos, x)


def phrase_id(name: str) -> str | None:
    m = re.search(r"sp\d{2}", name.lower())
    return m.group(0) if m else None


def scan_dataset(dataset: Path) -> pd.DataFrame:
    rows = []
    groups = [("clean", None, dataset / "clean")]
    groups += [("noisy", lvl, dataset / "noisy" / NOISE / f"{lvl}dB") for lvl in LEVELS]
    for kind, level, folder in groups:
        for path in sorted(folder.glob("*.wav")):
            x, info = read_wav(path)
            peak = float(np.max(np.abs(x)))
            rows.append({
                "file": path.name,
                "path": path.relative_to(dataset).as_posix(),
                "type": kind,
                "noise": NOISE if kind == "noisy" else "",
                "snr_nominal_db": level if level is not None else "",
                "phrase_id": phrase_id(path.name),
                "duration_s": round(info["duration_s"], 4),
                "sample_rate_hz": info["sample_rate_hz"],
                "channels": info["channels"],
                "bit_depth": info["bit_depth"],
                "n_samples": info["n_samples"],
                "min_amplitude": info["min_amplitude"],
                "max_amplitude": info["max_amplitude"],
                "peak_dbfs": round(db(peak), 2),
                "rms": round(rms(x), 6),
                "rms_dbfs": round(db(rms(x)), 2),
                "clipped": bool(info["max_amplitude"] >= info["full_scale"] - 1
                                or info["min_amplitude"] <= -info["full_scale"]),
            })
    table = pd.DataFrame(rows)
    clean_ids = set(table.loc[table["type"] == "clean", "phrase_id"])
    table["has_clean_pair"] = [
        True if t == "clean" else (pid in clean_ids)
        for t, pid in zip(table["type"], table["phrase_id"])
    ]
    return table


def check_dataset(table: pd.DataFrame) -> list[str]:
    notes = []
    n_clean = int((table["type"] == "clean").sum())
    n_noisy = int((table["type"] == "noisy").sum())
    notes.append(f"Файлов всего: {len(table)} (clean: {n_clean}, noisy: {n_noisy}); ожидалось 150 = 30 + 120.")
    for lvl in LEVELS:
        n = int((table["snr_nominal_db"] == lvl).sum())
        notes.append(f"  {lvl} дБ: {n} файлов.")
    noisy = table[table["type"] == "noisy"]
    notes.append(f"Шумных записей с чистой парой: {int(noisy['has_clean_pair'].sum())} из {len(noisy)}.")
    for col, label in (("sample_rate_hz", "Частота дискретизации, Гц"),
                       ("channels", "Число каналов"), ("bit_depth", "Разрядность, бит")):
        notes.append(f"{label}: {sorted(table[col].unique().tolist())}.")
    notes.append(f"Длительность: от {table['duration_s'].min():.2f} до {table['duration_s'].max():.2f} с.")
    notes.append(f"Файлов с признаком клиппинга (отсчёт на границе шкалы): {int(table['clipped'].sum())}.")
    return notes


def load_pairs(dataset: Path, table: pd.DataFrame) -> dict:
    clean_paths = {r.phrase_id: dataset / r.path for r in table[table["type"] == "clean"].itertuples()}
    cache: dict[str, tuple[np.ndarray, int]] = {}
    pairs = {}
    for r in table[table["type"] == "noisy"].itertuples():
        if r.phrase_id not in clean_paths:
            continue
        if r.phrase_id not in cache:
            c, ci = read_wav(clean_paths[r.phrase_id])
            cache[r.phrase_id] = (c, ci["sample_rate_hz"])
        clean, rate_c = cache[r.phrase_id]
        noisy, ni = read_wav(dataset / r.path)
        if ni["sample_rate_hz"] != rate_c:
            raise ValueError(f"{r.file}: частота дискретизации не совпадает с эталоном")
        lag = find_lag(clean, noisy)
        pairs[(r.phrase_id, int(r.snr_nominal_db))] = (clean, noisy, rate_c, lag, r.file)
    return pairs


def amplitude_spectrum_db(x: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    win = signal.windows.hann(len(x))
    spec = np.abs(np.fft.rfft(x * win)) / (np.sum(win) / 2.0)
    return np.fft.rfftfreq(len(x), 1.0 / rate), 20.0 * np.log10(np.maximum(spec, EPS))


def plot_phrase(pid: str, pairs: dict, plots: Path) -> None:
    clean, _, rate, _, _ = pairs[(pid, LEVELS[0])]
    series = [("чистая запись", clean)] + [
        (f"шум, {lvl} дБ", align(clean, pairs[(pid, lvl)][1], pairs[(pid, lvl)][3])[1]) for lvl in LEVELS
    ]

    ymax = 1.05 * max(np.max(np.abs(x)) for _, x in series)
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 10), sharex=True, sharey=True)
    for ax, (label, x) in zip(axes, series):
        ax.plot(np.arange(len(x)) / rate, x, lw=0.6)
        ax.set_ylim(-ymax, ymax)
        ax.set_ylabel("Амплитуда,\nотн. ед.")
        ax.set_title(f"{pid}: {label}", fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Время, с")
    fig.suptitle(f"Осциллограммы, фраза {pid} (общая шкала амплитуды)")
    fig.tight_layout()
    fig.savefig(plots / f"{pid}_waveform.png", dpi=130)
    plt.close(fig)

    f_c, s_c = amplitude_spectrum_db(clean, rate)
    fig, axes = plt.subplots(len(LEVELS), 1, figsize=(10, 10), sharex=True, sharey=True)
    for ax, (label, x) in zip(axes, series[1:]):
        f, s = amplitude_spectrum_db(x, rate)
        ax.plot(f, s, lw=0.6, label=label)
        ax.plot(f_c, s_c, lw=0.6, color="0.45", alpha=0.8, label="чистая запись")
        ax.set_ylim(-120, -20)
        ax.set_ylabel("Уровень, дБ")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Частота, Гц")
    fig.suptitle(f"Амплитудный спектр FFT, фраза {pid} (общая шкала)")
    fig.tight_layout()
    fig.savefig(plots / f"{pid}_fft.png", dpi=130)
    plt.close(fig)

    specs = []
    for label, x in series:
        f, t, sxx = signal.spectrogram(x, fs=rate, window="hann", nperseg=256, noverlap=192)
        specs.append((label, f, t, 10.0 * np.log10(np.maximum(sxx, EPS))))
    vmax = max(s.max() for *_, s in specs)
    vmin = vmax - 70.0
    fig, axes = plt.subplots(len(specs), 1, figsize=(10, 12), sharex=True, sharey=True)
    for ax, (label, f, t, s) in zip(axes, specs):
        mesh = ax.pcolormesh(t, f, s, vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
        ax.set_ylabel("Частота, Гц")
        ax.set_title(f"{pid}: {label}", fontsize=10, loc="left")
    axes[-1].set_xlabel("Время, с")
    fig.suptitle(f"Спектрограммы, фраза {pid} (общая цветовая шкала)")
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    cax = fig.add_axes((0.895, 0.1, 0.02, 0.8))
    fig.colorbar(mesh, cax=cax, label="PSD, дБ/Гц")
    fig.savefig(plots / f"{pid}_spectrogram.png", dpi=130)
    plt.close(fig)


def welch_psd(x: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    return signal.welch(x, fs=rate, window="hann", nperseg=512, noverlap=256)


def analyse_noise(pairs: dict, results: Path, plots: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    psd_noise: dict[int, list] = {lvl: [] for lvl in LEVELS}
    psd_clean: list = []
    freqs = None
    seen_clean = set()
    for (pid, lvl), (clean, noisy, rate, lag, name) in sorted(pairs.items()):
        c, n = align(clean, noisy, lag)
        noise = n - c
        freqs, p_n = welch_psd(noise, rate)
        psd_noise[lvl].append(p_n)
        if pid not in seen_clean:
            seen_clean.add(pid)
            psd_clean.append(welch_psd(c, rate)[1])
        rows.append({
            "file": name, "phrase_id": pid, "snr_nominal_db": lvl,
            "lag_samples": lag, "length_diff_samples": len(noisy) - len(clean),
            "snr_measured_db": round(snr_db(c, n), 3),
            "noise_rms_dbfs": round(db(rms(noise)), 2),
            "clean_rms_dbfs": round(db(rms(c)), 2),
        })
    snr_all = pd.DataFrame(rows)
    snr_all.to_csv(results / "snr_all_noisy.csv", index=False, encoding="utf-8-sig")

    mean_clean = np.mean(psd_clean, axis=0)
    mean_noise = {lvl: np.mean(v, axis=0) for lvl, v in psd_noise.items() if v}

    band_rows = []
    for lo, hi in BANDS_HZ:
        m = (freqs >= lo) & (freqs < hi) if hi < freqs[-1] else (freqs >= lo)
        row = {"band_hz": f"{lo}-{hi}",
               "speech_power_share_pct": round(100 * mean_clean[m].sum() / mean_clean.sum(), 1)}
        for lvl, p in mean_noise.items():
            row[f"noise_share_pct_{lvl}dB"] = round(100 * p[m].sum() / p.sum(), 1)
            row[f"band_snr_db_{lvl}dB"] = round(10 * np.log10(mean_clean[m].sum() / max(p[m].sum(), EPS)), 2)
        band_rows.append(row)
    bands = pd.DataFrame(band_rows)
    bands.to_csv(results / "noise_bands.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(freqs, 10 * np.log10(np.maximum(mean_clean, EPS)), color="k", lw=1.8, label="речь (чистые записи)")
    for lvl, p in mean_noise.items():
        ax.plot(freqs, 10 * np.log10(np.maximum(p, EPS)), lw=1.2, label=f"шум, папка {lvl} дБ")
    for i, fc in enumerate(FILTER_BAND_HZ):
        ax.axvline(fc, color="0.4", ls="--", lw=1, label="границы полосы фильтра" if i == 0 else None)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("PSD, дБ/Гц")
    ax.set_title(f"Средняя спектральная плотность мощности речи и шума «{NOISE}» (все записи)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "noise_psd_by_level.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [snr_all.loc[snr_all["snr_nominal_db"] == lvl, "snr_measured_db"] for lvl in LEVELS]
    ax.boxplot(data, tick_labels=[str(l) for l in LEVELS])
    ax.plot(range(1, len(LEVELS) + 1), LEVELS, "o--", color="0.4", label="номинал папки")
    ax.set_xlabel("Номинальный SNR папки, дБ")
    ax.set_ylabel("Измеренный SNR, дБ")
    ax.set_title("Измеренный SNR по всем шумным записям")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "snr_measured_vs_nominal.png", dpi=130)
    plt.close(fig)
    return snr_all, bands


def comment_for(delta: float, clipped: bool) -> str:
    if clipped:
        return "после фильтра есть клиппинг, результат некорректен"
    if delta >= 1.0:
        return "фильтр помог: убран шум вне основной полосы речи"
    if delta >= 0.2:
        return "небольшое улучшение"
    if delta > -0.2:
        return "эффект около нуля: выигрыш по шуму равен потерям речи"
    return "ухудшение: потери речи вне полосы больше убранного шума"


def process_selected(pairs: dict, results: Path, plots: Path, processed: Path) -> pd.DataFrame:
    rows = []
    params = (f"Butterworth bandpass, order={FILTER_ORDER}, "
              f"{FILTER_BAND_HZ[0]:.0f}-{FILTER_BAND_HZ[1]:.0f} Hz, sosfiltfilt (zero-phase)")
    for pid in PHRASES:
        fig, axes = plt.subplots(len(LEVELS), 1, figsize=(10, 11), sharex=True, sharey=True)
        for ax, lvl in zip(axes, LEVELS):
            clean, noisy, rate, lag, name = pairs[(pid, lvl)]
            sos = design_filter(rate)
            filtered = apply_filter(noisy, sos)
            out_name = f"{Path(name).stem}_{BAND_TAG}.wav"
            write_wav16(processed / out_name, filtered, rate)

            c, n = align(clean, noisy, lag)
            _, y = align(clean, filtered, lag)
            snr_before, snr_after = snr_db(c, n), snr_db(c, y)
            snr_after_ref = snr_db(apply_filter(clean, sos)[:len(c)] if lag >= 0
                                   else apply_filter(clean, sos)[-lag:][:len(c)], y)
            delta = snr_after - snr_before
            peak_after = float(np.max(np.abs(filtered)))
            clipped = peak_after >= 1.0
            base = {"phrase_id": pid, "snr_nominal_db": lvl, "lag_samples": lag}
            rows.append({**base, "state": "до", "file": name, "filter": "нет",
                         "snr_db": round(snr_before, 3), "delta_snr_db": "",
                         "snr_vs_filtered_clean_db": "",
                         "peak_dbfs": round(db(np.max(np.abs(noisy))), 2),
                         "rms_dbfs": round(db(rms(noisy)), 2), "clipping": False,
                         "comment": "исходная шумная запись"})
            rows.append({**base, "state": "после", "file": out_name, "filter": params,
                         "snr_db": round(snr_after, 3), "delta_snr_db": round(delta, 3),
                         "snr_vs_filtered_clean_db": round(snr_after_ref, 3),
                         "peak_dbfs": round(db(peak_after), 2),
                         "rms_dbfs": round(db(rms(filtered)), 2), "clipping": clipped,
                         "comment": comment_for(delta, clipped)})

            for x, label, style in ((c, "чистая", dict(color="k", lw=1.4)),
                                    (n, "до фильтра", dict(lw=1.0)),
                                    (y, "после фильтра", dict(lw=1.0))):
                f, p = welch_psd(x, rate)
                ax.plot(f, 10 * np.log10(np.maximum(p, EPS)), label=label, **style)
            for fc in FILTER_BAND_HZ:
                ax.axvline(fc, color="0.4", ls="--", lw=0.8)
            ax.set_ylabel("PSD, дБ/Гц")
            ax.set_title(f"{pid}, {lvl} дБ: ΔSNR = {delta:+.2f} дБ", fontsize=10, loc="left")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        ymax = max(l.get_ydata().max() for a in axes for l in a.get_lines()[:3])
        axes[0].set_ylim(ymax - 80, ymax + 5)
        axes[-1].set_xlabel("Частота, Гц")
        fig.suptitle(f"Спектры до и после полосового фильтра, фраза {pid}")
        fig.tight_layout()
        fig.savefig(plots / f"{pid}_before_after_psd.png", dpi=130)
        plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(results / "experiments.csv", index=False, encoding="utf-8-sig")
    return table


def plot_filter_response(rate: int, plots: Path) -> None:
    sos = design_filter(rate)
    w, h = signal.sosfreqz(sos, worN=4096, fs=rate)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(w, 40 * np.log10(np.maximum(np.abs(h), EPS)), label="итог (два прохода, нулевая фаза)")
    ax.plot(w, 20 * np.log10(np.maximum(np.abs(h), EPS)), ls="--", label="один проход")
    for i, fc in enumerate(FILTER_BAND_HZ):
        ax.axvline(fc, color="0.4", ls=":", label="частоты среза" if i == 0 else None)
    ax.set_xlim(0, rate / 2)
    ax.set_ylim(-100, 5)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Усиление, дБ")
    ax.set_title(f"АЧХ полосового фильтра Баттерворта {FILTER_BAND_HZ[0]:.0f}-{FILTER_BAND_HZ[1]:.0f} Гц, порядок {FILTER_ORDER}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "filter_response.png", dpi=130)
    plt.close(fig)


def delta_snr(pairs: dict, key: tuple, band_hz: tuple[float, float]) -> float:
    clean, noisy, rate, lag, _ = pairs[key]
    y = apply_filter(noisy, design_filter(rate, band_hz))
    c, n = align(clean, noisy, lag)
    _, y = align(clean, y, lag)
    return snr_db(c, y) - snr_db(c, n)


def sweep_cutoff(pairs: dict, results: Path, plots: Path) -> pd.DataFrame:
    variants = [(lo, FILTER_BAND_HZ[1]) for lo in SWEEP_LOW_HZ] + [(FILTER_BAND_HZ[0], hi) for hi in SWEEP_HIGH_HZ]
    rows = []
    for band in dict.fromkeys(variants):
        row = {"low_hz": band[0], "high_hz": band[1]}
        every = []
        for lvl in LEVELS:
            deltas = [delta_snr(pairs, (pid, lvl), band) for pid in PHRASES]
            row[f"mean_delta_snr_db_{lvl}dB"] = round(float(np.mean(deltas)), 3)
            every += deltas
        row["mean_delta_snr_db_all"] = round(float(np.mean(every)), 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(results / "sweep_cutoff.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    parts = ((axes[0], table[table["high_hz"] == FILTER_BAND_HZ[1]], "low_hz",
              f"Нижняя частота среза, Гц (верхняя {FILTER_BAND_HZ[1]:.0f} Гц)"),
             (axes[1], table[table["low_hz"] == FILTER_BAND_HZ[0]], "high_hz",
              f"Верхняя частота среза, Гц (нижняя {FILTER_BAND_HZ[0]:.0f} Гц)"))
    for ax, part, col, label in parts:
        part = part.sort_values(col)
        for lvl in LEVELS:
            ax.plot(part[col], part[f"mean_delta_snr_db_{lvl}dB"], "o-", label=f"{lvl} дБ")
        ax.plot(part[col], part["mean_delta_snr_db_all"], "k--", lw=2, label="среднее")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Средний ΔSNR, дБ")
    axes[0].legend()
    fig.suptitle("Влияние границ полосы фильтра на ΔSNR (12 выбранных записей)")
    fig.tight_layout()
    fig.savefig(plots / "sweep_cutoff.png", dpi=130)
    plt.close(fig)
    return table


def filter_all(pairs: dict, results: Path) -> pd.DataFrame:
    rows = [{"phrase_id": pid, "snr_nominal_db": lvl,
             "delta_snr_db": round(delta_snr(pairs, (pid, lvl), FILTER_BAND_HZ), 3)}
            for (pid, lvl) in sorted(pairs)]
    table = pd.DataFrame(rows)
    table.to_csv(results / "filter_all_noisy.csv", index=False, encoding="utf-8-sig")
    return table


def write_summary(results: Path, notes: list[str], snr_all: pd.DataFrame, bands: pd.DataFrame,
                  exp: pd.DataFrame, sweep: pd.DataFrame, every: pd.DataFrame) -> str:
    lines = ["ЛР №3, вариант 05 «Городская улица» (street). Сводка измерений", "",
             "1. Проверка датасета"] + ["   " + n for n in notes]
    lines += ["", "2. Измеренный SNR по всем шумным записям (среднее ± стандартное отклонение)"]
    for lvl in LEVELS:
        s = snr_all.loc[snr_all["snr_nominal_db"] == lvl, "snr_measured_db"]
        lines.append(f"   папка {lvl:>2} дБ: {s.mean():6.2f} ± {s.std():.2f} дБ (мин {s.min():.2f}, макс {s.max():.2f})")
    lines.append(f"   Ненулевой сдвиг найден у {int((snr_all['lag_samples'] != 0).sum())} записей; "
                 f"разная длина с эталоном у {int((snr_all['length_diff_samples'] != 0).sum())}.")
    lines += ["", "3. Распределение мощности по полосам (доля, %) и SNR в полосе, дБ",
              bands.to_string(index=False)]
    lines += ["", f"4. Фильтр: полосовой Баттерворта, порядок {FILTER_ORDER}, полоса "
                  f"{FILTER_BAND_HZ[0]:.0f}-{FILTER_BAND_HZ[1]:.0f} Гц, фильтрация вперёд-назад (sosfiltfilt).",
              "", "5. SNR до и после на 12 выбранных записях"]
    after = exp[exp["state"] == "после"]
    before = exp[exp["state"] == "до"].set_index(["phrase_id", "snr_nominal_db"])["snr_db"]
    for r in after.itertuples():
        lines.append(f"   {r.phrase_id} {r.snr_nominal_db:>2} дБ: {before[(r.phrase_id, r.snr_nominal_db)]:6.2f} -> "
                     f"{r.snr_db:6.2f} дБ, ΔSNR {float(r.delta_snr_db):+.2f} дБ; {r.comment}")
    d = after["delta_snr_db"].astype(float)
    lines.append(f"   Средний ΔSNR: {d.mean():+.2f} дБ; улучшение у {int((d > 0).sum())} из {len(d)} записей; "
                 f"клиппинг после фильтра: {int(after['clipping'].sum())} записей.")
    for lvl in LEVELS:
        dl = after.loc[after["snr_nominal_db"] == lvl, "delta_snr_db"].astype(float)
        lines.append(f"   средний ΔSNR при {lvl:>2} дБ: {dl.mean():+.2f} дБ")
    best = sweep.loc[sweep["mean_delta_snr_db_all"].idxmax()]
    lines += ["", "6. Проверка границ полосы на 12 записях (средний ΔSNR, дБ)", sweep.to_string(index=False),
              f"   Наибольший средний ΔSNR: полоса {int(best['low_hz'])}-{int(best['high_hz'])} Гц "
              f"({best['mean_delta_snr_db_all']:+.2f} дБ).",
              "", "7. Тот же фильтр на всех шумных записях (средний ΔSNR, дБ; доля записей с улучшением)"]
    for lvl in LEVELS:
        dl = every.loc[every["snr_nominal_db"] == lvl, "delta_snr_db"]
        lines.append(f"   папка {lvl:>2} дБ: {dl.mean():+.2f} дБ; улучшение у {int((dl > 0).sum())} из {len(dl)}")
    text = "\n".join(lines) + "\n"
    (results / "summary.txt").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="ЛР №3, вариант 05 (street)")
    parser.add_argument("--dataset", type=Path, default=here / "dataset", help="папка dataset/ от загрузчика")
    parser.add_argument("--out", type=Path, default=here / "results", help="папка для результатов")
    args = parser.parse_args()
    dataset, results = args.dataset.resolve(), args.out.resolve()
    plots, processed = results / "plots", results / "processed"
    for folder in (results, plots, processed):
        folder.mkdir(parents=True, exist_ok=True)
    if not (dataset / "clean").is_dir():
        print(f"Не найдена папка {dataset / 'clean'}. Сначала запустите download_dataset.py.")
        return 1

    print("Этап 1: проверка датасета")
    table = scan_dataset(dataset)
    table.to_csv(results / "files_table.csv", index=False, encoding="utf-8-sig")
    notes = check_dataset(table)
    print("\n".join(notes))

    pairs = load_pairs(dataset, table)
    missing = [(p, l) for p in PHRASES for l in LEVELS if (p, l) not in pairs]
    if missing:
        print(f"Нет записей для выбранных фраз: {missing}")
        return 1

    print("Этап 2: графики для фраз", ", ".join(PHRASES))
    for pid in PHRASES:
        plot_phrase(pid, pairs, plots)

    print("Этап 3: анализ шума по всем шумным записям")
    snr_all, bands = analyse_noise(pairs, results, plots)

    print("Этапы 4-5: фильтрация 12 записей и оценка SNR")
    plot_filter_response(pairs[(PHRASES[0], LEVELS[0])][2], plots)
    exp = process_selected(pairs, results, plots, processed)
    sweep = sweep_cutoff(pairs, results, plots)
    every = filter_all(pairs, results)

    print()
    print(write_summary(results, notes, snr_all, bands, exp, sweep, every))
    print(f"Готово. Результаты: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
