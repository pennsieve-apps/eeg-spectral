"""
Spectral biomarkers of scalp EEG, per recording:

1. Welch PSD (MNE `psd_array_welch`: 4 s Hamming windows, 50 % overlap, 0.25 Hz bins) of every
   window that lies in an analysable interval (no `bad_segment`/`artifact` event, no seizure
   and SEIZURE_MARGIN_S around it), read a chunk at a time; each window's spectrum is kept so
   that windows can be rejected afterwards and grouped (eyes closed / open, WINDOWED blocks).
2. Automatic rejection (REJECT=true): channels flat or above REJECT_UV peak-to-peak in most
   windows are left out; windows where several channels exceed REJECT_UV, every channel is
   flat, or whose median 20-45 Hz power (muscle) is a robust outlier (z > 4 against the
   recording's own windows) are left out for every channel (high-frequency power, not
   broadband, so that sleep slow waves of a long recording are not taken for artifacts); a window where a single channel exceeds
   REJECT_UV (an electrode pop) is left out for that channel only. On eeg-prep output this
   removes little; on raw clinical exports it keeps pops and movement out of the averages.
3. Band power per channel (absolute µV², relative to broadband, log10), slowing ratios.
4. specparam / FOOOF (Donoghue et al. 2020) on each channel's mean PSD: aperiodic exponent and
   offset (fixed mode by default), periodic peaks, r_squared; fits below MIN_R_SQUARED flagged.
5. Posterior dominant rhythm (Lodder & van Putten 2011): the strongest specparam peak in
   PDR_RANGE (6-14 Hz) of the mean spectrum of O1/O2 (P3/P4 when O1/O2 are missing; P3-O1/P4-O2
   etc. on bipolar recordings), on eyes-closed windows when they are annotated, else on all
   kept windows that are not annotated sleep. Left and right separately for the asymmetry;
   reactivity = attenuation of PDR-band power with eyes open when both states are annotated.
6. Revised Brain Symmetry Index (van Putten 2007): for each frequency bin the mean power of the
   right and of the left channels of the homologous pairs, |R - L| / (R + L), averaged over the
   bins of 1-25 Hz (band `rbsi`), and over each band's bins; per homologous pair the same
   expression on the two channels.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

SPECTRAL_VERSION = "1.0"

BAND_PRESETS: dict[str, tuple[tuple[str, float, float], ...]] = {
    # conventions.md (scalp): delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45 Hz
    "conventions": (("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
                    ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0)),
}
BAND_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
RBSI_RANGE = (1.0, 25.0)          # van Putten 2007
FLAT_UV = 0.5                     # peak-to-peak below this in a 4 s window = flat
ROBUST_Z = 4.0                    # window rejection: median 20-45 Hz log power above median + 4 * 1.4826 MAD
MUSCLE_BAND = (20.0, 45.0)        # the band of that test (muscle artifact); cut to the fit range's top
STORE_IN_MEMORY_MB = 256          # per-window spectra above this go to a memmap in $TMPDIR


def parse_bands(value: str | None) -> tuple[tuple[str, float, float], ...]:
    """A preset name (conventions) or a JSON object {"delta": [1, 4], ...}."""
    v = (value or "conventions").strip()
    if v in BAND_PRESETS:
        return BAND_PRESETS[v]
    try:
        obj = json.loads(v)
    except json.JSONDecodeError as e:
        raise ValueError(f"BANDS must be one of {', '.join(BAND_PRESETS)} or a JSON object: {e}") from None
    if not isinstance(obj, dict) or not obj:
        raise ValueError("BANDS JSON must be a non-empty object of name: [low_hz, high_hz]")
    out = []
    for name, edges in obj.items():
        if not BAND_NAME_RE.match(name) or name in ("broadband", "rbsi"):
            raise ValueError(f"band name {name!r}: lowercase letters, digits and -, not 'broadband' or 'rbsi'")
        if not (isinstance(edges, (list, tuple)) and len(edges) == 2):
            raise ValueError(f"band {name}: [low_hz, high_hz]")
        lo, hi = float(edges[0]), float(edges[1])
        if not (0 < lo < hi):
            raise ValueError(f"band {name}: need 0 < low < high, got {lo}, {hi}")
        out.append((name, lo, hi))
    return tuple(sorted(out, key=lambda b: (b[1], b[2])))


def parse_range(name: str, value: str) -> tuple[float, float]:
    m = re.match(r"^\s*([0-9.]+)\s*[-,:]\s*([0-9.]+)\s*$", str(value))
    if not m:
        raise ValueError(f"{name} must look like 1-45, got {value!r}")
    lo, hi = float(m.group(1)), float(m.group(2))
    if not 0 <= lo < hi:
        raise ValueError(f"{name}: need 0 <= low < high, got {value!r}")
    return lo, hi


@dataclass(frozen=True)
class Params:
    window_s: float = 4.0
    overlap: float = 0.5
    bands: tuple[tuple[str, float, float], ...] = BAND_PRESETS["conventions"]
    fit_range: tuple[float, float] = (1.0, 45.0)
    aperiodic_mode: str = "fixed"
    max_n_peaks: int = 6
    peak_width_limits: tuple[float, float] = (1.0, 8.0)
    min_r_squared: float = 0.9
    pdr_range: tuple[float, float] = (6.0, 14.0)
    reject: bool = True
    reject_uv: float = 500.0
    chunk_s: float = 60.0
    windowed: bool = False
    windowed_s: float = 60.0

    def __post_init__(self):
        if self.window_s < 1:
            raise ValueError("WELCH_WINDOW_S must be at least 1 s")
        if not 0 <= self.overlap < 1:
            raise ValueError("WELCH_OVERLAP must be in [0, 1)")
        if self.aperiodic_mode not in ("fixed", "knee"):
            raise ValueError("APERIODIC_MODE must be fixed or knee")
        if self.max_n_peaks < 0:
            raise ValueError("MAX_N_PEAKS must be >= 0")
        if self.fit_range[0] <= 0:
            raise ValueError("FIT_RANGE must start above 0 Hz")
        if self.reject_uv <= 0:
            raise ValueError("REJECT_UV must be > 0")
        if self.windowed and self.windowed_s < self.window_s:
            raise ValueError("WINDOWED_S must be at least WELCH_WINDOW_S")

    @property
    def broadband(self) -> tuple[float, float]:
        return (min(b[1] for b in self.bands), max(b[2] for b in self.bands))

    @property
    def fmax(self) -> float:
        return max(self.fit_range[1], self.broadband[1], self.pdr_range[1], RBSI_RANGE[1])

    def as_dict(self) -> dict:
        return {"psd": "MNE psd_array_welch", "welch_window_s": self.window_s, "welch_overlap": self.overlap,
                "welch_window": "hamming", "bands": {n: [lo, hi] for n, lo, hi in self.bands},
                "broadband_hz": list(self.broadband), "specparam": {
                    "package": "fooof", "freq_range": list(self.fit_range), "aperiodic_mode": self.aperiodic_mode,
                    "peak_width_limits": list(self.peak_width_limits), "max_n_peaks": self.max_n_peaks,
                    "min_r_squared": self.min_r_squared},
                "pdr_range_hz": list(self.pdr_range), "rbsi_range_hz": list(RBSI_RANGE),
                "reject": self.reject, "reject_uv": self.reject_uv, "flat_uv": FLAT_UV, "robust_z": ROBUST_Z,
                "robust_z_band_hz": list(MUSCLE_BAND),
                "spectral_version": SPECTRAL_VERSION}


# --------------------------------------------------------------------------- windows

def subtract(intervals: list[tuple[float, float]], cut: tuple[float, float]) -> list[tuple[float, float]]:
    out = []
    c0, c1 = cut
    for a, b in intervals:
        if c1 <= a or c0 >= b:
            out.append((a, b))
            continue
        if c0 > a:
            out.append((a, c0))
        if c1 < b:
            out.append((c1, b))
    return out


def analysable(duration: float, bad: Sequence[tuple[float, float]], seizures: Sequence[tuple[float, float | None]],
               exclude_seizure: bool, margin: float) -> list[tuple[float, float]]:
    """The recording minus bad segments/artifacts and (optionally) seizures with a margin.
    A seizure without a known end is cut to the end of the recording."""
    iv = [(0.0, duration)]
    for a, b in bad:
        iv = subtract(iv, (a, b))
    if exclude_seizure:
        for on, dur in seizures:
            iv = subtract(iv, (on - margin, (on + dur + margin) if dur else duration))
    return [(a, b) for a, b in iv if b > a]


def window_starts(intervals_samples: Sequence[tuple[int, int]], nper: int, step: int) -> list[np.ndarray]:
    """Per interval, the start samples of the whole windows inside it."""
    out = []
    for a, b in intervals_samples:
        if b - a < nper:
            out.append(np.zeros(0, dtype=np.int64))
        else:
            out.append(a + np.arange(0, b - a - nper + 1, step, dtype=np.int64))
    return out


@dataclass
class Windows:
    """Per-window spectra of one recording: psd (n_win, n_ch, n_f) µV²/Hz, ptp (n_win, n_ch) µV."""
    freqs: np.ndarray
    starts_s: np.ndarray
    psd: np.ndarray
    ptp: np.ndarray
    window_s: float
    keep: np.ndarray = None  # type: ignore[assignment]       (n_win,) window kept for every channel
    ch_keep: np.ndarray = None  # type: ignore[assignment]    (n_win, n_ch) window kept for that channel
    channel_ok: np.ndarray = None  # type: ignore[assignment]
    _tmp: object = None

    def close(self):
        if self._tmp is not None:
            self.psd = None  # type: ignore[assignment]
            self._tmp.cleanup()  # type: ignore[attr-defined]
            self._tmp = None


def compute_windows(read: Callable[[int, int], np.ndarray], sfreq: float, n_ch: int,
                    intervals_s: Sequence[tuple[float, float]], params: Params,
                    progress: Callable[[int, int], None] | None = None) -> Windows:
    """Welch spectra of every whole window in the intervals, CHUNK_S at a time.
    read(start, stop) returns (n_ch, stop - start) µV."""
    from mne.time_frequency import psd_array_welch

    nper = int(round(params.window_s * sfreq))
    step = max(1, nper - int(round(nper * params.overlap)))
    ivs = [(int(math.ceil(a * sfreq)), int(math.floor(b * sfreq))) for a, b in intervals_s]
    starts = window_starts(ivs, nper, step)
    n_win = int(sum(len(s) for s in starts))
    freqs_all = np.fft.rfftfreq(nper, 1 / sfreq)
    fmask = freqs_all <= params.fmax + 1e-9
    freqs = freqs_all[fmask]
    nbytes = n_win * n_ch * len(freqs) * 4
    tmp = None
    if nbytes > STORE_IN_MEMORY_MB * 2**20:
        tmp = tempfile.TemporaryDirectory(prefix="eeg-spectral-")
        psd = np.lib.format.open_memmap(f"{tmp.name}/psd.npy", mode="w+", dtype=np.float32,
                                        shape=(n_win, n_ch, len(freqs)))
    else:
        psd = np.zeros((n_win, n_ch, len(freqs)), dtype=np.float32)
    ptp = np.zeros((n_win, n_ch), dtype=np.float32)
    all_starts = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
    per_chunk = max(1, int((params.chunk_s * sfreq - nper) // step) + 1)
    w = 0
    for s in starts:
        for c0 in range(0, len(s), per_chunk):
            block = s[c0: c0 + per_chunk]
            a, b = int(block[0]), int(block[-1]) + nper
            x = np.asarray(read(a, b), dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = psd_array_welch(x, sfreq, fmin=0.0, fmax=float(freqs[-1]), n_fft=nper, n_per_seg=nper,
                                       n_overlap=nper - step, window="hamming", average=None, verbose="error")
            # p: (n_ch, n_f, n_seg); n_seg == len(block) by construction
            psd[w: w + len(block)] = np.moveaxis(p[:, : len(freqs), : len(block)], -1, 0)
            idx = (block - a)[:, None] + np.arange(nper)[None, :]
            seg = x[:, idx]                                   # (n_ch, m, nper)
            ptp[w: w + len(block)] = np.ptp(seg, axis=-1).T
            w += len(block)
            if progress:
                progress(w, n_win)
    return Windows(freqs, all_starts / sfreq, psd, ptp, params.window_s, _tmp=tmp)


def reject(win: Windows, params: Params) -> dict:
    """Sets win.channel_ok and win.keep; returns counts for provenance."""
    n_win, n_ch = win.ptp.shape
    info = {"windows": int(n_win)}
    if n_win == 0:
        win.channel_ok = np.ones(n_ch, bool)
        win.keep = np.zeros(0, bool)
        win.ch_keep = np.zeros((0, n_ch), bool)
        return info
    if not params.reject:
        win.channel_ok = np.ones(n_ch, bool)
        win.keep = np.ones(n_win, bool)
        win.ch_keep = np.ones((n_win, n_ch), bool)
        info.update(rejected_windows=0, rejected_channels=[], rejected_channel_windows=0)
        return info
    high = win.ptp > params.reject_uv
    flat = win.ptp < FLAT_UV
    ch_bad = (high.mean(axis=0) > 0.5) | (flat.mean(axis=0) > 0.5)
    ok = ~ch_bad
    win.channel_ok = ok
    keep = np.ones(n_win, bool)
    if ok.any():
        n_ok = int(ok.sum())
        keep &= high[:, ok].sum(axis=1) < max(2, math.ceil(0.1 * n_ok))
        keep &= ~flat[:, ok].all(axis=1)
        hi = min(MUSCLE_BAND[1], params.fit_range[1])
        bb = (win.freqs >= min(MUSCLE_BAND[0], hi - 5)) & (win.freqs <= hi)
        lp = np.zeros(n_win)
        for i in range(0, n_win, 2048):
            x = np.asarray(win.psd[i: i + 2048][:, ok][:, :, bb], dtype=np.float64).mean(axis=2)
            lp[i: i + 2048] = np.log10(np.maximum(np.median(x, axis=1), 1e-30))
        med = np.median(lp[keep]) if keep.any() else np.median(lp)
        mad = 1.4826 * np.median(np.abs(lp - med)) if keep.any() else 0.0
        if mad > 0:
            keep &= (lp - med) / mad <= ROBUST_Z
    else:
        keep[:] = False
    win.keep = keep
    win.ch_keep = keep[:, None] & ~high & ~flat
    info.update(rejected_windows=int((~keep).sum()), rejected_channels=[int(i) for i in np.flatnonzero(ch_bad)],
                rejected_channel_windows=int((keep[:, None] & ~win.ch_keep)[:, ok].sum()))
    return info


def inside(starts: np.ndarray, length: float, intervals: Sequence[tuple[float, float]]) -> np.ndarray:
    """Windows lying wholly inside any of the intervals."""
    m = np.zeros(len(starts), bool)
    for a, b in intervals:
        m |= (starts >= a - 1e-9) & (starts + length <= b + 1e-9)
    return m


def overlapping(starts: np.ndarray, length: float, intervals: Sequence[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(starts), bool)
    for a, b in intervals:
        m |= (starts < b) & (starts + length > a)
    return m


def alpha_score(win: Windows, post: list[int], ant: list[int], params: Params) -> np.ndarray:
    """Per window: share of PDR_RANGE power in the broadband power at the posterior (PDR) channels,
    minus the same share at the frontal channels when there are any (the PDR is posterior;
    frontal alpha, spindles and eye artifacts are not). Higher = more like eyes-closed wake."""
    f = win.freqs
    pr = (f >= params.pdr_range[0]) & (f <= params.pdr_range[1])
    bb = (f >= params.broadband[0]) & (f <= params.broadband[1])

    def share(chs: list[int]) -> np.ndarray:
        out = np.zeros(len(win.starts_s))
        for i in range(0, len(out), 2048):
            x = np.asarray(win.psd[i: i + 2048][:, chs], dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[i: i + 2048] = np.nanmean(x[..., pr].sum(-1) / x[..., bb].sum(-1), axis=1)
        return out

    s = share(post)
    if ant:
        s = s - share(ant)
    return np.nan_to_num(s, nan=-np.inf)


def top_fraction(score: np.ndarray, mask: np.ndarray, fraction: float, minimum: int) -> np.ndarray:
    """The windows of mask whose score is in the top `fraction` (at least `minimum` windows)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask.copy()
    n = min(len(idx), max(minimum, int(math.ceil(fraction * len(idx)))))
    best = idx[np.argsort(score[idx])[::-1][:n]]
    out = np.zeros_like(mask)
    out[best] = True
    return out


def covered_seconds(starts: np.ndarray, length: float) -> float:
    """Length of the union of windows [s, s + length)."""
    if len(starts) == 0:
        return 0.0
    s = np.sort(starts)
    total, a, b = 0.0, s[0], s[0] + length
    for x in s[1:]:
        if x <= b:
            b = max(b, x + length)
        else:
            total += b - a
            a, b = x, x + length
    return float(total + b - a)


def mean_psd(win: Windows, mask: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Mean spectrum (n_ch, n_f) over the windows in mask (float64). mask (n_win,) selects windows;
    it is combined with win.ch_keep (when set) so that each channel leaves out its own bad windows.
    A channel without any window falls back to the windows of mask."""
    mask = np.asarray(mask, bool)
    w = mask[:, None] & (win.ch_keep if win.ch_keep is not None else True)
    empty = w.sum(axis=0) == 0
    if empty.any():
        w[:, empty] = mask[:, None]
    idx = np.flatnonzero(w.any(axis=1))
    acc = np.zeros(win.psd.shape[1:], dtype=np.float64)
    for i in range(0, len(idx), chunk):
        sel = idx[i: i + chunk]
        acc += np.einsum("wcf,wc->cf", np.asarray(win.psd[sel], dtype=np.float64), w[sel].astype(np.float64))
    return acc / np.maximum(1, w.sum(axis=0))[:, None]


# --------------------------------------------------------------------------- band power

def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """PSD integrated over [lo, hi] (trapezoid, edges included), per channel."""
    m = (freqs >= lo - 1e-9) & (freqs <= hi + 1e-9)
    if m.sum() < 2:
        raise ValueError(f"band {lo:g}-{hi:g} Hz has fewer than two frequency bins")
    return np.trapezoid(psd[..., m], freqs[m], axis=-1)


@dataclass
class BandTable:
    names: list[str]          # bands then 'broadband'
    abs_power: np.ndarray     # (n_ch, n_bands + 1) µV²
    rel_power: np.ndarray
    log_power: np.ndarray


def band_table(freqs: np.ndarray, psd: np.ndarray, params: Params) -> BandTable:
    psd = np.atleast_2d(psd)
    names = [b[0] for b in params.bands] + ["broadband"]
    edges = [(lo, hi) for _, lo, hi in params.bands] + [params.broadband]
    absp = np.stack([band_power(freqs, psd, lo, hi) for lo, hi in edges], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(absp[:, -1:] > 0, absp / absp[:, -1:], np.nan)
        logp = np.log10(np.where(absp > 0, absp, np.nan))
    return BandTable(names, absp, rel, logp)


def slowing(bt: BandTable) -> tuple[np.ndarray, np.ndarray]:
    """theta/alpha and (delta+theta)/(alpha+beta) per channel (NaN when a band is not defined)."""
    col = {n: i for i, n in enumerate(bt.names)}
    n = bt.abs_power.shape[0]
    nan = np.full(n, np.nan)
    p = bt.abs_power
    with np.errstate(invalid="ignore", divide="ignore"):
        ta = p[:, col["theta"]] / p[:, col["alpha"]] if {"theta", "alpha"} <= col.keys() else nan
        dt = ((p[:, col["delta"]] + p[:, col["theta"]]) / (p[:, col["alpha"]] + p[:, col["beta"]])
              if {"delta", "theta", "alpha", "beta"} <= col.keys() else nan)
    return ta, dt


# --------------------------------------------------------------------------- specparam

@dataclass
class Fit:
    ok: bool
    exponent: float = math.nan
    offset: float = math.nan
    knee: float = math.nan
    r_squared: float = math.nan
    error: float = math.nan
    peaks: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # cf, power (log10 above aperiodic), bandwidth
    freqs: np.ndarray | None = None
    aperiodic_log: np.ndarray | None = None   # log10 aperiodic component over freqs
    fit_range: tuple[float, float] = (math.nan, math.nan)
    message: str = ""


def fit_spectrum(freqs: np.ndarray, psd: np.ndarray, params: Params, fit_range: tuple[float, float] | None = None) -> Fit:
    """specparam (FOOOF 1.1) on one spectrum in µV²/Hz."""
    with warnings.catch_warnings(record=True):  # fooof 1.1 announces its rename to specparam on import,
        from fooof import FOOOF                  # after setting simplefilter('always'): record, don't show

    fr = fit_range or params.fit_range
    m = (freqs >= fr[0] - 1e-9) & (freqs <= fr[1] + 1e-9)
    if m.sum() < 10 or not np.all(np.isfinite(psd[m])) or np.any(psd[m] <= 0):
        return Fit(False, message="spectrum empty, non-positive or too short for a fit", fit_range=fr)
    fm = FOOOF(peak_width_limits=params.peak_width_limits, max_n_peaks=params.max_n_peaks,
               aperiodic_mode=params.aperiodic_mode, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fm.fit(freqs, psd, list(fr))
        except Exception as e:  # noqa: BLE001
            return Fit(False, message=f"fit failed: {e}", fit_range=fr)
    if not fm.has_model:
        return Fit(False, message="fit failed", fit_range=fr)
    ap = fm.aperiodic_params_
    if params.aperiodic_mode == "knee":
        offset, knee, exponent = ap
    else:
        (offset, exponent), knee = ap, math.nan
    return Fit(True, float(exponent), float(offset), float(knee), float(fm.r_squared_), float(fm.error_),
               np.asarray(fm.peak_params_, float).reshape(-1, 3), fm.freqs, np.asarray(fm._ap_fit), fr)


def strongest_peak(fit: Fit, lo: float, hi: float) -> np.ndarray | None:
    if not fit.ok or len(fit.peaks) == 0:
        return None
    p = fit.peaks[(fit.peaks[:, 0] >= lo) & (fit.peaks[:, 0] <= hi)]
    return p[np.argmax(p[:, 1])] if len(p) else None


def peak_amplitude_uv(freqs: np.ndarray, psd: np.ndarray, fit: Fit, peak: np.ndarray) -> float:
    """Peak-to-peak amplitude (µV) of a sinusoid with the power the peak adds above the aperiodic
    fit within cf ± bandwidth: 2 * sqrt(2) * sqrt(excess power)."""
    cf, _, bw = peak
    m = (fit.freqs >= cf - bw) & (fit.freqs <= cf + bw)
    if m.sum() < 2:
        return math.nan
    f = fit.freqs[m]
    p = np.interp(f, freqs, psd)
    excess = np.clip(p - 10 ** fit.aperiodic_log[m], 0, None)
    return float(2 * math.sqrt(2) * math.sqrt(np.trapezoid(excess, f)))


# --------------------------------------------------------------------------- asymmetry

def bsi(right: np.ndarray, left: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """(unsigned BSI, signed mean (R - L)/(R + L)) over the bins of [lo, hi]; right/left are spectra."""
    m = (freqs >= lo - 1e-9) & (freqs <= hi + 1e-9)
    r, l_ = right[m], left[m]
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (r - l_) / (r + l_)
    q = q[np.isfinite(q)]
    if len(q) == 0:
        return math.nan, math.nan
    return float(np.mean(np.abs(q))), float(np.mean(q))


def asymmetry_bands(params: Params) -> list[tuple[str, float, float]]:
    return list(params.bands) + [("rbsi", *RBSI_RANGE)]
