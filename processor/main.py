"""
eeg-spectral: spectral biomarkers of clinical scalp EEG.

  INPUT_DIR   the eeg-prep BIDS-EEG derivative (recordings.csv + EDF + sidecars), any BIDS-EEG
              dataset, or loose EDF/BDF/BrainVision/FIF/EEGLAB files; found recursively
  OUTPUT_DIR/eeg-spectral[/LABEL]/
      bandpower.csv  aperiodic.csv  peaks.csv  pdr.csv  asymmetry.csv  slowing.csv  features.csv
      windowed.csv (WINDOWED=true)  figures/*.png  summary.json  provenance.json

One recording at a time, read a chunk at a time; a recording that fails is logged and skipped.
Exit codes: 0 ok, 1 no usable input / every recording failed, 2 bad parameters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepare_environment() -> None:
    """Lambda-safe start: HOME and TMPDIR may be fresh, empty folders and everything else
    read-only. Writable caches go under TMPDIR; nothing is written to the image."""
    tmp = Path(os.environ.get("TMPDIR") or "/tmp")
    for var, default in (("TMPDIR", tmp), ("HOME", tmp / "home")):
        p = Path(os.environ.get(var) or default)
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        os.environ[var] = str(p)
    mpl = Path(os.environ["TMPDIR"]) / "mpl"
    baked = Path(os.environ.get("EEG_SPECTRAL_MPL_CACHE", "/opt/eeg-spectral/matplotlib"))
    if not os.environ.get("MPLCONFIGDIR") or not os.access(os.environ["MPLCONFIGDIR"], os.W_OK):
        mpl.mkdir(parents=True, exist_ok=True)
        if baked.is_dir():  # the font cache built at image build time: copy instead of rebuilding
            for f in baked.glob("fontlist-*.json"):
                target = mpl / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())
        os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_USE_CUDA", "false")


_prepare_environment()

from processor import resources as R  # noqa: E402  (no numpy import)

# Threads before numpy: THREADS=0 -> from the cgroup CPU quota, capped by memory.
_requested = os.environ.get("THREADS", "0").strip() or "0"
try:
    _THREADS, _THREADS_WHY = R.resolve_workers(int(_requested), per_worker_mb=300, reserve_mb=700)
except ValueError:
    _THREADS, _THREADS_WHY = 1, f"THREADS={_requested!r} is not an integer"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402

import numpy as np  # noqa: E402

from processor import channels as C  # noqa: E402
from processor import figures as F  # noqa: E402
from processor import inputs as I  # noqa: E402
from processor import spectral as S  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("eeg-spectral")
log.setLevel(logging.INFO)
for _noisy in ("mne", "matplotlib", "fooof"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

APP_NAME = "eeg-spectral"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRUE, FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
MIN_SFREQ = 100.0
MIN_PDR_WINDOWS = 5          # eyes-closed windows needed before they are used for the PDR (~12 s)
PDR_TOP_FRACTION = 0.25      # PDR_WINDOWS=alpha-richest without eyes-closed annotation: the alpha-richest quarter
PDR_MODES = ("auto", "alpha-richest")

CITATIONS = ["10.1038/s41593-020-00744-x", "10.1016/j.jneumeth.2011.06.008", "10.1016/j.cnp.2020.11.001",
             "10.1016/j.clinph.2007.07.019", "10.3389/fnins.2013.00267"]

DEFAULTS = {
    "BANDS": "conventions", "WELCH_WINDOW_S": "4", "WELCH_OVERLAP": "0.5", "FIT_RANGE": "1-45",
    "APERIODIC_MODE": "fixed", "MAX_N_PEAKS": "6", "PEAK_WIDTH_LIMITS": "1-8", "MIN_R_SQUARED": "0.9",
    "PDR_RANGE": "6-14", "PDR_WINDOWS": "auto", "EXCLUDE_SEIZURE": "true", "SEIZURE_MARGIN_S": "60", "REJECT": "true",
    "REJECT_UV": "500", "WINDOWED": "false", "WINDOWED_S": "60", "FIGURES": "true", "CHUNK_S": "60",
    "THREADS": "0", "LABEL": "",
}

HEADERS = {
    "bandpower": ["recording_id", "channel", "band", "abs_power", "rel_power", "log_power"],
    "aperiodic": ["recording_id", "channel", "exponent", "offset", "knee", "r_squared", "error", "n_peaks", "flag"],
    "peaks": ["recording_id", "channel", "cf_hz", "power", "bandwidth"],
    "pdr": ["recording_id", "pdr_hz", "pdr_amplitude_uv", "pdr_left_hz", "pdr_right_hz", "asymmetry", "reactivity",
            "n_eyes_closed_s", "method"],
    "asymmetry": ["recording_id", "pair", "band", "bsi", "signed"],
    "slowing": ["recording_id", "channel", "theta_alpha_ratio", "delta_theta_over_alpha_beta"],
    "features": ["recording_id", "channel", "feature", "value"],
    "windowed": ["recording_id", "channel", "window_start_s", "feature", "value"],
}


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("mne", "fooof", "numpy", "scipy", "matplotlib"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def _bool(name: str, v: str) -> bool:
    s = str(v).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {v!r}")


def read_params(env=os.environ) -> tuple[dict, S.Params, dict]:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    try:
        params = S.Params(
            window_s=float(raw["WELCH_WINDOW_S"]), overlap=float(raw["WELCH_OVERLAP"]),
            bands=S.parse_bands(raw["BANDS"]), fit_range=S.parse_range("FIT_RANGE", raw["FIT_RANGE"]),
            aperiodic_mode=raw["APERIODIC_MODE"].strip().lower(), max_n_peaks=int(raw["MAX_N_PEAKS"]),
            peak_width_limits=S.parse_range("PEAK_WIDTH_LIMITS", raw["PEAK_WIDTH_LIMITS"]),
            min_r_squared=float(raw["MIN_R_SQUARED"]), pdr_range=S.parse_range("PDR_RANGE", raw["PDR_RANGE"]),
            reject=_bool("REJECT", raw["REJECT"]), reject_uv=float(raw["REJECT_UV"]), chunk_s=float(raw["CHUNK_S"]),
            windowed=_bool("WINDOWED", raw["WINDOWED"]), windowed_s=float(raw["WINDOWED_S"]))
    except (TypeError, ValueError) as e:
        raise ValueError(str(e)) from None
    other = {"exclude_seizure": _bool("EXCLUDE_SEIZURE", raw["EXCLUDE_SEIZURE"]),
             "margin_s": float(raw["SEIZURE_MARGIN_S"]), "figures": _bool("FIGURES", raw["FIGURES"]),
             "label": raw["LABEL"].strip(), "pdr_windows": raw["PDR_WINDOWS"].strip().lower()}
    if other["pdr_windows"] not in PDR_MODES:
        raise ValueError(f"PDR_WINDOWS must be one of {', '.join(PDR_MODES)}")
    if other["margin_s"] < 0:
        raise ValueError("SEIZURE_MARGIN_S must be >= 0")
    if params.chunk_s < params.window_s:
        raise ValueError("CHUNK_S must be at least WELCH_WINDOW_S")
    if {"theta", "alpha"} - {b[0] for b in params.bands}:
        log.warning("BANDS without theta and alpha: slowing ratios will be empty")
    if other["label"] and not all(c.isalnum() or c in "-_" for c in other["label"]):
        raise ValueError("LABEL: letters, digits, - and _ only")
    try:
        int(raw["THREADS"])
    except ValueError:
        raise ValueError("THREADS must be an integer") from None
    return raw, params, other


def fmt(x, nd: int = 6) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "" if not math.isfinite(xf) else f"{xf:.{nd}g}"


def num(x, nd: int = 4):
    """A JSON number (None when not finite)."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, nd) if math.isfinite(xf) else None


# --------------------------------------------------------------------------- channels and montage

def select_channels(raw_names: list[str], rec: I.Recording):
    """(raw indices, canonical names, contacts by name, montage, reference, notes)."""
    montage, mapping, reference = C.classify(raw_names)
    notes = []
    dropped = [n for n in raw_names if n not in mapping]
    if dropped:
        notes.append(f"{len(dropped)} non-EEG or unrecognised channel(s) left out: {', '.join(dropped[:8])}"
                     + (" ..." if len(dropped) > 8 else ""))
    excluded = set()
    if rec.channels is not None:
        by = {c.name.strip().lower(): c for c in rec.channels}
        for raw_name in mapping:
            c = by.get(raw_name.strip().lower())
            if c is not None and (c.status == "bad" or c.type not in ("EEG", "")):
                excluded.add(raw_name)
        if excluded:
            notes.append(f"{len(excluded)} channel(s) marked bad in _channels.tsv: {', '.join(sorted(excluded))}")
    idx, names, contacts = [], [], {}
    for i, n in enumerate(raw_names):
        if n in mapping and n not in excluded:
            canon, cont = mapping[n]
            idx.append(i)
            names.append(canon)
            contacts[canon] = cont
    # montage label as conventions.md uses it
    if rec.montage:
        label = rec.montage
    elif montage == "bipolar-longitudinal":
        label = "bipolar-longitudinal"
    elif reference:
        label = "as-recorded"
    elif any(n.lower().endswith(("-avg", "_avg", " avg")) for n in raw_names):
        label = "average"
    else:
        label = "as-recorded"
    if not reference:
        reference = rec.reference or ("average" if label == "average" else "")
    return idx, names, contacts, label, reference, notes


def sidecar_reference(rec: I.Recording) -> str:
    js = I.sidecar(rec.path, "eeg.json")
    if not js:
        return ""
    try:
        ref = str(json.loads(js.read_text()).get("EEGReference", ""))
    except (OSError, ValueError):
        return ""
    return "average" if "average" in ref.lower() else ""


# --------------------------------------------------------------------------- one recording

def process(rec: I.Recording, params: S.Params, other: dict, out: Path, writers: dict) -> dict:
    import mne

    t0 = time.monotonic()
    raw = mne.io.read_raw(str(rec.path), preload=False, verbose="error")
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    rid = rec.recording_id
    if rec.source == "loose":
        I.load_annotations(rec, raw)
    if sfreq < MIN_SFREQ:
        raise ValueError(f"sampling rate {sfreq:g} Hz is below {MIN_SFREQ:g} Hz")
    idx, names, contacts, montage, reference, notes = select_channels(raw.ch_names, rec)
    if montage == "as-recorded" and not rec.montage and sidecar_reference(rec) == "average":
        montage, reference = "average", "average"
    if len(idx) < 2:
        raise ValueError(f"{len(idx)} usable EEG channel(s) (10-20/10-10 names) of {len(raw.ch_names)}")
    bipolar = montage == "bipolar-longitudinal"

    # fit range: below Nyquist and below a low-pass the channels already went through
    fr = params.fit_range
    hi_cut = [c.high_cutoff for c in (rec.channels or []) if c.high_cutoff and c.high_cutoff > 0]
    limit = min([sfreq / 2 * 0.9] + [min(hi_cut) * 0.9] if hi_cut else [sfreq / 2 * 0.9])
    if fr[1] > limit:
        fr = (fr[0], math.floor(limit))
        notes.append(f"fit range cut to {fr[0]:g}-{fr[1]:g} Hz (low-pass / Nyquist)")
    p = params if fr == params.fit_range else S.Params(**{**params.__dict__, "fit_range": fr})

    intervals = S.analysable(duration, rec.bad, rec.seizures, other["exclude_seizure"], other["margin_s"])
    last_log = [time.monotonic()]

    def progress(done: int, total: int):
        if time.monotonic() - last_log[0] > 30:
            log.info("%s: %d of %d windows", rid, done, total)
            last_log[0] = time.monotonic()

    def read(a: int, b: int) -> np.ndarray:
        return raw.get_data(picks=idx, start=a, stop=b, units="uV", verbose="error")

    win = S.compute_windows(read, sfreq, len(idx), intervals, p, progress)
    try:
        return _analyse(rec, raw, win, p, other, out, writers, names, contacts, montage, reference, notes,
                        intervals, sfreq, duration, bipolar, t0)
    finally:
        win.close()


def _analyse(rec, raw, win, p: S.Params, other, out: Path, writers, names, contacts, montage, reference, notes,
             intervals, sfreq, duration, bipolar, t0) -> dict:
    rid = rec.recording_id
    rinfo = S.reject(win, p)
    if rinfo.get("rejected_channels"):
        bad = [names[i] for i in rinfo["rejected_channels"]]
        notes.append(f"{len(bad)} channel(s) flat or above {p.reject_uv:g} µV in most windows, left out: {', '.join(bad)}")
    ok = np.flatnonzero(win.channel_ok)
    keep = win.keep
    if keep.sum() == 0:
        raise ValueError(f"no artifact-free {p.window_s:g} s window ({len(win.starts_s)} window(s) before rejection)")
    ch_names = [names[i] for i in ok]
    psd_all = S.mean_psd(win, keep)[ok]
    freqs = win.freqs
    used_s = S.covered_seconds(win.starts_s[keep], p.window_s)

    # band power and slowing
    bt = S.band_table(freqs, psd_all, p)
    for c, ch in enumerate(ch_names):
        for b, band in enumerate(bt.names):
            writers["bandpower"].writerow([rid, ch, band, fmt(bt.abs_power[c, b]), fmt(bt.rel_power[c, b]),
                                           fmt(bt.log_power[c, b])])
    ta, dtab = S.slowing(bt)
    for c, ch in enumerate(ch_names):
        writers["slowing"].writerow([rid, ch, fmt(ta[c]), fmt(dtab[c])])

    # specparam per channel
    fits = [S.fit_spectrum(freqs, psd_all[c], p) for c in range(len(ch_names))]
    low_fit = []
    alpha_peak = np.full(len(ch_names), np.nan)
    for c, (ch, fit) in enumerate(zip(ch_names, fits)):
        flag = "" if fit.ok else (fit.message or "fit failed")
        if fit.ok and fit.r_squared < p.min_r_squared:
            flag = f"r_squared<{p.min_r_squared:g}"
            low_fit.append(ch)
        writers["aperiodic"].writerow([rid, ch, fmt(fit.exponent), fmt(fit.offset), fmt(fit.knee), fmt(fit.r_squared),
                                       fmt(fit.error), len(fit.peaks) if fit.ok else "", flag])
        for cf, pw, bw in fit.peaks:
            writers["peaks"].writerow([rid, ch, fmt(cf), fmt(pw), fmt(bw)])
        pk = S.strongest_peak(fit, *p.pdr_range)
        if pk is not None:
            alpha_peak[c] = pk[0]
    if low_fit:
        notes.append(f"specparam r_squared < {p.min_r_squared:g} on {len(low_fit)} channel(s): {', '.join(low_fit)}")

    # posterior dominant rhythm
    pdr = posterior_dominant_rhythm(rec, win, p, ok, ch_names, contacts, freqs, bipolar, other["pdr_windows"])
    writers["pdr"].writerow([rid, fmt(pdr["pdr_hz"], 4), fmt(pdr["amplitude"], 4), fmt(pdr["left_hz"], 4),
                             fmt(pdr["right_hz"], 4), fmt(pdr["asymmetry"], 4), fmt(pdr["reactivity"], 4),
                             fmt(pdr["ec_s"], 6), pdr["method"]])
    if pdr["note"]:
        notes.append(pdr["note"])

    # revised brain symmetry index
    pairs = C.homologous_pairs(ch_names, contacts)
    pos = {ch: i for i, ch in enumerate(ch_names)}
    bands = S.asymmetry_bands(p)
    pair_bsi: dict[str, float] = {}
    rbsi = math.nan
    for left, right in pairs:
        for band, lo, hi in bands:
            u, s = S.bsi(psd_all[pos[right]], psd_all[pos[left]], freqs, lo, hi)
            writers["asymmetry"].writerow([rid, f"{left}/{right}", band, fmt(u), fmt(s)])
            if band == "rbsi":
                pair_bsi[left] = pair_bsi[right] = u
    if pairs:
        rmean = psd_all[[pos[r] for _, r in pairs]].mean(axis=0)
        lmean = psd_all[[pos[l_] for l_, _ in pairs]].mean(axis=0)
        for band, lo, hi in bands:
            u, s = S.bsi(rmean, lmean, freqs, lo, hi)
            writers["asymmetry"].writerow([rid, "recording", band, fmt(u), fmt(s)])
            if band == "rbsi":
                rbsi, rbsi_signed = u, s
    else:
        rbsi_signed = math.nan
        notes.append("no homologous left/right channel pair: no asymmetry index")

    # features.csv (shared long shape)
    feats: list[tuple[str, np.ndarray]] = []
    bands_only = [b for b in bt.names if b != "broadband"]
    for b, band in enumerate(bands_only):
        feats.append((f"relpow_{band}", bt.rel_power[:, b]))
    for b, band in enumerate(bands_only):
        feats.append((f"logpow_{band}", bt.log_power[:, b]))
    feats.append(("aperiodic_exponent", np.array([f.exponent for f in fits])))
    feats.append(("aperiodic_offset", np.array([f.offset for f in fits])))
    feats.append(("pdr_hz", alpha_peak))
    feats.append(("theta_alpha_ratio", ta))
    feats.append(("bsi", np.array([pair_bsi.get(ch, np.nan) for ch in ch_names])))
    for fname, vals in feats:
        for c, ch in enumerate(ch_names):
            writers["features"].writerow([rid, ch, fname, fmt(vals[c])])

    # windowed band power
    if p.windowed and writers.get("windowed") is not None:
        write_windowed(writers["windowed"], rid, win, keep, ok, ch_names, p)

    # figures
    figs = []
    if other["figures"]:
        figs = make_figures(out, rid, freqs, psd_all, ch_names, contacts, fits, bt, pdr, p, bipolar)

    secs = time.monotonic() - t0
    exps = np.array([f.exponent for f in fits if f.ok])
    post = [i for i, ch in enumerate(ch_names) if C.region(contacts[ch]) in ("occipital", "parietal")]
    a_col = bt.names.index("alpha") if "alpha" in bt.names else None
    headline = {
        "montage": montage, "n_channels": len(ch_names), "seconds_analysed": num(used_s, 1),
        "windows_used": int(keep.sum()), "windows_rejected": int(rinfo.get("rejected_windows", 0)),
        "channel_windows_rejected": int(rinfo.get("rejected_channel_windows", 0)),
        "pdr_hz": num(pdr["pdr_hz"], 2), "pdr_amplitude_uv": num(pdr["amplitude"], 1),
        "pdr_asymmetry": num(pdr["asymmetry"], 3), "pdr_reactivity": num(pdr["reactivity"], 3),
        "pdr_basis": pdr["basis"],
        "rbsi": num(rbsi, 4), "rbsi_signed": num(rbsi_signed, 4),
        "aperiodic_exponent_median": num(np.median(exps), 3) if len(exps) else None,
        "theta_alpha_ratio_median": num(np.nanmedian(ta), 3) if np.isfinite(ta).any() else None,
        "relpow_alpha_posterior": num(np.nanmean(bt.rel_power[post, a_col]), 3) if post and a_col is not None else None,
        "low_r_squared_channels": len(low_fit),
    }
    log.info("%s: %d EEG channels (%s) of %d, %g Hz, %.0f s of %.0f s analysed (%d windows, %d rejected), "
             "PDR %s Hz, rBSI %s, exponent %s, %.1fs", rid, len(ch_names), montage, len(raw.ch_names), sfreq, used_s,
             duration, int(keep.sum()), rinfo.get("rejected_windows", 0), fmt(pdr["pdr_hz"], 3), fmt(rbsi, 3),
             headline["aperiodic_exponent_median"], secs)
    return {"status": "ok", "seconds": round(secs, 2), "montage": montage, "reference": reference,
            "sfreq": sfreq, "duration_s": round(duration, 3), "n_channels_file": len(raw.ch_names),
            "channels": ch_names, "fit_range_hz": list(p.fit_range),
            "analysable_intervals_s": [[round(a, 3), round(b, 3)] for a, b in intervals],
            "seizures_excluded": len(rec.seizures) if other["exclude_seizure"] else 0,
            "bad_segments": len(rec.bad), **rinfo, "seconds_analysed": round(used_s, 2),
            "notes": notes + rec.notes, "_headline": headline, "_figures": figs}


def posterior_dominant_rhythm(rec, win, p: S.Params, ok, ch_names, contacts, freqs, bipolar, pdr_windows) -> dict:
    out = {"pdr_hz": math.nan, "amplitude": math.nan, "left_hz": math.nan, "right_hz": math.nan,
           "asymmetry": math.nan, "reactivity": math.nan, "ec_s": 0.0, "method": "", "note": "", "basis": "",
           "sides": {}}
    candidates = C.PDR_BIPOLAR if bipolar else C.PDR_REFERENTIAL
    have = set(ch_names)
    chosen = next(((l_, r) for l_, r in candidates if l_ in have and r in have), None) or \
        next(((l_ if l_ in have else None, r if r in have else None) for l_, r in candidates
              if l_ in have or r in have), None)
    if chosen is None:
        out["method"] = "no posterior channel (O1/O2, P3/P4 or their bipolar pairs)"
        out["note"] = "PDR not computed: " + out["method"]
        return out
    keep = win.keep
    ec = S.inside(win.starts_s, p.window_s, rec.eyes_closed) & keep
    eo = S.inside(win.starts_s, p.window_s, rec.eyes_open) & keep
    col = {ch: i for i, ch in enumerate(ch_names)}
    left, right = chosen
    sides = [s for s in (left, right) if s]
    p_mode = pdr_windows
    if ec.sum() >= MIN_PDR_WINDOWS:
        mask, basis = ec, "eyes-closed windows"
        out["ec_s"] = S.covered_seconds(win.starts_s[ec], p.window_s)
    else:
        why = "no eyes-closed annotation" if not rec.eyes_closed else f"fewer than {MIN_PDR_WINDOWS} eyes-closed windows"
        awake = keep & ~S.overlapping(win.starts_s, p.window_s, rec.sleep)
        if awake.sum() == 0:
            awake = keep
        slept = ", annotated sleep left out" if rec.sleep else ""
        if p_mode == "auto":
            mask, basis = awake, f"all artifact-free windows ({why}{slept})"
            out["note"] = f"PDR on {basis}: may mix eyes-open, drowsy and sleep EEG; review"
        else:
            post = [int(ok[col[s]]) for s in sides]
            ant = [int(ok[i]) for i, ch in enumerate(ch_names) if C.region(contacts[ch]) == "frontal"]
            score = S.alpha_score(win, post, ant, p)
            mask = S.top_fraction(score, awake, PDR_TOP_FRACTION, MIN_PDR_WINDOWS)
            basis = (f"the {PDR_TOP_FRACTION:.0%} of artifact-free windows with the most posterior "
                     f"{p.pdr_range[0]:g}-{p.pdr_range[1]:g} Hz power ({why}{slept})")
            out["note"] = (f"PDR on {basis}: a stand-in for eyes-closed wakefulness; drowsiness or sleep "
                           "rhythms can be picked up when the patient was never awake with eyes closed; review")
    out["basis"] = "eyes-closed" if mask is ec else ("all" if p_mode == "auto" else "alpha-richest")
    spec = S.mean_psd(win, mask)[ok]
    combined = spec[[col[s] for s in sides]].mean(axis=0)
    fit = S.fit_spectrum(freqs, combined, p)
    pk = S.strongest_peak(fit, *p.pdr_range)
    if pk is not None:
        out["pdr_hz"] = float(pk[0])
        out["amplitude"] = S.peak_amplitude_uv(freqs, combined, fit, pk)
    amps = {}
    for side, ch in (("left", left), ("right", right)):
        if not ch:
            continue
        f = S.fit_spectrum(freqs, spec[col[ch]], p)
        pks = S.strongest_peak(f, *p.pdr_range)
        hz = float(pks[0]) if pks is not None else math.nan
        amp = S.peak_amplitude_uv(freqs, spec[col[ch]], f, pks) if pks is not None else math.nan
        out[f"{side}_hz"] = hz
        amps[side] = amp
        out["sides"][side] = (ch, spec[col[ch]], hz, amp)
    if amps.get("left") is not None and amps.get("right") is not None:
        lsum = amps["left"] + amps["right"]
        if math.isfinite(lsum) and lsum > 0:
            out["asymmetry"] = (amps["left"] - amps["right"]) / lsum
    if ec.sum() >= MIN_PDR_WINDOWS and eo.sum() >= MIN_PDR_WINDOWS:
        lo, hi = ((out["pdr_hz"] - 1, out["pdr_hz"] + 1) if math.isfinite(out["pdr_hz"])
                  else next(((b[1], b[2]) for b in p.bands if b[0] == "alpha"), p.pdr_range))
        idx = [col[s] for s in sides]
        p_ec = S.band_power(freqs, S.mean_psd(win, ec)[ok][idx].mean(axis=0), lo, hi)
        p_eo = S.band_power(freqs, S.mean_psd(win, eo)[ok][idx].mean(axis=0), lo, hi)
        if p_ec > 0:
            out["reactivity"] = float(1 - p_eo / p_ec)
    chans = "/".join(sides)
    out["method"] = (f"specparam peak {p.pdr_range[0]:g}-{p.pdr_range[1]:g} Hz, {chans}, {basis}"
                     + ("" if pk is not None else "; no peak found"))
    return out


def write_windowed(w, rid, win, keep, ok, ch_names, p: S.Params) -> None:
    """Relative band power and theta/alpha ratio per WINDOWED_S block of kept windows."""
    if len(win.starts_s) == 0:
        return
    block = np.floor(win.starts_s / p.windowed_s).astype(int)
    for b in np.unique(block[keep]):
        m = keep & (block == b)
        bt = S.band_table(win.freqs, S.mean_psd(win, m)[ok], p)
        ta, _ = S.slowing(bt)
        t = b * p.windowed_s
        for c, ch in enumerate(ch_names):
            for j, band in enumerate(bt.names[:-1]):
                w.writerow([rid, ch, fmt(t), f"relpow_{band}", fmt(bt.rel_power[c, j])])
            w.writerow([rid, ch, fmt(t), "theta_alpha_ratio", fmt(ta[c])])


def make_figures(out: Path, rid, freqs, psd, ch_names, contacts, fits, bt, pdr, p: S.Params, bipolar) -> list[dict]:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs = []
    regions = [C.region(contacts[ch]) for ch in ch_names]
    reg_fits = {}
    for reg in set(regions):
        sel = [i for i, r in enumerate(regions) if r == reg]
        reg_fits[reg] = S.fit_spectrum(freqs, psd[sel].mean(axis=0), p)
    name = f"{rid}_psd.png"
    F.psd_by_region(figdir / name, rid, freqs, psd, regions, reg_fits, p.fit_range)
    figs.append({"path": f"figures/{name}", "caption": (
        f"{rid}: mean power spectrum of each scalp region (log-log), with the aperiodic (1/f) component fitted by "
        "specparam as a dashed line. Bumps above the dashed line are oscillations (alpha near 8-13 Hz, beta); a "
        "raised low-frequency end in one region points to focal slowing; a steeper slope (larger exponent) means "
        f"relatively more slow activity. {F.RESEARCH}")})
    a_col = bt.names.index("alpha") if "alpha" in bt.names else None
    panels = []
    if a_col is not None:
        panels.append(("relative alpha power (8-13 Hz / broadband)", bt.rel_power[:, a_col], "viridis"))
    panels.append(("aperiodic exponent", np.array([f.exponent for f in fits]), "magma"))
    name = f"{rid}_topomaps.png"
    if F.topomaps(figdir / name, rid, ch_names, contacts, panels, bipolar):
        figs.append({"path": f"figures/{name}", "caption": (
            f"{rid}: scalp maps of relative alpha power (left) and of the aperiodic exponent (right) per channel. "
            "Alpha is normally highest over the occipital and parietal electrodes; a one-sided loss of alpha or a "
            "one-sided change of the exponent is an asymmetry worth checking against the raw EEG. "
            + ("Bipolar pairs are drawn at the midpoint of their two electrodes. " if bipolar else "")
            + F.RESEARCH)})
    if pdr["sides"]:
        name = f"{rid}_pdr.png"
        F.pdr(figdir / name, rid, freqs, pdr["sides"], p.pdr_range,
              {"eyes-closed": "eyes closed", "all": "all artifact-free windows"}.get(
                  pdr["basis"], "alpha-richest quarter of the windows"))
        hz = pdr["pdr_hz"]
        figs.append({"path": f"figures/{name}", "caption": (
            f"{rid}: posterior dominant rhythm. Spectra of the left and right posterior channels; the dotted lines "
            f"mark each side's peak in {p.pdr_range[0]:g}-{p.pdr_range[1]:g} Hz"
            + (f" (combined {hz:.2f} Hz)" if math.isfinite(hz) else " (no peak found)") + ". "
            "In adults a waking PDR below about 8 Hz, a left-right difference above about 1 Hz, or a large amplitude "
            "difference are findings to review. "
            + {"eyes-closed": "Computed on annotated eyes-closed EEG. ",
               "all": "No eyes-closed annotation: computed on all artifact-free EEG, which may include eyes-open, "
                      "drowsiness or sleep. "}.get(pdr["basis"],
               "No eyes-closed annotation: computed on the quarter of the artifact-free EEG with the most posterior "
               "alpha-range power, a stand-in for relaxed wakefulness with eyes closed; check that the patient "
               "was awake in some of the recording. ") + F.RESEARCH)})
    return figs


# --------------------------------------------------------------------------- main

def peak_rss_mb() -> float:
    for f in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return round(int(Path(f).read_text().strip()) / 2**20, 1)
        except (OSError, ValueError):
            continue
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 1024, 1)


def main() -> int:
    t_start = time.monotonic()
    started = datetime.now(timezone.utc)
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    try:
        raw_params, params, other = read_params()
    except ValueError as e:
        log.error("parameters: %s", e)
        return 2
    out = output_dir / APP_NAME / other["label"] if other["label"] else output_dir / APP_NAME
    out.mkdir(parents=True, exist_ok=True)
    log.info("%s %s: INPUT_DIR=%s OUTPUT_DIR=%s threads=%d (%s)", APP_NAME, app_version(), input_dir, out,
             _THREADS, _THREADS_WHY)
    recs = I.find_recordings(input_dir) if input_dir.is_dir() else []
    if not recs:
        log.error("no recording found under %s (recordings.csv, *_eeg.edf/.bdf/.vhdr/.fif/.set or loose files)",
                  input_dir)
        return 1
    log.info("%d recording(s) found", len(recs))

    def rel(pth: Path) -> str:
        try:
            return str(pth.relative_to(input_dir))
        except ValueError:
            return str(pth)

    per_rec: dict[str, dict] = {}
    headlines: dict[str, dict] = {}
    figures: list[dict] = []
    warnings_out: list[str] = []
    n_ok = 0
    names = [k for k in HEADERS if k != "windowed" or params.windowed]
    handles = {k: open(out / f"{k}.csv", "w", newline="", encoding="utf-8") for k in names}
    try:
        writers = {k: csv.writer(h, lineterminator="\n") for k, h in handles.items()}
        for k, w in writers.items():
            w.writerow(HEADERS[k])
        for i, rec in enumerate(recs, 1):
            log.info("[%d/%d] %s (%s)", i, len(recs), rec.recording_id, rel(rec.path))
            try:
                res = process(rec, params, other, out, writers)
                headlines[rec.recording_id] = res.pop("_headline")
                figures.extend(res.pop("_figures"))
                per_rec[rec.recording_id] = {"path": rel(rec.path), **res}
                for n in res["notes"]:
                    warnings_out.append(f"{rec.recording_id}: {n}")
                n_ok += 1
            except Exception as e:  # noqa: BLE001 - a recording that fails is logged and skipped
                log.error("%s: failed: %s", rec.recording_id, e)
                log.debug(traceback.format_exc())
                per_rec[rec.recording_id] = {"path": rel(rec.path), "status": "failed", "error": str(e)}
                warnings_out.append(f"{rec.recording_id}: failed: {e}")
            for h in handles.values():
                h.flush()
    finally:
        for h in handles.values():
            h.close()

    finished = datetime.now(timezone.utc)
    provenance = {
        "app": APP_NAME, "version": app_version(), "variant": None, "tools": tool_versions(),
        "parameters": {**raw_params, "spectral": params.as_dict()},
        "inputs": sorted(rel(r.path) for r in recs),
        "started": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 2), "peak_rss_mb": peak_rss_mb(),
        "threads": {"requested": int(raw_params["THREADS"]), "used": _THREADS, "why": _THREADS_WHY},
        "recordings": per_rec, "citations": CITATIONS, "licence_notice": None,
    }
    with open(out / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False)
    tables = [
        {"path": "bandpower.csv", "caption": "Absolute (µV²), relative (share of broadband) and log10 power per channel and band."},
        {"path": "aperiodic.csv", "caption": "specparam aperiodic exponent and offset per channel, with the fit's r_squared; "
                                             f"fits below {params.min_r_squared:g} are flagged."},
        {"path": "peaks.csv", "caption": "Oscillatory peaks found by specparam per channel: centre frequency, power above the aperiodic fit (log10), bandwidth (Hz)."},
        {"path": "pdr.csv", "caption": "Posterior dominant rhythm per recording: frequency, amplitude, left/right frequency, amplitude asymmetry (L-R)/(L+R), reactivity."},
        {"path": "asymmetry.csv", "caption": "Revised Brain Symmetry Index per homologous pair and band, and for the whole recording (pair = recording); band rbsi = 1-25 Hz. signed > 0: more power on the right."},
        {"path": "slowing.csv", "caption": "Slowing ratios per channel: theta/alpha and (delta+theta)/(alpha+beta) power."},
        {"path": "features.csv", "caption": "Per-channel features in the shared long shape (recording_id, channel, feature, value)."},
    ]
    if params.windowed:
        tables.append({"path": "windowed.csv", "caption": f"Relative band power and theta/alpha ratio per {params.windowed_s:g} s block."})
    summary = {"app": APP_NAME, "version": app_version(), "recordings": headlines, "figures": figures,
               "tables": tables, "warnings": warnings_out,
               "notice": "Spectral biomarkers are research outputs, to be reviewed by a clinician; they are not a "
                         "clinical EEG interpretation."}
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("%d of %d recording(s) done in %.1fs, peak memory %.0f MB; wrote %s", n_ok, len(recs),
             time.monotonic() - t_start, provenance["peak_rss_mb"], out)
    if n_ok == 0:
        log.error("every recording failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
