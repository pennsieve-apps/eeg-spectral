"""
Finding the recordings under INPUT_DIR, by name and content, recursively, following symlinks:

1. the eeg-prep BIDS-EEG derivative: every recordings.csv (recording_id, path, montage, ...);
2. any BIDS-EEG dataset: *_eeg.edf/.bdf/.vhdr/.fif/.set not listed in a recordings.csv, with
   their _channels.tsv and _events.tsv (trial_type, or SzCORE-style HED `eventType`);
3. loose EEG files (.edf/.bdf/.vhdr/.fif/.set without a BIDS name), with their EDF+
   annotations and, for CHB-MIT, the seizures listed in a `*-summary.txt` next to them.
"""
from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("eeg-spectral")

SIGNAL_EXTS = (".edf", ".bdf", ".vhdr", ".fif", ".set")
NA = {"", "n/a", "na", "nan", "none"}
SEIZURE_TEXT = re.compile(r"seizure|\bsz\b|^sz|onset|ictal", re.I)


def walk_files(root: Path) -> list[Path]:
    """Every file under root, following symlinked folders and files (INPUT_DIR entries are symlinks)."""
    found: list[Path] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("."):
                p = Path(dirpath) / name
                if p.is_file():
                    found.append(p)
    return found


def read_table(path: Path, delimiter: str | None = None) -> list[dict[str, str]]:
    """Rows of a TSV/CSV as dicts (BOM stripped, keys and values stripped, n/a -> '')."""
    delim = delimiter or ("\t" if path.suffix.lower() == ".tsv" else ",")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = []
        for raw in csv.DictReader(fh, delimiter=delim):
            row = {}
            for k, v in raw.items():
                if k is None:
                    continue
                v = (v or "").strip()
                row[k.strip()] = "" if v.lower() in NA else v
            rows.append(row)
        return rows


def fnum(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def is_signal(p: Path) -> bool:
    return p.suffix.lower() in SIGNAL_EXTS


def bids_stem(path: Path) -> str | None:
    """BIDS stem of an *_eeg.<ext> file (without _desc-preproc), or None when not BIDS-named."""
    m = re.match(r"^(sub-[^_]+(?:_[a-z]+-[^_]+)*)_eeg\.[A-Za-z0-9]+$", path.name)
    if not m:
        return None
    return re.sub(r"_desc-preproc$", "", m.group(1))


def loose_id(path: Path) -> str:
    """A recording id for a file without a BIDS name, as eeg-prep makes one up:
    chb01_03.edf -> sub-chb01_task-eeg_run-03; patient.edf -> sub-patient_task-eeg."""
    stem = re.sub(r"[^A-Za-z0-9_-]", "", path.stem)
    m = re.match(r"^(.*?)[_-](\d+)$", stem)
    if m and m.group(1):
        sub = re.sub(r"[^A-Za-z0-9]", "", m.group(1)) or "x"
        return f"sub-{sub}_task-eeg_run-{m.group(2)}"
    return f"sub-{re.sub(r'[^A-Za-z0-9]', '', stem) or 'x'}_task-eeg"


def sidecar(signal_path: Path, suffix: str) -> Path | None:
    """<prefix>_<suffix> next to <prefix>_eeg.<ext> (suffix e.g. 'channels.tsv')."""
    m = re.match(r"^(.*)_eeg\.[A-Za-z0-9]+$", signal_path.name)
    if not m:
        return None
    p = signal_path.with_name(f"{m.group(1)}_{suffix}")
    return p if p.is_file() else None


@dataclass
class Channel:
    name: str
    type: str = "EEG"
    status: str = "good"
    status_description: str = ""
    contacts: tuple[str, ...] = ()
    hemisphere: str = ""
    region: str = ""
    high_cutoff: float | None = None
    low_cutoff: float | None = None


@dataclass
class Recording:
    recording_id: str
    path: Path
    subject: str = ""
    source: str = ""                       # recordings.csv | bids | loose
    montage: str = ""                      # from recordings.csv when known
    reference: str = ""
    kind: str = "unknown"
    channels: list[Channel] | None = None  # from _channels.tsv
    seizures: list[tuple[float, float | None]] = field(default_factory=list)
    bad: list[tuple[float, float]] = field(default_factory=list)          # bad_segment + artifact
    eyes_closed: list[tuple[float, float]] = field(default_factory=list)
    eyes_open: list[tuple[float, float]] = field(default_factory=list)
    sleep: list[tuple[float, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def subject_of(recording_id: str) -> str:
    m = re.match(r"^sub-([A-Za-z0-9]+)", recording_id)
    return m.group(1) if m else ""


def classify_event(trial_type: str) -> str | None:
    """seizure | bad | eyes_closed | eyes_open | sleep | None."""
    t = trial_type.strip().lower().replace(" ", "_").replace("-", "_")
    if not t:
        return None
    if t in ("bad_segment", "bad") or "artifact" in t or "artefact" in t or t.startswith("bad_"):
        return "bad"
    if t in ("eyes_closed", "ec", "eyes_close", "closed_eyes") or t.startswith("eyes_closed"):
        return "eyes_closed"
    if t in ("eyes_open", "eo", "open_eyes") or t.startswith("eyes_open"):
        return "eyes_open"
    if t == "sleep" or t in ("n1", "n2", "n3", "rem", "nrem"):
        return "sleep"
    if t.startswith("seizure") or t in ("ictal", "sz") or t.startswith("sz_") or t.startswith("sz"):
        return "seizure"
    return None


def add_event(rec: Recording, kind: str | None, onset: float, dur: float | None) -> None:
    if kind == "seizure":
        rec.seizures.append((onset, dur if dur and dur > 0 else None))
    elif kind and dur and dur > 0:
        getattr(rec, {"bad": "bad", "eyes_closed": "eyes_closed", "eyes_open": "eyes_open",
                      "sleep": "sleep"}[kind]).append((onset, onset + dur))


def load_sidecars(rec: Recording) -> None:
    """_channels.tsv and _events.tsv next to the signal file, when present."""
    ch = sidecar(rec.path, "channels.tsv")
    if ch:
        chans = []
        for r in read_table(ch):
            if not r.get("name"):
                continue
            contacts = tuple(c for c in (r.get("contacts") or "").split("|") if c)
            chans.append(Channel(r["name"], (r.get("type") or "EEG").upper(), (r.get("status") or "good").lower(),
                                 r.get("status_description", ""), contacts, r.get("hemisphere", ""),
                                 r.get("region", ""), fnum(r.get("high_cutoff")), fnum(r.get("low_cutoff"))))
        rec.channels = chans
    ev = sidecar(rec.path, "events.tsv")
    if ev:
        for r in read_table(ev):
            on = fnum(r.get("onset"))
            if on is None:
                continue
            tt = r.get("trial_type") or r.get("eventType") or r.get("event_type") or ""
            add_event(rec, classify_event(tt), on, fnum(r.get("duration")))


def load_annotations(rec: Recording, raw) -> None:
    """EDF+ (or other) annotations of a loose file: seizures, artifacts, eyes open/closed."""
    for a in raw.annotations:
        desc = str(a["description"])
        kind = classify_event(desc)
        if kind is None and SEIZURE_TEXT.search(desc):
            kind = "seizure"
        add_event(rec, kind, float(a["onset"]), float(a["duration"]))


def chbmit_summary(files: list[Path]) -> dict[str, list[tuple[float, float]]]:
    """Seizures of CHB-MIT `*-summary.txt` files: {edf file name: [(start_s, end_s), ...]}."""
    out: dict[str, list[tuple[float, float]]] = {}
    for p in files:
        if not p.name.endswith("-summary.txt"):
            continue
        current, start = None, None
        for line in p.read_text(errors="replace").splitlines():
            m = re.match(r"\s*File Name:\s*(\S+)", line)
            if m:
                current = m.group(1)
                out.setdefault(current, [])
                continue
            m = re.match(r"\s*Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)", line)
            if m:
                start = float(m.group(1))
                continue
            m = re.match(r"\s*Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)", line)
            if m and current and start is not None:
                out[current].append((start, float(m.group(1))))
                start = None
    return out


def find_recordings(input_dir: Path) -> list[Recording]:
    """Recordings listed in any recordings.csv, plus BIDS-EEG and loose signal files not listed."""
    files = walk_files(input_dir)
    recs: dict[str, Recording] = {}
    listed: set[str] = set()
    for idx in [p for p in files if p.name == "recordings.csv"]:
        for r in read_table(idx, ","):
            rid, rel = r.get("recording_id", ""), r.get("path", "")
            if not rid or not rel:
                continue
            path = idx.parent / rel
            if not path.is_file():
                log.warning("%s: %s listed in %s but not found", rid, rel, idx)
                continue
            listed.add(os.path.realpath(path))
            if rid in recs:
                log.warning("%s listed twice; using the first (%s)", rid, recs[rid].path)
                continue
            recs[rid] = Recording(rid, path, subject=r.get("subject") or subject_of(rid), source="recordings.csv",
                                  montage=r.get("montage", ""), reference=r.get("reference", ""),
                                  kind=(r.get("kind") or "unknown").lower())
    # a BrainVision header (.vhdr) is the recording; skip its .eeg/.vmrk; skip .fif parts (-1.fif)
    summaries = chbmit_summary(files)
    signals = [p for p in files if is_signal(p) and os.path.realpath(p) not in listed
               and not re.search(r"-\d+\.fif$", p.name)]
    # a raw BIDS dataset's derivatives/ folder holds processed copies (e.g. ds003555's HFO clips):
    # skipped when there are recordings outside it
    in_deriv = [p for p in signals if "derivatives" in p.relative_to(input_dir).parts[:-1]]
    if in_deriv and (len(in_deriv) < len(signals) or recs):
        log.info("%d signal file(s) under derivatives/ skipped (the raw recordings are used)", len(in_deriv))
        signals = [p for p in signals if p not in in_deriv]
    for p in signals:
        stem = bids_stem(p)
        rid, source = (stem, "bids") if stem else (loose_id(p), "loose")
        if rid in recs:
            log.warning("%s: %s duplicates a recording already found (%s); skipped", rid, p, recs[rid].path)
            continue
        rec = Recording(rid, p, subject=subject_of(rid), source=source)
        for s, e in summaries.get(p.name, []):
            rec.seizures.append((s, e - s))
        recs[rid] = rec
    for rec in recs.values():
        load_sidecars(rec)
    return [recs[k] for k in sorted(recs)]
