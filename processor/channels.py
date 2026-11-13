"""
Scalp EEG channel names: canonical 10-20/10-10 spelling, montage (referential or bipolar),
hemisphere, region, homologous (mirror) channels, and template positions.

eeg-prep has done this already for its derivative (its _channels.tsv carries `contacts`,
`hemisphere`, `region`); this module does the same for plain BIDS-EEG datasets and loose EDF
files, so that eeg-spectral runs on its own:

  'EEG Fp1-Ref' -> 'Fp1'    'O1-Avg' -> 'O1'    'T3' -> 'T7'    'FP1-F7' -> 'Fp1-F7' (bipolar)
"""
from __future__ import annotations

import re
import warnings
from functools import lru_cache

import numpy as np

# Old 10-20 names (and their MNE spelling) -> 10-10 names, as eeg-prep canonicalises them.
OLD_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
# Reference suffixes and prefixes a referential channel name may carry.
REF_SUFFIX = re.compile(r"[-_ ]?(avg|ref|average|car|le|ar|a1a2|m1m2)$", re.I)  # words, not electrodes
PREFIX = re.compile(r"^(eeg|eog)[ _:-]*", re.I)
NON_EEG = re.compile(r"^(ecg|ekg|emg|eog|resp|pleth|spo2|sao2|pulse|hr|dc\d*|trig|trigger|status|stim|"
                     r"edf annotations|annotations|photic|ibi|bursts|suppr|vns|loc|roc|chin|x\d*|-|\.|--\d*)$", re.I)
REGIONS = (("Fp", "frontal"), ("AF", "frontal"), ("FT", "temporal"), ("FC", "central"), ("F", "frontal"),
           ("TP", "temporal"), ("T", "temporal"), ("CP", "parietal"), ("C", "central"), ("PO", "occipital"),
           ("P", "parietal"), ("O", "occipital"), ("I", "occipital"), ("A", "temporal"), ("M", "temporal"))


@lru_cache(maxsize=1)
def montage():
    """MNE's 10-05 template (named colin27_1005 from MNE 1.13, standard_1005 before)."""
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return mne.channels.make_standard_montage("colin27_1005")
        except ValueError:
            return mne.channels.make_standard_montage("standard_1005")


@lru_cache(maxsize=1)
def known_names() -> dict[str, str]:
    """lowercase -> canonical spelling of every 10-05 label, plus the old 10-20 names."""
    out = {n.lower(): n for n in montage().ch_names}
    for old, new in OLD_NAMES.items():
        out[old.lower()] = new
    return out


@lru_cache(maxsize=1)
def positions() -> dict[str, np.ndarray]:
    """Template head coordinates (m) of every 10-05 label."""
    return {k: np.asarray(v, float) for k, v in montage().get_positions()["ch_pos"].items()}


def electrode(name: str) -> str | None:
    """Canonical 10-20/10-10 name of a single electrode label, or None when not one."""
    n = PREFIX.sub("", name.strip())
    n = re.sub(r"\s+", "", n)
    k = known_names()
    if n.lower() in k:
        return k[n.lower()]
    m = REF_SUFFIX.search(n)
    if m and m.start() > 0:
        base = n[: m.start()]
        if base.lower() in k:
            return k[base.lower()]
    return None


def canonical(name: str) -> tuple[str | None, tuple[str, ...]]:
    """(canonical name, contacts): 'Fp1', ('Fp1',) or 'Fp1-F7', ('Fp1', 'F7'); (None, ()) when not EEG."""
    raw = name.strip()
    if not raw or NON_EEG.match(PREFIX.sub("", raw)):
        return None, ()
    e = electrode(raw)
    if e:
        return e, (e,)
    # bipolar pair: two electrode labels joined by '-' (CHB-MIT adds '-0'/'-1' to duplicates)
    body = re.sub(r"-\d+$", "", PREFIX.sub("", raw))
    parts = body.split("-")
    if len(parts) == 2:
        a, b = electrode(parts[0]), electrode(parts[1])
        if a and b and a != b:
            return f"{a}-{b}", (a, b)
    return None, ()


def hemisphere(contacts: tuple[str, ...]) -> str:
    """L (odd), R (even), Z (midline 'z', or a pair spanning both sides)."""
    sides = set()
    for c in contacts:
        m = re.search(r"(\d+)$", c)
        sides.add("Z" if not m else ("L" if int(m.group(1)) % 2 else "R"))
    return sides.pop() if len(sides) == 1 else "Z"


def region(contacts: tuple[str, ...]) -> str:
    """Region of a channel; a midline electrode is 'midline'; a bipolar pair takes its second
    (more posterior in the longitudinal montage) contact, so P7-O1 is occipital."""
    c = contacts[-1] if contacts else ""
    if c.lower().endswith("z"):
        return "midline"
    for prefix, reg in REGIONS:
        if c.startswith(prefix):
            return reg
    return "other"


def mirror_electrode(e: str) -> str | None:
    m = re.match(r"^([A-Za-z]+)(\d+)$", e)
    if not m:
        return e if e.lower().endswith("z") else None
    n = int(m.group(2))
    return f"{m.group(1)}{n + 1 if n % 2 else n - 1}"


def mirror(contacts: tuple[str, ...]) -> str | None:
    """Name of the homologous channel on the other side (Fp1-F7 -> Fp2-F8); None for midline."""
    ms = [mirror_electrode(c) for c in contacts]
    if any(m is None for m in ms) or tuple(ms) == contacts:
        return None
    return "-".join(ms)


def position(contacts: tuple[str, ...]) -> np.ndarray | None:
    """Template position of an electrode, or the midpoint of a bipolar pair."""
    pos = positions()
    pts = [pos[c] for c in contacts if c in pos]
    if len(pts) != len(contacts) or not pts:
        return None
    return np.mean(pts, axis=0)


def homologous_pairs(names: list[str], contacts: dict[str, tuple[str, ...]]) -> list[tuple[str, str]]:
    """(left, right) channel pairs present in `names`, left first, in the order of `names`."""
    have = set(names)
    out = []
    for n in names:
        c = contacts.get(n, ())
        if hemisphere(c) != "L":
            continue
        m = mirror(c)
        if m and m in have:
            out.append((n, m))
        elif m and len(c) == 2 and "-".join(reversed(m.split("-"))) in have:
            # a bipolar mirror stored the other way round (CHB-MIT: T7-FT9 and FT10-T8): the same
            # derivation with the opposite sign, so the same power spectrum
            out.append((n, "-".join(reversed(m.split("-")))))
    return out


def classify(raw_names: list[str]) -> tuple[str, dict[str, tuple[str, tuple[str, ...]]], str]:
    """
    Montage of a recording from its channel names, and each EEG channel's canonical name.

    Returns (montage, {raw name: (canonical name, contacts)}, reference): montage is
    'referential' or 'bipolar-longitudinal'. Pairs that all share their second contact
    ('Fp1-A1', 'F3-A1', ...) are referential to that electrode, so they become 'Fp1', 'F3'
    with reference 'A1'; pairs with varying second contacts ('Fp1-F7', 'F7-T7') are bipolar.
    Channels that are not EEG (or not 10-20/10-10 labels) are left out. A recording with both
    single electrodes and pairs keeps whichever kind is the majority.
    """
    parsed = {n: canonical(n) for n in raw_names}
    parsed = {n: v for n, v in parsed.items() if v[0]}
    pairs = {n: v for n, v in parsed.items() if len(v[1]) == 2}
    singles = {n: v for n, v in parsed.items() if len(v[1]) == 1}
    reference = ""
    if pairs and len(pairs) >= len(singles):
        seconds = {v[1][1] for v in pairs.values()}
        if len(seconds) == 1 and len(pairs) > 2:
            reference = seconds.pop()
            out = {n: (v[1][0], (v[1][0],)) for n, v in pairs.items()}
            return "referential", _dedupe(out), reference
        return "bipolar-longitudinal", _dedupe(pairs), reference
    return "referential", _dedupe(singles), reference


def _dedupe(m: dict[str, tuple[str, tuple[str, ...]]]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Keep the first raw channel of each canonical name (CHB-MIT repeats T8-P8) and of each pair
    whose reverse is already there."""
    seen, out = set(), {}
    for raw, v in m.items():
        rev = "-".join(reversed(v[1])) if len(v[1]) == 2 else None
        if v[0] in seen or (rev and rev in seen):   # CHB-MIT has T7-P7 and P7-T7 (the same pair negated)
            continue
        seen.add(v[0])
        out[raw] = v
    return out


# Posterior dominant rhythm channels, in order of preference: (left, right).
PDR_REFERENTIAL = (("O1", "O2"), ("P3", "P4"))
PDR_BIPOLAR = (("P3-O1", "P4-O2"), ("P7-O1", "P8-O2"), ("C3-P3", "C4-P4"))
