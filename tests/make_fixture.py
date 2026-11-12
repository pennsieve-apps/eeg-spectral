"""
A tiny synthetic scalp EEG input for the tests, `make run` and the release review
(app.yml `review.generate`), with planted, known spectral properties:

  <out>/eeg/       an eeg-prep-style BIDS-EEG derivative (conventions.md):
                   dataset_description.json, recordings.csv, and per recording
                   *_desc-preproc_{eeg.edf, eeg.json, channels.tsv, events.tsv}
    sub-SYN01  referential, 19 channels 10-20, common average reference, 256 Hz, 360 s
               - background: independent 1/f noise per electrode with exponent 1.5 (PSD ~ f^-1.5)
               - posterior alpha at 9.5 Hz, eyes closed 0-150 s and 210-360 s, eyes open 150-210 s
                 (alpha at 20 %): weights O2 1.0, O1 0.7 (left weaker), P4 0.6, P3 0.45, ...
               - left temporal slowing: 1.5-3.5 Hz noise on T7 (and less on F7, P7)
               - a `bad_segment` 60-70 s with a 400 µV slow artifact on Fp1/Fp2 (must be left out)
               - an unannotated 800 µV electrode pop on T8 at 100 s (auto rejection)
               - a seizure at 300 s (15 s, 4 Hz, 150 µV on every channel; must be left out, margin 60 s)
               - Cz marked bad in _channels.tsv (60 Hz noise in the file)
    sub-SYN02  bipolar-only (CHB-MIT style 'FP1-F7' names, 18 longitudinal pairs), 256 Hz, 240 s,
               same background and alpha (eyes closed throughout, not annotated), no events
  <out>/plain/     a plain BIDS-EEG dataset as the BIDS Siena database has it: 'Fp1-Avg' names with
                   the old T3/T4/T5/T6, an EKG channel, no _channels.tsv, HED-SCORE seizure rows
                   (eventType sz_foc_ia) in _events.tsv; sub-SIE01, 150 s
  <out>/loose/     chb99_01.edf: loose bipolar EDF and chb99-summary.txt with one seizure
  <out>/truth.json the planted values

Usage: python -m tests.make_fixture <out-dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SEED = 42
FS = 256.0
EXPONENT = 1.5
ALPHA_HZ = 9.5
ELECTRODES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4",
              "P8", "O1", "O2"]
OLD = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
ALPHA_W = {"O2": 1.0, "O1": 0.7, "P4": 0.6, "P3": 0.45, "Pz": 0.55, "P8": 0.45, "P7": 0.35, "C4": 0.15, "C3": 0.12,
           "Cz": 0.12, "T8": 0.1, "T7": 0.08}
DELTA_W = {"T7": 1.0, "F7": 0.5, "P7": 0.5}
BIPOLAR = [("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"), ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"),
           ("P3", "O1"), ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"), ("Fp2", "F4"), ("F4", "C4"),
           ("C4", "P4"), ("P4", "O2"), ("Fz", "Cz"), ("Cz", "Pz")]


def powerlaw(n: int, rng: np.random.Generator, exponent: float, std: float) -> np.ndarray:
    """Gaussian noise with PSD ~ f^-exponent (flat below 0.5 Hz), `std` µV."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    f = np.maximum(f, 0.5)
    x = np.fft.irfft(spec * f ** (-exponent / 2), n)
    return x / x.std() * std


def bandnoise(n: int, rng: np.random.Generator, lo: float, hi: float, std: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std() * std


def alpha_source(n: int, rng: np.random.Generator, amp: float, eyes_open: list[tuple[float, float]]) -> np.ndarray:
    """A 9.5 Hz rhythm with slow amplitude and phase wander; amplitude x0.2 with eyes open."""
    t = np.arange(n) / FS
    phase = 2 * np.pi * ALPHA_HZ * t + np.cumsum(rng.normal(0, 0.02, n))
    env = 1 + 0.25 * bandnoise(n, rng, 0.05, 0.5, 1.0)
    gain = np.ones(n)
    for a, b in eyes_open:
        gain[int(a * FS): int(b * FS)] = 0.2
    return amp * np.sqrt(2) * env * gain * np.sin(phase)


def electrodes(duration: float, rng, eyes_open=(), delta=True, alpha_amp=30.0) -> dict[str, np.ndarray]:
    n = int(duration * FS)
    a = alpha_source(n, rng, alpha_amp, list(eyes_open))
    d = bandnoise(n, rng, 1.5, 3.5, 14.0)
    out = {}
    for e in ELECTRODES:
        x = powerlaw(n, rng, EXPONENT, 15.0) + ALPHA_W.get(e, 0.04) * a
        if delta:
            x = x + DELTA_W.get(e, 0.0) * d
        out[e] = x
    return out


def write_edf(path: Path, data: np.ndarray, names: list[str], fs: float = FS) -> None:
    from edfio import Edf, EdfSignal

    sigs = []
    for x, name in zip(data, names):
        r = max(1.0, float(np.max(np.abs(x))) * 1.1)
        sigs.append(EdfSignal(np.asarray(x, float), sampling_frequency=fs, label=name, physical_dimension="uV",
                              physical_range=(-r, r)))
    Edf(sigs).write(path)


def write_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def hemi(contacts: list[str]) -> str:
    s = {("Z" if c[-1] in "zZ" else ("L" if int(c[-1]) % 2 else "R")) for c in contacts}
    return s.pop() if len(s) == 1 else "Z"


CH_COLS = ["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency", "status", "status_description",
           "contacts", "hemisphere", "region", "original_name"]
EV_COLS = ["onset", "duration", "trial_type", "source", "channels"]
REC_COLS = ["recording_id", "subject", "session", "task", "run", "path", "sfreq", "n_channels", "n_good", "duration_s",
            "montage", "reference", "kind", "start_time", "n_seizures", "source_format", "age", "sex"]


def write_derivative_recording(root: Path, sub: str, data: np.ndarray, names: list[str], contacts: list[list[str]],
                               montage: str, bad=(), events=(), original=None) -> dict:
    stem = f"sub-{sub}_ses-01_task-rest_run-01"
    d = root / f"sub-{sub}" / "ses-01" / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    write_edf(d / f"{stem}_desc-preproc_eeg.edf", data, names)
    write_tsv(d / f"{stem}_desc-preproc_channels.tsv", CH_COLS, [
        {"name": n, "type": "EEG", "units": "µV", "low_cutoff": "0.5", "high_cutoff": f"{FS / 2:g}",
         "sampling_frequency": f"{FS:g}", "status": "bad" if n in bad else "good",
         "status_description": "planted bad channel" if n in bad else "n/a", "contacts": "|".join(c),
         "hemisphere": hemi(c), "region": "n/a", "original_name": (original or names)[i]}
        for i, (n, c) in enumerate(zip(names, contacts))])
    write_tsv(d / f"{stem}_desc-preproc_events.tsv", EV_COLS, [
        {"onset": f"{o:g}", "duration": f"{du:g}", "trial_type": t, "source": "annotation", "channels": "n/a"}
        for o, du, t in events])
    (d / f"{stem}_desc-preproc_eeg.json").write_text(json.dumps({
        "SamplingFrequency": FS, "EEGReference": "average" if montage == "average" else "bipolar",
        "PowerLineFrequency": 50, "RecordingDuration": data.shape[1] / FS, "EEGChannelCount": len(names)}, indent=2))
    return {"recording_id": stem, "subject": sub, "session": "01", "task": "rest", "run": "01",
            "path": f"sub-{sub}/ses-01/eeg/{stem}_desc-preproc_eeg.edf", "sfreq": f"{FS:g}", "n_channels": len(names),
            "n_good": len(names) - len(bad), "duration_s": f"{data.shape[1] / FS:g}", "montage": montage,
            "reference": "average" if montage == "average" else "", "kind": "routine", "start_time": "",
            "n_seizures": sum(1 for e in events if e[2] == "seizure"), "source_format": "bids-edf", "age": "",
            "sex": ""}


def referential(rng) -> tuple[np.ndarray, list[str], dict]:
    dur = 360.0
    eo = [(150.0, 210.0)]
    el = electrodes(dur, rng, eyes_open=eo)
    n = int(dur * FS)
    t = np.arange(n) / FS
    # bad_segment 60-70 s: slow 400 µV artifact on Fp1/Fp2
    i0, i1 = int(60 * FS), int(70 * FS)
    for e in ("Fp1", "Fp2"):
        el[e][i0:i1] += 400 * np.sin(2 * np.pi * 1.2 * t[i0:i1])
    # unannotated electrode pop on T8 at 100 s
    j = int(100 * FS)
    el["T8"][j: j + int(0.8 * FS)] += 800 * np.exp(-np.arange(int(0.8 * FS)) / (0.15 * FS))
    # seizure at 300 s, 15 s of 4 Hz on every channel
    k0, k1 = int(300 * FS), int(315 * FS)
    for e in ELECTRODES:
        el[e][k0:k1] += 150 * np.sin(2 * np.pi * 4.0 * t[k0:k1])
    # Cz is bad: replace with line noise; average reference over the good channels
    el["Cz"] = 40 * np.sin(2 * np.pi * 60 * t) + rng.normal(0, 2, n)
    good = [e for e in ELECTRODES if e != "Cz"]
    avg = np.mean([el[e] for e in good], axis=0)
    data = np.array([el[e] - avg if e != "Cz" else el[e] for e in ELECTRODES])
    events = [(0.0, 150.0, "eyes_closed"), (150.0, 60.0, "eyes_open"), (210.0, 150.0, "eyes_closed"),
              (60.0, 10.0, "bad_segment"), (300.0, 15.0, "seizure")]
    return data, events, {"duration_s": dur}


def bipolar(rng, duration: float = 240.0) -> tuple[np.ndarray, list[str], list[list[str]]]:
    el = electrodes(duration, rng, delta=False)
    data = np.array([el[a] - el[b] for a, b in BIPOLAR])
    names = [f"{a.upper()}-{b.upper()}" for a, b in BIPOLAR]
    return data, names, [[a, b] for a, b in BIPOLAR]


def main(out: Path) -> None:
    rng = np.random.default_rng(SEED)
    out = Path(out)
    root = out / "eeg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "eeg-spectral synthetic fixture", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "eeg-spectral tests/make_fixture.py"}]}, indent=2))
    rows = []
    data, events, info = referential(rng)
    rows.append(write_derivative_recording(root, "SYN01", data, ELECTRODES, [[e] for e in ELECTRODES], "average",
                                           bad={"Cz"}, events=events))
    bdata, bnames, bcontacts = bipolar(rng)
    canon = [f"{a}-{b}" for a, b in BIPOLAR]
    rows.append(write_derivative_recording(root, "SYN02", bdata, canon, bcontacts, "bipolar-longitudinal",
                                           original=bnames))
    with open(root / "recordings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REC_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(root / "participants.tsv", "w", encoding="utf-8") as fh:
        fh.write("participant_id\tage\tsex\nsub-SYN01\tn/a\tn/a\nsub-SYN02\tn/a\tn/a\n")

    # plain BIDS, Siena style
    plain = out / "plain"
    d = plain / "sub-SIE01" / "ses-01" / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    (plain / "dataset_description.json").write_text(json.dumps({"Name": "Siena-style", "BIDSVersion": "1.8.0"}))
    el = electrodes(150.0, rng, delta=False)
    avg = np.mean([el[e] for e in ELECTRODES], axis=0)
    pdata = np.array([el[e] - avg for e in ELECTRODES] + [rng.normal(0, 200, int(150 * FS))])
    pnames = [f"{OLD.get(e, e)}-Avg" for e in ELECTRODES] + ["EKG"]
    stem = "sub-SIE01_ses-01_task-szMonitoring_run-01"
    write_edf(d / f"{stem}_eeg.edf", pdata, pnames)
    (d / f"{stem}_eeg.json").write_text(json.dumps({"EEGReference": "Common Average", "SamplingFrequency": FS}))
    with open(d / f"{stem}_events.tsv", "w", encoding="utf-8") as fh:
        fh.write("onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n"
                 "100.00\t20.00\tsz_foc_ia\tn/a\tT4,T6\t2016-01-01 02:18:17\t150.00\n")

    # loose bipolar EDF with a CHB-MIT summary
    loose = out / "loose"
    loose.mkdir(parents=True, exist_ok=True)
    ldata, lnames, _ = bipolar(rng, 120.0)
    write_edf(loose / "chb99_01.edf", ldata, lnames + [])
    (loose / "chb99-summary.txt").write_text(
        "File Name: chb99_01.edf\nFile Start Time: 10:00:00\nFile End Time: 10:02:00\n"
        "Number of Seizures in File: 1\nSeizure Start Time: 80 seconds\nSeizure End Time: 95 seconds\n")

    truth = {"alpha_hz": ALPHA_HZ, "exponent": EXPONENT, "fs": FS,
             "SYN01": {"recording_id": rows[0]["recording_id"], "duration_s": info["duration_s"],
                       "eyes_closed": [[0, 150], [210, 360]], "eyes_open": [[150, 210]], "bad_segment": [60, 70],
                       "seizure": [300, 315], "pop": [100, 100.8], "bad_channel": "Cz",
                       "weaker_alpha_side": "left", "slowing_channel": "T7"},
             "SYN02": {"recording_id": rows[1]["recording_id"], "duration_s": 240.0},
             "SIE01": {"recording_id": stem, "seizure": [100, 120]},
             "chb99": {"recording_id": "sub-chb99_task-eeg_run-01", "seizure": [80, 95]}}
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"fixture written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/fixture"))
