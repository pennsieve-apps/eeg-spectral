"""The synthetic fixture through the processor (no Docker): planted values are recovered."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from processor import channels as C
from processor import inputs as I
from processor import main as M
from processor import spectral as S
from tests import make_fixture


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="session")
def fixture(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("fixture")
    make_fixture.main(d)
    return d


def run(monkeypatch, input_dir: Path, output_dir: Path, **env) -> int:
    for k in M.DEFAULTS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session")
def derivative_out(fixture, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture / "eeg", out, WINDOWED="true") == 0
    finally:
        mp.undo()
    return out / "eeg-spectral"


@pytest.fixture(scope="session")
def truth(fixture) -> dict:
    return json.loads((fixture / "truth.json").read_text())


def by(rs, **kw):
    return [r for r in rs if all(r[k] == v for k, v in kw.items())]


# ---------------------------------------------------------------- planted values

def test_aperiodic_exponent_recovered(derivative_out, truth):
    ap = rows(derivative_out / "aperiodic.csv")
    for key in ("SYN01", "SYN02"):
        rid = truth[key]["recording_id"]
        ex = np.array([float(r["exponent"]) for r in by(ap, recording_id=rid)])
        assert len(ex) == 18
        assert abs(np.median(ex) - truth["exponent"]) < 0.1, (key, ex)
        assert np.all(np.abs(ex - truth["exponent"]) < 0.25), (key, ex)
        assert all(float(r["r_squared"]) > 0.9 and r["flag"] == "" for r in by(ap, recording_id=rid))


def test_pdr_frequency_asymmetry_reactivity(derivative_out, truth):
    pdr = {r["recording_id"]: r for r in rows(derivative_out / "pdr.csv")}
    ref = pdr[truth["SYN01"]["recording_id"]]
    assert abs(float(ref["pdr_hz"]) - truth["alpha_hz"]) < 0.25
    assert abs(float(ref["pdr_left_hz"]) - truth["alpha_hz"]) < 0.25
    assert abs(float(ref["pdr_right_hz"]) - truth["alpha_hz"]) < 0.25
    assert float(ref["asymmetry"]) < -0.1           # planted: left (O1) weaker than right (O2)
    assert float(ref["reactivity"]) > 0.8           # eyes open: alpha at 20 % amplitude (4 % power)
    # eyes closed 0-150 and 210-360, minus the bad segment 60-70 and the seizure exclusion from 240 s
    assert abs(float(ref["n_eyes_closed_s"]) - 170) <= 4
    assert "eyes-closed" in ref["method"] and "O1/O2" in ref["method"]
    bip = pdr[truth["SYN02"]["recording_id"]]
    assert abs(float(bip["pdr_hz"]) - truth["alpha_hz"]) < 0.25
    assert "P3-O1/P4-O2" in bip["method"] and "no eyes-closed annotation" in bip["method"]
    assert bip["reactivity"] == ""


def test_band_power_sums_and_alpha_topography(derivative_out, truth):
    bp = rows(derivative_out / "bandpower.csv")
    rid = truth["SYN01"]["recording_id"]
    for ch in ("O2", "Fp1", "T7"):
        rel = [float(r["rel_power"]) for r in by(bp, recording_id=rid, channel=ch) if r["band"] != "broadband"]
        assert abs(sum(rel) - 1) < 1e-6
        bb = by(bp, recording_id=rid, channel=ch, band="broadband")[0]
        assert float(bb["rel_power"]) == pytest.approx(1.0)
        assert float(bb["log_power"]) == pytest.approx(math.log10(float(bb["abs_power"])), abs=1e-4)
    alpha = {r["channel"]: float(r["rel_power"]) for r in by(bp, recording_id=rid, band="alpha")}
    assert max(alpha, key=alpha.get) == "O2"
    assert alpha["O2"] > alpha["O1"] > alpha["Fp1"]


def test_bad_channel_and_seizure_and_bad_segment_left_out(derivative_out, truth):
    prov = json.loads((derivative_out / "provenance.json").read_text())
    rec = prov["recordings"][truth["SYN01"]["recording_id"]]
    assert "Cz" not in rec["channels"] and len(rec["channels"]) == 18
    # 0-60 and 70-240 s analysable (bad segment 60-70; seizure at 300 s with a 60 s margin to the end)
    assert rec["analysable_intervals_s"] == [[0.0, 60.0], [70.0, 240.0]]
    assert rec["seizures_excluded"] == 1 and rec["bad_segments"] == 1
    assert rec["montage"] == "average"
    bip = prov["recordings"][truth["SYN02"]["recording_id"]]
    assert bip["montage"] == "bipolar-longitudinal" and bip["channels"][0] == "Fp1-F7"


def test_electrode_pop_rejected_for_its_channel_only(derivative_out, truth):
    prov = json.loads((derivative_out / "provenance.json").read_text())
    rec = prov["recordings"][truth["SYN01"]["recording_id"]]
    assert rec["rejected_channel_windows"] >= 1        # T8 at 100 s: windows 96-100, 98-102, 100-104
    ap = {r["channel"]: float(r["exponent"]) for r in rows(derivative_out / "aperiodic.csv")
          if r["recording_id"] == truth["SYN01"]["recording_id"]}
    assert abs(ap["T8"] - truth["exponent"]) < 0.1


def test_asymmetry_and_slowing_point_to_left_temporal(derivative_out, truth):
    rid = truth["SYN01"]["recording_id"]
    asym = rows(derivative_out / "asymmetry.csv")
    delta = {r["pair"]: (float(r["bsi"]), float(r["signed"])) for r in by(asym, recording_id=rid, band="delta")}
    assert max(delta, key=lambda k: delta[k][0]) == "T7/T8"
    assert delta["T7/T8"][1] < -0.3                    # signed (R-L)/(R+L): more delta on the left
    rec = by(asym, recording_id=rid, pair="recording")
    assert {r["band"] for r in rec} == {"delta", "theta", "alpha", "beta", "gamma", "rbsi"}
    assert 0 < float(by(rec, band="rbsi")[0]["bsi"]) < 0.2
    sl = {r["channel"]: r for r in by(rows(derivative_out / "slowing.csv"), recording_id=rid)}
    assert float(sl["T7"]["delta_theta_over_alpha_beta"]) > 1.3 * float(sl["T8"]["delta_theta_over_alpha_beta"])


def test_features_shape(derivative_out, truth):
    ft = rows(derivative_out / "features.csv")
    assert list(ft[0]) == ["recording_id", "channel", "feature", "value"]
    names = {r["feature"] for r in ft}
    expected = {f"relpow_{b}" for b in ("delta", "theta", "alpha", "beta", "gamma")} | \
        {f"logpow_{b}" for b in ("delta", "theta", "alpha", "beta", "gamma")} | \
        {"aperiodic_exponent", "aperiodic_offset", "pdr_hz", "theta_alpha_ratio", "bsi"}
    assert names == expected
    rid = truth["SYN01"]["recording_id"]
    f = {(r["channel"], r["feature"]): r["value"] for r in by(ft, recording_id=rid)}
    assert f[("Fz", "bsi")] == ""                      # midline: no homologous channel
    assert f[("O1", "bsi")] == f[("O2", "bsi")] != ""
    assert abs(float(f[("O2", "pdr_hz")]) - truth["alpha_hz"]) < 0.25


def test_windowed(derivative_out, truth):
    w = rows(derivative_out / "windowed.csv")
    rid = truth["SYN01"]["recording_id"]
    starts = sorted({float(r["window_start_s"]) for r in by(w, recording_id=rid)})
    assert starts == [0.0, 60.0, 120.0, 180.0]
    eo = [float(r["value"]) for r in by(w, recording_id=rid, channel="O2", feature="relpow_alpha")]
    assert eo[3] < eo[0]                               # 180-240: half eyes open


def test_summary_and_provenance(derivative_out, truth):
    s = json.loads((derivative_out / "summary.json").read_text())
    assert s["app"] == "eeg-spectral" and s["version"] == "0.1.0"
    head = s["recordings"][truth["SYN01"]["recording_id"]]
    assert isinstance(head["pdr_hz"], float) and isinstance(head["rbsi"], float)
    assert len(s["figures"]) == 6
    for fig in s["figures"]:
        assert (derivative_out / fig["path"]).is_file() and len(fig["caption"]) > 80
        assert "research output" in fig["caption"].lower()
    for t in s["tables"]:
        assert (derivative_out / t["path"]).is_file()
    prov = json.loads((derivative_out / "provenance.json").read_text())
    assert set(prov) == {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                         "wall_seconds", "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"}
    assert prov["tools"]["fooof"] and prov["citations"]
    assert set(prov["threads"]) == {"requested", "used", "why"}


# ---------------------------------------------------------------- other inputs

def test_plain_bids_siena_style(fixture, tmp_path, monkeypatch, truth):
    assert run(monkeypatch, fixture / "plain", tmp_path) == 0
    out = tmp_path / "eeg-spectral"
    prov = json.loads((out / "provenance.json").read_text())
    rec = prov["recordings"][truth["SIE01"]["recording_id"]]
    assert rec["montage"] == "average"
    assert "T7" in rec["channels"] and "P8" in rec["channels"] and "T3" not in rec["channels"]
    assert "EKG" not in " ".join(rec["channels"]) and len(rec["channels"]) == 19
    assert rec["analysable_intervals_s"] == [[0.0, 40.0]]   # sz_foc_ia at 100 s, 60 s margin
    pdr = rows(out / "pdr.csv")[0]
    assert abs(float(pdr["pdr_hz"]) - truth["alpha_hz"]) < 0.3


def test_loose_bipolar_edf_with_chbmit_summary(fixture, tmp_path, monkeypatch, truth):
    assert run(monkeypatch, fixture / "loose", tmp_path, FIGURES="false") == 0
    out = tmp_path / "eeg-spectral"
    prov = json.loads((out / "provenance.json").read_text())
    rec = prov["recordings"][truth["chb99"]["recording_id"]]
    assert rec["montage"] == "bipolar-longitudinal" and rec["seizures_excluded"] == 1
    assert rec["analysable_intervals_s"] == [[0.0, 20.0]]
    assert not (out / "figures").exists()


def test_everything_side_by_side(fixture, tmp_path, monkeypatch):
    """Ports merged at the root of INPUT_DIR, as symlinks."""
    inp = tmp_path / "in"
    inp.mkdir()
    for d in ("eeg", "plain", "loose"):
        os.symlink(fixture / d, inp / d)
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false") == 0
    prov = json.loads((tmp_path / "out" / "eeg-spectral" / "provenance.json").read_text())
    assert len(prov["recordings"]) == 4 and all(r["status"] == "ok" for r in prov["recordings"].values())


def test_label_adds_a_level(fixture, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture / "loose", tmp_path, LABEL="second", FIGURES="false") == 0
    assert (tmp_path / "eeg-spectral" / "second" / "features.csv").is_file()


def test_no_input_exit_1(tmp_path, monkeypatch):
    (tmp_path / "in").mkdir()
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out") == 1


@pytest.mark.parametrize("env", [{"WELCH_OVERLAP": "1.5"}, {"BANDS": "{bad"}, {"FIT_RANGE": "45-1"},
                                 {"APERIODIC_MODE": "lorentz"}, {"REJECT": "maybe"}, {"THREADS": "x"},
                                 {"PDR_WINDOWS": "sometimes"}])
def test_bad_parameters_exit_2(fixture, tmp_path, monkeypatch, env):
    assert run(monkeypatch, fixture / "loose", tmp_path, **env) == 2


def test_failed_recording_is_skipped(fixture, tmp_path, monkeypatch):
    inp = tmp_path / "in"
    inp.mkdir()
    os.symlink(fixture / "loose" / "chb99_01.edf", inp / "chb99_01.edf")
    (inp / "broken_02.edf").write_bytes(b"not an edf")
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false") == 0
    prov = json.loads((tmp_path / "out" / "eeg-spectral" / "provenance.json").read_text())
    assert prov["recordings"]["sub-broken_task-eeg_run-02"]["status"] == "failed"


# ---------------------------------------------------------------- units

def test_channel_names():
    assert C.canonical("EEG Fp1-Ref") == ("Fp1", ("Fp1",))
    assert C.canonical("T3-Avg") == ("T7", ("T7",))
    assert C.canonical("FP1-F7") == ("Fp1-F7", ("Fp1", "F7"))
    assert C.canonical("T8-P8-0") == ("T8-P8", ("T8", "P8"))
    assert C.canonical("ECG")[0] is None and C.canonical("EKG")[0] is None
    mont, mapping, ref = C.classify(["Fp1-A1", "F3-A1", "O1-A1", "O2-A1"])
    assert mont == "referential" and ref == "A1" and mapping["O1-A1"][0] == "O1"
    assert C.classify(["FP1-F7", "F7-T7", "FZ-CZ"])[0] == "bipolar-longitudinal"
    assert C.mirror(("Fp1", "F7")) == "Fp2-F8" and C.mirror(("Fz", "Cz")) is None
    assert C.hemisphere(("O1",)) == "L" and C.region(("P7", "O1")) == "occipital"


def test_event_classes():
    assert I.classify_event("sz_foc_ia") == "seizure"
    assert I.classify_event("bad_segment") == "bad" and I.classify_event("artifact") == "bad"
    assert I.classify_event("eyes_closed") == "eyes_closed" and I.classify_event("Eyes Open") == "eyes_open"
    assert I.classify_event("bckg") is None


def test_rbsi_definition():
    f = np.arange(0, 30.25, 0.25)
    right, left = np.full_like(f, 2.0), np.full_like(f, 1.0)
    u, s = S.bsi(right, left, f, 1, 25)
    assert u == pytest.approx(1 / 3) and s == pytest.approx(1 / 3)
    u, s = S.bsi(left, right, f, 1, 25)
    assert s == pytest.approx(-1 / 3)


def test_specparam_on_known_spectrum():
    f = np.arange(0, 60.25, 0.25)
    fs = np.maximum(f, 0.25)
    psd = 10 ** (2 - 1.8 * np.log10(fs)) + 30 * np.exp(-((f - 10.5) ** 2) / (2 * 0.8 ** 2))
    fit = S.fit_spectrum(f, psd, S.Params())
    assert fit.ok and abs(fit.exponent - 1.8) < 0.1
    pk = S.strongest_peak(fit, 6, 14)
    assert pk is not None and abs(pk[0] - 10.5) < 0.2


def test_pdr_alpha_richest_windows(fixture, tmp_path, monkeypatch, truth):
    assert run(monkeypatch, fixture / "eeg", tmp_path, PDR_WINDOWS="alpha-richest", FIGURES="false") == 0
    pdr = {r["recording_id"]: r for r in rows(tmp_path / "eeg-spectral" / "pdr.csv")}
    bip = pdr[truth["SYN02"]["recording_id"]]
    assert abs(float(bip["pdr_hz"]) - truth["alpha_hz"]) < 0.25 and "most posterior" in bip["method"]
    # annotated eyes-closed EEG wins over the heuristic
    assert "eyes-closed windows" in pdr[truth["SYN01"]["recording_id"]]["method"]


def test_derivatives_folder_of_a_raw_dataset_skipped(fixture, tmp_path, monkeypatch):
    import shutil

    inp = tmp_path / "ds"
    shutil.copytree(fixture / "plain", inp)
    d = inp / "derivatives" / "sub-SIE01" / "ses-01" / "eeg"
    d.mkdir(parents=True)
    shutil.copy(next((fixture / "plain").rglob("*_eeg.edf")), d / "sub-SIE01_ses-01_task-hfo_run-01_eeg.edf")
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false") == 0
    prov = json.loads((tmp_path / "out" / "eeg-spectral" / "provenance.json").read_text())
    assert list(prov["recordings"]) == ["sub-SIE01_ses-01_task-szMonitoring_run-01"]


def test_reversed_bipolar_pair_dropped():
    mont, mapping, _ = C.classify(["FP1-F7", "F7-T7", "T7-P7", "P7-T7", "T7-FT9", "FT9-FT10", "T8-P8-0", "T8-P8-1"])
    assert mont == "bipolar-longitudinal"
    assert "P7-T7" not in mapping and "T8-P8-1" not in mapping and mapping["T7-FT9"][0] == "T7-FT9"


def test_edf_without_physical_dimension_read_as_microvolts(tmp_path, monkeypatch, truth):
    """ds003555's EDFs leave the physical dimension empty; MNE then assumes volts."""
    from edfio import Edf, EdfSignal

    rng = np.random.default_rng(1)
    el = make_fixture.electrodes(120.0, rng, delta=False)
    inp = tmp_path / "in"
    inp.mkdir()
    sigs = [EdfSignal(el[e] - np.mean(list(el.values()), axis=0), sampling_frequency=make_fixture.FS, label=e,
                      physical_dimension="", physical_range=(-500, 500)) for e in make_fixture.ELECTRODES]
    Edf(sigs).write(inp / "child_01.edf")
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false") == 0
    out = tmp_path / "out" / "eeg-spectral"
    prov = json.loads((out / "provenance.json").read_text())
    rec = prov["recordings"]["sub-child_task-eeg_run-01"]
    assert any("physical dimension" in n for n in rec["notes"]) and rec["rejected_windows"] == 0
    assert abs(float(rows(out / "pdr.csv")[0]["pdr_hz"]) - truth["alpha_hz"]) < 0.3


def test_lowpass_claim_checked_against_spectrum(fixture, tmp_path, monkeypatch, truth):
    """A _channels.tsv high_cutoff of 30 Hz the data do not show (BIDS Siena) keeps the 1-45 Hz fit."""
    import shutil

    inp = tmp_path / "eeg"
    shutil.copytree(fixture / "eeg", inp)
    for ch in inp.rglob("*SYN02*_channels.tsv"):
        ch.write_text(ch.read_text().replace("\t128\t", "\t30\t"))
    assert run(monkeypatch, inp, tmp_path / "out", FIGURES="false") == 0
    rec = json.loads((tmp_path / "out" / "eeg-spectral" / "provenance.json").read_text())["recordings"][
        truth["SYN02"]["recording_id"]]
    assert rec["fit_range_hz"] == [1.0, 45.0] and any("not seen in the spectrum" in n for n in rec["notes"])
