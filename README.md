# eeg-spectral: spectral biomarkers of clinical scalp EEG

eeg-spectral is built on **specparam (FOOOF)**, developed by Thomas Donoghue, Bradley Voytek and
colleagues in the Voytek Lab at UC San Diego ([repository](https://github.com/fooof-tools/fooof),
[paper](https://doi.org/10.1038/s41593-020-00744-x)), and on **MNE-Python**, developed by
Alexandre Gramfort and the MNE community ([repository](https://github.com/mne-tools/mne-python),
[paper](https://doi.org/10.3389/fnins.2013.00267)). The posterior dominant rhythm follows the
method of **Shaun Lodder and Michel van Putten** (University of Twente;
[paper](https://doi.org/10.1016/j.jneumeth.2011.06.008)), with the large-cohort validation of
**Ivan Zibrandtsen and Troels Kjaer** ([paper](https://doi.org/10.1016/j.cnp.2020.11.001)) as the
reference, and the hemispheric asymmetry is **Michel van Putten's revised Brain Symmetry Index**
([paper](https://doi.org/10.1016/j.clinph.2007.07.019)). Pennsieve packages these methods as an app;
it did not develop them. The demo data are the **Siena Scalp EEG Database** of Paolo Detti and
colleagues at the University of Siena (BIDS conversion by Jonathan Dan et al., EPFL), the
**CHB-MIT Scalp EEG Database** of Ali Shoeb and John Guttag (MIT, with Children's Hospital Boston),
and **OpenNeuro ds003555** of Dorottya Cserpan, Johannes Sarnthein and colleagues (University
Hospital Zurich).

One step of the **Scalp EEG Interictal Biomarkers** workflow
(`eeg-prep → {eeg-spectral, eeg-connectivity, eeg-sleep → eeg-spikes} → eeg-report`).

> **Research output.** Every number this app writes (PDR frequency, asymmetry, slowing ratios,
> aperiodic exponent) is a quantitative EEG measure for research, to be reviewed by a clinician
> against the raw EEG. It is not a clinical EEG interpretation.

## What it does

**For the epileptologist.** The app summarises the background of each recording in the numbers an
EEG reader estimates by eye: the frequency, amplitude, symmetry and reactivity of the **posterior
dominant rhythm** (PDR); **focal or lateralised slowing** (theta/alpha and
(delta+theta)/(alpha+beta) power ratios per channel, and the revised Brain Symmetry Index per
homologous electrode pair and band, with the side); and the **power in each band** per channel. Three
figures per recording show where to look: the spectrum of each scalp region, scalp maps of alpha
power and of the 1/f exponent, and the left and right posterior spectra with the PDR marked.
Seizures (plus 60 s on each side), annotated artifacts and bad segments, and automatically detected
artifacts are left out, so the numbers describe interictal background.

**For the researcher.** Each channel's power spectrum is split by specparam into an **aperiodic
(1/f) component** (exponent and offset; the exponent is a proposed marker of cortical
excitation/inhibition balance) and **oscillatory peaks** (centre frequency, power above the 1/f
component, bandwidth). Everything is written in long tables keyed by `recording_id` and `channel`,
plus the shared `features.csv` (`relpow_<band>`, `logpow_<band>`, `aperiodic_exponent`,
`aperiodic_offset`, `pdr_hz`, `theta_alpha_ratio`, `bsi`) that the iEEG apps and a future
normative step read.

### Method

1. **Input.** The eeg-prep derivative (`recordings.csv`, `*_desc-preproc_eeg.edf`, `_channels.tsv`,
   `_events.tsv`), any BIDS-EEG dataset, or loose EDF/BDF/BrainVision/FIF/EEGLAB files, found
   recursively. Channel names are canonicalised as eeg-prep does (`EEG Fp1-Ref`, `Fp1-Avg` → `Fp1`;
   `T3/T4/T5/T6` → `T7/T8/P7/P8`); non-EEG channels and channels marked bad are left out.
   **Referential and bipolar-only recordings both work**: pairs such as `FP1-F7` are kept as
   bipolar channels (`montage` = `bipolar-longitudinal` in `provenance.json`), and pairs that all
   share their second electrode (`Fp1-A1`, `F3-A1`, …) are read as referential to it.
2. **Windows.** 4 s Hamming windows with 50 % overlap (0.25 Hz resolution) over the analysable
   EEG: the recording minus `bad_segment`/`artifact` events and minus each annotated seizure with
   `SEIZURE_MARGIN_S` (60 s) before and after. Read 60 s at a time, so memory does not grow with
   the length of the recording.
3. **Automatic rejection** (`REJECT=true`): a window is left out when at least two channels (or
   10 %) exceed `REJECT_UV` (500 µV) peak-to-peak, when every channel is flat, or when its median
   20–45 Hz power is a robust outlier (median + 4 × 1.4826 MAD over the recording's windows:
   muscle); a window where a single channel exceeds `REJECT_UV` (an electrode pop) is left out for
   that channel only; a channel flat or above `REJECT_UV` in most windows is left out. The
   outlier test uses 20–45 Hz, not broadband power, so that sleep slow waves in long recordings
   are not taken for artifacts.
4. **Spectra** (MNE `psd_array_welch`), averaged over the kept windows per channel.
5. **Band power**: absolute (µV², trapezoid over the band, edges included), relative (share of the
   broadband 1–45 Hz power) and log10; bands delta 1–4, theta 4–8, alpha 8–13, beta 13–30,
   gamma 30–45 Hz.
6. **specparam** (fooof 1.1.1) on each channel's spectrum: fixed aperiodic mode, 1–45 Hz,
   `peak_width_limits` 1–8 Hz, `max_n_peaks` 6; `r_squared` is reported and fits below 0.9 are
   flagged.
7. **Posterior dominant rhythm** (after Lodder & van Putten 2011): the strongest specparam peak in
   6–14 Hz of the mean spectrum of O1 and O2 (P3/P4 when O1/O2 are missing; on bipolar recordings
   P3-O1/P4-O2, then P7-O1/P8-O2, then C3-P3/C4-P4), on **annotated eyes-closed EEG** when there
   are at least 5 windows of it, otherwise on all kept windows outside annotated sleep, with a
   warning (`PDR_WINDOWS=alpha-richest` uses the quarter of those windows with the most posterior
   6–14 Hz power instead). Left and right are fitted separately: `pdr_left_hz`, `pdr_right_hz`,
   and the amplitude asymmetry (L − R)/(L + R). The amplitude is the peak-to-peak amplitude of a
   sinusoid carrying the power the peak adds above the 1/f fit within ± its bandwidth. Reactivity,
   when eyes-open and eyes-closed EEG are both annotated, is 1 − (power within ± 1 Hz of the PDR
   with eyes open / with eyes closed): 0.5 means the PDR attenuates by half with eye opening.
8. **Revised Brain Symmetry Index** (van Putten 2007): at each frequency bin, the mean power of
   the right channels and of the left channels of the homologous pairs present, |R − L| / (R + L),
   averaged over the bins of 1–25 Hz (`rbsi`, 0 = symmetric, 1 = one side silent) and over each
   band's bins; per pair (Fp1/Fp2, F7/F8, …; on bipolar recordings Fp1-F7/Fp2-F8, …) the same
   expression on the two channels. The `signed` column is the mean of (R − L)/(R + L): positive
   means more power on the right, so lateralised left slowing gives a negative delta value.
9. **Slowing ratios** per channel from the absolute band powers.

## Ports

| Port | Direction | Content |
|---|---|---|
| `eeg` | input (subfolder) | eeg-prep's BIDS-EEG derivative, any BIDS-EEG dataset, or loose EEG files |
| `spectral` | output | `eeg-spectral/` with the tables, `figures/`, `summary.json`, `provenance.json` |

## Parameters

| Name | Default | What it does |
|---|---|---|
| `BANDS` | `conventions` | delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz, or a JSON object `{"delta": [1, 4], ...}` |
| `WELCH_WINDOW_S` | `4` | Welch window length (s) |
| `WELCH_OVERLAP` | `0.5` | Welch window overlap (fraction) |
| `FIT_RANGE` | `1-45` | specparam frequency range (Hz); cut below Nyquist and below a low-pass in `_channels.tsv` |
| `APERIODIC_MODE` | `fixed` | specparam aperiodic model: `fixed` or `knee` |
| `MAX_N_PEAKS` | `6` | most peaks specparam fits |
| `PEAK_WIDTH_LIMITS` | `1-8` | peak bandwidth limits (Hz) |
| `MIN_R_SQUARED` | `0.9` | fits below are flagged |
| `PDR_RANGE` | `6-14` | where the PDR peak is searched (Hz) |
| `PDR_WINDOWS` | `auto` | without eyes-closed annotations: `auto` = all artifact-free windows; `alpha-richest` = the quarter with most posterior 6–14 Hz power |
| `EXCLUDE_SEIZURE` | `true` | leave annotated seizures out |
| `SEIZURE_MARGIN_S` | `60` | seconds left out before and after each seizure |
| `REJECT` | `true` | automatic artifact rejection (above) |
| `REJECT_UV` | `500` | peak-to-peak threshold per 4 s window (µV) |
| `WINDOWED` | `false` | also write `windowed.csv` (time course) |
| `WINDOWED_S` | `60` | block length of `windowed.csv` (s) |
| `FIGURES` | `true` | write the figures |
| `CHUNK_S` | `60` | seconds read at a time |
| `THREADS` | `0` | BLAS threads; 0 = from the container's CPU quota, capped by memory |
| `LABEL` | (empty) | write to `eeg-spectral/<LABEL>/` |

## Outputs (under `OUTPUT_DIR/eeg-spectral/`)

| File | Columns |
|---|---|
| `bandpower.csv` | recording_id, channel, band (delta … gamma, broadband), abs_power (µV²), rel_power, log_power |
| `aperiodic.csv` | recording_id, channel, exponent, offset, knee (knee mode only), r_squared, error, n_peaks, flag |
| `peaks.csv` | recording_id, channel, cf_hz, power (log10 above the aperiodic fit), bandwidth (Hz) |
| `pdr.csv` | recording_id, pdr_hz, pdr_amplitude_uv, pdr_left_hz, pdr_right_hz, asymmetry, reactivity, n_eyes_closed_s, method |
| `asymmetry.csv` | recording_id, pair (`T7/T8`, or `recording`), band (bands + `rbsi` 1–25 Hz), bsi, signed |
| `slowing.csv` | recording_id, channel, theta_alpha_ratio, delta_theta_over_alpha_beta |
| `features.csv` | recording_id, channel, feature, value (shared long shape; `bsi` is the channel's pair rBSI, empty on the midline) |
| `windowed.csv` | recording_id, channel, window_start_s, feature, value (`WINDOWED=true`) |
| `figures/<recording_id>_{psd,topomaps,pdr}.png` | spectrum per region with the 1/f fit; topomaps of relative alpha power and of the exponent; PDR left vs right |
| `summary.json` | headline numbers per recording, the figures with captions (in order), tables, warnings |
| `provenance.json` | app, version, tools, parameters, inputs, timings, peak memory, threads, per-recording status, montage, intervals and rejection counts, citations |

A recording that fails (unreadable, too few EEG channels, sampling rate below 100 Hz) is logged and
listed as failed; the step fails only when every recording fails or nothing is found (exit 1); bad
parameters exit 2.

## Runtime and results on the demo data

Measured in Docker with **amd64 emulation on an M-series Mac** (so slower than on AWS), image
`eeg-spectral` 0.1.0 (135 MB compressed, 303 MB of Python packages). One recording at a time;
almost all of the time is reading the EDF and the FFTs, single-threaded.

| Input | Recordings | Wall time | Peak memory (cgroup) |
|---|---|---|---|
| Synthetic fixture (`make lambda-check`) | 2 (6 and 4 min) | 4.8 s | 204 MB |
| BIDS Siena, all of it, raw BIDS (`--cpus 2 --memory 2g`) | 41 runs, 0.5-13.5 h, 41 ok | 156-188 s | 835-898 MB (two runs) |
| Siena sub-01 run-00 alone, 13.5 h, Lambda-style (`--cpus 1`, read-only, uid 65534) | 1 | 16 s | 624 MB |
| CHB-MIT chb01_01/03/04/15/16, bipolar, loose EDF + summary | 5 x 1 h, 5 ok | 16-27 s | 369-460 MB |
| OpenNeuro ds003555 sub-01..03, 3 h of sleep EEG at 1024 Hz | 3, 3 ok | 25-47 s | 465 MB |

The cgroup peak includes the page cache of the EDF being read; per-window spectra above 256 MB
(recordings longer than about 10 h at 19 channels) go to a memory-mapped file in `$TMPDIR`. The
app is sized at **1 vCPU, 2 GB, 900 s**, within the Lambda limits: a 24 h, 25-channel recording is
estimated at under a minute.

On eeg-prep's Siena derivative (13 runs it had written, 0.6-13.5 h): 13 of 13 ok in 51 s, 718 MB;
far fewer windows rejected (median 1.4 %, eeg-prep had already cut bad segments and removed
artifact components); PDR found in 10 of 13 (7.7-12.8 Hz); aperiodic exponents lower than on the
raw files (median 0.99 vs 1.34), as expected after eeg-prep's filtering and ICA.

What it found on the raw BIDS inputs:

- **Siena** (adults with focal epilepsy, seizure-monitoring recordings, mostly without
  eyes-closed annotations): a PDR peak in 34 of 41 runs, median 8.8 Hz (6.5-13.5 Hz; the
  values above 12 Hz are sleep spindles in recordings made largely asleep, see Limitations);
  recording rBSI median 0.057 (0.015-0.209); median aperiodic exponent per run 1.34
  (0.78-2.04); 6 % of windows rejected in the median run, 45 % in the worst (raw, unfiltered
  clinical data with 5 mV artifacts). The side of the delta-band asymmetry (`signed`) matched the
  side of the electrodes in the seizure annotations in 9 of 13 patients with one-sided annotations
  (exploratory; small asymmetries, no clinical claim).
- **CHB-MIT chb01** (bipolar only): PDR 9.9-11.0 Hz in 4 of 5 hours; the reversed duplicate pair
  `P7-T7` and the `T8-P8-1` copy are left out, 21 of 23 channels used.
- **ds003555** (children, sleep EEG): PDR peaks at 12-13 Hz in two children are spindles, as
  expected for sleep; the EDFs carry no physical dimension and are read as µV (noted per recording).
- **Synthetic fixture**: planted 9.5 Hz alpha recovered at 9.51 Hz (referential) and 9.50 Hz
  (bipolar); planted exponent 1.5 recovered at 1.499 and 1.501 (median over channels); planted
  left-weaker alpha gives asymmetry -0.26; planted eyes-open attenuation gives reactivity 0.95;
  planted left temporal delta gives the largest delta BSI at T7/T8 with signed -0.49 (left).

Outputs on the demo data are in `workflow-scalp-eeg/data/work/{siena,chbmit,ds003555}/eeg-spectral/`
(Siena: on eeg-prep's derivative; the raw-BIDS run of all 41 runs is in `siena/eeg-spectral-raw-bids/`).


## Limitations

- **Not a clinical read.** The PDR is a spectral peak, not the visually identified rhythm: on
  recordings without annotated eyes-closed wakefulness (most long-term monitoring) it is computed
  over whatever state the patient was in, and drowsiness, sleep spindles (12–14 Hz) or a
  posterior theta can be reported instead; `method` and the warnings say which EEG was used. A
  missing PDR (`pdr_hz` empty) means no peak was found in 6–14 Hz, not that the rhythm is absent.
- Long recordings are averaged over wake and sleep; band powers and slowing ratios of a 12-hour
  recording are not those of a routine waking EEG. Use `WINDOWED=true` for the time course, or a
  hypnogram-aware analysis.
- The reference changes the numbers: amplitudes and the topography differ between average,
  linked-ear and bipolar montages, and between eeg-prep output and raw exports. Compare
  recordings in the same montage (`provenance.json` says which).
- Bipolar channels mix two electrodes; homologous bipolar pairs are mirrored electrode by
  electrode, and topomaps place each pair at the midpoint of its electrodes.
- The automatic rejection is deliberately simple (amplitude, flat, muscle-band outliers). It does
  not remove eye movements or ECG; run eeg-prep (ICA + ICLabel) first for that.
- The PDR method follows Lodder & van Putten 2011 and Zibrandtsen & Kjaer 2021 in spirit (a peak
  frequency of the posterior spectrum); it is not their software, and it has been checked here on
  synthetic data with known values and on the demo databases, not against clinical reads.
- The aperiodic exponent depends on the fit range; with a 1–45 Hz fixed-mode fit it is comparable
  only to exponents computed the same way.

## Licences

Our code: Apache-2.0 (`LICENSE`). Bundled components and their licences: `THIRD_PARTY.md`
(fooof Apache-2.0, MNE-Python BSD-3-Clause, NumPy/SciPy BSD-3-Clause, Matplotlib PSF-based; all
permissive). The demo databases are not bundled: Siena CC BY 4.0 (BIDS conversion; PhysioNet
original ODC-By 1.0), CHB-MIT ODC-By 1.0, ds003555 CC0.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in `CITATION.cff`.

- Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T, Lara AH, Wallis JD,
  Knight RT, Shestyuk A, Voytek B. Parameterizing neural power spectra into periodic and aperiodic
  components. *Nature Neuroscience* 23(12):1655–1665 (2020). doi:10.1038/s41593-020-00744-x,
  PMID 33230329 (10.1038/s41593-020-00744-x).
- Lodder SS, van Putten MJAM. Automated EEG analysis: Characterizing the posterior dominant
  rhythm. *Journal of Neuroscience Methods* 200(1):86–93 (2011). doi:10.1016/j.jneumeth.2011.06.008,
  PMID 21722667 (10.1016/j.jneumeth.2011.06.008).
- van Putten MJAM. The revised brain symmetry index. *Clinical Neurophysiology* 118(11):2362–2367
  (2007). doi:10.1016/j.clinph.2007.07.019, PMID 17888719 (10.1016/j.clinph.2007.07.019).
- Zibrandtsen IC, Kjaer TW. Fully automatic peak frequency estimation of the posterior dominant
  rhythm in a large retrospective hospital EEG cohort. *Clinical Neurophysiology Practice* 6:1–9
  (2021). doi:10.1016/j.cnp.2020.11.001, PMID 33385100 (10.1016/j.cnp.2020.11.001).
- Gramfort A et al. MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience* 7:267
  (2013). doi:10.3389/fnins.2013.00267, PMID 24431986 (10.3389/fnins.2013.00267).
- The Voytek Lab. FOOOF (specparam) 1.1.1. https://github.com/fooof-tools/fooof (Apache-2.0).

Demo data:

- Detti P. Siena Scalp EEG Database (version 1.0.0). PhysioNet (2020). doi:10.13026/5d4a-j060;
  Detti P et al. EEG Synchronization Analysis for Seizure Prediction: A Study on Data of
  Noninvasive Recordings. *Processes* 8(7):846 (2020). doi:10.3390/pr8070846; Dan J, Detti P.
  BIDS Siena Scalp EEG Database (v1.0.0). EPFL / Zenodo (2024). doi:10.5281/zenodo.10640762.
- Shoeb A. CHB-MIT Scalp EEG Database. PhysioNet (2010). doi:10.13026/C2K01R; Shoeb AH.
  Application of machine learning to epileptic seizure onset detection and treatment. PhD thesis,
  MIT (2009). https://dspace.mit.edu/handle/1721.1/54669.
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215
  (2000). doi:10.1161/01.CIR.101.23.e215, PMID 10851218 (10.1161/01.CIR.101.23.e215).
- Cserpan D, Boran E, Rosch R, Lo Biundo SP, Ramantani G, Sarnthein J. Dataset of EEG recordings
  of pediatric patients with epilepsy based on the 10-20 system. OpenNeuro (2021).
  doi:10.18112/openneuro.ds003555.v1.0.1; Cserpan D et al. Scalp high-frequency oscillation rates
  are higher in younger children. *Brain Communications* 3(2):fcab052 (2021).
  doi:10.1093/braincomms/fcab052, PMID 33870193 (10.1093/braincomms/fcab052).

## Acknowledgements

specparam/FOOOF is developed by the Voytek Lab (UC San Diego); MNE-Python by the MNE-Python
community. The Siena Scalp EEG Database was recorded at the Unit of Neurology and Neurophysiology of
the University of Siena and published on PhysioNet; the CHB-MIT database was collected at
Children's Hospital Boston and published on PhysioNet. PhysioNet asks users to cite Goldberger et
al. 2000 together with each database.

## Development

    python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
    make test PYTHON=.venv/bin/python        # pytest on the synthetic fixture, no Docker
    make run                                  # Docker over data/fixture/eeg -> data/output/
    make lambda-check                         # read-only image, uid 65534, HOME/TMPDIR under /tmp
    make run-demo DEMO=../workflow-scalp-eeg/data/work/siena/eeg-prep/eeg

The fixture (`tests/make_fixture.py`, also the release review's input) plants a 9.5 Hz posterior
alpha (weaker on the left, attenuated with eyes open), a 1/f background with exponent 1.5, left
temporal delta, a bad segment, an electrode pop, a seizure and a bad channel, in a referential
(average reference) and a bipolar-only recording, plus a Siena-style plain BIDS dataset and a
loose CHB-MIT-style EDF; the tests assert that each is recovered.
