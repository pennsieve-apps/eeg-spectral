# Sample input

eeg-spectral reads, recursively under `INPUT_DIR`:

1. the `eeg` output of **eeg-prep** (a BIDS-EEG derivative with `recordings.csv`) - the normal case
   in the Scalp EEG Interictal Biomarkers workflow;
2. any **BIDS-EEG dataset** (`*_eeg.edf/.bdf/.vhdr/.fif/.set`, with `_channels.tsv` and
   `_events.tsv` when present, including SzCORE-style HED `eventType` seizure rows);
3. **loose EEG files** (EDF/EDF+, BDF, BrainVision, FIF, EEGLAB), referential (`Fp1`, `EEG Fp1-Ref`,
   `Fp1-A1`, `T3`) or bipolar (`FP1-F7`), with EDF+ annotations and CHB-MIT `*-summary.txt` seizure lists.

## Synthetic (no download)

    python -m tests.make_fixture data/fixture      # data/fixture/eeg is the eeg-prep-style input

## Real recordings used to test this app

- **BIDS Siena Scalp EEG Database** (Zenodo doi:10.5281/zenodo.10640762, CC BY 4.0; from PhysioNet
  doi:10.13026/5d4a-j060): 14 adults with focal epilepsy, 41 runs, 19 channels in average
  reference at 256 Hz. Download the Zenodo archive and point `INPUT_DIR` at `BIDS_Siena/`, or one
  `sub-XX/` folder of it.
- **CHB-MIT Scalp EEG Database** (PhysioNet doi:10.13026/C2K01R, ODC-By 1.0): bipolar-only EDFs,
  e.g. `chb01/chb01_03.edf` plus `chb01/chb01-summary.txt`
  (`https://physionet.org/files/chbmit/1.0.0/chb01/`).
- **OpenNeuro ds003555** (doi:10.18112/openneuro.ds003555.v1.0.1, CC0): sleep EEG of children with
  epilepsy, BIDS-EEG (`openneuro-py download --dataset ds003555 --include sub-01`).

In the workflow repo, `scripts/fetch-sample-data.sh` stages all three under `data/`.
