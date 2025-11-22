
# SES / ES Bearing Fault Analysis Pipeline (Python)

This repository provides a minimal, reproducible Python pipeline for
envelope analysis of rolling element bearings, adapted from a Master's
dissertation on the influence of epistemic and aleatoric experimental
uncertainties in rolling element condition monitoring.

The pipeline allows you to:

1. Load an experimental (or simulated) vibration dataset (X, Y, or Z axis).
2. Compute Squared Envelope Spectrum (SES) or standard Envelope Spectrum (ES).
3. Investigate:
   - Hilbert vs rectification demodulation,
   - presence vs absence of an informed frequency band around the BPFO,
   - NFFT (FFT length) effects on 1x and 2x BPFO amplitudes,
   - full vs segmented signals (e.g. 10 x 1 s segments),
   - convex hull of BPFO amplitudes in the (1xBPFO, 2xBPFO) plane.

## Structure

- `ses_core.py`
  Core functions:
  - `ses(...)` – SES/ES computation with Hilbert or rectification and optional band search;
  - `compute_BPFO_SNR(...)` – signal-to-noise ratio of BPFO harmonics;
  - `analyse_ses(...)` – NFFT sweep for full and segmented signals;
  - `convexhull(...)` – convex hull plotting and area measure;
  - `load_excel_signal(...)` – load one axis from a bearing test Excel file.

- `ses_pipeline_example.ipynb`
  Example Jupyter notebook that walks through:
  1. Loading a single test from Excel;
  2. Visualising raw, rectified, and Hilbert-demodulated signals;
  3. Computing SES/ES for full and segmented signals;
  4. Comparing Hilbert vs rectification and band vs no-band selection;
  5. Zooming around BPFO;
  6. Running an NFFT study and plotting 1x/2x BPFO amplitude vs NFFT;
  7. Computing convex hulls for 1x vs 2x BPFO amplitudes.

## Requirements

Install dependencies (for example, in a fresh environment):

```bash
pip install numpy scipy matplotlib pandas openpyxl
```

## Usage

1. Unzip the project and open a terminal in the extracted folder.
2. Launch Jupyter Lab or Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

3. Open `ses_pipeline_example.ipynb` and follow the cells from top to bottom.
   You only need to edit the cell where you specify the path to your Excel file
   and the bearing fault frequency (BPFO).

The code is written so that it can be adapted to both experimental and
simulated datasets, provided that you construct a time-series signal and a
sampling frequency `fs`.
