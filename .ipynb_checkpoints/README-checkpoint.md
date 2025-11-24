
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

1. Launch Jupyter Lab or Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

2. Open `ses_pipeline_example.ipynb` and follow the cells from top to bottom.
   You only need to edit the cell where you specify the path to your Excel file
   and the bearing fault frequency (BPFO).

The code is written so that it can be adapted to both experimental and
simulated datasets, provided that you construct a time-series signal and a
sampling frequency `fs`.


## SES Pipeline Structure

This repository is organised around a reusable SES/ES core and two analysis
workflows:

- `ses_core.py`  
  Core implementation of the Squared Envelope Spectrum (SES) and standard
  Envelope Spectrum (ES), including:
  - demodulation via rectification and Hilbert transform
  - banded peak search around BPFO-related frequencies
  - NFFT and segmentation studies
  - convex hull analysis of 1× and 2× BPFO amplitudes for segmented signals

- `ses_pipeline_example.ipynb`  
  Example notebook showing how to apply the SES pipeline to **experimental
  bearing vibration data**. It demonstrates:
  - loading raw acceleration signals and speed from test files
  - computing SES/ES for full and segmented signals
  - visualising BPFO and its harmonics in the frequency domain
  - convex hull construction for 1×BPFO vs 2×BPFO peak amplitudes.

- Simulated segmentation study (inside `ses_pipeline_example.ipynb`)  
  A companion section in the notebook that performs a **simulated segmentation
  study** to validate the SES pipeline. It includes:
  - generation of synthetic bearing-like signals (impulse trains convolved
    with a decaying sinusoidal impulse response)
  - visualisation of impulse trains and bearing responses
  - systematic sweep of segmentation window length and NFFT values
  - comparison of 1×BPFO SES peaks between:
    - the full 10 s signal, and
    - a single 1 s segment repeated to fill 10 s
  - time-domain plots showing full vs repeated segment for each window length.

The simulated study is designed to mirror the structure of the experimental
pipeline and to confirm that the chosen segmentation strategy does not
introduce artificial uncertainty in the SES-based BPFO amplitudes.
