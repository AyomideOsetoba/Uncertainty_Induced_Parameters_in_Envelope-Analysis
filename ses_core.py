
import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft as scifft
from scipy.spatial import ConvexHull
from typing import List, Dict, Tuple
import pandas as pd


# def ses(
#     signal: np.ndarray,
#     nfft: int,
#     fs: float,
#     bpfo_list: List[float],
#     hilb: str = "t",
#     p_type: str = "SES",
#     band: bool = True,
#     band_half_width_hz: float = 1.0,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute SES/ES and extract BPFO-related peaks.

#     Parameters
#     ----------
#     signal : np.ndarray
#         Time-domain vibration signal.
#     nfft : int
#         FFT length.
#     fs : float
#         Sampling frequency [Hz].
#     bpfo_list : list of float
#         List of BPFO-related frequencies [Hz], e.g. [1*BPFO, 2*BPFO, 3*BPFO].
#     hilb : {'t', 'f'}
#         't' -> Hilbert demodulation (|analytic(signal)|).
#         'f' -> rectification demodulation (|signal|).
#     p_type : {'SES', 'ES'}
#         'SES' -> squared envelope spectrum.
#         'ES'  -> standard envelope spectrum.
#     band : bool
#         If True, search for maximum in a ±band_half_width_hz band
#         around each BPFO. If False, use the FFT bin at BPFO.
#     band_half_width_hz : float
#         Half-width of the frequency band for band search.

#     Returns
#     -------
#     ses_peaks : np.ndarray
#         SES/ES peak amplitudes at each BPFO-related frequency.
#     idx_peaks : np.ndarray
#         Indices in the spectrum where these peaks were taken.
#     ses_spec : np.ndarray
#         One-sided SES/ES spectrum (amplitude, with factor 2 applied).
#     """
#     signal = np.asarray(signal)

#     # --- Envelope (demodulation) ---
#     if hilb == "t":
#         env = np.abs(hilbert(signal))
#         env = env - 0
#     else:
#         env = np.abs(signal)
#         # env = env - np.mean(env)

#     if p_type.upper() == "SES":
#         env = env**2  # squared envelope

#     # Original behaviour:
#     #  - Hilbert branch: no DC removal (you used "- 0")
#     #  - Rectified branch: subtract mean
#     if hilb != "t":
#         env = env - np.mean(env)

#     # --- FFT and one-sided spectrum ---
#     spec = np.abs(scifft(env, nfft)) / len(signal)
#     ses_spec = spec[: nfft // 2] * 2  # one-sided, amplitude scaled by 2
#     freq = np.arange(nfft // 2) * (fs / nfft)

#     ses_peaks = []
#     idx_peaks = []

#     for bpfo in bpfo_list:
#         if band:
#             mask = (freq >= bpfo - band_half_width_hz) & (freq <= bpfo + band_half_width_hz)
#             idx_band = np.where(mask)[0]

#             if idx_band.size > 0:
#                 local_idx = np.argmax(ses_spec[idx_band])
#                 idx_peak = idx_band[local_idx]
#                 ses_peak = ses_spec[idx_peak]
#             else:
#                 idx_peak = -1
#                 ses_peak = 0.0
#         else:
#             idx_peak = int(round(bpfo * nfft / fs))
#             if 0 <= idx_peak < len(ses_spec):
#                 ses_peak = ses_spec[idx_peak]
#             else:
#                 idx_peak = -1
#                 ses_peak = 0.0

#         idx_peaks.append(idx_peak)
#         ses_peaks.append(ses_peak)

#     return np.array(ses_peaks), np.array(idx_peaks), ses_spec


def ses(signal, nfft, fs, bpfo_list, hilb, p_type, bw, band=True):
    """
    Compute the Squared Envelope Spectrum (SES) or Envelope Spectrum (ES)
    and extract the amplitudes at specified BPFO-related frequencies.

    Parameters
    ----------
    signal : array_like
        Time-domain signal (1D array).
    nfft : int
        FFT length.
    fs : float
        Sampling frequency [Hz].
    bpfo_list : list of float
        List of BPFO-related frequencies [Hz] at which peaks are extracted
        (e.g. [BPFO, 2*BPFO, 3*BPFO]).
    hilb : {'t', 'f'}
        Demodulation method:
        - 't' : Hilbert transform demodulation (|analytic(signal)| or its square),
        - 'f' : rectification demodulation (|signal| or its square).
    p_type : {'SES', 'ES'}
        Spectrum type:
        - 'SES' : squared envelope spectrum,
        - 'ES'  : standard envelope spectrum.
    bw: Int
        bandwith in Hz.
    band : bool, optional
        If True, search for the maximum in a ±1 Hz band around each BPFO.
        If False, use the single FFT bin at the BPFO frequency.

    Returns
    -------
    ses_peaks : np.ndarray
        Peak amplitudes at each BPFO-related frequency.
    idx_peaks : np.ndarray
        Indices (in the one-sided spectrum) where these peaks are located.
    ses : np.ndarray
        One-sided SES/ES spectrum (amplitude, with factor 2 applied).
    """
    # Demodulation and envelope construction
    if hilb == 't':
        if p_type == 'SES':
            hilb_env = np.abs(hilbert(signal))**2
            hilb_env_dc = hilb_env - 0  # keep original behaviour (no DC removal)
            ses = np.abs(scifft(hilb_env_dc, nfft)) / len(signal)
        elif p_type == 'ES':
            hilb_env = np.abs(hilbert(signal))
            hilb_env_dc = hilb_env - 0  # keep original behaviour (no DC removal)
            ses = np.abs(scifft(hilb_env_dc, nfft)) / len(signal)
    else:
        if p_type == 'SES':
            env = np.abs(signal)**2
            env_dc = env - np.mean(env)  # remove DC from envelope
            ses = np.abs(scifft(env_dc, nfft)) / len(signal)
        elif p_type == 'ES':
            env = np.abs(signal)
            env_dc = env - np.mean(env)  # remove DC from envelope
            ses = np.abs(scifft(env_dc, nfft)) / len(signal)

    # One-sided spectrum (frequency axis and amplitude)
    freq = np.arange(nfft // 2) * (fs / nfft)
    ses = ses[:nfft // 2]
    ses = ses * 2  # double-sided → one-sided amplitude correction

    ses_peaks = []
    idx_peaks = []

    for bpfo in bpfo_list:
        if band:
            # bw Hz band around BPFO
            bpfo_band = (freq >= bpfo - bw) & (freq <= bpfo + bw)
            idx_band = np.where(bpfo_band)[0]

            if len(idx_band) > 0:
                idx_peak = idx_band[np.argmax(ses[idx_band])]
                ses_peak = ses[idx_peak]
            else:
                ses_peak = 0
                idx_peak = -1
        else:
            # Use the nearest FFT bin at BPFO
            idx_peak = round(bpfo * nfft / fs)
            if 1 <= idx_peak < len(ses):
                ses_peak = ses[idx_peak]
            else:
                ses_peak = 0
                idx_peak = -1

        ses_peaks.append(ses_peak)
        idx_peaks.append(idx_peak)

    return np.array(ses_peaks), np.array(idx_peaks), ses




def compute_BPFO_SNR(frequency, SES, BPFO, N):
    """
    Compute the Signal-to-Noise Ratio (SNR) for BPFO in the SES/ES.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency axis [Hz].
    SES : np.ndarray
        Spectrum values (SES or ES).
    BPFO : float
        Ball Pass Frequency Outer race [Hz].
    N : int
        Number of harmonics to consider.

    Returns
    -------
    noise_power : float
        Estimated median noise power.
    SNR_BPFO_dB : float
        Signal-to-noise ratio in dB.
    save_signal : list of float
        List of peak amplitudes at the considered harmonics.
    """
    BPFO_harmonics = np.arange(1, N + 1) * BPFO
    signal_power = 0.0
    save_signal = []
    max_peak_locations = []

    for harmonic in BPFO_harmonics:
        # ±5 Hz search band
        mask = (frequency >= harmonic - 5) & (frequency <= harmonic + 5)
        SES_in_band = SES[mask]
        freq_in_band = frequency[mask]

        if SES_in_band.size > 0:
            local_idx = np.argmax(SES_in_band)
            max_peak = SES_in_band[local_idx]
            signal_power += max_peak
            save_signal.append(max_peak)
            max_peak_locations.append(freq_in_band[local_idx])

    if not max_peak_locations:
        raise ValueError("No significant peaks found. Check SES input.")

    noise_lower = 0.5 * max_peak_locations[0]
    noise_upper = 1.5 * max_peak_locations[-1]
    noise_mask = (frequency >= noise_lower) & (frequency <= noise_upper)
    noise_values = SES[noise_mask]

    if noise_values.size == 0:
        raise ValueError("No noise data found in the selected range. Check frequency input.")

    noise_power = np.median(noise_values)
    SNR_BPFO = signal_power / noise_power
    SNR_BPFO_dB = 10 * np.log10(SNR_BPFO)

    return noise_power, SNR_BPFO_dB, save_signal


# def analyse_ses(
#     y_cases: List[np.ndarray],
#     fs: float,
#     segment_sec: float,
#     bpfo_list: List[float],
#     nfft_values: List[int],
#     p_type: str = "SES",
#     band: bool = True,
# ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[list]], List[float]]:
#     """
#     Run SES/ES analysis over multiple cases, NFFT values, and segments.

#     Parameters
#     ----------
#     y_cases : list of np.ndarray
#         List of time-domain signals, one per case.
#     fs : float
#         Sampling frequency [Hz].
#     segment_sec : float
#         Segment length [s] for segmentation (e.g. 1 s).
#     bpfo_list : list of float
#         List of BPFO-related frequencies [Hz] (typically [BPFO, 2*BPFO]).
#     nfft_values : list of int
#         List of FFT lengths (e.g. [2**i for i in range(15, 24)]).
#     p_type : {'SES', 'ES'}
#         Spectrum type to compute.
#     band : bool
#         If True, apply band search around each BPFO; otherwise use direct bin.

#     Returns
#     -------
#     results_full : dict
#         results_full['rect'][case_idx] -> array of shape (H, K)
#         results_full['hilbert'][case_idx] -> same shape,
#         where H = len(bpfo_list), K = len(nfft_values).
#     results_seg : dict
#         results_seg['rect'][case_idx] -> list of length H;
#             each element is a list of length S (segments);
#             each segment entry is an array of length K (over NFFT).
#         results_seg['hilbert'][case_idx] -> same structure.
#     bpfo_list : list of float
#         Echo of input bpfo_list (for convenience).
#     """
#     methods = {"rect": "f", "hilbert": "t"}
#     H = len(bpfo_list)
#     K = len(nfft_values)

#     results_full: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
#     results_seg: Dict[str, List[list]] = {m: [] for m in methods}

#     seg_len = int(round(fs * segment_sec))

#     for y in y_cases:
#         # Prepare segments: truncate to an integer number of segments
#         total_samples = len(y)
#         num_segments = total_samples // seg_len
#         y_trunc = y[: num_segments * seg_len]

#         segments = [
#             y_trunc[i * seg_len : (i + 1) * seg_len] for i in range(num_segments)
#         ]

#         for method_name, hilb_flag in methods.items():
#             # --- full signal matrix: H × K ---
#             full_matrix = np.zeros((H, K))
#             # --- segmented: list[H][segment_index] = np.array(len K) ---
#             seg_matrix: List[List[np.ndarray]] = [
#                 [np.zeros(K) for _ in range(num_segments)] for _ in range(H)
#             ]

#             for k, nfft in enumerate(nfft_values):
#                 # Full signal SES/ES
#                 ses_peaks_full, _, _ = ses(
#                     y, nfft, fs, bpfo_list, hilb=hilb_flag, p_type=p_type, band=band
#                 )
#                 full_matrix[:, k] = ses_peaks_full

#                 # Each segment
#                 for s_idx, seg in enumerate(segments):
#                     ses_peaks_seg, _, _ = ses(
#                         seg, nfft, fs, bpfo_list, hilb=hilb_flag, p_type=p_type, band=band
#                     )
#                     for h in range(H):
#                         seg_matrix[h][s_idx][k] = ses_peaks_seg[h]

#             results_full[method_name].append(full_matrix)
#             results_seg[method_name].append(seg_matrix)

#     return results_full, results_seg, bpfo_list


from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

def analyse_ses(
    y_cases: List[np.ndarray],
    fs: float,
    segment_sec: float,
    bpfo_list: List[float],
    nfft_values: List[int],
    p_type: str,
    no_case: int,
    bw: int,
    band: bool = True,
    start_index: int = 0,
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[int, np.ndarray]], List[float]]:
    """
    Analyse SES/ES sensitivity to NFFT and segmentation for multiple cases.

    This reproduces your original implementation (used in the dissertation):
      - Full 10 s signal and 1 s non-overlapping segments.
      - Peak extraction at each BPFO-related frequency for each NFFT.
      - Results stored as dicts keyed by case_idx.
      - Excel export for the last NFFT only.

    Parameters
    ----------
    y_cases : list of np.ndarray
        List of time-domain signals.
        For case_idx = 0, the function uses y_cases[start_index + 0].
    fs : float
        Sampling frequency [Hz].
    segment_sec : float
        Segment length in seconds (e.g. 1.0 for 1 s).
    bpfo_list : list of float
        BPFO-related frequencies [Hz] (e.g. [BPFO, 2*BPFO]).
    nfft_values : list of int
        NFFT values to sweep (e.g. [2**15, ..., 2**23]).
    p_type : {'SES', 'ES'}
        Spectrum type to compute in ses().
    no_case : int
        Number of cases to process (case_idx = 0, ..., no_case - 1).
    bw: int
        bandwidth in Hz
    band : bool, optional
        If True (default), use ±1 Hz band search around each BPFO.
        If False, use the exact FFT bin.
    start_index : int, optional
        Offset in y_cases where the first case is taken from.
        - Use 0 for normal indexing (case 0 → y_cases[0]).
        - Use 2 to reproduce your old y_cases[case_idx+2] behaviour.

    Returns
    -------
    results_full : dict
        results_full['rect'][case_idx] -> array (H, K),
        results_full['hilbert'][case_idx] -> array (H, K),
        where H = len(bpfo_list), K = len(nfft_values).
    results_seg : dict
        results_seg['rect'][case_idx] -> array (H, S, K),
        results_seg['hilbert'][case_idx] -> array (H, S, K),
        where S is the number of segments (floor(10 s / segment_sec)).
    bpfo_list : list of float
        Echo of the input BPFO list for convenience.
    """
    results_full: Dict[str, Dict[int, np.ndarray]] = {'rect': {}, 'hilbert': {}}
    results_seg: Dict[str, Dict[int, np.ndarray]] = {'rect': {}, 'hilbert': {}}

    for method in ['f', 't']:   # 'f' = rectification, 't' = Hilbert
        for case_idx in range(no_case):
            idx = start_index + case_idx
            if idx >= len(y_cases):
                raise IndexError(
                    f"Case index {idx} out of range for y_cases of length {len(y_cases)}. "
                    f"Check no_case={no_case} and start_index={start_index}."
                )

            signal = y_cases[idx]

            # Segmentation
            seg_len = int(fs * segment_sec)
            num_segments = len(signal) // seg_len

            peak_matrix_full = []   # will become (K, H) then transposed -> (H, K)
            peak_matrix_seg = []    # will become (K, S, H) then -> (H, S, K)

            for nfft in nfft_values:
                # --- Full-signal SES/ES for all harmonics ---
                p_full, _, _ = ses(signal, nfft, fs, bpfo_list, method, p_type, bw, band=band)
                peak_matrix_full.append(p_full)

                # --- Segmented SES/ES ---
                seg_peaks = []
                for i in range(num_segments):
                    segment = signal[i * seg_len:(i + 1) * seg_len]
                    p_seg, _, _ = ses(segment, nfft, fs, bpfo_list, method, p_type, bw, band=band)
                    seg_peaks.append(p_seg)

                peak_matrix_seg.append(seg_peaks)

            # Convert to arrays with consistent shapes
            peak_matrix_full = np.array(peak_matrix_full).T
            # shape: (H, K)

            peak_matrix_seg = np.transpose(np.array(peak_matrix_seg), (2, 1, 0))
            # shape: (H, S, K)

            key = 'rect' if method == 'f' else 'hilbert'
            results_full[key][case_idx] = peak_matrix_full
            results_seg[key][case_idx] = peak_matrix_seg

            # Save Excel for the last NFFT value only
            last_idx = len(nfft_values) - 1
            df = pd.DataFrame()

            for h, bpfo in enumerate(bpfo_list):
                full_peaks = peak_matrix_full[h, last_idx]
                seg_peaks_last_nfft = peak_matrix_seg[h, :, last_idx]

                df[f'Full_{bpfo:.1f}Hz'] = [full_peaks]
                for s, val in enumerate(seg_peaks_last_nfft):
                    df[f'Seg{s+1}_{bpfo:.1f}Hz'] = [val]

            df.to_excel(f'ses_peaks_case{case_idx + 1}_{key}.xlsx', index=False)

    return results_full, results_seg, bpfo_list




def convexhull(points: np.ndarray, color: str, ax, alpha_fill: float = 0.15, lw: float = 2.0) -> float:
    """
    Plot the convex hull of a 2D point cloud and return sqrt(area).

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        2D points (e.g. [1*BPFO amplitude, 2*BPFO amplitude]).
    color : str
        Colour for hull boundary and fill.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    alpha_fill : float
        Transparency of the hull fill.
    lw : float
        Line width for hull boundary.

    Returns
    -------
    sqrt_area : float
        Square root of the hull area (2D area).
    """
    points = np.asarray(points)
    hull = ConvexHull(points)

    hull_vertices = hull.vertices
    polygon = points[hull_vertices]
    polygon = np.vstack([polygon, polygon[0]])  # close polygon

    ax.fill(
        polygon[:, 0],
        polygon[:, 1],
        facecolor=color,
        edgecolor=color,
        alpha=alpha_fill,
        linewidth=lw,
    )

    area = hull.volume  # in 2D, ConvexHull.volume is area
    sqrt_area = np.sqrt(area)

    return sqrt_area


def load_excel_signal(
    filepath: str,
    axis: str = "Y",
    fs: float = 48_000.0,
    duration: float = 10.0,
    header_rows_to_skip: int = 35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single test from an Excel file and return time vector, selected axis, and speed.

    Assumes the sheet has columns in the order:
    Time, Speed, X - Axis, Y - Axis, Z - Axis,
    and that first 'header_rows_to_skip' rows are discarded.

    Parameters
    ----------
    filepath : str
        Path to the Excel file (e.g. 'Test 1.xlsx').
    axis : {'X', 'Y', 'Z'}
        Which acceleration axis to extract.
    fs : float
        Sampling frequency [Hz].
    duration : float
        Duration to extract [s].
    header_rows_to_skip : int
        Number of initial rows to discard.

    Returns
    -------
    t : np.ndarray
        Time vector [s] (length = duration * fs).
    x : np.ndarray
        Selected axis data (length = duration * fs).
    speed : np.ndarray
        Speed signal corresponding to the same samples.
    """
    df_raw = pd.read_excel(filepath, engine="openpyxl")
    df = df_raw.iloc[header_rows_to_skip:].copy()

    # Normalise column names (truncate if fewer exist)
    df.columns = ["Time", "Speed", "X - Axis", "Y - Axis", "Z - Axis"][: len(df.columns)]

    n_samples = int(round(duration * fs))
    df = df.iloc[:n_samples]

    t = df["Time"].to_numpy()
    speed = df["Speed"].to_numpy()

    axis_map = {
        "X": "X - Axis",
        "Y": "Y - Axis",
        "Z": "Z - Axis",
    }
    col_name = axis_map[axis.upper()]

    x = df[col_name].to_numpy()

    return t, x, speed
