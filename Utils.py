import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# ============================================================
# 2. Combined EEG + trigger processing
# ============================================================
def get_eeg_data_segmented_(
        csv_path,
        sample_rate=500,
        window_size=1.0,
        overlap=0.5,
        data_slice=None,
        debug = True,
        filter = False,
):
    """
    Load EEG SSVEP CSV, discretize trigger channel, and segment EEG into
    equal-length windows with overlap, ensuring no window crosses label boundaries.

    Args:
        csv_path       : path to .csv EEG recording
        sample_rate    : Hz
        window_size    : seconds per window
        overlap        : fraction overlap between windows (e.g. 0.5 = 50%)
        rest_threshold : below this trigger value = rest
        n_clusters     : number of non-rest trigger levels (None = auto)
        data_slice     : optional slice for trimming the data (e.g. slice(1700, 22000))

    Returns:
        X : np.ndarray, shape (n_windows, n_channels, n_samples)
            EEG data segments
        y : np.ndarray, shape (n_windows,)
            Integer labels per segment
        centers : np.ndarray
            Detected trigger plateau centers
        trigger_discrete : np.ndarray
            Discretized trigger signal (for optional plotting/debug)
    """

    # --- Load data ---
    df = pd.read_csv(csv_path)
    eeg_cols = [c for c in df.columns if "EEG" in c]
    eeg     = df[eeg_cols].values.T  # shape (C, T)
    trigger = df["Trigger"].values
    print(f"Loaded: EEG shape {eeg.shape}, Trigger shape {trigger.shape}")

    # --- Optional slicing ---
    if data_slice is not None:
        eeg     = eeg[:, data_slice]
        trigger = trigger[data_slice]
        print(f"After slicing: EEG shape {eeg.shape}, Trigger shape {trigger.shape} \n")



    # --- Preprocessing: Bandpass + Notch filter ---
    from scipy.signal import butter, filtfilt, iirnotch

    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        """
        Apply Butterworth bandpass filter to multi-channel EEG.
        data: np.ndarray (channels, samples)
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=1)

    def notch_filter(data, freq, fs, quality=30):
        """
        Apply notch filter (e.g. 50 Hz) to remove line noise.
        """
        b, a = iirnotch(w0=freq, Q=quality, fs=fs)
        return filtfilt(b, a, data, axis=1)

    # --- Apply preprocessing ---
    if filter:
        eeg = notch_filter(eeg, freq=50, fs=sample_rate)
        eeg = bandpass_filter(eeg, lowcut=5, highcut=40, fs=sample_rate)

    

    # --- Discretize trigger automatically ---
    # trigger_discrete, centers = discretize_trigger_auto(
    #     trigger,
    #     n_clusters=n_clusters,
    #     rest_threshold=rest_threshold
    # )
    # print(f"Detected {len(centers)} unique trigger plateaus: {centers}")

    # --- Compute windowing parameters ---
    win_len = int(window_size * sample_rate)
    step    = int(win_len * (1 - overlap))
    print(f"Window length: {win_len} samples ({window_size}s), Step: {step} samples")

    # --- Helper: Detect contiguous label regions ---
    def contiguous_regions(labels):
        regions = []
        start = 0
        current = labels[0]
        for i in range(1, len(labels)):
            if labels[i] != current:
                regions.append((current, start, i))
                start = i
                current = labels[i]
        regions.append((current, start, len(labels)))
        return regions

    # --- Create windows ---
    X, y = [], []
    # regions = contiguous_regions(trigger_discrete)
    regions = contiguous_regions(trigger)

    if debug:
        print("\n=== Detected contiguous regions (label, start, end, length) ===")
        for (label, start, end) in regions:
            print(f"Label {label:<3} | Start: {start:<6} | End: {end:<6} | Len: {end-start}")
        print("Total regions detected:", len(regions))

    for label, start, end in regions:
        region_len = end - start
        if region_len < win_len:
            # too short to form even one window
            print(f"Skipping region {label} [{start}:{end}] (length {region_len}) - too short for one window")
            continue

        # cut overlapping windows within this region only
        for w_start in range(start, end - win_len + 1, step):
            w_end = w_start + win_len
            X.append(eeg[:, w_start:w_end])
            y.append(label)

    X = np.stack(X)
    y = np.array(y, dtype=int)

    print(f"\n Extracted {len(y)} windows, shape={X.shape}, classes={np.unique(y)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    # return X, y, centers, trigger_discrete
    return X, y, eeg, trigger



def get_eeg_data_segmented(
        csv_path,
        sample_rate=500,
        window_size=1.0,
        overlap=0.5,
        data_slice=None,
        verbose=True,
        filter=False,
        save_npz_path=None,
        relabel_map=None,
):
    """
    Load EEG SSVEP CSV, segment EEG into windows, and optionally relabel classes.

    Args:
        csv_path       : path to .csv EEG recording
        sample_rate    : Hz
        window_size    : seconds per window
        overlap        : fraction overlap between windows (e.g. 0.5 = 50%)
        data_slice     : optional slice for trimming (e.g. slice(1700, 22000))
        verbose        : print segmentation info
        filter         : apply notch + bandpass filtering
        save_npz_path  : if given, saves X and y (relabelled if requested)
        relabel_map    : dict for optional relabeling, e.g.
                         {0: 0.0, 1: 7.0, 2: 10.5, 3: 12.0, 4: 15.2, 5: 18.1}

    Returns:
        X : np.ndarray, shape (n_windows, n_channels, n_samples)
        y : np.ndarray, shape (n_windows,)
        eeg : np.ndarray, raw EEG data (channels, samples)
        trigger : np.ndarray, trigger channel (samples,)
    """

    # --- Load data ---
    df = pd.read_csv(csv_path)
    eeg_cols = [c for c in df.columns if "EEG" in c]
    eeg     = df[eeg_cols].values.T  # shape (C, T)
    trigger = df["Trigger"].values
    print(f"Loaded: EEG shape {eeg.shape}, Trigger shape {trigger.shape}")

    # --- Optional slicing ---
    if data_slice is not None:
        eeg = eeg[:, data_slice]
        trigger = trigger[data_slice]
        print(f"After slicing: EEG shape {eeg.shape}, Trigger shape {trigger.shape}\n")

    # --- Optional filtering ---
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low, high = lowcut / nyq, highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=1)

    def notch_filter(data, freq, fs, quality=30):
        b, a = iirnotch(w0=freq, Q=quality, fs=fs)
        return filtfilt(b, a, data, axis=1)

    if filter:
        eeg = notch_filter(eeg, freq=50, fs=sample_rate)
        eeg = bandpass_filter(eeg, lowcut=5, highcut=40, fs=sample_rate)

    # --- Compute windowing parameters ---
    win_len = int(window_size * sample_rate)
    step = int(win_len * (1 - overlap))
    print(f"Window length: {win_len} samples ({window_size}s), Step: {step} samples")

    # --- Helper: contiguous regions of same label ---
    def contiguous_regions(labels):
        regions, start, current = [], 0, labels[0]
        for i in range(1, len(labels)):
            if labels[i] != current:
                regions.append((current, start, i))
                start, current = i, labels[i]
        regions.append((current, start, len(labels)))
        return regions

    # --- Segment EEG by trigger regions ---
    X, y = [], []
    regions = contiguous_regions(trigger)

    # if verbose:
    #     print("\n=== Detected contiguous regions (label, start, end, length) ===")
    #     for (label, start, end) in regions:
    #         print(f"Label {label:<3} | Start: {start:<6} | End: {end:<6} | Len: {end-start}")
    #     print("Total regions detected:", len(regions))
    
    if verbose:
        print("\n=== Detected contiguous regions (label, start, end, length) ===")
        for (label, start, end) in regions:
            display_label = relabel_map.get(label, label) if relabel_map else label
            print(f"Label {display_label:<6} | Start: {start:<6} | End: {end:<6} | Len: {end-start}")
        print("Total regions detected:", len(regions))


    for label, start, end in regions:
        region_len = end - start
        if region_len < win_len:
            print(f"Skipping region {label} [{start}:{end}] (len={region_len}) - too short")
        for w_start in range(start, end - win_len + 1, step):
            X.append(eeg[:, w_start:w_start + win_len])
            y.append(label)

    X = np.stack(X)
    y = np.array(y, dtype=int)

    print(f"\nExtracted {len(y)} windows, shape={X.shape}, classes={np.unique(y)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # --- Optional relabeling ---
    if relabel_map is not None:
        y = np.vectorize(relabel_map.get)(y)
        print(f"Relabeled classes using mapping: {relabel_map}")
        print(f"New label set: {np.unique(y)}")

    # --- Optional save ---
    if save_npz_path:
        np.savez(save_npz_path, X=X, y=y)
        print(f"Saved segmented EEG data to: {save_npz_path}")

    return X, y, eeg, trigger










def load_and_concat_ssvep_datasets_(
    datasets,
    sample_rate=500,
    window_size=1.0,
    overlap=0.5,
    debug=False,
    filter=False,
):
    """
    Load and combine multiple SSVEP EEG datasets from a list of CSV files.

    Args:
        datasets : list[tuple[str, slice or None]]
            List of (csv_path, data_slice) pairs.
            Example: [("file1.csv", slice(1700,22000)), ("file2.csv", None)]
        sample_rate : int
            Sampling rate in Hz.
        window_size : float
            Length of each EEG window (seconds).
        overlap : float
            Fraction overlap between windows (0–1).
        debug : bool
            Print debug info for each file.

    Returns:
        X_all : np.ndarray, shape (sum_i n_windows_i, n_channels, n_samples)
        y_all : np.ndarray, shape (sum_i n_windows_i,)
        triggers_all : list[np.ndarray]
    """

    X_list, y_list, eeg_list, trigger_list = [], [], [], []

    print(f"\n=== Loading {len(datasets)} EEG files ===")
    for i, (path, dslice) in enumerate(datasets, start=1):
        print(f"\n[{i}/{len(datasets)}] Processing {path}")
        if dslice is not None:
            print(f"   → Using slice {dslice.start}:{dslice.stop}")
        else:
            print("   → No slicing applied")
        try:
            X, y, eeg, trigger = get_eeg_data_segmented(
                csv_path=path,
                sample_rate=sample_rate,
                window_size=window_size,
                overlap=overlap,
                data_slice=dslice,
                debug=debug,
                filter=filter,
            )
            X_list.append(X)
            y_list.append(y)
            eeg_list.append(eeg)
            trigger_list.append(trigger)
            print(f"Added {X.shape[0]} windows from {path}")
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not X_list:
        raise RuntimeError("No valid EEG datasets could be loaded.")

    # --- Shape consistency check ---
    n_channels, n_samples = X_list[0].shape[1:]
    for X in X_list:
        if X.shape[1:] != (n_channels, n_samples):
            raise ValueError(
                f"Shape mismatch: expected {(n_channels, n_samples)}, got {X.shape[1:]}"
            )

    # --- Concatenate ---
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    eeg_all = np.concatenate(eeg_list, axis=1)
    trigger_all = np.concatenate(trigger_list, axis=0)

    
    print(f"\nCombined all datasets: X={X_all.shape}, y={y_all.shape}, eeg={eeg_all.shape}, trigger={trigger_all.shape}")
    print(f"Classes present: y = {np.unique(y_all)}, trigger = {np.unique(trigger_all)}")
    return X_all, y_all, eeg_all, trigger_all







def load_and_concat_ssvep_datasets(
    datasets,
    sample_rate=500,
    window_size=1.0,
    overlap=0.5,
    verbose=True,
    filter=False,
):
    """
    Load and combine multiple SSVEP EEG datasets from CSV files,
    each with its own optional slice and label mapping.

    Args:
        datasets : list of tuples
            Each entry is either:
                (csv_path, data_slice)
            or
                (csv_path, data_slice, relabel_map)
            Example:
                [
                    ("file1.csv", slice(1700,22000), {0:0,1:7,2:10.5,3:12,4:15.2,5:18.1}),
                    ("file2.csv", None, {0:0,1:5,2:8.6,3:11,4:13.4,5:17}),
                ]
        sample_rate : int
            Sampling rate in Hz.
        window_size : float
            Length of each EEG window (seconds).
        overlap : float
            Fraction overlap between windows (0–1).
        debug : bool
            Print debug info for each file.
        filter : bool
            Apply bandpass + notch filtering.

    Returns:
        X_all : np.ndarray, shape (sum_i n_windows_i, n_channels, n_samples)
        y_all : np.ndarray, shape (sum_i n_windows_i,)
        eeg_all : np.ndarray, concatenated raw EEG signals
        trigger_all : np.ndarray, concatenated trigger signal
    """

    X_list, y_list, eeg_list, trigger_list = [], [], [], []

    print(f"\n=== Loading {len(datasets)} EEG files ===")
    for i, entry in enumerate(datasets, start=1):
        # Handle both (path, slice) and (path, slice, map)
        if len(entry) == 3:
            path, dslice, relabel_map = entry
        elif len(entry) == 2:
            path, dslice = entry
            relabel_map = None
        else:
            raise ValueError(
                "Each dataset entry must be (csv_path, data_slice[, relabel_map])"
            )

        print(f"\n[{i}/{len(datasets)}] Processing {path}")
        if dslice is not None:
            print(f"   → Using slice {dslice.start}:{dslice.stop}")
        else:
            print("   → No slicing applied")
        if relabel_map is not None:
            print(f"   → Applying custom mapping: {relabel_map}")

        try:
            X, y, eeg, trigger = get_eeg_data_segmented(
                csv_path=path,
                sample_rate=sample_rate,
                window_size=window_size,
                overlap=overlap,
                data_slice=dslice,
                verbose=verbose,
                filter=filter,
                relabel_map=relabel_map,
            )
            X_list.append(X)
            y_list.append(y)
            eeg_list.append(eeg)
            trigger_list.append(trigger)
            print(f"Added {X.shape[0]} windows from {path}")
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not X_list:
        raise RuntimeError("No valid EEG datasets could be loaded.")

    # --- Shape consistency check ---
    n_channels, n_samples = X_list[0].shape[1:]
    for X in X_list:
        if X.shape[1:] != (n_channels, n_samples):
            raise ValueError(
                f"Shape mismatch: expected {(n_channels, n_samples)}, got {X.shape[1:]}"
            )

    # --- Concatenate ---
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    eeg_all = np.concatenate(eeg_list, axis=1)
    trigger_all = np.concatenate(trigger_list, axis=0)

    print(
        f"\n   Combined all datasets:"
        f"\n   X={X_all.shape}, y={y_all.shape}"
        f"\n   eeg={eeg_all.shape}, trigger={trigger_all.shape}"
    )
    print(f"   Classes present: y={np.unique(y_all)}, trigger={np.unique(trigger_all)}")

    return X_all, y_all, eeg_all, trigger_all









def downsample_label(X, y, target_label=0, keep_ratio=0.1, seed=42):
    """
    Randomly keep only a fraction of samples from the given target_label.
    For example, keep_ratio=0.1 keeps 10% of all label 0 windows.

    Returns:
        X_balanced, y_balanced
    """
    np.random.seed(seed)

    # find indices per class
    idx_target = np.where(y == target_label)[0]
    idx_other  = np.where(y != target_label)[0]

    # randomly select subset of label-0 indices
    n_keep = int(len(idx_target) * keep_ratio)
    keep_idx_target = np.random.choice(idx_target, n_keep, replace=False)

    # combine and shuffle
    keep_idx = np.concatenate([keep_idx_target, idx_other])
    np.random.shuffle(keep_idx)

    return X[keep_idx], y[keep_idx]






# Load the logged metrics
# df = pd.read_csv("logs/EEGClassifier_Validation/version_14/metrics.csv")

def plot_training_metrics(csv_path):
    df = pd.read_csv(csv_path)

    # Filter for epochs (drop NaNs and duplicates)
    df = df.dropna(subset=["val_acc"])
    df = df.groupby("epoch", as_index=False).mean()

    # Plot validation accuracy
    plt.figure(figsize=(16,5))
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", marker='^', linestyle=':', markersize=2)
    if "train_acc" in df.columns:
        plt.plot(df["epoch"], df["train_acc"], label="Training Accuracy", linestyle="--", alpha=0.7)
   
    plt.plot([df["epoch"].min(), df["epoch"].max()], [0.70, 0.70], 'r--', label="70% Accuracy")
    plt.plot(df["val_acc"].idxmax(), df["val_acc"].max(), 'go', label=f"Best Val Acc {df['val_acc'].max():.3f}")
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation vs Training Accuracy over Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# For testing: 

# print(f"X shape: {X.shape}, y shape: {y.shape}")
# plt.figure(figsize=(10,4))
# plt.plot(X[37, 0, :])  # 27th window, first channel, all 500 samples 
# plt.show()
# plt.figure(figsize=(10,4))
# plt.plot(y[37, :])
# plt.show()
