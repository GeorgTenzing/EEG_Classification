import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch


# ============================================================
# Combined EEG processing, segmentation, and relabeling
# ============================================================
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
        remove_labels=None,
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

    # --- Necessary slicing ---
    if data_slice is not None:
        eeg = eeg[:, data_slice]
        trigger = trigger[data_slice]
        print(f"After slicing: EEG shape {eeg.shape}, Trigger shape {trigger.shape}\n")

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
    
    if verbose:
        print("\n=== Detected contiguous regions (label, start, end, length) ===")
        for (label, start, end) in regions:
            display_label = relabel_map.get(label, label) if relabel_map else label
            print(f"Label {display_label:<6} | Start: {start:<6} | End: {end:<6} | Len: {end-start}")
        print("Total regions detected:", len(regions))


    for label, start, end in regions:
        region_len = end - start
        if region_len < win_len:
            display_label = relabel_map.get(label, label) if relabel_map else label
            print(f"Skipping region {display_label} [{start}:{end}] (len={region_len}) - too short")
        for w_start in range(start, end - win_len + 1, step):
            X.append(eeg[:, w_start:w_start + win_len])
            y.append(label)

    X = np.stack(X)
    y = np.array(y, dtype=int)

    print(f"\nExtracted {len(y)} windows, shape={X.shape}, classes={np.unique(y)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")


    # --- Optional removal of specific labels (e.g., remove label 0) ---
    if remove_labels:
        mask = ~np.isin(y, remove_labels)  # remove_labels can be list
        X = X[mask]
        y = y[mask]
        
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







# ============================================================
# Load and concatenate multiple SSVEP datasets with optional relabeling
# ============================================================
def load_and_concat_ssvep_datasets(
    datasets,
    sample_rate=500,
    window_size=1.0,
    overlap=0.5,
    verbose=True,
    filter=False,
    remove_labels=None,
    relabel_map = {0:0, 2:1, 3:2, 4:3, 5:4},
    target_label=None,
    keep_ratio=0.2,
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
        target_label : int
            Label to downsample (e.g., 0 for non-SSVEP).
        keep_ratio : float
            Fraction of target_label windows to keep.
        remove_labels : list of int
            List of labels to remove from final dataset.

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
                remove_labels=remove_labels,
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
    X_all       = np.concatenate(X_list, axis=0)
    y_all       = np.concatenate(y_list, axis=0)
    eeg_all     = np.concatenate(eeg_list, axis=1)
    trigger_all = np.concatenate(trigger_list, axis=0)

    # --- Summary ---
    print(
        f"\n   Combined all datasets:"
        f"\n   X={X_all.shape}, y={y_all.shape}"
        f"\n   eeg={eeg_all.shape}, trigger={trigger_all.shape}"
    )
    print(f"   Classes present: y={np.unique(y_all)}, trigger={np.unique(trigger_all)}")

    # --- Optional: Balance classes by downsampling target label ---
    if target_label is not None:
        print(f"\n--- Downsampling label {target_label} to keep ratio {keep_ratio} ---")
        X_bal, y_bal = downsample_label(X_all, y_all, target_label=target_label, keep_ratio=keep_ratio)
    else:
        X_bal, y_bal = X_all, y_all
    print("Before downsample:", np.unique(y_all, return_counts=True))
    print(X_all.shape, y_all.shape)
    print("After downsample:",  np.unique(y_bal, return_counts=True))
    print(X_bal.shape, y_bal.shape)

    return X_all, y_all, X_bal, y_bal, eeg_all, trigger_all, 



def downsample_label(X, y, target_label=0, keep_ratio=0.2, seed=42):
    """
    Randomly keep only a fraction of samples from the given target_label.
    For example, keep_ratio=0.2 keeps 20% of all label 0 windows.

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
    
    X_bal = X[keep_idx]
    y_bal = y[keep_idx]

    return X_bal, y_bal





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



def plot_all_results(results, test_results=None):
    for name, info in results.items():
        print(f"\nPlotting {name}: Test Accuracy = {test_results[name]['test_acc']:.3f}")
        plot_training_metrics(info["metrics_path"])
        
