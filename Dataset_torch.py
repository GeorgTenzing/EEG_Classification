import torch
import numpy as np
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import MelSpectrogram



# ============================================================
# 1. Unified preprocessing + dataset class
# ============================================================
class EEGDataset(Dataset):
    """
    EEG dataset with built-in preprocessing:
      - trial-wise normalization
      - channel selection (now 8 EEG channels)
      - conversion to tensors
    """
    def __init__(self, X, y, occipital_slice=None):
        """
        Args:
            X (np.ndarray): EEG data (n_windows, n_channels, n_samples)
            y (np.ndarray): Labels   (n_windows, n_samples)
            occipital_slice (slice): Optional channel selection slice
        """

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        n_windows = X.shape[0]

        # --- Normalize each windows individually ---
        for i in range(n_windows):
            mean = X[i].mean()
            std  = X[i].std() if X[i].std() != 0 else 1.0
            X[i] = (X[i] - mean) / std

        # --- Channel selection (for your setup: 8 channels total) ---
        if occipital_slice is not None:
            X = X[:, occipital_slice, :]

        # --- Convert to tensors ---
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return self.X[idx].unsqueeze(0), self.y[idx]
        return self.X[idx], self.y[idx]
    

import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, iirnotch


class EEGDataset_with_filters(Dataset):

    def __init__(
        self,
        X, y,
        occipital_slice=None,
        notch_50=False,
        custom_filter_fn=None,   # lambda x: filtered_x
        sample_rate=500
    ):
        """
        Args:
            X : (n_windows, n_channels, n_samples)
            y : labels
            notch_50 : bool → apply 50 Hz notch
            custom_filter_fn : function(x) → x, applied per window
        """

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # -----------------------------
        # Filter helpers
        # -----------------------------
        def apply_notch(x):
            b, a = iirnotch(50, Q=30, fs=sample_rate)
            return filtfilt(b, a, x, axis=-1)

        def apply_highpass(x, cutoff):
            nyq = sample_rate * 0.5
            b, a = butter(4, cutoff / nyq, btype='high')
            return filtfilt(b, a, x, axis=-1)

        # -----------------------------
        # Apply filters to all windows
        # -----------------------------
        for i in range(X.shape[0]):

            window = X[i]  # shape (C, T)

            if notch_50:
                window = apply_notch(window)

            X[i] = window

        # ---------------------------
        # Normalize per-window
        # ---------------------------
        for i in range(X.shape[0]):
            mean = X[i].mean()
            std = X[i].std() if X[i].std() != 0 else 1.0
            X[i] = (X[i] - mean) / std

        # Optional channel selection
        if occipital_slice is not None:
            X = X[:, occipital_slice, :]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]







class EEGDataset_mel(Dataset):
    def __init__(
        self, 
        X, y, 
        occipital_slice=None,
        transform="mel",          # "mel", "fft", "stft", None
        sample_rate=500,
        n_mels=64
    ):
        """
        Args:
            transform: One of {"mel", "fft", "stft", None}
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        n_windows = X.shape[0]

        # Normalize each window
        for i in range(n_windows):
            m = X[i].mean()
            s = X[i].std() if X[i].std() != 0 else 1.0
            X[i] = (X[i] - m) / s

        # Optional channel selection
        if occipital_slice is not None:
            X = X[:, occipital_slice, :]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        # Store parameters
        self.transform = transform
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Pre-create torchaudio transforms
        if transform == "mel":
            self.mel = MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                normalized=True,
                hop_length=32,      # small → more frames
            )

    # -------------------------------------------------------
    # Transformation functions
    # -------------------------------------------------------
    def apply_mel(self, x):
        """Apply MelSpectrogram per channel."""
        mel_list = []
        for c in range(x.shape[0]):
            mel = self.mel(x[c].unsqueeze(0))  # (1, n_mels, frames)
            mel_list.append(mel.squeeze(0))
        return torch.stack(mel_list)          # (channels, n_mels, frames)

    def apply_fft(self, x):
        """Apply magnitude FFT per channel."""
        # x: (channels, samples)
        Xf = torch.fft.rfft(x, dim=-1)         # complex
        mag = torch.abs(Xf)                    # magnitude
        return mag                              # (channels, freq_bins)

    def apply_stft(self, x):
        """Short-Time Fourier Transform per channel."""
        stft_list = []
        for c in range(x.shape[0]):
            Z = torch.stft(
                x[c],
                n_fft=128,
                hop_length=64,
                window=torch.hann_window(128),
                return_complex=True
            )
            stft_list.append(torch.abs(Z))     # magnitude: (freq, frames)
        return torch.stack(stft_list)          # (channels, freq, frames)

    # -------------------------------------------------------
    # __getitem__ uses the selected transform
    # -------------------------------------------------------
    def __getitem__(self, idx):
        x = self.X[idx]

        if self.transform == "mel":
            x = self.apply_mel(x)

        elif self.transform == "fft":
            x = self.apply_fft(x)

        elif self.transform == "stft":
            x = self.apply_stft(x)

        return x, self.y[idx]

    def __len__(self):
        return len(self.y)



from scipy.signal import iirnotch, filtfilt, savgol_filter, medfilt, detrend

class EEGDataset_mel_with_filters(Dataset):
    """
    EEG dataset with optional filters and Mel/STFT spectrogram transforms.
    """

    def __init__(
        self,
        X, y,
        occipital_slice=None,
        filters=["notch", "savgol"],        # ["notch", "savgol", ...]
        notch_50=False,
        transform="mel",      # "mel", "stft", or None
        sample_rate=500,
        n_mels=64,
    ):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.transform = transform
        self.sample_rate = sample_rate

        if filters is None:
            filters = []

        # ------------------------------------------------
        # GOOD FILTERS
        # ------------------------------------------------
        def notch(x):
            b, a = iirnotch(50, Q=30, fs=sample_rate)
            return filtfilt(b, a, x, axis=-1)

        def savgol(x):
            return savgol_filter(x, 31, 3, axis=-1)

        def median(x):
            return medfilt(x, kernel_size=(1, 5))

        def remove_trend(x):
            return detrend(x, axis=-1)

        def hann_taper(x):
            w = np.hanning(x.shape[-1])
            return x * w

        filter_map = {
            "notch": notch,
            "savgol": savgol,
            "median": median,
            "detrend": remove_trend,
            "hann": hann_taper,
        }

        for f in filters:
            if f not in filter_map:
                raise ValueError(f"Invalid filter: {f}")

        # ------------------------------------------------
        # Apply filters
        # ------------------------------------------------
        for i in range(X.shape[0]):
            window = X[i]
            for f in filters:
                window = filter_map[f](window)
            X[i] = window

        # ------------------------------------------------
        # Channel selection
        # ------------------------------------------------
        if occipital_slice is not None:
            X = X[:, occipital_slice, :]

        # ------------------------------------------------
        # Normalization (global)
        # ------------------------------------------------
        for i in range(X.shape[0]):
            m = X[i].mean()
            s = X[i].std() if X[i].std() != 0 else 1.0
            X[i] = (X[i] - m) / s

        # ------------------------------------------------
        # Prepare spectrogram transforms
        # ------------------------------------------------
        if transform == "mel":
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=256,
                hop_length=32,
                n_mels=n_mels,
                normalized=True,
            )

        elif transform == "stft":
            self.n_fft = 256
            self.hop_length = 32

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    # ------------------------------------------------
    def __len__(self):
        return len(self.y)

    # ------------------------------------------------
    def __getitem__(self, idx):
        x = self.X[idx]

        # ----------------------------
        # MEL SPECTROGRAM
        # ----------------------------
        if self.transform == "mel":
            # x: (C, T)
            out = []
            for c in range(x.shape[0]):
                mel = self.mel(x[c].unsqueeze(0)).squeeze(0)
                out.append(mel)
            x = torch.stack(out)    # (C=8, mel_bins, frames)

        # ----------------------------
        # STFT SPECTROGRAM
        # ----------------------------
        elif self.transform == "stft":
            out = []
            for c in range(x.shape[0]):
                S = torch.stft(
                    x[c],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=torch.hann_window(self.n_fft),
                    return_complex=True,
                )
                out.append(torch.abs(S))
            x = torch.stack(out)    # (C=8, freq_bins, frames)

        return x, self.y[idx]
