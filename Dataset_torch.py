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
