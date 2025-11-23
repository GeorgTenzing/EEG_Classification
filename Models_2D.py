import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics.classification import Accuracy, ConfusionMatrix
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from Models import BaseModel


class EEGClassifier_mel(BaseModel):
    """
    Classifier for MelSpectrogram and STFT spectrogram EEG representations.
    """
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None, dropout=0.3, class_weights=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels, class_weights)

        # C, H, W = in_shape  # channels, freq/mels, frames

        # -----------------------------------------------------------
        # Feature extractor: Conv2D CNN
        # -----------------------------------------------------------
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Compress to 1x1
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Dropout(dropout),
        )

        # -----------------------------------------------------------
        # Classification head
        # -----------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # -----------------------------------------------------------
    # Forward
    # -----------------------------------------------------------
    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.features(x)
        x = self.classifier(x)
        return x
    


class EEGNet_for_mel(BaseModel): # working
    """
    EEGNet v4 (Lawhern et al., 2018) — adapted for your BCI SSVEP dataset
    Compatible with EEG input of shape (batch, n_channels, n_samples)
    """
    def __init__(
        self,
        in_channels=8,
        n_samples=500,
        num_classes=6,
        dropout_rate=0.4, # 0.5
        F1=16, # 8
        D=4, # 2
        F2=64,
        kernel_length=256, # 64
        LR=1e-3,
        WEIGHT_DECAY=1e-5,
        class_labels=None,
        class_weights=None
        
    ):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels, class_weights)
        
        if F2 is None:
            F2 = F1 * D

        # ------------------------------------------------------------
        # EEGNet feature extractor
        # ------------------------------------------------------------
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2),
                      bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (in_channels, 1),
                      groups=F1,
                      bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16),
                      padding=(0, 8),
                      groups=F1 * D,
                      bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, n_samples)
            dummy = self.forward_features(dummy)
            self.flatten_dim = dummy.shape[1]

        # ------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------
        self.classifier = nn.Linear(self.flatten_dim, num_classes)

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    # ------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------
    def forward_features(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        # Input: (batch, channels, samples)
        # x = x.unsqueeze(1)  # (batch, 1, channels, samples)
        B, C, H, W = x.shape   # x is mel spectrogram input
        x = x.reshape(B, 1, C, H * W)
        feats = self.forward_features(x)
        out = self.classifier(feats)
        return out 
    
    
    

# ------------------------------------------------------------
# Residual 2D block
# ------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, se=True):
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 1×1 projection if dimensions change
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride) \
                        if (in_channels != out_channels or downsample) else nn.Identity()

        # Optional squeeze-and-excitation (channel attention)
        if se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.se is not None:
            out = out * self.se(out)

        out += identity
        out = self.relu(out)
        return out


# ------------------------------------------------------------
# MUCH STRONGER MEL/STFT CLASSIFIER
# ------------------------------------------------------------
class EEGClassifier_mel_with_res(BaseModel):
    """
    High-accuracy classifier for MelSpectrogram/STFT spectrogram EEG.
    Uses residual blocks + SE attention.
    """
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3,
                 WEIGHT_DECAY=1e-5, class_labels=None, class_weights=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels, class_weights)

        # --------------------------------------------------------
        # Hierarchical encoder (ResNet-like)
        # --------------------------------------------------------
        self.encoder = nn.Sequential(
            ResBlock(in_channels, 32, downsample=False),
            ResBlock(32, 64, downsample=True),
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 256, downsample=True),
        )

        # Trim variable spectrogram sizes into 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # --------------------------------------------------------
        # Classifier
        # --------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    # ------------------------------------------------------------
    def forward(self, x):
        x = self.encoder(x)    # ResNet-SE encoder
        x = self.pool(x)
        x = self.classifier(x)
        return x






class DepthwiseSeparableConv(nn.Module):
    """MobileNet-style depthwise + pointwise conv."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class EEGClassifier_mel_small(BaseModel):
    """
    Tiny Mel/STFT classifier (<300k parameters)
    Works with input shapes: (B, 8, H, W)
    """

    def __init__(
        self,
        in_channels=8,
        num_classes=6,
        LR=1e-3,
        WEIGHT_DECAY=1e-5,
        class_labels=None,
        dropout=0.25,
        class_weights=None,
    ):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels, class_weights)

        self.in_channels = in_channels

        # ---------------------------
        # Block 1
        # ---------------------------
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ---------------------------
        # Depthwise separable blocks
        # ---------------------------
        self.block2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.block3 = DepthwiseSeparableConv(64, 128, stride=2)
        self.block4 = DepthwiseSeparableConv(128, 256, stride=2)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------------------
        # Classifier
        # ---------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    # -------------------------------------------------------
    # Forward with shape checking and auto-fix
    # -------------------------------------------------------
    def forward(self, x):

        # Ensure input has shape (B, C, H, W)
        if x.ndim == 3:
            raise ValueError(
                f"EEGClassifier_mel_small received raw EEG of shape {x.shape}. "
                "You must use Mel/STFT spectrograms: (B, C, H, W)."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels but got {x.shape[1]}. "
                "Make sure Mel/STFT transform is applied BEFORE the model."
            )

        # Pass through blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)
        x = self.classifier(x)
        return x
