import numpy as np

import torch
# torch.set_float32_matmul_precision('high')        # for better performance on some hardware
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics.classification import Accuracy, ConfusionMatrix
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.nn.functional as F
import torchaudio

from Models import BaseModel


class TCNModel_withBase(BaseModel): # working, 
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)
        layers = []
        in_ch, out_ch = in_channels, 32
        for d in [1, 2, 4, 8]:
            layers += [nn.Conv1d(in_ch, out_ch, 3, padding=d, dilation=d),
                       nn.BatchNorm1d(out_ch), nn.ReLU()]
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(out_ch, num_classes)
    def forward(self, x):
        x = self.tcn(x)           # (B, out_ch, T)
        x = x.mean(-1)            # global average pooling
        return self.head(x)









class BiLSTMModel_withBase(BaseModel): # working, but ass 
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, chans=8, hidden_size=64, num_layers=2, dropout=0.25, class_labels=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)
        self.hidden_size = hidden_size

        # Temporal feature extractor (optional 1D conv front-end)
        self.encoder = nn.Sequential(
            nn.Conv1d(chans, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.encoder(x)               # (B, 64, T')
        x = x.permute(0, 2, 1)            # (B, T', 64)
        x, _ = self.lstm(x)               # (B, T', 2*hidden)
        x = x[:, -1, :]                   # last time step
        
        return self.fc(x)

    
class ShallowConvNet_withBase(BaseModel): # working, ass
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, chans=8, samples=500, class_labels=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)

        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), bias=False),       # temporal filters
            nn.Conv2d(40, 40, (chans, 1), bias=False),   # spatial filters
            nn.BatchNorm2d(40),
            nn.ELU(),                                    # nonlinearity
            nn.AvgPool2d((1, 75), stride=(1, 15))
        )

        # Compute flattened feature size dynamically (safer)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, chans, samples)
            out = self.net(dummy)
            flatten_dim = out.shape[1] * out.shape[2] * out.shape[3]

        self.classifier = nn.Linear(flatten_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)      # (B, 1, chans, samples)
        x = self.net(x)
        x = torch.square(x)     # elementwise square (like nn.Square)
        x = torch.log(x + 1e-6) # elementwise log (stabilized)
        x = x.flatten(start_dim=1)
        return self.classifier(x)






class EEGTransformer_withBase(BaseModel):  # working, ass
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, chans=8, samples=500, embed_dim=64, nhead=4, class_labels=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)
        self.conv = nn.Conv1d(chans, embed_dim, 8, 4, 2)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 1)   # (B, T, embed)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])



class SpectroCNN_withBase(BaseModel):
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, chans=8, samples=500, n_fft=128, hop_length=32):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY)

        self.spectrogram = torchaudio.transformsSpectrogram(
            n_fft=n_fft, hop_length=hop_length, power=2.0
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(chans, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, chans, samples)
        # Compute per-channel spectrograms
        specs = [self.spectrogram(x[:, c]) for c in range(x.shape[1])]
        x = torch.stack(specs, dim=1)      # (B, chans, freq, time)
        x = self.cnn(x)
        return self.fc(x)


class TCNAttentionModel_withBase(BaseModel):
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, chans=8, hidden=64, class_labels=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)
        self.tcn = nn.Sequential(
            nn.Conv1d(chans, hidden, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.tcn(x)           # (B, hidden, T)
        x = x.permute(0, 2, 1)    # (B, T, hidden)
        attn_out, _ = self.attn(x, x, x)
        x = attn_out.mean(dim=1)
        return self.fc(x)


class EEGTransformerLite_withBase(BaseModel):
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, chans=8, samples=500, embed_dim=64, nhead=4, class_labels=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)
        self.proj = nn.Conv1d(chans, embed_dim, 8, 4, 2)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, 128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 1)  # (B, T, embed)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

    
    
    
    
    
    
    
    








import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset


class Conv1d_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5, stride=1, padding=2): 
        super().__init__()
        self.conv     = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activate =  nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activate(self.conv(x))

    

LAYER_KINDS = {
        "conv", "pool", "drop", "bottleneck", "out" # working blocks
    }

LAYER_FACTORY = {
    "conv":       Conv1d_block,
    "bottleneck": Conv1d_block,
    "deconv":  lambda ic, oc: nn.ConvTranspose1d(ic, oc, kernel_size=2, stride=2),
    "out":     lambda ic, oc: nn.Conv1d(ic, oc, kernel_size=1),
    "pool":    lambda *a:     nn.MaxPool1d(2),
    "drop":    lambda *a:     nn.Dropout(a[0]),
}

sol_6307  = [
    # Encoder
    ("conv", 1, 1, "save1"),
    ("pool",),
    ("conv", 1, 2),
      ("drop", 0.25, "save2"),
    ("pool",),
    ("conv", 2, 4, "save3"),
    ("pool",),
    ("conv", 4, 8, "save4"),
    ("pool",),
    ("conv", 8, 16, "save5"),
    ("pool",),
    ("conv", 16, 32, "save6"),
    # ("pool",),
    # ("conv", 32, 64, "save7"), 
    
    # Bottleneck
    ("pool",),
    # ("bottleneck", 64, 128),
    ("bottleneck", 43, 64),
    
      ("drop", 0.5),

    # Decoder
    # ("deconv", 128, 64, "save7"),
    ("deconv", 64, 32, "save6"),
    ("deconv", 32, 16, "save5"),
    ("deconv", 16, 8, "save4"),
    ("deconv", 8, 4, "save3"),
      ("drop", 0.25),
    ("deconv", 4, 2, "save2"),
    ("deconv", 2, 1, "save1"),

    # Output
    ("out", 1, 1)
]

# --- Main Model ---
class FlexibleUNet1D(BaseModel):
    def __init__(
        self,
        user_spec = sol_6307,
        in_channels=8,
        num_classes=6,
        base=16,
        LR=1e-3,
        WEIGHT_DECAY=1e-5,
        class_labels=None,
    ):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)
        # self.save_hyperparameters()
        self.base = base
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.LR = LR
        self.WEIGHT_DECAY = WEIGHT_DECAY

        # --- Translate user-friendly spec ---
        self.config_spec = translate_spec(user_spec)
        self.layers = nn.ModuleList()

        for i, (category, kind, *args) in enumerate(self.config_spec):
            if category == "layer":
                factory = LAYER_FACTORY[kind]
                if kind in ["pool", "drop"]:
                    self.layers.append(factory(*args))
                else:
                    in_mult, out_mult, *rest = args
                    in_c = in_channels if i == 0 else in_mult * base
                    out_c = num_classes if kind == "out" else out_mult * base
                    self.layers.append(factory(in_c, out_c))
                    
        # --- Add global pooling and classification head ---
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier  = nn.Linear(base * 2, num_classes)  # adjust *2 if last concat doubles channels

    # ---------------------------------------------------------------------
    def forward(self, x):
        """
        x: Tensor of shape (B, C, T)
        """
        context = {}
        i = 0

        for category, kind, *args in self.config_spec:
            if category == "layer":
                x = self.layers[i](x)
                i += 1
                if args and isinstance(args[-1], str):
                    # store feature map for skip connection
                    context[args[-1]] = x.clone()

            elif category == "transform":
                if kind == "concat":
                    skip_name = args[0]
                    skip_feat = context[skip_name]

                    # --- handle size mismatch (e.g., pooling) ---
                    if skip_feat.shape[-1] != x.shape[-1]:
                        skip_feat = nn.functional.interpolate(
                            skip_feat, size=x.shape[-1], mode="linear", align_corners=False
                        )
                    x = torch.cat([x, skip_feat], dim=1)
        # --- Global pooling and classification ---
        x = self.global_pool(x)   # (B, C, 1)
        x = x.squeeze(-1)         # (B, C)
        x = self.classifier(x)    # (B, num_classes)
        return x
    
        

def translate_spec(user_spec: list) -> list:
    """
    Expand a user-friendly model spec into a normalized config spec.
    Converts tuples like ("deconv", in, out, "skip") into a full
    sequence of deconv + concat + conv blocks.
    """
    config = []

    for step in user_spec:
        kind = step[0]

        # --- deconv shortcut expansion ---
        if kind == "deconv" and len(step) == 4:
            _, in_mult, out_mult, skip = step
            config.extend([
                ("layer", "deconv", in_mult, out_mult),
                ("transform", "concat", skip),
                ("layer", "conv", out_mult * 2, out_mult),
            ])
            continue

        # --- regular layer kinds ---
        if kind in LAYER_KINDS:
            config.append(("layer", *step))
        else:
            raise ValueError(f"Unknown spec kind: {kind}")

    # optional pretty-print
    print("\nConfig Spec:")
    for i, (cat, kind, *args) in enumerate(config):
        print(f"{i:02d}: {cat:<9} {kind:<10} {args}")

    return config







