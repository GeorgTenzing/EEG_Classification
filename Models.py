import numpy as np
import matplotlib.pyplot as plt

import torch
# torch.set_float32_matmul_precision('high')        # for better performance on some hardware
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchmetrics.classification import Accuracy, ConfusionMatrix
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping



# Hierachical model definition example  
class BaseModel(pl.LightningModule):
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # self.save_hyperparameters()
        self.lr = LR
        self.weight_decay = WEIGHT_DECAY
        self.criterion = nn.CrossEntropyLoss()
        self.class_labels = class_labels
        # self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_cm   = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    # Subclasses must define these:
    def forward(self, x):
        raise NotImplementedError("Implement in subclass")
    
    # ------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------
    def training_step(self, batch):
        X, y = batch
        preds = self(X)
        
        # # Define class weights (you can tune the first one for label 0)
        # # weights = torch.tensor([0.5, 1, 1, 1, 1, 1], dtype=torch.float32, device=self.device)
        # weights = torch.ones(self.num_classes, dtype=torch.float32, device=self.device)
        # weights[0] = 0.5
        # # Use weighted cross-entropy
        # criterion = nn.CrossEntropyLoss(weight=weights)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, y)
        
        # loss = nn.functional.cross_entropy(preds, y)
        
        acc = self.train_acc(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------
    def validation_step(self, batch):
        X, y = batch
        preds = self(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, y)
        acc = self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, logger=True, on_step=False)

    # ------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------
    def test_step(self, batch):
        X, y = batch
        preds = self(X)
        loss = nn.functional.cross_entropy(preds, y)
        acc = self.test_acc(preds, y)
        
        # Store confusion matrix for later
        self.test_cm.update(preds, y)
        
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.trainer.max_epochs, 
        #     eta_min=5e-6
        # )
        return optimizer
    
    def on_test_epoch_end(self):
        # --- Compute confusion matrix ---
        cm = self.test_cm.compute().cpu().numpy()
        num_classes = cm.shape[0]

        # --- Normalize confusion matrix row-wise to get percentages ---
        # Each row sums to 1 (or 100%)
        cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

        # --- Class labels ---
        class_labels = self.class_labels

        # --- Compute per-class accuracies ---
        class_acc = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_acc):
            self.log(f"class_{i}_acc", acc)
            print(f"Class {class_labels[i]} accuracy: {acc:.3f}")

        # --- Plot percentage confusion matrix ---
        fig, ax = plt.subplots(figsize=(6, 5))

        # Percent-based heatmap
        im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)

        ax.set_title("Confusion Matrix (% per true class)")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        # Tick marks
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

        # Annotate each cell with PERCENT (not counts)
        for i in range(num_classes):
            for j in range(num_classes):
                value = cm_percent[i, j]
                color = "white" if value > 50 else "black"
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=color, fontsize=10)

        # Colorbar (0% to 100%)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Percentage (%)")

        fig.tight_layout()
        plt.show()

        # --- Reset confusion matrix ---
        self.test_cm.reset()
















class EEGClassifier(BaseModel): # working, 
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)

        # --- Feature extractor ---
        self.features = nn.Sequential(
            # --- Block 1 ---
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # --- Block 2 ---
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # --- Block 3 ---
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # --- Block 4 ---
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
                # nn.ReLU(),
                # nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
                # nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    

class MNISTNet(BaseModel): # working, not bad 70% val acc
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)
        self.drop_p, self.drop_final = 0.15, 0.15
        self.seq = [
            [in_channels, 64, 128, 64, 128, 64],
            [64, 128, 256, 128],
            [128, 256, 128, 64, 128]
        ]

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_c),
                # nn.SiLU()
                nn.ReLU()
            )

        def maxpool_dropout(pool_kernel=2, drop_p=self.drop_p):
            return nn.Sequential(
                nn.MaxPool1d(pool_kernel),
                nn.Dropout(drop_p),
            )

        def conv_block_sequence(channels):
            layers = []
            for in_c, out_c in zip(channels, channels[1:]):
                layers.append(conv_block(in_c, out_c))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block_sequence(self.seq[0]),
            maxpool_dropout(),
            conv_block_sequence(self.seq[1]),
            maxpool_dropout(),
            conv_block_sequence(self.seq[2]),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self.seq[-1][-1]),
            nn.Dropout(self.drop_final),
            nn.Linear(self.seq[-1][-1], num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    
    
    

class EEGNet(BaseModel): # working
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
        F1=32, # 8, 32
        D=8,     # 2, 8
        F2=128, # 128 
        kernel_length=256, # 64
        LR=1e-3,
        WEIGHT_DECAY=1e-5,
        class_labels=None
        
    ):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels)
        
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
        x = x.unsqueeze(1)  # (batch, 1, channels, samples)
        feats = self.forward_features(x)
        out = self.classifier(feats)
        return out