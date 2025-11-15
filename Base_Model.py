import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, ConfusionMatrix



# Hierachical model definition example  
class BaseModel(pl.LightningModule):
    def __init__(self, in_channels=8, num_classes=6, LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None, class_weights=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # self.save_hyperparameters()
        self.lr = LR
        self.weight_decay = WEIGHT_DECAY
        self.class_labels = class_labels
        self.class_weights = class_weights

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
        
        if self.class_weights is None:
            loss_fn = nn.CrossEntropyLoss()
        else:
            w = torch.tensor(self.class_weights, dtype=torch.float32, device=self.device)
            loss_fn = nn.CrossEntropyLoss(weight=w)
        
        preds = self(X)
        loss = loss_fn(preds, y)
        
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
        im = ax.imshow(cm_percent, interpolation='nearest', cmap='jet', vmin=0, vmax=100)

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


