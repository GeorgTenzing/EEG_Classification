from pyclbr import Class
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

from Base_Model import BaseModel_new


# ---------------------------------------------------------------------------
# 1. Small EEG Encoder (TCN-style)
# Turns (B, C, T) → (B, D)
# ---------------------------------------------------------------------------
class TCNModel_v1_outch64_GELU_head2_small(nn.Module): 
    def __init__(self, in_channels=8, hidden=16, D=128):
        super().__init__()
        
        layers = []
        in_ch, out_ch = in_channels, hidden
        for d in [1, 2, 4, 8, 16, 32]:
            layers += [nn.Conv1d(in_ch, out_ch, 3, padding=d, dilation=d), 
                       nn.BatchNorm1d(out_ch), 
                       nn.GELU()]
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        # temporal global average pooling + linear projection
        self.fc = nn.Linear(hidden, D)   # D = embedding dimension

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.tcn(x)           # (B, out_ch, T)
        x = x.mean(-1)            # global average pooling
        return self.fc(x)    # (B, D)

# ---------------------------------------------------------------------------
# 2. TRM Shared Tiny Network
# This is the single network used for BOTH z and y updates.
# It is a lightweight MLP (2-layer) as recommended in the TRM paper.
# ---------------------------------------------------------------------------
class TRMSharedNet(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D)
        )

    def forward(self, mixed):
        return self.net(mixed)





# ---------------------------------------------------------------------------
# 3. Full TRM Model for EEG
# ---------------------------------------------------------------------------

class TRM_EEG_Model_v1(BaseModel_new):
    def __init__(self, 
                 in_channels=8, 
                 D=128,    
                 n_inner=1,       # inner recursions inside deep supervision
                 T_outer=30,       # deep supervision steps 
                 num_classes=6, 
                 LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None, class_weights=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels, class_weights)

        # save config
        self.D = D
        self.num_classes = num_classes
        self.n_inner = n_inner
        self.T_outer = T_outer

        # 1) EEG encoder: (B, C, T) → (B, D)
        self.encoder = TCNModel_v1_outch64_GELU_head2_small(in_channels=in_channels, D=D)

        # 2) latent y and z init (learned parameters)
        self.y_init = nn.Parameter(torch.randn(1, D))
        self.z_init = nn.Parameter(torch.randn(1, D))

        # 3) shared TRM tiny network
        self.shared_net = TRMSharedNet(D)

        # 4) output classifier head
        self.output_head = nn.Linear(D, num_classes)

    # ----------------------------------------------------------------
    # Single recursion step
    # z ← f(x, y, z)
    # y ← f(y, z)  (using same network, but concatenating inputs correctly)
    # ----------------------------------------------------------------
    def recurse_once(self, x_embed, y, z):
        # Update z first: z ← f(x + y + z)
        z_new = self.shared_net(x_embed + y + z)

        # Update y next: y ← f(y + z_new)
        y_new = self.shared_net(y + z_new)

        return y_new, z_new

    # ----------------------------------------------------------------
    # Forward pass (deep supervision)
    # ----------------------------------------------------------------
    def forward(self, x):
        # -------------------------------------------------------------
        # x: (B, C, T)
        # -------------------------------------------------------------
        B = x.shape[0]

        # 1) encode EEG → x_embed: (B, D)
        x_embed = self.encoder(x) 

        # 2) initialize y and z (repeat for batch dimension)
        y = self.y_init.repeat(B, 1)  # (B, D)
        z = self.z_init.repeat(B, 1)  # (B, D)

        # 3) deep supervision outer loop
        logits_list = []

        for t in range(self.T_outer):

            # inner recursion loop
            for i in range(self.n_inner):
                y, z = self.recurse_once(x_embed, y, z)

            # classifier output at this deep supervision step
            logits = self.output_head(y)   # (B, num_classes)
            # logits_list.append(logits)

        # 4) Return final output for training/inference
        # return logits_list         # return list of predictions for each DS step
        return logits     # return final prediction only


class TRM_EEG_Model_v2(BaseModel_new):
    def __init__(self, 
                 in_channels=8, 
                 D=128,    
                 n_inner=1,       # inner recursions inside deep supervision
                 T_outer=30,       # deep supervision steps 
                 num_classes=6, 
                 LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None, class_weights=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels, class_weights)

        # save config
        self.D = D
        self.num_classes = num_classes
        self.n_inner = n_inner
        self.T_outer = T_outer

        # 1) EEG encoder: (B, C, T) → (B, D)
        self.encoder = TCNModel_v1_outch64_GELU_head2_small(in_channels=in_channels, D=D)

        # 2) latent y and z init (learned parameters)
        self.y_init = nn.Parameter(torch.randn(1, D))
        self.z_init = nn.Parameter(torch.randn(1, D))

        # 3) shared TRM tiny network
        self.shared_net = TRMSharedNet(D)

        # 4) output classifier head
        self.output_head = nn.Linear(D, num_classes)

    # ----------------------------------------------------------------
    # Single recursion step
    # z ← f(x, y, z)
    # y ← f(y, z)  (using same network, but concatenating inputs correctly)
    # ----------------------------------------------------------------
    def recurse_once(self, x_embed, y, z):
        # Update z first: z ← f(x + y + z)
        z_new = self.shared_net(x_embed + y + z)
        return z_new

    # ----------------------------------------------------------------
    # Forward pass (deep supervision)
    # ----------------------------------------------------------------
    def forward(self, x):
        # -------------------------------------------------------------
        # x: (B, C, T)
        # -------------------------------------------------------------
        B = x.shape[0]

        # 1) encode EEG → x_embed: (B, D)
        x_embed = self.encoder(x) 

        # 2) initialize y and z (repeat for batch dimension)
        y = self.y_init.repeat(B, 1)  # (B, D)
        z = self.z_init.repeat(B, 1)  # (B, D)

        # 3) deep supervision outer loop
        logits_list = []

        for t in range(self.T_outer):

            # inner recursion loop
            for i in range(self.n_inner):
                z = self.recurse_once(x_embed, y, z)
                
            y = self.shared_net(y + z)

            # classifier output at this deep supervision step
            logits = self.output_head(y)   # (B, num_classes)
            # logits_list.append(logits)

        # 4) Return final output for training/inference
        # return logits_list         # return list of predictions for each DS step
        return logits     # return final prediction only
    




class TRM_EEG_Model_v1_2(BaseModel_new):
    def __init__(self, 
                 in_channels=8, 
                 D=128,    
                 n_inner=1,       # inner recursions inside deep supervision
                 T_outer=25,       # deep supervision steps 
                 num_classes=6, 
                 LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None, class_weights=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels, class_weights)

        # save config
        self.D = D
        self.num_classes = num_classes
        self.n_inner = n_inner
        self.T_outer = T_outer

        # 1) EEG encoder: (B, C, T) → (B, D)
        self.encoder = TCNModel_v1_outch64_GELU_head2_small(in_channels=in_channels, D=D)

        # 2) latent y and z init (learned parameters)
        self.y_init = nn.Parameter(torch.randn(1, D))
        self.z_init = nn.Parameter(torch.randn(1, D))

        # 3) shared TRM tiny network
        self.shared_net = TRMSharedNet(D)

        # 4) output classifier head
        self.output_head = nn.Linear(D, num_classes)

    # ----------------------------------------------------------------
    # Single recursion step
    # z ← f(x, y, z)
    # y ← f(y, z)  (using same network, but concatenating inputs correctly)
    # ----------------------------------------------------------------
    def recurse_once(self, x_embed, y, z):
        # Update z first: z ← f(x + y + z)
        z_new = self.shared_net(x_embed + y + z)

        # Update y next: y ← f(y + z_new)
        y_new = self.shared_net(y + z_new)

        return y_new, z_new

    # ----------------------------------------------------------------
    # Forward pass (deep supervision)
    # ----------------------------------------------------------------
    def forward(self, x):
        # -------------------------------------------------------------
        # x: (B, C, T)
        # -------------------------------------------------------------
        B = x.shape[0]

        # 1) encode EEG → x_embed: (B, D)
        x_embed = self.encoder(x) 

        # 2) initialize y and z (repeat for batch dimension)
        y = self.y_init.repeat(B, 1)  # (B, D)
        z = self.z_init.repeat(B, 1)  # (B, D)

        # 3) deep supervision outer loop
        logits_list = []

        for t in range(self.T_outer):

            # inner recursion loop
            for i in range(self.n_inner):
                y, z = self.recurse_once(x_embed, y, z)

            # classifier output at this deep supervision step
            logits = self.output_head(y)   # (B, num_classes)
            # logits_list.append(logits)

        # 4) Return final output for training/inference
        # return logits_list         # return list of predictions for each DS step
        return logits     # return final prediction only





class TRM_EEG_Model_v1_3(BaseModel_new):
    def __init__(self, 
                 in_channels=8, 
                 D=65,    
                 n_inner=1,       # inner recursions inside deep supervision
                 T_outer=30,       # deep supervision steps 
                 num_classes=6, 
                 LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None, class_weights=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels, class_weights)

        # save config
        self.D = D
        self.num_classes = num_classes
        self.n_inner = n_inner
        self.T_outer = T_outer

        # 1) EEG encoder: (B, C, T) → (B, D)
        self.encoder = TCNModel_v1_outch64_GELU_head2_small(in_channels=in_channels, D=D)

        # 2) latent y and z init (learned parameters)
        self.y_init = nn.Parameter(torch.randn(1, D))
        self.z_init = nn.Parameter(torch.randn(1, D))

        # 3) shared TRM tiny network
        self.shared_net = TRMSharedNet(D)

        # 4) output classifier head
        self.output_head = nn.Linear(D, num_classes)

    # ----------------------------------------------------------------
    # Single recursion step
    # z ← f(x, y, z)
    # y ← f(y, z)  (using same network, but concatenating inputs correctly)
    # ----------------------------------------------------------------
    def recurse_once(self, x_embed, y, z):
        # Update z first: z ← f(x + y + z)
        z_new = self.shared_net(x_embed + y + z)

        # Update y next: y ← f(y + z_new)
        y_new = self.shared_net(y + z_new)

        return y_new, z_new

    # ----------------------------------------------------------------
    # Forward pass (deep supervision)
    # ----------------------------------------------------------------
    def forward(self, x):
        # -------------------------------------------------------------
        # x: (B, C, T)
        # -------------------------------------------------------------
        B = x.shape[0]

        # 1) encode EEG → x_embed: (B, D)
        x_embed = self.encoder(x) 

        # 2) initialize y and z (repeat for batch dimension)
        y = self.y_init.repeat(B, 1)  # (B, D)
        z = self.z_init.repeat(B, 1)  # (B, D)

        # 3) deep supervision outer loop
        logits_list = []

        for t in range(self.T_outer):

            # inner recursion loop
            for i in range(self.n_inner):
                y, z = self.recurse_once(x_embed, y, z)

            # classifier output at this deep supervision step
            logits = self.output_head(y)   # (B, num_classes)
            # logits_list.append(logits)

        # 4) Return final output for training/inference
        # return logits_list         # return list of predictions for each DS step
        return logits     # return final prediction only



class TRM_EEG_Model_v1_3(BaseModel_new):
    def __init__(self, 
                 in_channels=8, 
                 D=256,    
                 n_inner=1,       # inner recursions inside deep supervision
                 T_outer=20,       # deep supervision steps 
                 num_classes=6, 
                 LR=1e-3, WEIGHT_DECAY=1e-5, class_labels=None, class_weights=None):
        super().__init__(in_channels, num_classes, LR, WEIGHT_DECAY, class_labels, class_weights)

        # save config
        self.D = D
        self.num_classes = num_classes
        self.n_inner = n_inner
        self.T_outer = T_outer

        # 1) EEG encoder: (B, C, T) → (B, D)
        self.encoder = TCNModel_v1_outch64_GELU_head2_small(in_channels=in_channels, D=D)

        # 2) latent y and z init (learned parameters)
        self.y_init = nn.Parameter(torch.randn(1, D))
        self.z_init = nn.Parameter(torch.randn(1, D))

        # 3) shared TRM tiny network
        self.shared_net = TRMSharedNet(D)

        # 4) output classifier head
        self.output_head = nn.Linear(D, num_classes)

    # ----------------------------------------------------------------
    # Single recursion step
    # z ← f(x, y, z)
    # y ← f(y, z)  (using same network, but concatenating inputs correctly)
    # ----------------------------------------------------------------
    def recurse_once(self, x_embed, y, z):
        # Update z first: z ← f(x + y + z)
        z_new = self.shared_net(x_embed + y + z)

        # Update y next: y ← f(y + z_new)
        y_new = self.shared_net(y + z_new)

        return y_new, z_new

    # ----------------------------------------------------------------
    # Forward pass (deep supervision)
    # ----------------------------------------------------------------
    def forward(self, x):
        # -------------------------------------------------------------
        # x: (B, C, T)
        # -------------------------------------------------------------
        B = x.shape[0]

        # 1) encode EEG → x_embed: (B, D)
        x_embed = self.encoder(x) 

        # 2) initialize y and z (repeat for batch dimension)
        y = self.y_init.repeat(B, 1)  # (B, D)
        z = self.z_init.repeat(B, 1)  # (B, D)

        # 3) deep supervision outer loop
        logits_list = []

        for t in range(self.T_outer):

            # inner recursion loop
            for i in range(self.n_inner):
                y, z = self.recurse_once(x_embed, y, z)

            # classifier output at this deep supervision step
            logits = self.output_head(y)   # (B, num_classes)
            # logits_list.append(logits)

        # 4) Return final output for training/inference
        # return logits_list         # return list of predictions for each DS step
        return logits     # return final prediction only



