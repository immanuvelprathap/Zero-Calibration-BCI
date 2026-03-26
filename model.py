# import torch
# import torch.nn as nn

# class EEGNet(nn.Module):
#     def __init__(self, n_classes=2, channels=64, samples=481):
#         super(EEGNet, self).__init__()
        
#         # --- BLOCK 1: MULTI-SCALE FILTER BANK ---
#         # We use two different kernel sizes to catch both Alpha and Beta waves
#         self.temp_conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
#         self.temp_conv2 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False)
        
#         self.bn1 = nn.BatchNorm2d(32) # Combined 16 + 16 filters
        
#         # --- BLOCK 2: SPATIAL CONVOLUTION (Depthwise) ---
#         self.depth_conv = nn.Conv2d(32, 64, (channels, 1), groups=32, bias=False)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.elu = nn.ELU()
#         self.avg_pool1 = nn.AvgPool2d((1, 4))
#         self.dropout1 = nn.Dropout(0.3)
        
#         # --- BLOCK 3: SEPARABLE CONVOLUTION ---
#         self.sep_conv = nn.Conv2d(64, 64, (1, 16), padding=(0, 8), groups=64, bias=False)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.avg_pool2 = nn.AvgPool2d((1, 8))
#         self.dropout2 = nn.Dropout(0.3)
        
#         # --- BLOCK 4: CLASSIFIER ---
#         # Calculate the flattened size dynamically
#         self.feature_size = 64 * (samples // 4 // 8) 
#         self.fc = nn.Linear(self.feature_size, n_classes)

#     def forward(self, x):
#         # Add the channel dimension [Batch, 1, Channels, Samples]
#         if len(x.shape) == 3:
#             x = x.unsqueeze(1)
            
#         # Parallel Temporal Filters
#         x1 = self.temp_conv1(x)
#         x2 = self.temp_conv2(x)
#         x = torch.cat((x1, x2), dim=1) # Combine features
        
#         x = self.bn1(x)
#         x = self.elu(self.depth_conv(x))
#         x = self.bn2(x)
#         x = self.avg_pool1(x)
#         x = self.dropout1(x)
        
#         x = self.elu(self.sep_conv(x))
#         x = self.bn3(x)
#         x = self.avg_pool2(x)
#         x = self.dropout2(x)
        
#         x = x.view(x.size(0), -1) # Flatten
#         return self.fc(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.ELU(),
            nn.Linear(in_channels // 4, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EEGNet(nn.Module):
    def __init__(self, n_classes=2, channels=64, samples=481):
        super(EEGNet, self).__init__()
        
        # 1. Multi-Scale Filter Bank
        self.temp_conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.temp_conv2 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 2. Attention Layer (The 80% Push)
        self.attention = AttentionBlock(32)
        
        # 3. Spatial Convolution
        self.depth_conv = nn.Conv2d(32, 64, (channels, 1), groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.35)
        
        # 4. Separable Convolution
        self.sep_conv = nn.Conv2d(64, 64, (1, 16), padding=(0, 8), groups=64, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.35)
        
        self.feature_size = 64 * (samples // 4 // 8) 
        self.fc = nn.Linear(self.feature_size, n_classes)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        
        x = torch.cat((self.temp_conv1(x), self.temp_conv2(x)), dim=1)
        x = self.bn1(x)
        
        # Apply Attention
        x = self.attention(x)
        
        x = self.elu(self.depth_conv(x))
        x = self.bn2(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        x = self.elu(self.sep_conv(x))
        x = self.bn3(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        return self.fc(x)