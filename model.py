import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
    Designed to process (Batch, Channels, Time) data.
    """
    def __init__(self, n_classes=2, channels=64, samples=481, 
                 dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        # ----------------------------------------------------------------------
        # Block 1: Temporal Convolution (Bandpass filters)
        # ----------------------------------------------------------------------
        # Learns frequency filters independently of where they happen on the head.
        self.block1 = nn.Sequential(
            # Input: (Batch, 1, Channels, Time) -> Output: (Batch, F1, Channels, Time)
            nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # ----------------------------------------------------------------------
        # Block 2: Depthwise Spatial Convolution
        # ----------------------------------------------------------------------
        # Learns the optimal electrode weights for each frequency band.
        self.block2 = nn.Sequential(
            # Depthwise Conv: groups=F1 makes it learn spatial filters per temporal filter
            nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            # Average pooling over time shrinks the sequence length by 4
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # ----------------------------------------------------------------------
        # Block 3: Separable Convolution
        # ----------------------------------------------------------------------
        # Optimally combines spatio-temporal features using 1x1 pointwise convolutions.
        self.block3 = nn.Sequential(
            # Depthwise over time
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            # Pointwise (1x1) over filters
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            # Average pooling over time shrinks the sequence length by 8
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # ----------------------------------------------------------------------
        # Block 4: Classification
        # ----------------------------------------------------------------------
        # Calculate the number of features after the two pooling operations
        out_samples = (samples // 4) // 8
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * out_samples, n_classes)
        )

    def forward(self, x):
        # EEG data comes in as (Batch, Channels, Time)
        # Conv2d expects (Batch, Color_Channels, Height, Width)
        # We add a dummy "color" dimension of 1.
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    # --- Quick Test ---
    print("Testing EEGNet Architecture...")
    
    # Create a dummy tensor matching our PhysioNet dimensions
    # Batch Size = 16, Channels = 64, Time Samples = 481
    dummy_input = torch.randn(16, 64, 481)
    
    # Initialize the model
    model = EEGNet(n_classes=2, channels=64, samples=481)
    
    # Pass the dummy data through the network
    output = model(dummy_input)
    
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Batch, Classes)")
    print("Network forward pass successful! Ready for training.")