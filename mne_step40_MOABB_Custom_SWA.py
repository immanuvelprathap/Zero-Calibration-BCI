import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skorch import NeuralNetClassifier
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import CrossSessionEvaluation
import warnings

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. YOUR CUSTOM ARCHITECTURE (From Step 36)
# High Dropout, Elu activations, optimized pooling
# ==========================================
class EEGNet_Final(nn.Module):
    def __init__(self, channels=22, samples=1001, n_classes=4):
        super(EEGNet_Final, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.separable = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(0.5) # Heavy structural regularization
        
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, channels, samples)
            dummy_x = self.pooling1(self.batchnorm2(self.depthwise(self.batchnorm1(self.conv1(dummy_x)))))
            dummy_x = self.pooling2(self.batchnorm3(self.separable(dummy_x)))
            self.flat_size = dummy_x.numel()
            
        self.classifier = nn.Linear(self.flat_size, n_classes)

    def forward(self, x):
        # The 64-to-32 bit bridge fix
        x = x.float() 
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = F.elu(self.batchnorm1(self.conv1(x)))
        x = F.elu(self.batchnorm2(self.depthwise(x)))
        x = self.pooling1(x)
        x = self.dropout(x)
        x = F.elu(self.batchnorm3(self.separable(x)))
        x = self.pooling2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==========================================
# 2. THE MOABB REFEREE
# ==========================================
def run_custom_dl_benchmark():
    print("🚀 INITIALIZING MOABB REFEREE FOR CUSTOM ARCHITECTURE...")
    
    dataset = BNCI2014_001()
    dataset.subject_list = [1] 
    paradigm = MotorImagery(n_classes=4, fmin=8, fmax=30, resample=250)

    # We use your advanced AdamW optimizer with Weight Decay to prevent overfitting
    net = NeuralNetClassifier(
        EEGNet_Final,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,           # Upgraded from Adam to AdamW
        optimizer__weight_decay=0.02,    # Heavy penalty on wild weight shifts
        lr=0.001,
        batch_size=64,
        max_epochs=200,                  # Allowing more time to settle
        train_split=None, 
        device=device,
        verbose=0
    )
    
    pipelines = {"Custom EEGNet (AdamW)": net}

    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=dataset,
        overwrite=True,
        hdf5_path=None, 
        n_jobs=1 
    )

    print("⚖️ RUNNING CROSS-SESSION EVALUATION (Training Custom Architecture... hold the line)...")
    results = evaluation.process(pipelines)
    
    print("\n" + "="*50)
    print("✅ CUSTOM DEEP LEARNING BENCHMARK (SUBJECT 1)")
    print("="*50)
    for index, row in results.iterrows():
        print(f"Pipeline: {row['pipeline']}")
        print(f"Target Test Session: {row['session']}")
        print(f"Strict Accuracy: {row['score'] * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_custom_dl_benchmark()