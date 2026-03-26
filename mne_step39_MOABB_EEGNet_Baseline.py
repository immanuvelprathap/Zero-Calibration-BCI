import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import CrossSessionEvaluation
import warnings

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. THE VANILLA EEGNET (Lawhern et al. 2018)
# No Mixup, No SWA, No Euclidean Alignment
# ==========================================
class Vanilla_EEGNet(nn.Module):
    def __init__(self, channels=22, samples=1001, n_classes=4):
        super(Vanilla_EEGNet, self).__init__()
        # MOABB feeds data as (Batch, Channels, Time). PyTorch Conv2D needs (Batch, 1, Channels, Time).
        # We handle this expansion in the forward pass.
        
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (channels, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.separable = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(0.5)
        
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, channels, samples)
            dummy_x = self.pooling1(self.batchnorm2(self.depthwise(self.batchnorm1(self.conv1(dummy_x)))))
            dummy_x = self.pooling2(self.batchnorm3(self.separable(dummy_x)))
            self.flat_size = dummy_x.numel()
            
        self.classifier = nn.Linear(self.flat_size, n_classes)

    # def forward(self, x):
    #     # Skorch/MOABB passes 3D tensors. Add the empty "1" channel dimension for Conv2D.
    #     if len(x.shape) == 3:
    #         x = x.unsqueeze(1) 
            
    #     x = F.elu(self.batchnorm1(self.conv1(x)))
    #     x = F.elu(self.batchnorm2(self.depthwise(x)))
    #     x = self.pooling1(x)
    #     x = self.dropout(x)
    #     x = F.elu(self.batchnorm3(self.separable(x)))
    #     x = self.pooling2(x)
    #     x = self.dropout(x)
    #     x = x.view(x.size(0), -1)
    #     return self.classifier(x)
    def forward(self, x):
        # 1. Cast the heavy 64-bit numpy data to 32-bit PyTorch floats
        x = x.float() 
        
        # 2. Skorch/MOABB passes 3D tensors. Add the empty "1" channel dimension for Conv2D.
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
# 2. THE MOABB REFEREE PIPELINE
# ==========================================
def run_dl_benchmark():
    print("🚀 INITIALIZING MOABB REFEREE FOR DEEP LEARNING...")
    
    dataset = BNCI2014_001()
    dataset.subject_list = [1] ####################   For changing Subject ID do it here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  ###############################
    # Notice the resample=250. This creates 1001 samples for a 4-second epoch.
    paradigm = MotorImagery(n_classes=4, fmin=8, fmax=30, resample=250)

    # Wrap our PyTorch model so MOABB can use it like a Scikit-Learn model
    net = NeuralNetClassifier(
        Vanilla_EEGNet,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        batch_size=64,
        max_epochs=150,
        train_split=None, # MOABB handles the train/test splitting manually
        device=device,
        verbose=0 # Keep console clean
    )
    
    pipelines = {"Vanilla EEGNet": net}

    # The Strict Cross-Session Referee (Train Day 1, Test Day 2)
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=dataset,
        overwrite=True,
        hdf5_path=None, 
        n_jobs=1 
    )

    print("⚖️ RUNNING CROSS-SESSION EVALUATION (Training Vanilla EEGNet... this will take a moment)...")
    results = evaluation.process(pipelines)
    
    print("\n" + "="*50)
    print("✅ OFFICIAL DEEP LEARNING BENCHMARK (SUBJECT 1)") ###################   For changing Subject ID do it here!!!!!!!!!!!!!!!!print("✅ OFFICIAL DEEP LEARNING BENCHMARK (SUBJECT 2)")!!!!!!!!!!!!!!!!!!!!  ###############################
    print("="*50)
    for index, row in results.iterrows():
        print(f"Pipeline: {row['pipeline']}")
        print(f"Target Test Session: {row['session']}")
        print(f"Strict Accuracy: {row['score'] * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_dl_benchmark()