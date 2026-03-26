import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from model import EEGNet

def train_loso_pipeline(data_path, epochs=100, batch_size=32): # epochs=60 to 100 - more parameters need more time to converge
    print(f"Loading aligned data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    mask = np.isin(data['y'], ['left_hand', 'right_hand'])
    X, y, meta = data['X'][mask], data['y'][mask], data['meta'][mask]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    subjects = meta['subject'].unique()
    
    # --- 🔥 THE CUDA BRIDGE 🔥 ---
    # This automatically detects your RTX A2000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Initializing Hardware: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    # -----------------------------
    
    loso_accuracies = []
    
    for test_sub in subjects:
        print(f"\n--- Testing Unseen Subject {test_sub} ---")
        
        test_idx = meta['subject'] == test_sub
        X_train, y_train = X[~test_idx], y_encoded[~test_idx]
        X_test, y_test = X[test_idx], y_encoded[test_idx]
        
        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                                torch.tensor(y_train, dtype=torch.long)), 
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                               torch.tensor(y_test, dtype=torch.long)), 
                                 batch_size=batch_size, shuffle=False)
        
        # Initialize Model and SHIP IT TO THE GPU
        model = EEGNet(n_classes=2)
        model.to(device) # <--- Moving the neural network to VRAM
        
        # V2 Logic: Standard CrossEntropy (No label smoothing)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # change your loss function to include Label Smoothing. This helps significantly with generalization across different subjects.
        #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        # Inside train_predictor.py
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3) #slower learning prevents the model from "overshooting" the optimal weights
        
        # Training Loop
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                # SHIP BATCHES TO THE GPU
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                
        # Evaluation Loop
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                # SHIP BATCHES TO THE GPU
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        acc = 100 * correct / total
        loso_accuracies.append(acc)
        print(f"Result: {acc:.2f}%")
        
    print(f"\nFINAL AVG LOSO ACCURACY: {np.mean(loso_accuracies):.2f}%")

if __name__ == "__main__":
    train_loso_pipeline('dataset/bci/processed/physionet_mi_aligned.pkl')

