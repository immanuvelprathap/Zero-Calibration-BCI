import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from model import EEGNet

def train_loso_pipeline(data_path, epochs=30, batch_size=16):
    # 1. Load the aligned data
    print(f"Loading aligned data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    X = data['X']        # (Trials, Channels, Time)
    y = data['y']        # String labels (e.g., 'left_hand', 'right_hand')
    meta = data['meta']  # Subject metadata
    
    # 2. Encode string labels to integers (0 and 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes detected: {le.classes_}")
    
    subjects = meta['subject'].unique()
    print(f"Found {len(subjects)} subjects for LOSO Cross-Validation.")
    
    loso_accuracies = []
    
    # 3. Leave-One-Subject-Out Loop
    for test_sub in subjects:
        print(f"\n{'-'*50}")
        print(f"Testing on Unseen Subject {test_sub} (Zero-Calibration)")
        print(f"{'-'*50}")
        
        # Split data: Test is the current subject, Train is everyone else
        test_idx = meta['subject'] == test_sub
        train_idx = ~test_idx
        
        X_train, y_train = X[train_idx], y_encoded[train_idx]
        X_test, y_test = X[test_idx], y_encoded[test_idx]
        
        # Convert to PyTorch Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 4. Initialize Model, Loss, and Optimizer for this fold
        model = EEGNet(n_classes=len(le.classes_))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 5. Training Loop
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")
                
        # 6. Evaluation (Zero-Calibration inference)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        acc = 100 * correct / total
        loso_accuracies.append(acc)
        print(f"--> Subject {test_sub} Zero-Calibration Accuracy: {acc:.2f}%")
        
    # 7. Final Results
    print(f"\n{'='*50}")
    print(f"FINAL LOSO ACCURACY: {np.mean(loso_accuracies):.2f}% ± {np.std(loso_accuracies):.2f}%")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Point it to our mathematically aligned dataset
    aligned_data_path = 'dataset/bci/processed/physionet_mi_aligned.pkl'
    train_loso_pipeline(aligned_data_path)