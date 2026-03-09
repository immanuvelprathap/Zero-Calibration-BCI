import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import os

def run_evaluation(data_path):
    # 1. Load Data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # --- BINARY FILTER (Optional but recommended for clearer clusters) ---
    # To see all 5 classes, comment out the next 4 lines
    mask = np.isin(data['y'], ['left_hand', 'right_hand'])
    X = data['X'][mask]
    y = data['y'][mask]
    subjects = data['meta']['subject'][mask]
    
    # 2. Generate t-SNE Visualization
    print(f"Generating t-SNE for {X.shape[0]} trials...")
    X_flat = X.reshape(X.shape[0], -1)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_flat)
    
    plt.figure(figsize=(12, 7))
    # Color by Subject to show domain alignment, Shape by Class (Left/Right)
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=subjects, style=y, palette='bright', s=100)
    
    plt.title("Riemannian Manifold: t-SNE Distribution\n(Subject overlap indicates successful Domain Adaptation)")
    
    # FIX: Changed loc='2' to loc='upper left'
    plt.legend(title="Subjects / Tasks", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('result/tsne_manifold.png')
    print("Success! Plot saved to: result/tsne_manifold.png")

if __name__ == "__main__":
    if not os.path.exists('result'):
        os.makedirs('result')
    run_evaluation('dataset/bci/processed/physionet_mi_aligned.pkl')