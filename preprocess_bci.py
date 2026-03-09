import os
import pickle
import numpy as np
from scipy.linalg import fractional_matrix_power
from pathlib import Path

def euclidean_alignment(X, meta):
    """
    Applies Riemannian Domain Adaptation (Euclidean Alignment).
    X: shape (Trials, Channels, Time) -> (n, c, t)
    meta: DataFrame containing subject IDs corresponding to each trial
    """
    print("\nApplying Riemannian Domain Adaptation (Euclidean Alignment)...")
    aligned_X = np.zeros_like(X)
    subjects = meta['subject'].unique()
    
    for sub in subjects:
        # 1. Isolate the current subject's trials
        idx = meta['subject'] == sub
        sub_X = X[idx]
        
        # 2. Compute trial covariances: R_i = X_i * X_i^T
        # n = trials, i = channels, j = channels, t = time
        # This computes the dot product across the time dimension for every channel pair
        covs = np.einsum('nit,njt->nij', sub_X, sub_X)
        
        # 3. Compute Subject Mean Covariance: R_bar
        mean_cov = np.mean(covs, axis=0)
        
        # 4. Compute Alignment Transform: R_bar^(-1/2)
        # We use np.real() because fractional_matrix_power can return tiny imaginary 
        # artifacts due to numerical floating-point precision limits.
        r_inv_half = np.real(fractional_matrix_power(mean_cov, -0.5))
        
        # 5. Apply transformation: X_tilde = R_bar^(-1/2) * X
        # i = channels, j = channels, n = trials, t = time
        aligned_X[idx] = np.einsum('ij,njt->nit', r_inv_half, sub_X)
        
    print("Alignment complete! All subject spatial covariances are centered at the Identity Matrix (I).")
    return aligned_X

if __name__ == "__main__":
    # Define paths
    input_file = 'dataset/bci/raw/physionet_mi_raw.pkl'
    output_dir = 'dataset/bci/processed'
    
    print(f"Loading raw BCI data from {input_file}...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
        
    X = data['X']        # Shape: (Trials, Channels, Time)
    y = data['y']        # Shape: (Trials,) labels
    meta = data['meta']  # DataFrame containing Subject IDs
    
    print(f"Loaded {X.shape[0]} trials across {len(meta['subject'].unique())} subjects.")
    
    # Execute the Euclidean Alignment
    aligned_X = euclidean_alignment(X, meta)
    
    # Save the mathematically aligned data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, 'physionet_mi_aligned.pkl')
    
    print(f"\nSaving mathematically aligned data to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump({'X': aligned_X, 'y': y, 'meta': meta}, f)
    print("Save successful.")