import os
import pickle
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, filtfilt  # <--- NEW IMPORT
from pathlib import Path

# --- NEW FILTER FUNCTION ---
def bandpass_filter(data, lowcut=8.0, highcut=30.0, fs=160.0, order=4):
    """
    Filters the EEG data to keep only the 8-30 Hz range (Mu and Beta bands).
    fs=160.0 is the standard sampling rate for the PhysioNet MI dataset.
    """
    print(f"\n[Phase 0: DSP] Applying Butterworth Bandpass Filter ({lowcut}-{highcut} Hz)...")
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter along the time axis (the last dimension)
    filtered_data = filtfilt(b, a, data, axis=-1)
    print("✅ Filtering Complete!")
    return filtered_data
# ---------------------------

# def euclidean_alignment(X, meta):
#     """
#     Applies Euclidean Alignment (EA) to center subject covariances at Identity (I).
#     """
#     print("\n[Phase 1] Applying Euclidean Alignment...")
#     aligned_X = np.zeros_like(X)
#     subjects = meta['subject'].unique()
    
#     for sub in subjects:
#         idx = meta['subject'] == sub
#         sub_X = X[idx]
        
#         # 1. Compute trial covariances: R_i = X_i * X_i^T / (T-1)
#         n_time = sub_X.shape[-1]
#         covs = np.einsum('nit,njt->nij', sub_X, sub_X) / (n_time - 1)
        
#         # 2. Compute Subject Mean Covariance
#         mean_cov = np.mean(covs, axis=0)
        
#         # 3. Add Epsilon for Numerical Stability (Prevents Singular Matrix errors)
#         mean_cov += 1e-6 * np.eye(mean_cov.shape[0])
        
#         # 4. Compute Alignment Transform: R_bar^(-1/2)
#         r_inv_half = np.real(fractional_matrix_power(mean_cov, -0.5))
        
#         # 5. Apply transformation: X_tilde = R_bar^(-1/2) * X
#         aligned_X[idx] = np.einsum('ij,njt->nit', r_inv_half, sub_X)
    
#     # 6. Global Z-score scaling
#     print("Applying Global Standardization...")
#     aligned_X = (aligned_X - np.mean(aligned_X)) / (np.std(aligned_X) + 1e-7)

#     print("✅ Space Alignment Complete!")
#     return aligned_X
def euclidean_alignment(X, meta):
    print("\n[Phase 1: V2] Applying Euclidean Alignment + Local Z-Score...")
    aligned_X = np.zeros_like(X)
    subjects = meta['subject'].unique()
    
    for sub in subjects:
        idx = meta['subject'] == sub
        sub_X = X[idx]
        
        n_time = sub_X.shape[-1]
        covs = np.einsum('nit,njt->nij', sub_X, sub_X) / (n_time - 1)
        
        mean_cov = np.mean(covs, axis=0)
        mean_cov += 1e-6 * np.eye(mean_cov.shape[0])
        
        r_inv_half = np.real(fractional_matrix_power(mean_cov, -0.5))
        
        # Apply EA transformation
        sub_aligned = np.einsum('ij,njt->nit', r_inv_half, sub_X)
        
        # V2 FIX: Local Z-Score (Scale only this subject)
        sub_aligned = (sub_aligned - np.mean(sub_aligned)) / (np.std(sub_aligned) + 1e-7)
        
        aligned_X[idx] = sub_aligned
    
    # Notice: NO global Z-score at the end anymore!
    print("✅ Local Alignment and Scaling Complete!")
    return aligned_X

# 
if __name__ == "__main__":
    # 1. Load your raw data (adjust path if needed)
    print("Loading raw PhysioNet data...")
    with open('dataset/bci/processed/physionet_mi_aligned.pkl', 'rb') as f:
        data = pickle.load(f)
        
    X = data['X']
    y = data['y']
    meta = data['meta']
    
    # 2. 🔥 NEW: Apply the Bandpass Filter first! 🔥
    X_filtered = bandpass_filter(X)
    
    # 3. Apply Euclidean Alignment on the CLEANED data
    X_aligned = euclidean_alignment(X_filtered, meta)
    
    # 4. Save the new V3 dataset
    data['X'] = X_aligned
    save_path = 'dataset/bci/processed/physionet_mi_aligned.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"\n🎉 V3 Preprocessing Complete! Saved to {save_path}")