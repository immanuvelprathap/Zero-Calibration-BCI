import os
import pickle
from pathlib import Path
import mne
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

# Suppress excessive MNE warnings
mne.set_log_level('WARNING')

def download_and_epoch():
    # We use 2 classes for the IEEE paper (Binary Classification)
    print("Initializing MOABB Paradigm (8-30 Hz Bandpass)...")
    paradigm = MotorImagery(n_classes=2, fmin=8.0, fmax=30.0)
    
    print("Loading PhysioNet Motor Imagery Dataset...")
    dataset = PhysionetMI()
    
    # SCALING UP: Using 20 subjects to improve generalization
    subjects_to_test = list(range(1, 21)) 
    print(f"Fetching data for {len(subjects_to_test)} subjects. This will take a few minutes...")
    
    # Fetch the data (Trials, Channels, Time)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects_to_test)
    
    print(f"\nSuccessfully extracted {X.shape[0]} discrete trials.")
    print(f"Data tensor shape: {X.shape} (Trials, Channels, Time)")
    
    # Save the raw data
    save_dir = Path('dataset/bci/raw')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'physionet_mi_raw.pkl'
    
    print(f"Saving data to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump({'X': X, 'y': labels, 'meta': meta}, f)
        
    print("Download and save complete!")

if __name__ == "__main__":
    download_and_epoch()