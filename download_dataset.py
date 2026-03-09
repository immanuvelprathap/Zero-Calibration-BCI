import os
import pickle
from pathlib import Path
import mne
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

# Suppress excessive MNE warnings to keep the terminal clean
mne.set_log_level('WARNING')

def download_and_epoch():
    print("Initializing MOABB Paradigm...")
    # This automatically applies an 8.0 - 30.0 Hz bandpass filter (Mu and Beta rhythms)
    # and slices the continuous EEG into discrete trials.
    paradigm = MotorImagery(n_classes=2, fmin=8.0, fmax=30.0)
    
    print("Loading PhysioNet Motor Imagery Dataset...")
    dataset = PhysionetMI()
    
    # We start with just 3 subjects for fast testing. 
    # To run on the full dataset later, change this to: list(range(1, 110))
    subjects_to_test = [1, 2, 3] 
    print(f"Fetching data for subjects: {subjects_to_test}. This might take a minute...")
    
    # Fetch the data
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects_to_test)
    
    print(f"\nSuccessfully extracted {X.shape[0]} discrete trials.")
    print(f"Data tensor shape: {X.shape} (Trials, Channels, Time)")
    
    # Create the directory structure if it doesn't exist
    save_dir = Path('dataset/bci/raw')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'physionet_mi_raw.pkl'
    
    print(f"Saving data to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump({'X': X, 'y': labels, 'meta': meta}, f)
        
    print("Download and save complete!")

if __name__ == "__main__":
    download_and_epoch()