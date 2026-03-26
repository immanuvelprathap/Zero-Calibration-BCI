import mne
from mne.preprocessing import ICA
from moabb.datasets import BNCI2014_001
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def clean_and_inspect_session(session_name, runs_dict):
    print(f"\n🧠 PROCESSING {session_name.upper()}...")
    
    # 1. Extract and Concatenate only the runs from THIS specific session
    raw_list = [raw.copy() for run_name, raw in runs_dict.items()]
    raw_session = mne.concatenate_raws(raw_list)
    raw_session.load_data()
    
    # 2. Critical ICA Pre-processing: 1Hz Highpass filter
    # ICA mathematically fails if there are slow baseline drifts (sweat/movement)
    print("   -> Applying 1-40 Hz Bandpass filter for stable ICA...")
    raw_session.filter(l_freq=1.0, h_freq=40.0, picks='eeg')

    # 3. Initialize ICA (The "Halve the Channels" Rule)
    # BCI IV 2a has 22 EEG channels. We use 11 components to avoid overfitting noise.
    print("   -> Fitting ICA (11 Components)...")
    ica = ICA(n_components=11, random_state=42, max_iter='auto')
    
    # Fit only on the EEG channels (ignoring the 3 EOG channels for the spatial filter)
    ica.fit(raw_session, picks='eeg')

    # 4. Automatic Artifact Detection (Sanity Check)
    # BCI IV 2a actually has 3 EOG channels built-in. We can ask MNE to 
    # automatically find which of our 11 components correlate with eye blinks.
    eog_ch_names = [ch for ch in raw_session.ch_names if 'EOG' in ch.upper()]
    if eog_ch_names:
        print(f"   -> Found EOG channels: {eog_ch_names}. Automating blink detection...")
        eog_indices, eog_scores = ica.find_bads_eog(raw_session, ch_name=eog_ch_names)
        ica.exclude = eog_indices
        print(f"   -> Automatically flagged components as blinks: {eog_indices}")
    else:
        print("   -> No explicit EOG channels found for automated routing.")

    # 5. Visual Inspection (Alex's Manual Step)
    print(f"   -> Launching Topomap visualizer for {session_name}...")
    print("   -> Close the plot window to continue the script.")
    
    # This plots the spatial map of the components. 
    # Eye blinks will look like massive red/blue blobs at the very front of the head.
    fig = ica.plot_components(title=f'ICA Components - {session_name}', show=True)
    
    # Apply the ICA to clean the data
    raw_cleaned = ica.apply(raw_session.copy())
    print(f"✅ {session_name.upper()} successfully cleaned.")
    
    return raw_cleaned

if __name__ == "__main__":
    print("🚀 DOWNLOADING BCI IV 2A (SUBJECT 1)...")
    dataset = BNCI2014_001()
    # MOABB returns a nested dict: data[subject][session][run]
    subject_data = dataset.get_data(subjects=[1])[1] 

    cleaned_sessions = {}

    # Loop strictly by Session to respect spatial non-stationarity
    for session_name, runs in subject_data.items():
        cleaned_raw = clean_and_inspect_session(session_name, runs)
        cleaned_sessions[session_name] = cleaned_raw

    print("\n🎉 ALL SESSIONS CLEANED INDEPENDENTLY.")
    print("Next step: Epoching this clean data and passing it to our 95% SWA Model.")