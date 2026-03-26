import mne
from mne.preprocessing import ICA
from moabb.datasets import BNCI2014_001
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

def clean_and_inspect_session(session_name, runs_dict):
    print(f"\n🧠 PROCESSING {session_name.upper()}...")
    
    # 1. Extract and Concatenate only the runs from THIS specific session
    raw_list = [raw.copy() for run_name, raw in runs_dict.items()]
    raw_session = mne.concatenate_raws(raw_list)
    raw_session.load_data()
    
    # 2. Bandpass filter for stable ICA
    print("   -> Applying 1-40 Hz Bandpass filter...")
    raw_session.filter(l_freq=1.0, h_freq=40.0, picks='eeg')

    # 3. Fit ICA (11 Components to prevent overfitting)
    print("   -> Fitting ICA (11 Components)...")
    ica = ICA(n_components=11, random_state=42, max_iter='auto')
    ica.fit(raw_session, picks='eeg')

    # 4. Automatic Artifact Detection
    eog_ch_names = [ch for ch in raw_session.ch_names if 'EOG' in ch.upper()]
    if eog_ch_names:
        print(f"   -> Automating blink detection using {eog_ch_names}...")
        eog_indices, eog_scores = ica.find_bads_eog(raw_session, ch_name=eog_ch_names)
        ica.exclude = eog_indices
        print(f"   -> Flagged Bad Components (Blinks/Artifacts): {eog_indices}")
        
        # --- AUDIT PLOT 1: EOG SCORES ---
        print("   -> Generating Audit 1: EOG Correlation Scores...")
        fig_scores = ica.plot_scores(eog_scores, title=f'EOG Scores - {session_name}', show=False)
        fig_scores.savefig(f"audit_sub1_{session_name}_eog_scores.png", bbox_inches='tight')
        plt.close(fig_scores)
    else:
        print("   -> No explicit EOG channels found.")

    # --- AUDIT PLOT 2: TOPOMAPS ---
    print("   -> Generating Audit 2: Component Topomaps...")
    fig_components = ica.plot_components(title=f'ICA Components - {session_name}', show=False)
    # Handle MNE's list return type for components
    if isinstance(fig_components, list):
        fig_components[0].savefig(f"audit_sub1_{session_name}_topomaps.png", bbox_inches='tight')
        for f in fig_components: plt.close(f)
    else:
        fig_components.savefig(f"audit_sub1_{session_name}_topomaps.png", bbox_inches='tight')
        plt.close(fig_components)

    # --- AUDIT PLOT 3: SIGNAL OVERLAY (BEFORE VS AFTER) ---
    print("   -> Generating Audit 3: Before & After Signal Overlay...")
    # This plots the raw signal vs the cleaned signal for visual confirmation
    fig_overlay = ica.plot_overlay(raw_session, exclude=ica.exclude, picks='eeg', show=False)
    fig_overlay.savefig(f"audit_sub1_{session_name}_signal_overlay.png", bbox_inches='tight')
    plt.close(fig_overlay)

    # 5. Apply the ICA to clean the data
    raw_cleaned = ica.apply(raw_session.copy())
    print(f"✅ {session_name.upper()} successfully cleaned and Audited.")
    
    return raw_cleaned

if __name__ == "__main__":
    print("🚀 DOWNLOADING BCI IV 2A (SUBJECT 1)...")
    dataset = BNCI2014_001()
    subject_data = dataset.get_data(subjects=[1])[1] 

    cleaned_sessions = {}

    # Loop strictly by Session
    for session_name, runs in subject_data.items():
        cleaned_raw = clean_and_inspect_session(session_name, runs)
        cleaned_sessions[session_name] = cleaned_raw

    print("\n🎉 ICA PIPELINE COMPLETE. AUDIT FILES SAVED TO DIRECTORY.")
    print("Check your folder for the 'audit_sub1_...' PNG files to show Alex.")