# Zero-Calibration Brain-Computer Interfaces via Riemannian Domain Adaptation

**Authors:** Immanuvel Prathap Sagayaraju, Filip Nikolic, Harsh Nayak  
**Program:** MSc in Artificial Intelligence in Medicine (AIM), University of Bern  

---

## 🧠 The Problem: The Calibration Bottleneck (Domain Shift)
The translation of Brain-Computer Interfaces (BCIs) from laboratory settings to real-world clinical applications is severely hindered by the **calibration bottleneck**. 

Electroencephalogram (EEG) signals are highly sensitive to individual physiological differences—such as skull thickness, cortical folding, and exact electrode placement. Because of this, a deep learning model trained on "Subject A" will exhibit catastrophic performance degradation when applied to "Subject B." This phenomenon, known as **domain shift**, currently necessitates 30 to 60 minutes of tedious, subject-specific recalibration before a BCI can be used by a new patient.

## 💡 Our Approach: Zero-Calibration via Riemannian Geometry
This project proposes a zero-calibration framework that combines **Riemannian geometry** with a **depthwise separable convolutional neural network (EEGNet)**. By mathematically aligning the data before it touches the neural network, we effectively erase the physiological bias of individual users.

### Architecture Pipeline
![Architecture Diagram](DL%20Project%20Unibe%20Old.jpg)
<!--- <img src="DL Project Unibe Old.jpg" alt="Architecture Diagram" width="100%"> ---> 
*(Note: Ensure the architecture diagram image is named correctly in the repository).*

Our pipeline decouples the calibration problem into two distinct phases:

1. **Data Acquisition & Epoching (MOABB & MNE-Python):** Continuous EEG signals from the PhysioNet Motor Imagery dataset are bandpass-filtered (8–30 Hz) to isolate the $\mu$ and $\beta$ motor rhythms, and sliced into discrete trials $X \in \mathbb{R}^{C \times T}$.

2. **Riemannian Domain Adaptation (Euclidean Alignment):**
   EEG spatial covariance matrices reside on a curved, non-Euclidean Symmetric Positive-Definite (SPD) manifold. We compute the trial covariance $R_i = X_i X_i^T$ and the subject's mean resting-state covariance $\bar{R}$. We then apply Euclidean Alignment:
   $$\tilde{X}_i = \bar{R}^{-1/2} X_i$$
   This mathematically projects inter-subject spatial covariances to a shared Identity matrix ($I$), standardizing the spatial variance across all subjects.

3. **Subject-Invariant Feature Extraction (EEGNet):**
   The aligned data $\tilde{X}_i$ is fed into EEGNet. Using temporal, depthwise spatial, and separable convolutions, the network learns optimal features without overfitting to the noise of a specific subject's raw spatial covariance.

4. **Rigorous Validation (LOSO):**
   The entire architecture is evaluated using a **Leave-One-Subject-Out (LOSO)** cross-validation loop. We train on $N-1$ subjects and test on $1$ completely unseen subject, proving true zero-shot inference capabilities.

---

## 📂 Repository Structure

```text
Zero-Calibration-BCI/
│
├── dataset/                    # Ignored in Git (Data storage)
│   └── bci/
│       ├── raw/                # Raw PhysioNet .pkl files
│       └── processed/          # Aligned data (Identity Matrix centered)
│
├── save/                       # Ignored in Git (Model checkpoints)
├── result/                     # Ignored in Git (Evaluation plots/metrics)
│
├── download_dataset.py         # Script 1: Fetch and epoch data via
