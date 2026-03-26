from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import CrossSessionEvaluation
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings

warnings.filterwarnings("ignore")

def run_baseline_benchmark():
    print("🚀 INITIALIZING MOABB REFEREE ENVIRONMENT...")
    
    dataset = BNCI2014_001()
    dataset.subject_list = [1] 
    paradigm = MotorImagery(n_classes=4, fmin=8, fmax=30)

    pipeline = make_pipeline(
        CSP(n_components=8, log=True, cov_est='epoch'),
        LDA()
    )
    pipelines = {"CSP + LDA": pipeline}

    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=dataset,
        overwrite=True,
        hdf5_path=None, 
        n_jobs=1 
    )

    print("⚖️ RUNNING CROSS-SESSION EVALUATION (Train: Session T -> Test: Session E)...")
    results = evaluation.process(pipelines)
    
    print("\n" + "="*50)
    print("✅ OFFICIAL MOABB BENCHMARK RESULTS (SUBJECT 1)")
    print("="*50)
    # FIXED: Iterate through the entire results dataframe directly
    for index, row in results.iterrows():
        print(f"Pipeline: {row['pipeline']}")
        print(f"Target Test Session: {row['session']}")
        print(f"Strict Accuracy: {row['score'] * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_baseline_benchmark()