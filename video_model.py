import subprocess
import os
import pandas as pd

def predict_video_folder(test_dir: str) -> dict:
    """
    Runs the DeepFake prediction script on a folder of test videos and returns predictions.

    Args:
        test_dir (str): Path to folder containing .mp4 test videos.

    Returns:
        dict: {filename: ("Real"/"Fake", probability)} or {"error": "..."}
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    script_path = os.path.join(base_dir, "models", "dfdc_deepfake_challenge", "predict_folder.py")
    weights_path = os.path.join(base_dir, "models", "dfdc_deepfake_challenge", "weights", "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36.pth")
    output_csv = os.path.join(base_dir, "models", "dfdc_deepfake_challenge", "submission.csv")

    if not os.path.exists(script_path):
        return {"error": f"Prediction script not found at {script_path}"}
    
    try:
        subprocess.run(
            [
                "python", script_path,
                "--test-dir", test_dir,
                "--output", output_csv,
                "--models", weights_path
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        return {"error": f"Prediction failed: {e}"}

    if not os.path.exists(output_csv):
        return {"error": "Prediction CSV not found"}

    df = pd.read_csv(output_csv)
    results = {}
    for _, row in df.iterrows():
        label = float(row['label']) if 'label' in row else float(row['prediction'])
        predicted_class = "Fake" if label >= 0.5 else "Real"
        filename = row.get('filename', 'unknown.mp4')
        results[filename] = (predicted_class, round(label, 4))

    return results
