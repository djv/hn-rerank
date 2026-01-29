from pathlib import Path
import subprocess
import shutil

def export_tuned():
    tuned_dir = Path("tuned_model")
    if not tuned_dir.exists():
        print("Tuned model not found. Run tune_embeddings.py first.")
        return

    onnx_dir = Path("onnx_model")
    if onnx_dir.exists():
        print("Backing up existing model...")
        shutil.move(str(onnx_dir), "onnx_model_backup")

    print(f"Exporting tuned model from {tuned_dir} to ONNX...")
    
    # Use optimum-cli to export the local model
    try:
        subprocess.check_call(
            [
                "optimum-cli",
                "export",
                "onnx",
                "--model",
                str(tuned_dir),
                "--task",
                "feature-extraction",
                "--optimize",
                "O3", 
                str(onnx_dir),
            ]
        )
        print("Export complete! The app will now use the fine-tuned model.")
        
    except subprocess.CalledProcessError as e:
        print(f"Export failed: {e}")
        if Path("onnx_model_backup").exists():
            print("Restoring backup...")
            if onnx_dir.exists():
                shutil.rmtree(onnx_dir)
            shutil.move("onnx_model_backup", str(onnx_dir))

if __name__ == "__main__":
    export_tuned()
