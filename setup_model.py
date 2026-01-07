from pathlib import Path
import subprocess

def setup():
    model_dir = Path("onnx_model")
    if (model_dir / "model.onnx").exists():
        print("Model already exists.")
        return

    print("Setting up model (requires internet and ~450MB space)...")
    model_id = "BAAI/bge-base-en-v1.5"
    
    # We use optimum to export to ONNX. 
    # Check if optimum-cli is available

    print(f"Exporting {model_id} to ONNX...")
    subprocess.check_call([
        "optimum-cli", "export", "onnx", 
        "--model", model_id, 
        "--task", "feature-extraction",
        "--optimize", "O3",
        "--trust-remote-code",
        str(model_dir)
    ])
    
    print("Setup complete.")

if __name__ == "__main__":
    setup()
