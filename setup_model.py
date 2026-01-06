import os
import sys
from pathlib import Path
import subprocess

def setup():
    model_dir = Path("bge_model")
    quantized_model = model_dir / "model_quantized.onnx"

    if quantized_model.exists():
        print("Model already exists and is quantized.")
        return

    print("Model missing. Starting setup process...")

    # 1. Export
    print("Running export_onnx.py...")
    try:
        subprocess.check_call([sys.executable, "export_onnx.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error exporting model: {e}")
        sys.exit(1)

    # 2. Quantize
    print("Running quantize_onnx.py...")
    try:
        subprocess.check_call([sys.executable, "quantize_onnx.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error quantizing model: {e}")
        sys.exit(1)

    print("Setup complete! You can now run the application.")

if __name__ == "__main__":
    setup()
