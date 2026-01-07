from pathlib import Path
import subprocess
import sys

def setup():
    model_dir = Path("onnx_model")
    if (model_dir / "model_quantized.onnx").exists():
        print("Model already exists.")
        return

    print("Setting up model (requires internet and ~500MB space)...")
    model_id = "nomic-ai/nomic-embed-text-v1.5"
    
    # We use optimum to export to ONNX. 
    # Adding it temporarily if not present.
    try:
        import importlib.util
        if importlib.util.find_spec("optimum") is None:
            raise ImportError
    except ImportError:
        print("Installing optimum and onnxruntime...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optimum[onnxruntime]"])

    print(f"Exporting {model_id} to ONNX...")
    subprocess.check_call([
        "optimum-cli", "export", "onnx", 
        "--model", model_id, 
        "--task", "feature-extraction",
        "--optimize", "O3",
        str(model_dir)
    ])
    
    # Quantize for CPU performance
    if (model_dir / "model.onnx").exists() and not (model_dir / "model_quantized.onnx").exists():
        print("Quantizing model...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input=str(model_dir / "model.onnx"),
            model_output=str(model_dir / "model_quantized.onnx"),
            weight_type=QuantType.QUInt8
        )
    print("Setup complete.")

if __name__ == "__main__":
    setup()
