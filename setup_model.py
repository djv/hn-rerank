from pathlib import Path
import subprocess

MODEL_EXPORT_EXTRA_HINT = (
    "setup_model.py requires the 'model-export' extra. "
    "Run: uv sync --extra model-export"
)


def setup():
    model_dir = Path("onnx_model")
    if (model_dir / "model.onnx").exists():
        print("Model already exists.")
        return

    print("Setting up model (requires internet and ~450MB space)...")
    model_id = "BAAI/bge-base-en-v1.5"

    print(f"Exporting {model_id} to ONNX...")
    try:
        subprocess.check_call(
            [
                "optimum-cli",
                "export",
                "onnx",
                "--model",
                model_id,
                "--task",
                "feature-extraction",
                "--optimize",
                "O3",
                "--trust-remote-code",
                str(model_dir),
            ]
        )
    except FileNotFoundError as exc:
        raise SystemExit(MODEL_EXPORT_EXTRA_HINT) from exc

    print("Setup complete.")


if __name__ == "__main__":
    setup()
