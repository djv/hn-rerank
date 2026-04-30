from pathlib import Path
import subprocess

from api.model_metadata import (
    BGE_SMALL_CLS_QUERY_SPEC,
    load_model_spec,
    write_model_spec,
)

MODEL_EXPORT_EXTRA_HINT = (
    "setup_model.py requires the 'model-export' extra. "
    "Run: uv sync --extra model-export"
)


def setup():
    model_dir = Path("onnx_model")
    if (model_dir / "model.onnx").exists():
        if load_model_spec(model_dir) == BGE_SMALL_CLS_QUERY_SPEC:
            print("Model already exists.")
            return
        raise SystemExit(
            "Existing onnx_model is not BGE-small CLS/query. "
            "Move or remove onnx_model, then rerun setup_model.py."
        )
        return

    print("Setting up model (requires internet and ~150MB space)...")
    model_id = BGE_SMALL_CLS_QUERY_SPEC.model_id

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

    write_model_spec(model_dir, BGE_SMALL_CLS_QUERY_SPEC)
    print("Setup complete.")


if __name__ == "__main__":
    setup()
