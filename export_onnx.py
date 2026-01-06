from optimum.exporters.onnx import main_export
from pathlib import Path

MODEL_ID = "BAAI/bge-small-en-v1.5"
OUTPUT_DIR = Path("bge_model")


def export():
    if (OUTPUT_DIR / "model.onnx").exists():
        print(f"Model already exported to {OUTPUT_DIR}. Skipping export.")
        return

    print(f"Exporting {MODEL_ID} to ONNX...")
    main_export(
        model_name_or_path=MODEL_ID,
        output=OUTPUT_DIR,
        task="feature-extraction",
        opset=None,
        device="cpu",
        trust_remote_code=True,
    )
    print("Export complete.")


if __name__ == "__main__":
    export()
