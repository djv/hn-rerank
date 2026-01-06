from optimum.exporters.onnx import main_export
from pathlib import Path

MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
OUTPUT_DIR = Path("onnx_model")


def export():
    if OUTPUT_DIR.exists():
        print(f"Directory {OUTPUT_DIR} already exists. Skipping export.")
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
