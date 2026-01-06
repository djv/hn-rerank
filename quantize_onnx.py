import sys
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path


def quantize(input_path: str, output_path: str):
    input_model = Path(input_path)
    output_model = Path(output_path)

    if not input_model.exists():
        print(f"Input model {input_path} not found.")
        return

    print(f"Quantizing {input_model} to Int8...")
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QUInt8,
    )
    print(f"Quantization complete. Saved to {output_model}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default fallback
        quantize("onnx_model/model.onnx", "onnx_model/model_quantized.onnx")
    else:
        quantize(sys.argv[1], sys.argv[2])
