# Model Provenance

Current production embedding artifact:
- Active path: `onnx_model/`
- Preserved copy: `archive/models/onnx_model_prod_2026-01-31_b244bb8b/`
- `model.onnx` SHA256: `b244bb8b65de59a101e807843c12033d6302247126ad2d0efb62bab5e09e26ab`
- File timestamp: `2026-01-31 20:03:36 UTC`

What is known:
- The active ONNX artifact was created on January 31, 2026.
- It postdates the current `tuned_model/` checkpoint timestamp (`2026-01-27`) and predates the later `v10-tuned` metadata change committed on February 1, 2026.
- The repo includes a RunPod workflow in `finetune_runpod.py` that exports `onnx_model/` remotely, downloads `onnx_model.tar.gz`, and extracts it locally. The production artifact layout and timestamps are consistent with that flow.

What is not confirmed:
- The exact checkpoint label for the active production ONNX artifact.
- Whether the artifact was installed via `finetune_runpod.py` directly or by an equivalent manual export/import flow.

Practical conclusion:
- Treat the current production artifact as a distinct versioned model, not as the same thing as the current `tuned_model/` checkpoint and not as the stock baseline export.
