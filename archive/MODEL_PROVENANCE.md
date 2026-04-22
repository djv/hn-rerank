# Model Provenance

Current production embedding artifact:
- Active path: `onnx_model/`
- Active model: `intfloat/e5-base-v2`
- Active metadata: `onnx_model/hn_embedding_model.json`
- Active `model.onnx` SHA256: `400309c22deacd385419da5557e012ad341394a32fe949f23df6ddb9110619d3`
- Promotion benchmark: `runs/benchmarks/biencoder_bakeoff_20260422.json`
- Preserved previous production copy: `archive/models/onnx_model_prod_2026-01-31_b244bb8b/`
- Previous production SHA256: `b244bb8b65de59a101e807843c12033d6302247126ad2d0efb62bab5e09e26ab`

What is known:
- Production was switched to the E5 artifact exported into `archive/models/e5_base_v2/` and then copied into `onnx_model/` on 2026-04-22 UTC.
- Both ranking and clustering now load the same live `onnx_model/` artifact.
- The previous production artifact remains preserved with explicit runtime metadata for rollback.

Rollback:
- Replace `onnx_model/` with `archive/models/onnx_model_prod_2026-01-31_b244bb8b/`.
- Restore `EMBEDDING_MODEL_VERSION` and `CLUSTER_EMBEDDING_MODEL_VERSION` in `api/constants.py` to `prod-2026-01-31` if a same-day rollback needs cache separation.
