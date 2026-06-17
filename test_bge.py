import time
from api.model_metadata import BIENCODER_BAKEOFF_SPECS, write_model_spec
from api.rerank import get_embeddings
import json

print("Loading dataset...")
with open("tests/snapshots/baseline_full.json") as f:
    data = json.load(f)

texts = []
for c in data["candidates"]:
    texts.append(c["title"] + " " + c.get("text_content", ""))
for c in data["train_stories"]:
    texts.append(c["title"] + " " + c.get("text_content", ""))
for c in data.get("test_stories", []):
    texts.append(c["title"] + " " + c.get("text_content", ""))

texts = list(set(texts)) # unique texts
print(f"Loaded {len(texts)} unique texts.")

spec = BIENCODER_BAKEOFF_SPECS["bge_base_official"]
write_model_spec(".cache/bge_test", spec)

print("Starting embedding with BGE...")
start = time.time()
# monkey patch the cache dir and model loader
import api.rerank
api.rerank.CACHE_DIR = __import__('pathlib').Path(".cache/bge_test")
api.rerank.EMBEDDING_MODEL_DIR = api.rerank.CACHE_DIR
api.rerank._MODEL_CACHE = {}

def progress(curr, total):
    if curr % 100 == 0 or curr == total:
        print(f"  Embedded {curr}/{total}...")

get_embeddings(texts, progress_callback=progress)
end = time.time()
print(f"Done in {end - start:.2f} seconds.")
