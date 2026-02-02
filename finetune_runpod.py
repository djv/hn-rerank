#!/usr/bin/env -S uv run
"""Automated fine-tuning on RunPod GPU cloud.

Clones repo from GitHub - no manual file uploads needed.

Usage:
    export RUNPOD_API_KEY="your_key"
    uv run finetune_runpod.py

Get API key from: https://www.runpod.io/console/user/settings
"""

import os
import sys
import time
import subprocess
from pathlib import Path

import runpod

# Config
GPU_TYPE = "NVIDIA GeForce RTX 3090"
CONTAINER = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
VOLUME_SIZE = 10
POD_NAME = "hn-finetune"
GITHUB_REPO = "https://github.com/djv/hn-rerank.git"

# Training script - clones from GitHub
TRAIN_SCRIPT = f"""
set -e
cd /workspace
git clone {GITHUB_REPO} repo
cd repo

pip install -q sentence-transformers optimum onnx onnxruntime

python tune_embeddings.py --epochs 3 --batch-size 32

python export_tuned.py

tar -czf /workspace/onnx_model.tar.gz -C /workspace/repo onnx_model/
echo "TRAINING_COMPLETE"
"""


def main():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: Set RUNPOD_API_KEY environment variable")
        print("Get key from: https://www.runpod.io/console/user/settings")
        sys.exit(1)

    runpod.api_key = api_key
    pod_id = None

    try:
        print(f"Creating pod with {GPU_TYPE}...")
        pod = runpod.create_pod(
            name=POD_NAME,
            image_name=CONTAINER,
            gpu_type_id=GPU_TYPE,
            volume_in_gb=VOLUME_SIZE,
            ports="22/tcp",
        )
        pod_id = pod["id"]
        print(f"Pod created: {pod_id}")

        # Wait for pod
        print("Waiting for pod to start...")
        status = None
        for _ in range(60):
            status = runpod.get_pod(pod_id)
            if status.get("desiredStatus") == "RUNNING" and status.get("runtime"):
                runtime = status.get("runtime", {})
                if runtime.get("uptimeInSeconds", 0) > 10:
                    break
            time.sleep(5)
            print(".", end="", flush=True)
        else:
            print("\nTimeout waiting for pod")
            sys.exit(1)

        print("\nPod running!")

        # Get SSH info
        runtime = status["runtime"]
        ssh_host = runtime.get("publicIp")
        ssh_port = 22
        for port_info in runtime.get("ports", []):
            if port_info.get("privatePort") == 22:
                ssh_port = port_info.get("publicPort", 22)
                if port_info.get("ip"):
                    ssh_host = port_info["ip"]
                break

        if not ssh_host:
            print("Could not get SSH host")
            sys.exit(1)

        print(f"SSH: root@{ssh_host}:{ssh_port}")

        # Run training via SSH
        print("Starting training (~3 min on RTX 4090)...")
        result = subprocess.run(
            [
                "ssh", "-p", str(ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=30",
                f"root@{ssh_host}",
                TRAIN_SCRIPT
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

        if "TRAINING_COMPLETE" not in result.stdout:
            print(f"Training may have failed: {result.stderr[-500:]}")
            sys.exit(1)

        # Download result
        print("Downloading model...")
        subprocess.run(
            [
                "scp", "-P", str(ssh_port),
                "-o", "StrictHostKeyChecking=no",
                f"root@{ssh_host}:/workspace/onnx_model.tar.gz",
                "."
            ],
            check=True,
            timeout=120,
        )

        # Extract
        if Path("onnx_model").exists():
            import shutil
            shutil.move("onnx_model", "onnx_model_backup_local")

        subprocess.run(["tar", "-xzf", "onnx_model.tar.gz"], check=True)
        Path("onnx_model.tar.gz").unlink()

        print("\n✓ Model saved to onnx_model/")
        print("Update api/constants.py: EMBEDDING_MODEL_VERSION = 'v9-tuned'")

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if pod_id:
            print("Terminating pod...")
            try:
                runpod.terminate_pod(pod_id)
                print("✓ Pod terminated")
            except Exception as e:
                print(f"Warning: Could not terminate pod {pod_id}: {e}")
                print("Check: https://www.runpod.io/console/pods")


if __name__ == "__main__":
    main()
