#!/usr/bin/env -S uv run
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.model_metadata import BIENCODER_BAKEOFF_SPECS, write_model_spec  # noqa: E402

MODEL_EXPORT_EXTRA_HINT = (
    "export_biencoder.py requires the 'model-export' extra. "
    "Run: uv sync --extra model-export"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a named bi-encoder preset to an isolated ONNX directory."
    )
    parser.add_argument(
        "preset",
        choices=sorted(BIENCODER_BAKEOFF_SPECS),
        help="Named bakeoff preset to export",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: archive/models/<preset>)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = BIENCODER_BAKEOFF_SPECS[args.preset]
    output_dir = args.output_dir or (Path("archive/models") / args.preset)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        spec.model_id,
        "--task",
        "feature-extraction",
        "--optimize",
        "O3",
    ]
    if spec.trust_remote_code:
        command.append("--trust-remote-code")
    command.append(str(output_dir))

    try:
        subprocess.check_call(command)
    except FileNotFoundError as exc:
        raise SystemExit(MODEL_EXPORT_EXTRA_HINT) from exc

    write_model_spec(output_dir, spec)
    print(f"Exported {args.preset} to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
