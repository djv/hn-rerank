#!/usr/bin/env python3
"""Generate the HN Rerank dashboard (one-shot)."""

import argparse
import asyncio
import logging
import sys
from pipeline import Config, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HN Rerank dashboard")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    config = Config.load(args.config)
    try:
        asyncio.run(run_pipeline(config))
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
