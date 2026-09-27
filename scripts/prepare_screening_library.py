"""Create a deterministic 30k ChEMBL screening-library snapshot."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.library import ScreeningLibrarySpec, build_screening_library
from src.train import load_config, project_root_from_config_path, resolve_path

LOGGER = logging.getLogger("prepare_screening_library")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    root = project_root_from_config_path(args.config)
    library_cfg = config["data"]["screening_library"]
    generator_cfg = library_cfg["generator"]

    output_path = resolve_path(root, library_cfg["input_path"])
    manifest_path = resolve_path(root, library_cfg["manifest"])
    spec = ScreeningLibrarySpec(
        expected_release=str(generator_cfg["expected_release"]),
        target_size=int(generator_cfg["target_size"]),
        candidate_pool_size=int(generator_cfg["candidate_pool_size"]),
        page_size=int(generator_cfg["page_size"]),
        seed=int(generator_cfg["seed"]),
        refuse_release_mismatch=bool(generator_cfg.get("refuse_release_mismatch", True)),
        timeout_seconds=float(generator_cfg.get("timeout_seconds", 60.0)),
        max_retries=int(generator_cfg.get("max_retries", 5)),
    )

    if output_path.exists() and manifest_path.exists():
        LOGGER.info("Screening-library snapshot already exists; leaving it unchanged: %s", output_path)
        LOGGER.info("Delete both snapshot and manifest only if you intentionally want to regenerate them.")
        return

    manifest = build_screening_library(output_path, manifest_path, spec=spec)
    LOGGER.info("Wrote %d screening molecules to %s", manifest["selected_count"], output_path)
    LOGGER.info("Wrote screening-library manifest to %s", manifest_path)
    LOGGER.info("Snapshot SHA-256: %s", manifest["snapshot_sha256"])


if __name__ == "__main__":
    main()
