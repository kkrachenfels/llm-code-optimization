#!/usr/bin/env python3
"""Download and prepare PolybenchC benchmarks."""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_polybench(output_dir: Path):
    """
    Download PolybenchC from GitHub.

    Args:
        output_dir: Directory to clone the repository
    """
    repo_url = "https://github.com/MatthiasJReisinger/PolybenchC.git"

    if output_dir.exists():
        logger.info(f"PolybenchC already exists at {output_dir}")
        return

    logger.info(f"Cloning PolybenchC from {repo_url}...")
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(output_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully cloned PolybenchC to {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e.stderr}")
        sys.exit(1)


def verify_structure(polybench_dir: Path):
    """Verify the PolybenchC directory structure."""
    expected_categories = [
        'linear-algebra',
        'datamining',
        'stencils',
        'medley'
    ]

    logger.info("Verifying directory structure...")
    found_categories = []

    for category in expected_categories:
        category_path = polybench_dir / category
        if category_path.exists():
            found_categories.append(category)
            # Count benchmarks in this category
            benchmarks = [d for d in category_path.rglob('*.c')]
            logger.info(f"  {category}: {len(benchmarks)} benchmarks")

    if not found_categories:
        logger.error("No benchmark categories found!")
        sys.exit(1)

    logger.info(f"Found {len(found_categories)} categories")


def main():
    """Main function."""
    # Default location: datasets/polybench
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "datasets" / "polybench"

    logger.info("=" * 80)
    logger.info("PolybenchC Download and Setup")
    logger.info("=" * 80)

    # Create datasets directory
    output_dir.parent.mkdir(exist_ok=True)

    # Download
    download_polybench(output_dir)

    # Verify
    verify_structure(output_dir)

    logger.info("=" * 80)
    logger.info("Setup complete!")
    logger.info(f"PolybenchC location: {output_dir}")
    logger.info("")
    logger.info("To use with training:")
    logger.info(f"  python train_simple.py --dataset polybench --dataset-path {output_dir} ...")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
