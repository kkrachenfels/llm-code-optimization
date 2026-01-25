#!/usr/bin/env python3
"""Download and prepare cBench benchmarks for training."""

import subprocess
import sys
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Benchmarks to extract from the ctuning-programs repository
BENCHMARKS = [
    'automotive_bitcount',
    'automotive_basicmath',
    'network_dijkstra',
    'telecomm_CRC32',
    'automotive_qsort1',
    'security_sha',
    'security_blowfish',
    'telecomm_adpcm',
]

REPO_URL = "https://github.com/ctuning/ctuning-programs.git"


def download_cbench(output_dir: Path):
    """
    Download cBench programs from the ctuning-programs repository.

    Args:
        output_dir: Directory to place the benchmark sources
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info(f"cBench directory already exists at {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone the repository to a temporary location
    clone_dir = output_dir.parent / '_ctuning_clone'

    if not clone_dir.exists():
        logger.info(f"Cloning {REPO_URL} (this may take a while)...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Clone complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            sys.exit(1)
    else:
        logger.info(f"Using existing clone at {clone_dir}")

    # Extract relevant benchmark source directories
    programs_dir = clone_dir / 'program'
    if not programs_dir.exists():
        # Try alternate structure
        programs_dir = clone_dir

    extracted = 0
    for bench_name in BENCHMARKS:
        # Search for the benchmark directory
        bench_src = None
        candidates = [
            programs_dir / bench_name / 'src',
            programs_dir / f'cbench-{bench_name}' / 'src',
            clone_dir / bench_name / 'src',
        ]

        for candidate in candidates:
            if candidate.exists():
                bench_src = candidate
                break

        if bench_src is None:
            # Try glob search
            matches = list(clone_dir.rglob(f'*{bench_name}*'))
            for m in matches:
                src_dir = m / 'src' if m.is_dir() else None
                if src_dir and src_dir.exists():
                    bench_src = src_dir
                    break
                elif m.is_dir():
                    # Check if it contains .c files directly
                    if list(m.glob('*.c')):
                        bench_src = m
                        break

        if bench_src is None:
            logger.warning(f"  {bench_name}: NOT FOUND")
            continue

        # Copy source files to output directory
        dest_dir = output_dir / bench_name / 'src'
        dest_dir.mkdir(parents=True, exist_ok=True)

        c_files = list(bench_src.glob('*.c'))
        h_files = list(bench_src.glob('*.h'))

        for f in c_files + h_files:
            shutil.copy2(f, dest_dir / f.name)

        logger.info(f"  {bench_name}: {len(c_files)} .c files, {len(h_files)} .h files")
        extracted += 1

    # Clean up clone
    logger.info("Cleaning up temporary clone...")
    shutil.rmtree(clone_dir, ignore_errors=True)

    return extracted


def verify_compilation(output_dir: Path):
    """Try to compile each benchmark to verify it's usable."""
    logger.info("Verifying compilation...")
    results = {}

    for bench_dir in sorted(output_dir.iterdir()):
        if not bench_dir.is_dir():
            continue

        src_dir = bench_dir / 'src'
        if not src_dir.exists():
            src_dir = bench_dir

        c_files = list(src_dir.glob('*.c'))
        if not c_files:
            results[bench_dir.name] = 'NO_SOURCES'
            continue

        # Try to compile all .c files together
        cmd = ['gcc', '-O2', '-std=c11', '-fsyntax-only'] + \
              [f'-I{src_dir}'] + \
              [str(f) for f in c_files]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                results[bench_dir.name] = 'OK'
            else:
                results[bench_dir.name] = 'COMPILE_ERROR'
                logger.debug(f"  {bench_dir.name} errors: {result.stderr[:200]}")
        except (subprocess.TimeoutExpired, Exception) as e:
            results[bench_dir.name] = f'ERROR: {e}'

    return results


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "datasets" / "cbench"

    logger.info("=" * 80)
    logger.info("cBench Download and Setup")
    logger.info("=" * 80)

    # Download
    extracted = download_cbench(output_dir)

    if extracted == 0 and not any(output_dir.iterdir()):
        logger.error("No benchmarks were extracted. Check repository structure.")
        sys.exit(1)

    # Verify compilation
    results = verify_compilation(output_dir)

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info("-" * 40)
    ok_count = 0
    for name, status in sorted(results.items()):
        logger.info(f"  {name:30s} {status}")
        if status == 'OK':
            ok_count += 1
    logger.info("-" * 40)
    logger.info(f"  {ok_count}/{len(results)} benchmarks compile successfully")
    logger.info("=" * 80)
    logger.info("")
    logger.info("To use with training:")
    logger.info(f"  python train_simple.py --dataset cbench --dataset-path {output_dir} ...")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
