#!/usr/bin/env python3
"""
run_sdg_pipeline.py — Master orchestrator for the SDG extraction pipeline.

Runs all steps in order:
  1. Extract text from PDFs
  2. Split text into semantic passages
  3. Match SDG keywords → DuckDB
  4. Generate ground truth Excel

Usage:
    python scripts/run_sdg_pipeline.py [--step STEP] [--force] [--workers N]

Steps: extract, split, match, ground_truth, all (default: all)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def run_step(name: str, cmd: list[str], cwd: str = "."):
    """Run a pipeline step and check for errors."""
    print(f"\n{'='*70}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] Step '{name}' failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n[OK] Step '{name}' completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="SDG Extraction Pipeline Orchestrator")
    parser.add_argument("--step", default="all",
                        choices=["extract", "split", "match", "ground_truth", "all"],
                        help="Which step to run (default: all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers for text extraction")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if outputs exist")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Ground truth sample size")
    args = parser.parse_args()

    python = sys.executable
    steps_to_run = []

    if args.step in ("all", "extract"):
        cmd = [python, str(SCRIPTS_DIR / "extract_text.py"), "--workers", str(args.workers)]
        if args.force:
            cmd.append("--force")
        steps_to_run.append(("1. Extract Text from PDFs", cmd))

    if args.step in ("all", "split"):
        cmd = [python, str(SCRIPTS_DIR / "split_passages.py")]
        if args.force:
            cmd.append("--force")
        steps_to_run.append(("2. Split Text into Passages", cmd))

    if args.step in ("all", "match"):
        cmd = [python, str(SCRIPTS_DIR / "sdg_keyword_match.py")]
        if args.force:
            cmd.append("--force")
        steps_to_run.append(("3. SDG Keyword Matching → DuckDB", cmd))

    if args.step in ("all", "ground_truth"):
        cmd = [python, str(SCRIPTS_DIR / "generate_ground_truth.py"),
               "--sample", str(args.sample)]
        steps_to_run.append(("4. Generate Ground Truth Excel", cmd))

    print(f"\n{'#'*70}")
    print(f"  SDG EXTRACTION PIPELINE")
    print(f"  Steps to run: {len(steps_to_run)}")
    print(f"{'#'*70}")

    total_time = 0
    for name, cmd in steps_to_run:
        elapsed = run_step(name, cmd)
        total_time += elapsed

    print(f"\n{'#'*70}")
    print(f"  ALL STEPS COMPLETE — Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
