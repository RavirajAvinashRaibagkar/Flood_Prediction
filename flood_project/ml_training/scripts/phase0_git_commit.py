"""
phase0_git_commit.py  --  FloodSentinel AI
==========================================
Creates git commits per phase as required by the project rules.
Run after each phase to record a clean commit.

Usage:
  python phase0_git_commit.py --phase 0  (after Phase 0 is complete)
  python phase0_git_commit.py --phase 1  (after Phase 1 baselines)
"""
import sys
import subprocess
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent

PHASE_MESSAGES = {
    0: (
        "Phase 0: Fix task, splits, and normalization\n\n"
        "- Switch from patch classification to pixel-wise segmentation\n"
        "- Download and use official Sen1Floods11 GCS split CSVs\n"
        "- Assert zero chip-ID overlap across train/val/test/bolivia\n"
        "- Compute train-only normalization stats (VV dB, VH dB, VV-VH dB)\n"
        "- Flood pixel prevalence per split (ignoring -1/nodata)\n"
        "- pHash deduplicate check across splits\n"
        "- Verified dataset license from local LICENSE.txt (CC BY 4.0)\n"
        "- Documented BUGs 9-13 in BUGS_FIXED.md\n"
        "- Fail loudly: no silent except/pass in any new script"
    ),
    1: (
        "Phase 1: Classical baselines on segmentation task\n\n"
        "- Otsu threshold on VV and VH (per-chip adaptive)\n"
        "- Fixed threshold VV=-16dB, VH=-22dB\n"
        "- Majority-class baseline\n"
        "- All metrics ignore nodata (label=-1)\n"
        "- Bootstrap 95% CI resampled by chip\n"
        "- Report on val/test/bolivia separately"
    ),
    3: (
        "Phase 3: U-Net segmentation training\n\n"
        "- U-Net with pretrained encoder via segmentation_models_pytorch\n"
        "- BCE + Dice loss (ignore_index=-1)\n"
        "- AdamW, warmup + cosine, layer-wise LR decay\n"
        "- AMP, EMA weights, early stopping on VAL IoU only\n"
        "- Threshold calibrated on VAL, frozen for test/bolivia\n"
        "- 5 seeds [42, 7, 99, 123, 314], mean +/- 95% CI"
    ),
}


def run_git(args: list) -> tuple:
    result = subprocess.run(
        ["git"] + args,
        cwd=str(BASE.parent.parent.parent),   # project root
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=list(PHASE_MESSAGES.keys()))
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()

    msg = PHASE_MESSAGES[opts.phase]
    print(f"Git commit message:\n{'-'*50}\n{msg}\n{'-'*50}")

    if opts.dry_run:
        print("[DRY RUN] No git operations performed.")
        return

    # Check git status
    rc, stdout, stderr = run_git(["status", "--short"])
    if rc != 0:
        raise RuntimeError(f"git status failed: {stderr}")
    print(f"\nGit status:\n{stdout}")

    if not stdout.strip():
        print("[SKIP] No changes to commit.")
        return

    # Stage all changes
    rc, stdout, stderr = run_git(["add", "-A"])
    if rc != 0:
        raise RuntimeError(f"git add failed: {stderr}")

    # Commit
    rc, stdout, stderr = run_git(["commit", "-m", msg])
    if rc != 0:
        raise RuntimeError(f"git commit failed:\n{stderr}")
    print(f"\n[OK] Committed Phase {opts.phase}:\n{stdout}")

    # Show log
    rc, out, _ = run_git(["log", "--oneline", "-5"])
    print(f"\nRecent commits:\n{out}")


if __name__ == "__main__":
    main()
