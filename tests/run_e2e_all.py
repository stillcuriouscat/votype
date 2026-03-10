#!/usr/bin/env python3
"""Run all E2E test layers in sequence.

Auto-starts daemon and Kitty if needed.
Runs: L2 Virtual → L2 Real SSH → L2 Real Vertex → L1 Real Pipeline.

Usage:
    python tests/run_e2e_all.py                      # All layers
    python tests/run_e2e_all.py --skip-l1             # L2 only (no audio)
    python tests/run_e2e_all.py --post-processor gemini-fix  # L1 with gemini-fix
    python tests/run_e2e_all.py --verbose
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

E2E_FEATURES = Path(__file__).parent / "e2e_features.json"
VENV_ACTIVATE = Path.home() / ".local/share/voice-input/venv/bin/activate"


def run_test(name, script, args=None, timeout=180):
    """Run a test script and return (success, output)."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    cmd = f"source {VENV_ACTIVATE} && python {script}"
    if args:
        cmd += " " + " ".join(args)

    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=False, text=True, timeout=timeout,
    )
    return result.returncode == 0


def print_feature_summary():
    """Print e2e_features.json summary."""
    if not E2E_FEATURES.exists():
        return
    try:
        data = json.loads(E2E_FEATURES.read_text())
    except (json.JSONDecodeError, OSError):
        return

    print(f"\n{'='*60}")
    print("  E2E Feature Checklist Summary")
    print(f"{'='*60}")

    by_layer = {}
    for f in data["features"]:
        layer = f["layer"]
        by_layer.setdefault(layer, []).append(f)

    total_pass = 0
    total_fail = 0
    for layer in sorted(by_layer.keys()):
        features = by_layer[layer]
        passed = sum(1 for f in features if f["passes"])
        failed = len(features) - passed
        total_pass += passed
        total_fail += failed
        status = "\033[32mALL PASS\033[0m" if failed == 0 else f"\033[31m{failed} FAIL\033[0m"
        print(f"  {layer}: {passed}/{len(features)} {status}")
        for f in features:
            icon = "\033[32m✓\033[0m" if f["passes"] else "\033[31m✗\033[0m"
            err = f" ({f['last_error']})" if f.get("last_error") else ""
            print(f"    {icon} {f['id']}{err}")

    print(f"\n  Total: {total_pass}/{total_pass + total_fail}")
    print(f"  Last run: {data.get('last_run', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all E2E test layers",
    )
    parser.add_argument("--skip-l1", action="store_true",
                       help="Skip L1 Real tests (no audio needed)")
    parser.add_argument("--skip-l2-real", action="store_true",
                       help="Skip L2 Real tests (no SSH needed)")
    parser.add_argument("--post-processor", default="gemini-fix",
                       choices=["haiku-fix", "gemini-fix"],
                       help="Post-processor for L1 test (default: gemini-fix)")
    parser.add_argument("--duration", type=int, default=8,
                       help="Recording duration for L1 test (default: 8)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    test_dir = Path(__file__).parent
    v_flag = ["--verbose"] if args.verbose else []
    results = {}
    start = time.time()

    # Layer 1: L2 Virtual (no external deps)
    results["L2 Virtual"] = run_test(
        "L2 Virtual E2E (vocab, guards, presets)",
        test_dir / "test_e2e_virtual_ssh.py",
        args=v_flag,
    )

    # Layer 2: L2 Real SSH chain (needs SSH)
    if not args.skip_l2_real:
        results["L2 Real SSH"] = run_test(
            "L2 Real SSH Chain (Haiku)",
            test_dir / "test_e2e_ssh_chain.py",
            args=v_flag,
            timeout=120,
        )

        results["L2 Real Vertex"] = run_test(
            "L2 Real Vertex AI Chain (Gemini)",
            test_dir / "test_e2e_vertex_chain.py",
            args=v_flag,
            timeout=120,
        )

    # Layer 3: L1 Real Pipeline (needs daemon + Kitty + audio)
    if not args.skip_l1:
        results["L1 Real Pipeline"] = run_test(
            f"L1 Real Pipeline ({args.post_processor})",
            test_dir / "test_e2e_ssh_haiku.py",
            args=v_flag + [
                "--post-processor", args.post_processor,
                "--duration", str(args.duration),
            ],
            timeout=300,
        )

    elapsed = time.time() - start

    # Summary
    print_feature_summary()

    print(f"\n{'='*60}")
    print("  Test Suite Results")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        icon = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
        print(f"  {icon}  {name}")
        if not passed:
            all_pass = False
    print(f"\n  Elapsed: {elapsed:.1f}s")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
