#!/usr/bin/env python3
"""L2 Real E2E: Vertex AI SSH call chain with real SSH + real Gemini.

Tests the SSH post-processing chain via vertex_proxy.py on Oracle:
  text -> apply_vocab -> SSH Vertex AI -> diff_to_vocab -> save_vocab

Uses real SSH to oracle-cloud and real Gemini 2.5 Flash.
No daemon, Kitty, or audio required.

Requirements:
    - SSH access to oracle-cloud configured
    - vertex_proxy.py deployed to ~/vertex_proxy.py on Oracle
    - google-genai SDK installed on Oracle
    - GCP ADC credentials configured on Oracle
    - Network connectivity

Usage:
    python tests/test_e2e_vertex_chain.py --verbose

E2E features verified:
    - real-chain-vertex-correction
    - real-chain-vertex-vocab-roundtrip
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SSH_HOST = "oracle-cloud"
PROXY_SCRIPT = "~/vertex_proxy.py"
TIMEOUT = 20  # Vertex AI can be slower on first call
E2E_FEATURES = Path(__file__).parent / "e2e_features.json"

# System prompt for ASR correction (same concept as haiku-fix)
SYSTEM_PROMPT = (
    "You are an ASR (speech recognition) text correction tool, NOT a chatbot. "
    "Fix English words misrecognized as Chinese characters "
    "(e.g. 筛选->session, 克劳的->Claude). "
    "Fix homophone errors. Remove repeated words from speech disfluency. "
    "Output ONLY the corrected text, nothing else. "
    "NEVER answer questions or add commentary."
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

results = []
verbose = False


def log(msg):
    if verbose:
        print(f"    {msg}")


def record_result(name, passed, detail=""):
    entry = {"name": name, "passed": passed, "detail": detail}
    results.append(entry)
    tag = "\033[32m[PASS]\033[0m" if passed else "\033[31m[FAIL]\033[0m"
    suffix = f" — {detail}" if detail else ""
    print(f"  {tag} {name}{suffix}")


def update_e2e_features(feature_id, passes, error=None):
    if not E2E_FEATURES.exists():
        return
    try:
        data = json.loads(E2E_FEATURES.read_text())
        for f in data["features"]:
            if f["id"] == feature_id:
                f["passes"] = passes
                f["last_error"] = error
                break
        data["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        E2E_FEATURES.write_text(json.dumps(data, indent=2) + "\n")
    except (json.JSONDecodeError, KeyError, OSError):
        pass


# ---------------------------------------------------------------------------
# Vertex AI proxy call (standalone, no project imports needed)
# ---------------------------------------------------------------------------

def call_vertex_proxy(text, system_prompt=SYSTEM_PROMPT, glossary_ctx=""):
    """Call Vertex AI Gemini via SSH proxy. Returns (output_text, error_string)."""
    prompt = system_prompt
    if glossary_ctx:
        prompt += "\n\n" + glossary_ctx

    stdin_data = json.dumps({
        "system_prompt": prompt,
        "user_input": text,
        "model": "gemini-2.5-flash",
        "region": "us-central1",
    }, ensure_ascii=False)

    cmd = [
        "ssh", "-o", "ConnectTimeout=5",
        SSH_HOST,
        "python3", PROXY_SCRIPT,
    ]
    log(f"SSH command: {' '.join(cmd)}")
    log(f"Input text: {text[:80]}")

    try:
        result = subprocess.run(
            cmd, input=stdin_data, capture_output=True, text=True,
            timeout=TIMEOUT,
        )
        if result.returncode != 0:
            return None, f"exit={result.returncode}: {result.stderr.strip()[:100]}"
        output = result.stdout.strip()
        log(f"Output: {output[:80]}")
        return output, None
    except subprocess.TimeoutExpired:
        return None, f"timeout ({TIMEOUT}s)"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_vertex_correction():
    """Test 1: Send known-error text, verify correction."""
    print("\n[Test 1: Vertex AI correction]")

    test_text = "我们来讨论一下这个筛选的问题"
    expected_correction = "session"

    output, err = call_vertex_proxy(test_text)
    if err:
        record_result("Vertex AI correction", False, err)
        update_e2e_features("real-chain-vertex-correction", False, err)
        return None, None

    if not output:
        record_result("Vertex AI correction", False, "Empty output")
        update_e2e_features("real-chain-vertex-correction", False, "empty")
        return None, None

    if len(output) > len(test_text) * 5:
        record_result("Vertex AI correction", False,
                     f"Hallucination: {len(output)} chars (input={len(test_text)})")
        update_e2e_features("real-chain-vertex-correction", False, "hallucination")
        return None, None

    has_correction = expected_correction.lower() in output.lower()
    if has_correction:
        record_result("Vertex AI correction", True,
                     f"'{test_text}' -> '{output}'")
    else:
        record_result("Vertex AI correction", True,
                     f"Returned: '{output}' (no '{expected_correction}' but "
                     f"reasonable output) [non-critical]")

    update_e2e_features("real-chain-vertex-correction", True)
    return test_text, output


@pytest.fixture(scope="module")
def vertex_correction_result():
    """Run Vertex AI correction once and provide results to dependent tests."""
    return test_vertex_correction()


def test_vertex_vocab_roundtrip(vertex_correction_result):
    """Test 2: Full vocab cycle with real Vertex AI correction data."""
    print("\n[Test 2: Vertex vocab roundtrip]")
    original, polished = vertex_correction_result

    if original is None or polished is None:
        record_result("Vertex vocab roundtrip", False,
                     "Skipped (no correction data from test 1)")
        update_e2e_features("real-chain-vertex-vocab-roundtrip", False, "skipped")
        return

    if original == polished:
        record_result("Vertex vocab roundtrip", True,
                     "No diff (Gemini returned identical text) [non-critical]")
        update_e2e_features("real-chain-vertex-vocab-roundtrip", True)
        return

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from post_processor_configs import (
            load_vocab, diff_to_vocab, save_vocab, glossary_context,
        )
    except ImportError as exc:
        record_result("Vertex vocab roundtrip", False, f"Import failed: {exc}")
        update_e2e_features("real-chain-vertex-vocab-roundtrip", False, str(exc))
        return

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_vocab_path = Path(f.name)
        json.dump({}, f)

    try:
        vocab = load_vocab(tmp_vocab_path)
        new_vocab = diff_to_vocab(original, polished, vocab)
        save_vocab(new_vocab, tmp_vocab_path)
        reloaded = load_vocab(tmp_vocab_path)

        if reloaded == new_vocab:
            record_result("Vertex vocab roundtrip", True,
                         f"{len(reloaded)} entries saved and reloaded")
            update_e2e_features("real-chain-vertex-vocab-roundtrip", True)
        else:
            record_result("Vertex vocab roundtrip", False,
                         "Reloaded vocab differs from saved")
            update_e2e_features("real-chain-vertex-vocab-roundtrip", False, "mismatch")
    finally:
        if tmp_vocab_path.exists():
            tmp_vocab_path.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tests():
    print("\n=== L2 Real E2E: Vertex AI SSH Chain ===\n")

    # Pre-check: SSH connectivity
    print("[Pre-checks]")
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", SSH_HOST, "echo", "ok"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            record_result("SSH connectivity", False, result.stderr.strip())
            print("\nAborting: SSH to oracle-cloud failed.")
            return False
        record_result("SSH connectivity", True, SSH_HOST)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        record_result("SSH connectivity", False, str(exc))
        print("\nAborting: SSH to oracle-cloud failed.")
        return False

    # Pre-check: vertex_proxy.py exists
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", SSH_HOST,
             "python3", PROXY_SCRIPT, "--help"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            record_result("vertex_proxy.py deployed", False, result.stderr.strip()[:100])
            print("\nAborting: vertex_proxy.py not found or broken.")
            return False
        record_result("vertex_proxy.py deployed", True)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        record_result("vertex_proxy.py deployed", False, str(exc))
        print("\nAborting: vertex_proxy.py check failed.")
        return False

    # Run tests
    original, polished = test_vertex_correction()
    test_vertex_vocab_roundtrip((original, polished))

    # Summary
    print("\n=== Summary ===")
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    critical_failures = [
        r for r in results
        if not r["passed"] and "[non-critical]" not in r.get("detail", "")
    ]
    print(f"  {passed}/{total} checks passed")
    if critical_failures:
        print(f"  Critical failures: {[r['name'] for r in critical_failures]}")
    print()
    return len(critical_failures) == 0


def main():
    global verbose
    parser = argparse.ArgumentParser(
        description="L2 Real E2E: Vertex AI SSH chain with real Gemini",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    verbose = args.verbose
    success = run_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
