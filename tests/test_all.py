import sys
import time
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, r"C:\ecg_arrhythmia")

SEP  = "=" * 55
SEP2 = "-" * 55
TEST_FILES = [
    ("Phase 5 — Inference Pipeline", r"tests\test_inference.py",   None),
    ("Phase 6 — AI Agent",           r"tests\test_agent.py",       None),
    ("Phase 9 — Integration",        r"tests\test_integration.py", "--no-api"),
]

def run_test(label: str, script: str, extra_arg: str, with_api: bool) -> tuple:
    """
    Run one test file as a subprocess.
    Returns (passed, failed, duration_seconds).
    """
    cmd = [sys.executable, script]
    if "test_integration" in script and not with_api:
        if extra_arg:
            cmd.append(extra_arg)
    print(f"\n{SEP}")
    print(f"RUNNING: {label}")
    print(f"  Script : {script}")
    print(SEP2)
    t0   = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output = False,
        cwd            = r"C:\ecg_arrhythmia",
    )
    dur  = time.perf_counter() - t0
    success = proc.returncode == 0
    return success, dur

def check_api_running() -> bool:
    """Check if FastAPI is running on port 8000."""
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-api", action="store_true",
                        help="Include API endpoint tests (requires uvicorn running)")
    args = parser.parse_args()
    print(f"\n{SEP}")
    print("ECG Arrhythmia Detection System")
    print("Master Test Suite — All Phases")
    print(SEP)
    print(f"  Date     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  API tests: {'YES' if args.with_api else 'NO (use --with-api to enable)'}")

    if args.with_api and not check_api_running():
        print("\nWARNING: --with-api set but API is not running.")
        print("Start it first:")
        print("  uvicorn api.main:app --host 127.0.0.1 --port 8000")
        print("Continuing without API tests...\n")
        args.with_api = False
    results  = []
    total_t  = 0.0
    for label, script, extra in TEST_FILES:
        success, dur = run_test(label, script, extra, args.with_api)
        results.append((label, success, dur))
        total_t += dur
    print(f"\n{SEP}")
    print("MASTER TEST SUITE RESULTS")
    print(SEP2)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    for label, ok, dur in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {label:<35} ({dur:.1f}s)")

    print(SEP2)
    print(f"  Total    : {len(results)} test suites")
    print(f"  Passed   : {passed}")
    print(f"  Failed   : {failed}")
    print(f"  Duration : {total_t:.1f}s")
    print(SEP)
    if failed == 0:
        print("\nALL TESTS PASSED")
        print("System is ready for Phase 11 — Report & Demo.")
    else:
        print(f"\n{failed} suite(s) failed — check output above.")
    print()
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())