"""
run.py
C:/ecg_arrhythmia/run.py

Single command to launch the entire ECG system.

    cd C:/ecg_arrhythmia
    python run.py

Starts:
    FastAPI  → http://127.0.0.1:8000   (backend + API)
    Streamlit→ http://127.0.0.1:8501   (dashboard)

Options:
    python run.py --api-only
    python run.py --dashboard-only
    python run.py --port-api 8000 --port-dash 8501
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT      = Path(__file__).resolve().parent
API_HOST  = os.getenv("API_HOST",       "127.0.0.1")
API_PORT  = int(os.getenv("API_PORT",   "8000"))
DASH_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

_processes = []


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║        ECG Arrhythmia Detection System               ║
║        6-Layer AI Pipeline — Capstone 2024-25        ║
╠══════════════════════════════════════════════════════╣
║  FastAPI   → http://127.0.0.1:8000                  ║
║  Docs      → http://127.0.0.1:8000/docs             ║
║  Dashboard → http://127.0.0.1:8501                  ║
╚══════════════════════════════════════════════════════╝
""")


def start_api(host: str, port: int) -> subprocess.Popen:
    """Start FastAPI server with uvicorn."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", host,
        "--port", str(port),
        "--reload",
        "--log-level", "info",
    ]
    print(f"[run] Starting FastAPI on http://{host}:{port} ...")
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    _processes.append(proc)
    return proc


def start_dashboard(port: int) -> subprocess.Popen:
    """Start Streamlit dashboard."""
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "src" / "layer5_dashboard" / "app.py"),
        "--server.port",    str(port),
        "--server.address", "127.0.0.1",
        "--server.headless","true",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"[run] Starting Streamlit on http://127.0.0.1:{port} ...")
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    _processes.append(proc)
    return proc


def wait_for_api(host: str, port: int, timeout: int = 60) -> bool:
    """Poll until FastAPI /health returns 200 or timeout."""
    import urllib.request
    url     = f"http://{host}:{port}/health"
    elapsed = 0
    print(f"[run] Waiting for API to be ready", end="", flush=True)
    while elapsed < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            print(" ✓")
            return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)
            elapsed += 2
    print(" ✗ (timeout)")
    return False


def shutdown(signum=None, frame=None):
    """Gracefully stop all child processes."""
    print("\n[run] Shutting down all services...")
    for p in _processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            p.kill()
    print("[run] All services stopped.")
    sys.exit(0)


def monitor(processes: list):
    """Background thread — restarts a crashed process and alerts."""
    while True:
        time.sleep(5)
        for p in processes:
            if p.poll() is not None:
                print(f"\n[run] WARNING: Process {p.pid} exited unexpectedly.")


def main():
    parser = argparse.ArgumentParser(description="ECG System Launcher")
    parser.add_argument("--api-only",       action="store_true")
    parser.add_argument("--dashboard-only", action="store_true")
    parser.add_argument("--port-api",  type=int, default=API_PORT)
    parser.add_argument("--port-dash", type=int, default=DASH_PORT)
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print_banner()

    procs = []

    if not args.dashboard_only:
        api_proc = start_api(API_HOST, args.port_api)
        procs.append(api_proc)
        # Wait for API to be ready before starting dashboard
        # so dashboard can call the API on first load
        ready = wait_for_api(API_HOST, args.port_api, timeout=90)
        if not ready:
            print("[run] API did not start in time. Check logs above.")
            shutdown()

    if not args.api_only:
        time.sleep(2)
        dash_proc = start_dashboard(args.port_dash)
        procs.append(dash_proc)
        time.sleep(3)

    print("\n[run] All services running. Press Ctrl+C to stop.\n")
    print(f"  API docs  → http://{API_HOST}:{args.port_api}/docs")
    print(f"  Dashboard → http://127.0.0.1:{args.port_dash}")
    print(f"  Health    → http://{API_HOST}:{args.port_api}/health\n")

    # Monitor thread
    t = threading.Thread(target=monitor, args=(procs,), daemon=True)
    t.start()

    # Block main thread until Ctrl+C
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
