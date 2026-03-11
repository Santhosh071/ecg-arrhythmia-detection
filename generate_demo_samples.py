"""
generate_demo_samples.py
C:/ecg_arrhythmia/generate_demo_samples.py

Creates small .npy sample files from your existing test data.
Run once to generate demo files for testing the dashboard and API.

Run:
    cd C:/ecg_arrhythmia
    python generate_demo_samples.py

Output files (saved to C:/ecg_arrhythmia/samples/):
    sample_10beats.npy      ← 10 beats  — quick API test
    sample_50beats.npy      ← 50 beats  — dashboard demo
    sample_200beats.npy     ← 200 beats — full demo
    sample_mixed.npy        ← 50 beats with all 8 classes guaranteed
    sample_critical.npy     ← 30 beats, lots of V and E (high risk demo)
    sample_normal.npy       ← 30 beats, all normal (low risk demo)
    sample_raw_signal.npy   ← raw 1D ECG signal (tests preprocessing path)
    timestamps_50.npy       ← matching timestamps for sample_50beats.npy
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\ecg_arrhythmia")

PROCESSED = Path(r"D:\ecg_project\datasets\mitbih\processed\test")
OUTPUT    = Path(r"C:\ecg_arrhythmia\samples")
FS        = 360.0
BEAT_LEN  = 187

CLASS_NAMES = {
    0: "Normal (N)", 1: "LBBB (L)", 2: "RBBB (R)",
    3: "Atrial Premature (A)", 4: "PVC (V)",
    5: "Paced Beat (/)", 6: "Ventricular Escape (E)", 7: "Fusion Beat (F)"
}


def make_timestamps(n: int) -> np.ndarray:
    """Generate realistic R-peak timestamps for n beats."""
    return np.arange(n, dtype=float) * (BEAT_LEN / FS)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load test data
    beats_path  = PROCESSED / "beats.npy"
    labels_path = PROCESSED / "labels.npy"

    if not beats_path.exists():
        print(f"ERROR: {beats_path} not found.")
        print("Run Phase 2 preprocessing first.")
        return

    beats  = np.load(str(beats_path))
    labels = np.load(str(labels_path))
    print(f"Loaded test set: {beats.shape[0]} beats")

    # ── sample_10beats.npy ───────────────────────────────────────────────
    idx = np.random.choice(len(beats), 10, replace=False)
    np.save(str(OUTPUT / "sample_10beats.npy"), beats[idx])
    print("  sample_10beats.npy    — 10 random beats")

    # ── sample_50beats.npy ───────────────────────────────────────────────
    idx = np.random.choice(len(beats), 50, replace=False)
    np.save(str(OUTPUT / "sample_50beats.npy"),   beats[idx])
    np.save(str(OUTPUT / "timestamps_50.npy"),    make_timestamps(50))
    print("  sample_50beats.npy    — 50 random beats")
    print("  timestamps_50.npy     — matching timestamps")

    # ── sample_200beats.npy ──────────────────────────────────────────────
    idx = np.random.choice(len(beats), 200, replace=False)
    np.save(str(OUTPUT / "sample_200beats.npy"), beats[idx])
    print("  sample_200beats.npy   — 200 random beats")

    # ── sample_mixed.npy — all 8 classes guaranteed ──────────────────────
    mixed_beats = []
    for cls in range(8):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue
        n_take = min(7, len(cls_idx))
        chosen = np.random.choice(cls_idx, n_take, replace=False)
        mixed_beats.append(beats[chosen])
        print(f"    Class {cls} ({CLASS_NAMES[cls]:30s}): {n_take} beats added")

    mixed = np.vstack(mixed_beats)
    np.random.shuffle(mixed)
    np.save(str(OUTPUT / "sample_mixed.npy"), mixed)
    print(f"  sample_mixed.npy      — {len(mixed)} beats (all 8 classes)")

    # ── sample_critical.npy — high risk: lots of V and E ─────────────────
    crit_beats = []
    # Normal beats for context
    normal_idx = np.where(labels == 0)[0]
    crit_beats.append(beats[np.random.choice(normal_idx, 10, replace=False)])
    # PVC (V) — class 4
    v_idx = np.where(labels == 4)[0]
    if len(v_idx) >= 12:
        crit_beats.append(beats[np.random.choice(v_idx, 12, replace=False)])
    # Ventricular Escape (E) — class 6
    e_idx = np.where(labels == 6)[0]
    if len(e_idx) >= 5:
        crit_beats.append(beats[np.random.choice(e_idx, min(5, len(e_idx)), replace=False)])
    # Fusion (F) — class 7
    f_idx = np.where(labels == 7)[0]
    if len(f_idx) >= 3:
        crit_beats.append(beats[np.random.choice(f_idx, 3, replace=False)])

    critical = np.vstack(crit_beats)
    np.random.shuffle(critical)
    np.save(str(OUTPUT / "sample_critical.npy"), critical)
    print(f"  sample_critical.npy   — {len(critical)} beats (high risk: V+E+F dominant)")

    # ── sample_normal.npy — all normal, low risk ──────────────────────────
    normal_idx = np.where(labels == 0)[0]
    chosen     = np.random.choice(normal_idx, 30, replace=False)
    np.save(str(OUTPUT / "sample_normal.npy"), beats[chosen])
    print(f"  sample_normal.npy     — 30 normal beats (expect Low risk)")

    # ── sample_raw_signal.npy — raw 1D ECG ───────────────────────────────
    # Build a realistic synthetic raw signal from real beats
    chosen     = np.random.choice(len(beats), 60, replace=False)
    raw_signal = beats[chosen].flatten()   # 60 * 187 = 11220 samples
    np.save(str(OUTPUT / "sample_raw_signal.npy"), raw_signal)
    print(f"  sample_raw_signal.npy — raw 1D signal ({len(raw_signal)} samples)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nAll files saved to: {OUTPUT}")
    print("\nHow to use each file:")
    print("  Dashboard upload    → sample_50beats.npy or sample_200beats.npy")
    print("  Demo (normal only)  → sample_normal.npy   (shows green / Low risk)")
    print("  Demo (critical)     → sample_critical.npy (shows red / Critical risk)")
    print("  Demo (all classes)  → sample_mixed.npy    (shows all class types)")
    print("  API /predict test   → sample_10beats.npy  (quick curl test)")
    print("  API /signal/quality → sample_raw_signal.npy")

    # Print file sizes
    print("\nFile sizes:")
    for f in sorted(OUTPUT.glob("*.npy")):
        size = f.stat().st_size / 1024
        arr  = np.load(str(f))
        print(f"  {f.name:<30} {arr.shape}  {size:.1f} KB")


if __name__ == "__main__":
    np.random.seed(42)   # reproducible samples
    main()
