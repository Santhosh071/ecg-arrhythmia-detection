import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\ecg_arrhythmia")
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)
import torch
from src.layer3_models.inference import (
    ModelLoader, predict_beats, BEAT_LENGTH, CLASS_NAMES, SHORT_NAMES
)

MODELS_DIR = Path(r"D:\ecg_project\models")
DATA_DIR   = Path(r"D:\ecg_project\datasets\mitbih\processed")
DEVICE     = torch.device("cpu")
SEP        = "=" * 55
SEP2       = "-" * 55


def load_test_data(split: str = "test") -> tuple:
    """Load beats and labels from processed split folder."""
    beats  = np.load(str(DATA_DIR / split / "beats.npy"))
    labels = np.load(str(DATA_DIR / split / "labels.npy"))
    return beats.astype(np.float32), labels.astype(int)


def evaluate_cnn(loader: ModelLoader, beats: np.ndarray, labels: np.ndarray,
                 batch_size: int = 128) -> dict:
    """
    Evaluate CNN classifier on test set.
    Returns accuracy, per-class F1, macro F1, weighted F1, confusion matrix.
    """
    print(f"\n{SEP}\nCNN CLASSIFIER EVALUATION\n{SEP2}")
    all_preds = []
    all_confs = []
    loader.cnn.eval()
    with torch.no_grad():
        for start in range(0, len(beats), batch_size):
            x     = torch.tensor(beats[start:start+batch_size],
                                  dtype=torch.float32).unsqueeze(1)
            probs = torch.softmax(loader.cnn(x), dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_confs.extend(confs.tolist())
    preds   = np.array(all_preds)
    confs   = np.array(all_confs)
    acc     = accuracy_score(labels, preds)
    f1_mac  = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_wgt  = f1_score(labels, preds, average="weighted", zero_division=0)
    cm      = confusion_matrix(labels, preds, labels=list(range(8)))
    per_class = {}
    for cls in range(8):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        per_class[cls] = {
            "name"     : CLASS_NAMES[cls],
            "short"    : SHORT_NAMES[cls],
            "support"  : int(mask.sum()),
            "precision": round(float(precision_score(labels==cls, preds==cls, zero_division=0)), 4),
            "recall"   : round(float(recall_score(labels==cls,    preds==cls, zero_division=0)), 4),
            "f1"       : round(float(f1_score(labels==cls,        preds==cls, zero_division=0)), 4),
        }
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Macro F1        : {f1_mac:.4f}")
    print(f"  Weighted F1     : {f1_wgt:.4f}")
    print(f"  Mean Confidence : {confs.mean()*100:.1f}%")
    print(f"\n  Per-Class Results:")
    print(f"  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print(f"  {SEP2}")
    for cls, m in per_class.items():
        print(f"  {m['name']:<45} {m['precision']:>6.4f} {m['recall']:>6.4f} "
              f"{m['f1']:>6.4f} {m['support']:>6}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    header = "       " + " ".join(f"{SHORT_NAMES[i]:>5}" for i in range(8))
    print(f"  {header}")
    for i in range(8):
        row = " ".join(f"{cm[i,j]:>5}" for j in range(8))
        print(f"  {SHORT_NAMES[i]:>5}  {row}")
    return {
        "model"          : "CNN Classifier",
        "accuracy"       : round(acc,    4),
        "macro_f1"       : round(f1_mac, 4),
        "weighted_f1"    : round(f1_wgt, 4),
        "mean_confidence": round(float(confs.mean()), 4),
        "per_class"      : per_class,
        "confusion_matrix": cm.tolist(),
        "predictions"    : preds,
        "confidences"    : confs,
    }


def evaluate_autoencoder(model, threshold: float, model_name: str,
                         beats: np.ndarray, labels: np.ndarray,
                         val_mean: float = None, val_std: float = None,
                         z_thresh: float = None,
                         batch_size: int = 128) -> dict:
    """
    Evaluate one autoencoder for anomaly detection.
    Normal (class 0) = negative, Any arrhythmia = positive.
    Returns precision, recall, F1, ROC-AUC, FPR, FNR.
    """
    print(f"\n{SEP}\n{model_name.upper()} EVALUATION\n{SEP2}")
    binary_labels = (labels != 0).astype(int)
    errors        = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(beats), batch_size):
            x    = torch.tensor(beats[start:start+batch_size], dtype=torch.float32)
            errs = model.reconstruction_error(x).cpu().numpy()
            errors.extend(errs.tolist())
    errors = np.array(errors)
    if val_mean is not None and val_std is not None and val_std > 1e-12:
        scores = (errors - val_mean) / val_std
    else:
        scores = errors
    if z_thresh is not None and val_mean is not None:
        preds = (scores > z_thresh).astype(int)
    else:
        preds = (errors > threshold).astype(int)
    prec = precision_score(binary_labels, preds, zero_division=0)
    rec  = recall_score(binary_labels,    preds, zero_division=0)
    f1   = f1_score(binary_labels,        preds, zero_division=0)
    acc  = accuracy_score(binary_labels,  preds)
    try:
        auc = roc_auc_score(binary_labels, errors)
    except Exception:
        auc = 0.0
    cm             = confusion_matrix(binary_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    print(f"  Threshold       : {threshold:.6f}")
    if z_thresh is not None:
        print(f"  Z-score thresh  : {z_thresh}")
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"  False Pos Rate  : {fpr*100:.2f}%")
    print(f"  False Neg Rate  : {fnr*100:.2f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    normal_errs  = errors[binary_labels == 0]
    anomaly_errs = errors[binary_labels == 1]
    print(f"\n  Reconstruction Error Distribution:")
    print(f"    Normal  — mean={normal_errs.mean():.6f}  std={normal_errs.std():.6f}"
          f"  max={normal_errs.max():.6f}")
    print(f"    Anomaly — mean={anomaly_errs.mean():.6f}  std={anomaly_errs.std():.6f}"
          f"  max={anomaly_errs.max():.6f}")
    return {
        "model"             : model_name,
        "threshold"         : threshold,
        "accuracy"          : round(acc,  4),
        "precision"         : round(prec, 4),
        "recall"            : round(rec,  4),
        "f1"                : round(f1,   4),
        "roc_auc"           : round(auc,  4),
        "false_pos_rate"    : round(fpr,  4),
        "false_neg_rate"    : round(fnr,  4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "normal_mean_error" : round(float(normal_errs.mean()),  6),
        "anomaly_mean_error": round(float(anomaly_errs.mean()), 6),
        "errors"            : errors,
        "predictions"       : preds,
    }


def evaluate_combined(cnn_results: dict, trans_results: dict,
                      lstm_results: dict, labels: np.ndarray) -> dict:
    """
    Evaluate combined system:
      is_anomaly = transformer_flag OR lstm_flag OR (cnn != Normal)
    """
    print(f"\n{SEP}\nCOMBINED SYSTEM EVALUATION\n{SEP2}")
    binary_labels = (labels != 0).astype(int)
    cnn_flags     = (cnn_results["predictions"]   != 0).astype(int)
    trans_flags   = trans_results["predictions"]
    lstm_flags    = lstm_results["predictions"]
    combined      = ((cnn_flags == 1) | (trans_flags == 1) | (lstm_flags == 1)).astype(int)
    prec          = precision_score(binary_labels, combined, zero_division=0)
    rec           = recall_score(binary_labels,    combined, zero_division=0)
    f1            = f1_score(binary_labels,        combined, zero_division=0)
    acc           = accuracy_score(binary_labels,  combined)
    cm            = confusion_matrix(binary_labels, combined, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    print(f"  Decision rule   : CNN OR Transformer AE OR LSTM AE")
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  False Pos Rate  : {fpr*100:.2f}%")
    print(f"  False Neg Rate  : {fnr*100:.2f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    return {
        "model"         : "Combined System",
        "accuracy"      : round(acc,  4),
        "precision"     : round(prec, 4),
        "recall"        : round(rec,  4),
        "f1"            : round(f1,   4),
        "false_pos_rate": round(fpr,  4),
        "false_neg_rate": round(fnr,  4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def benchmark_speed(loader: ModelLoader, beats: np.ndarray) -> dict:
    """Measure inference time per beat at different batch sizes."""
    print(f"\n{SEP}\nINFERENCE SPEED BENCHMARK\n{SEP2}")
    results = {}
    for n in [10, 50, 100, 200, 500]:
        if n > len(beats):
            continue
        timestamps = np.arange(n, dtype=float) * (187 / 360)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            predict_beats(beats[:n], timestamps, loader, batch_size=64)
            times.append((time.perf_counter() - t0) * 1000)
        avg      = np.mean(times)
        per_beat = avg / n
        results[n] = {"total_ms": round(avg, 1), "per_beat_ms": round(per_beat, 3)}
        print(f"  {n:5d} beats → {avg:7.1f} ms total | {per_beat:.3f} ms/beat")
    return results


def main() -> dict:
    print(f"\n{SEP}")
    print("ECG System — Phase 10 Model Evaluation")
    print(SEP)
    print("\nLoading test data...")
    beats, labels = load_test_data("test")
    print(f"  Test set: {beats.shape[0]} beats | "
          f"{(labels==0).sum()} normal | {(labels!=0).sum()} anomaly")
    print("\nLoading models...")
    loader = ModelLoader(str(MODELS_DIR))
    loader.load_all()
    cnn_res   = evaluate_cnn(loader, beats, labels)
    trans_res = evaluate_autoencoder(
        model      = loader.transformer_ae,
        threshold  = loader.trans_threshold,
        model_name = "Transformer Autoencoder",
        beats      = beats,
        labels     = labels,
        val_mean   = loader.trans_val_mean,
        val_std    = loader.trans_val_std,
        z_thresh   = loader.trans_z_thresh,
    )
    lstm_res = evaluate_autoencoder(
        model      = loader.lstm_ae,
        threshold  = loader.lstm_threshold,
        model_name = "LSTM Autoencoder",
        beats      = beats,
        labels     = labels,
    )
    combined = evaluate_combined(cnn_res, trans_res, lstm_res, labels)
    speed    = benchmark_speed(loader, beats)
    print(f"\n{SEP}\nFINAL EVALUATION SUMMARY\n{SEP2}")
    print(f"  {'Model':<30} {'Accuracy':>9} {'F1':>8} {'ROC-AUC':>9}")
    print(f"  {SEP2}")
    print(f"  {'CNN Classifier':<30} {cnn_res['accuracy']*100:>8.2f}% "
          f"{cnn_res['weighted_f1']:>8.4f} {'N/A':>9}")
    print(f"  {'Transformer AE':<30} {trans_res['accuracy']*100:>8.2f}% "
          f"{trans_res['f1']:>8.4f} {trans_res['roc_auc']:>9.4f}")
    print(f"  {'LSTM AE':<30} {lstm_res['accuracy']*100:>8.2f}% "
          f"{lstm_res['f1']:>8.4f} {lstm_res['roc_auc']:>9.4f}")
    print(f"  {'Combined System':<30} {combined['accuracy']*100:>8.2f}% "
          f"{combined['f1']:>8.4f} {'N/A':>9}")
    print()
    all_results = {
        "cnn"           : {k: v for k, v in cnn_res.items()
                           if k not in ("predictions", "confidences")},
        "transformer_ae": {k: v for k, v in trans_res.items()
                           if k not in ("errors", "predictions")},
        "lstm_ae"       : {k: v for k, v in lstm_res.items()
                           if k not in ("errors", "predictions")},
        "combined"      : combined,
        "speed_benchmark": speed,
        "test_set_size" : int(len(beats)),
        "normal_count"  : int((labels == 0).sum()),
        "anomaly_count" : int((labels != 0).sum()),
    }
    return all_results, cnn_res, trans_res, lstm_res


if __name__ == "__main__":
    results, cnn_res, trans_res, lstm_res = main()
    out_path = Path(r"C:\ecg_arrhythmia\evaluation\eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Run evaluation/evaluation_report.py to generate PDF + CSV.")
