import sys
import json
import torch
from pathlib import Path

BEAT_LENGTH = 187
NUM_CLASSES = 8
DEVICE      = torch.device("cpu")
_MODELS_DIR = Path(r"D:/ecg_project/models")
_ARCH_DIR   = Path(r"C:/ecg_arrhythmia/src\layer3_models")

class ModelLoader:
    def __init__(self, models_dir: str = None):
        self.models_dir     = Path(models_dir) if models_dir else _MODELS_DIR
        self.cnn            = None
        self.transformer_ae = None
        self.lstm_ae        = None
        self.trans_threshold = None
        self.trans_val_mean  = None
        self.trans_val_std   = None
        self.trans_z_thresh  = None
        self.lstm_threshold  = None
        self.class_names     = {}
        self.is_loaded       = False

    def _import_architectures(self):
        for p in [str(_ARCH_DIR), str(self.models_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from cnn import ECGCNNClassifier
            from transformer_ae import TransformerAutoencoder
            from lstm_ae import LSTMAutoencoder
            return ECGCNNClassifier, TransformerAutoencoder, LSTMAutoencoder
        except ImportError as e:
            raise ImportError(
                f"Cannot import model architectures: {e}\n"
                f"Copy cnn.py, transformer_ae.py, lstm_ae.py to:\n  {_ARCH_DIR}"
            )

    def _load_state(self, path: Path, model, key: str = "model_state_dict"):
        bundle = torch.load(path, map_location=DEVICE)
        if isinstance(bundle, dict) and key in bundle:
            model.load_state_dict(bundle[key])
        else:
            model.load_state_dict(bundle)
        return bundle if isinstance(bundle, dict) else {}

    def _load_cnn(self, ECGCNNClassifier):
        cnn_dir = self.models_dir / "cnn_classifier"
        labels_path = cnn_dir / "class_labels.json"
        if labels_path.exists():
            with open(labels_path, encoding="utf-8") as f:
                self.class_names = {int(k): v for k, v in json.load(f).items()}
        else:
            from .result_schema import CLASS_NAMES
            self.class_names = CLASS_NAMES

        retrain  = cnn_dir / "cnn_retrain_classifier_best.pth"
        original = cnn_dir / "cnn_classifier_best.pth"
        cnn_path = retrain if retrain.exists() else original
        choice_path = cnn_dir / "final_model_choice.json"
        if choice_path.exists():
            with open(choice_path, encoding="utf-8") as f:
                choice = json.load(f)
            if "original" in choice.get("chosen_model", "").lower() and original.exists():
                cnn_path = original
        if not cnn_path.exists():
            raise FileNotFoundError(f"CNN weights not found in {cnn_dir}")
        cnn = ECGCNNClassifier(num_classes=NUM_CLASSES)
        self._load_state(cnn_path, cnn)
        cnn.eval()
        self.cnn = cnn
        print(f"[loader] CNN              : {cnn_path.name}")

    def _load_transformer_ae(self, TransformerAutoencoder):
        tae_dir  = self.models_dir / "transformer_ae"
        tae_path = tae_dir / "transformer_ae_v4_final.pth"
        if not tae_path.exists():
            tae_path = tae_dir / "transformer_ae_best.pth"
        if not tae_path.exists():
            raise FileNotFoundError(f"Transformer AE weights not found in {tae_dir}")
        bundle = torch.load(tae_path, map_location=DEVICE)
        if isinstance(bundle, dict) and "config" in bundle:
            cfg = bundle["config"]
            tae = TransformerAutoencoder(
                d_model         = cfg["d_model"],
                nhead           = cfg["nhead"],
                num_layers      = cfg["num_layers"],
                dim_feedforward = cfg["dim_feedforward"],
                dropout         = cfg.get("dropout", 0.1),
            )
            tae.load_state_dict(bundle["model_state_dict"])
            self.trans_threshold = bundle.get("threshold")
            self.trans_val_mean  = bundle.get("val_mean")
            self.trans_val_std   = bundle.get("val_std")
            self.trans_z_thresh  = bundle.get("threshold_z")
        else:
            tae = TransformerAutoencoder(
                d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1
            )
            tae.load_state_dict(bundle)
        thresh_path = tae_dir / "anomaly_threshold.json"
        if thresh_path.exists():
            with open(thresh_path, encoding="utf-8") as f:
                td = json.load(f)
            self.trans_threshold = td.get("threshold",   self.trans_threshold)
            self.trans_val_mean  = td.get("val_mean",    self.trans_val_mean)
            self.trans_val_std   = td.get("val_std",     self.trans_val_std)
            self.trans_z_thresh  = td.get("threshold_z", self.trans_z_thresh)
        if self.trans_threshold is None:
            self.trans_threshold = 0.001
        tae.eval()
        self.transformer_ae = tae
        print(f"[loader] Transformer AE   : {tae_path.name} | threshold={self.trans_threshold:.6f}")

    def _load_lstm_ae(self, LSTMAutoencoder):
        lstm_dir  = self.models_dir / "lstm_ae"
        lstm_path = lstm_dir / "lstm_ae_final.pth"
        if not lstm_path.exists():
            lstm_path = lstm_dir / "lstm_ae_best.pth"
        if not lstm_path.exists():
            raise FileNotFoundError(f"LSTM AE weights not found in {lstm_dir}")
        bundle = torch.load(lstm_path, map_location=DEVICE)
        if isinstance(bundle, dict) and "config" in bundle:
            cfg  = bundle["config"]
            lstm = LSTMAutoencoder(
                hidden_size = cfg["hidden_size"],
                num_layers  = cfg["num_layers"],
                dropout     = cfg.get("dropout", 0.2),
            )
            lstm.load_state_dict(bundle["model_state_dict"])
            self.lstm_threshold = bundle.get("threshold")
        else:
            lstm = LSTMAutoencoder(hidden_size=64, num_layers=2, dropout=0.2)
            lstm.load_state_dict(bundle)
        thresh_path = lstm_dir / "lstm_ae_threshold.json"
        if thresh_path.exists():
            with open(thresh_path, encoding="utf-8") as f:
                self.lstm_threshold = json.load(f).get("threshold", self.lstm_threshold)
        if self.lstm_threshold is None:
            self.lstm_threshold = 0.001
        lstm.eval()
        self.lstm_ae = lstm
        print(f"[loader] LSTM AE          : {lstm_path.name} | threshold={self.lstm_threshold:.6f}")

    def load_all(self) -> "ModelLoader":
        if not self.models_dir.exists():
            raise FileNotFoundError(f"models_dir not found: {self.models_dir}")
        print(f"[loader] Loading from {self.models_dir}")
        ECGCNNClassifier, TransformerAutoencoder, LSTMAutoencoder = self._import_architectures()
        self._load_cnn(ECGCNNClassifier)
        self._load_transformer_ae(TransformerAutoencoder)
        self._load_lstm_ae(LSTMAutoencoder)
        self.is_loaded = True
        print("[loader] All models ready.\n")
        return self

    def status(self) -> dict:
        return {
            "is_loaded"        : self.is_loaded,
            "cnn_ready"        : self.cnn is not None,
            "transformer_ready": self.transformer_ae is not None,
            "lstm_ready"       : self.lstm_ae is not None,
            "trans_threshold"  : self.trans_threshold,
            "trans_z_thresh"   : self.trans_z_thresh,
            "trans_val_mean"   : self.trans_val_mean,
            "trans_val_std"    : self.trans_val_std,
            "lstm_threshold"   : self.lstm_threshold,
            "num_classes"      : len(self.class_names),
        }