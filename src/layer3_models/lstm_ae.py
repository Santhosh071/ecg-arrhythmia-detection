import torch
import torch.nn as nn
import json

BEAT_LENGTH = 187
SAMPLE_RATE = 360


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, hidden, cell):
        out, _ = self.lstm(encoder_outputs, (hidden, cell))
        return self.output_layer(out).squeeze(-1)


class LSTMAutoencoder(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder   = LSTMEncoder(hidden_size, num_layers, dropout)
        self.decoder   = LSTMDecoder(hidden_size, num_layers, dropout)
        self.threshold = None

    def forward(self, x):
        x = x.unsqueeze(-1)
        enc_out, hidden, cell = self.encoder(x)
        return self.decoder(enc_out, hidden, cell)

    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=-1)

    def compute_threshold(self, val_errors):
        self.threshold = val_errors.mean().item() + 2 * val_errors.std().item()
        return self.threshold

    def detect_anomaly(self, x):
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Run compute_threshold() first.")
        self.eval()
        with torch.no_grad():
            errors = self.reconstruction_error(x)
        return errors > self.threshold, errors

    def save_threshold(self, path):
        with open(path, 'w') as f:
            json.dump({"threshold": self.threshold}, f)

    def load_threshold(self, path):
        with open(path, 'r') as f:
            self.threshold = json.load(f)["threshold"]

    def model_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LSTMAutoencoder] Params: {total:,} total | {trainable:,} trainable")
        print(f"  Input : (batch, {BEAT_LENGTH})")
        print(f"  Output: (batch, {BEAT_LENGTH})")
        print(f"  Threshold: {self.threshold}")


if __name__ == "__main__":
    try:
        model = LSTMAutoencoder()
        model.model_summary()
        x     = torch.randn(4, BEAT_LENGTH)
        recon = model(x)
        assert recon.shape == (4, BEAT_LENGTH)
        errors = model.reconstruction_error(x)
        model.compute_threshold(errors)
        is_anomaly, _ = model.detect_anomaly(x)
        print(f"  Anomalies detected: {is_anomaly.sum().item()} / {len(x)}")
        print("[PASS] lstm_ae.py ready for training.")
    except Exception as e:
        print(f"[FAIL] {e}")
