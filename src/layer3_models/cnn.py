import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = {
    0: "Normal (N)",
    1: "Left Bundle Branch Block (L)",
    2: "Right Bundle Branch Block (R)",
    3: "Atrial Premature Beat (A)",
    4: "Premature Ventricular Contraction (V)",
    5: "Paced Beat (/)",
    6: "Ventricular Escape Beat (E)",
    7: "Fusion Beat (F)",
}
NUM_CLASSES = 8
BEAT_LENGTH = 187
SAMPLE_RATE = 360


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class ECGCNNClassifier(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.4):
        super().__init__()
        self.conv1    = ConvBlock1D(1,   32, kernel_size=7)
        self.conv2    = ConvBlock1D(32,  64, kernel_size=5)
        self.conv3    = ConvBlock1D(64, 128, kernel_size=5)
        self.gap      = nn.AdaptiveAvgPool1d(1)
        self.fc1      = nn.Linear(128, 256)
        self.bn_fc1   = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2      = nn.Linear(256, 128)
        self.bn_fc2   = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output   = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.gap(x).squeeze(-1)
        x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.output(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs      = torch.softmax(self.forward(x), dim=-1)
            conf, pred = torch.max(probs, dim=-1)
        return pred, conf, [CLASS_NAMES[c.item()] for c in pred]

    def model_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ECGCNNClassifier] Params: {total:,} total | {trainable:,} trainable")
        print(f"  Input : (batch, 1, {BEAT_LENGTH})")
        print(f"  Output: (batch, {NUM_CLASSES})")


if __name__ == "__main__":
    try:
        model = ECGCNNClassifier()
        model.model_summary()
        x      = torch.randn(4, 1, BEAT_LENGTH)
        logits = model(x)
        assert logits.shape == (4, NUM_CLASSES)
        pred, conf, names = model.predict(x)
        print("[PASS] cnn.py ready for training.")
    except Exception as e:
        print(f"[FAIL] {e}")
