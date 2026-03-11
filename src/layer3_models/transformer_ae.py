import torch
import torch.nn as nn
import json

BEAT_LENGTH = 187
SAMPLE_RATE = 360

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=BEAT_LENGTH, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerAutoencoder(nn.Module):

    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_projection  = nn.Linear(1, d_model)
        self.output_projection = nn.Linear(d_model, 1)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.threshold = None

    def forward(self, x):
        x = x.unsqueeze(-1)                     
        x = self.pos_encoding(self.input_projection(x)) 
        memory = self.encoder(x)                
        out    = self.decoder(x, memory)        
        return self.output_projection(out).squeeze(-1)  
    
    def reconstruction_error(self, x):
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=-1)

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


if __name__ == "__main__":
    try:
        model = TransformerAutoencoder()
        model.model_summary()
        x     = torch.randn(4, BEAT_LENGTH)
        recon = model(x)
        assert recon.shape == (4, BEAT_LENGTH)
        errors = model.reconstruction_error(x)
        model.compute_threshold(errors)
        is_anomaly, _ = model.detect_anomaly(x)
    except Exception as e:
        print(f"[FAIL] {e}")