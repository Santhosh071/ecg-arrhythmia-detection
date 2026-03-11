import torch
import torch.nn as nn
import torch.nn.functional as F

BEAT_LENGTH  = 187
LATENT_DIM   = 32

class VAEEncoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BEAT_LENGTH, 128), nn.ReLU(),
            nn.Linear(128, 64),          nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h)

class VAEDecoder(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),   nn.ReLU(),
            nn.Linear(64, 128),          nn.ReLU(),
            nn.Linear(128, BEAT_LENGTH),
        )

    def forward(self, z):
        return self.net(z)
class ECGVariationalAutoencoder(nn.Module):

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = VAEEncoder(latent_dim)
        self.decoder    = VAEDecoder(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def generate(self, n_samples, device='cpu'):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            return self.decoder(z)

    def model_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

def vae_loss(recon, x, mu, log_var, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


if __name__ == "__main__":
    try:
        model = ECGVariationalAutoencoder()
        model.model_summary()
        x                    = torch.randn(4, BEAT_LENGTH)
        recon, mu, log_var   = model(x)
        assert recon.shape   == (4, BEAT_LENGTH)
        assert mu.shape      == (4, LATENT_DIM)
        loss, recon_l, kl_l  = vae_loss(recon, x, mu, log_var)
        synthetic            = model.generate(n_samples=8)
        assert synthetic.shape == (8, BEAT_LENGTH)
    except Exception as e:
        print(f"[FAIL] {e}")