import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from utils import (FILENAME_PREFIX, CHECKPOINT_DIR, SAMPLES_DIR,
                   _fmt_hms, _next_run_index, _save_checkpoint, _find_latest_checkpoint)


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(128)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(256)

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        self.dec_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.relu(self.enc_bn3(self.enc_conv3(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 256, 16, 16)
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.relu(self.dec_bn2(self.dec_conv2(x)))
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


class CelebAHFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"]
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label


def train_vae(
    epochs=20,
    batch_size=128,
    latent_dim=128,
    short_run=False,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    checkpoint_prefix: str = FILENAME_PREFIX,
):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    print("\n[INFO] Loading CelebA from Hugging Face…")
    celeba = load_dataset("flwrlabs/celeba")

    full_train = celeba["train"]
    split = full_train.train_test_split(test_size=0.1, seed=42)
    train_dataset = CelebAHFDataset(split["train"], transform=transform)
    val_dataset = CelebAHFDataset(split["test"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model + optimizer + scheduler
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.95, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    # Compute a run index (increments across runs)
    run_idx = _next_run_index(checkpoint_dir=checkpoint_dir, prefix=checkpoint_prefix)

    print("\n[INFO] Starting training…")
    t0_total = time.perf_counter()

    for epoch in range(epochs if not short_run else 1):
        t0_epoch = time.perf_counter()

        model.train()
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if short_run and batch_idx > 10:
                break

        # Average by number of batches for readability
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, _ in val_loader:
                val_data = val_data.to(device)
                recon, mu, logvar = model(val_data)
                val_loss += vae_loss(recon, val_data, mu, logvar).item()
        avg_val_loss = val_loss / len(val_loader)

        # Scheduler step & logs
        scheduler.step(avg_val_loss)
        elapsed_epoch = time.perf_counter() - t0_epoch
        print(f"\nEpoch {epoch + 1:03d}"
              f" | Train: {avg_train_loss:.4f}"
              f" | Val: {avg_val_loss:.4f}"
              f" | LR: {optimizer.param_groups[0]['lr']:.6f}"
              f" | [TIME] { _fmt_hms(elapsed_epoch) }")

        # Save checkpoint with epoch, run index, date
        _save_checkpoint(
            model=model,
            epoch=epoch + 1,
            run_idx=run_idx,
            checkpoint_dir=checkpoint_dir,
            prefix=checkpoint_prefix
        )

    elapsed_total = time.perf_counter() - t0_total
    print(f"\n[TIME] Full training took { _fmt_hms(elapsed_total) }")

    return model


def generate_faces_from_latest(
    latent_dim=128,
    num_samples=16,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    samples_dir: Path = SAMPLES_DIR,
    prefix: str = FILENAME_PREFIX
) -> Path | None:
    """
    Loads the latest checkpoint and saves a PNG with a name mirroring the checkpoint.
    E.g. ckpt: vae_E005_I003_D20250908-143012.pt -> samples/faces_vae_E005_I003_D20250908-143012_N16.png
    """
    latest = _find_latest_checkpoint(checkpoint_dir=checkpoint_dir, prefix=prefix)
    if latest is None:
        print("[WARN] No checkpoints found.")
        return None

    # Rebuild the model and load weights
    model = VAE(latent_dim=latent_dim).to(device)
    state = torch.load(latest, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Generate and save
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z).cpu()
        grid = utils.make_grid(samples, nrow=4)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")

        stem = latest.stem  # "vae_E###_I###_DYYYYMMDD-HHMMSS"
        out_path = samples_dir / f"faces_{stem}_N{num_samples}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Generated faces from {latest.name} -> {out_path}")
        return out_path


if __name__ == "__main__":
    print(f"[INFO] Using device: {device}")
    vae_model = train_vae(epochs=20, short_run=True)
    generate_faces_from_latest()
