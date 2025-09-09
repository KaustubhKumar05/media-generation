import os
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

from utils import (
    FILENAME_PREFIX, CHECKPOINT_DIR, SAMPLES_DIR,
    _fmt_hms, _next_run_index, _save_checkpoint, _find_latest_checkpoint
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 256
LR = 3e-4
NUM_WORKERS = os.cpu_count() - 2


class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()

        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
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
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
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
    dataset="flwrlabs/celeba",
    epochs=20,
    batch_size=BATCH_SIZE,
    latent_dim=64,
    short_run=False,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    checkpoint_prefix: str = FILENAME_PREFIX,
):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    print(f"\nLoading dataset from Hugging Face: {dataset}")
    celeba = load_dataset(dataset)

    full_train = celeba["train"]
    split = full_train.train_test_split(test_size=0.1, seed=42)
    train_dataset = CelebAHFDataset(split["train"], transform=transform)
    val_dataset = CelebAHFDataset(split["test"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    run_idx = _next_run_index(checkpoint_dir=checkpoint_dir, prefix=checkpoint_prefix)

    print("\nStarting training…")
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

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, _ in val_loader:
                val_data = val_data.to(device)
                recon, mu, logvar = model(val_data)
                val_loss += vae_loss(recon, val_data, mu, logvar).item()
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        elapsed_epoch = time.perf_counter() - t0_epoch

        print(f"Epoch {epoch + 1:03d} completed"
              f" | Train: {avg_train_loss:.2f}"
              f" | Val: {avg_val_loss:.2f}"
              f" | LR: {optimizer.param_groups[0]['lr']:.6f}"
              f" | [TIME] { _fmt_hms(elapsed_epoch) }\n")

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            _save_checkpoint(
                model=model,
                epoch=epoch + 1,
                run_idx=run_idx,
                checkpoint_dir=checkpoint_dir,
                prefix=checkpoint_prefix
            )
            generate_faces_from_latest(
                latent_dim=latent_dim,
                num_samples=16,
                checkpoint_dir=checkpoint_dir,
                samples_dir=SAMPLES_DIR,
                prefix=checkpoint_prefix
            )

    elapsed_total = time.perf_counter() - t0_total
    print(f"\n[TIME] Full training took { _fmt_hms(elapsed_total) }")

    return model


def generate_faces_from_latest(
    latent_dim=64,
    num_samples=16,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    samples_dir: Path = SAMPLES_DIR,
    prefix: str = FILENAME_PREFIX
) -> Path | None:
    latest = _find_latest_checkpoint(checkpoint_dir=checkpoint_dir, prefix=prefix)
    if latest is None:
        print("\nNo checkpoints found.")
        return None

    model = VAE(latent_dim=latent_dim).to(device)
    state = torch.load(latest, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z).cpu()
        grid = utils.make_grid(samples, nrow=4)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")

        stem = latest.stem
        out_path = samples_dir / f"faces_{stem}_N{num_samples}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"\nGenerated faces from {latest.name} -> {out_path}")
        return out_path


if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Number of CPU cores: {os.cpu_count()}")
    vae_model = train_vae(dataset="flwrlabs/celeba", epochs=50, short_run=False)
