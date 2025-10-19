# -*- coding: utf-8 -*-
"""
gan.py
GAN training script with HF upload support, improved checkpointing, and sampling.
Compatible with CPU, CUDA, and MPS.
"""

import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import HfApi, login

# ========================
# Configuration
# ========================
CHECKPOINT_DIR = Path("checkpoints")
SAMPLES_DIR = Path("samples")
FILENAME_PREFIX = "gan"
HF_REPO_ID = "kozonhf/uploads"
HF_SUBDIR = "gan"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 1024 if device == "cuda" else 256
G_LR = 1e-4
D_LR = 3e-4
NUM_WORKERS = os.cpu_count() - 2
SEED = 0
LATENT_DIM = 128
IMG_DIM = 128 * 128 * 3  # 128x128 RGB images

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ========================
# HF Upload Utilities
# ========================
hf_api = HfApi()


def hf_upload(local_path: str | Path,
              repo_id: str = HF_REPO_ID,
              subdir: str = HF_SUBDIR,
              repo_type: str = "model") -> str:
    """
    Upload any file (weights, images, logs) to a single HF repo.
    Files are organized by subdir in the repo.
    """
    local_path = Path(local_path)
    path_in_repo = f"{subdir}/{local_path.name}" if subdir else local_path.name
    commit_url = hf_api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded {local_path} → {repo_id}/{path_in_repo}")
    return commit_url


# ========================
# Training Utilities
# ========================
def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _next_run_index(checkpoint_dir: Path = CHECKPOINT_DIR, prefix: str = FILENAME_PREFIX) -> int:
    pattern = re.compile(rf"{re.escape(prefix)}_E\d+_I(\d+)_D\d{{8}}-\d{{6}}\.pt$")
    max_idx = 0
    for p in checkpoint_dir.glob(f"{prefix}_E*_I*_D*.pt"):
        m = pattern.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _find_latest_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR,
                            prefix: str = FILENAME_PREFIX) -> Path | None:
    candidates = list(checkpoint_dir.glob(f"{prefix}_E*_I*_D*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _save_checkpoint(model_state: dict, epoch: int, run_idx: int,
                     checkpoint_dir: Path = CHECKPOINT_DIR, prefix: str = FILENAME_PREFIX) -> Path:
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = checkpoint_dir / f"{prefix}_E{epoch:03d}_I{run_idx:03d}_D{datestr}.pt"

    # Save weights in FP16 for compact size (convert back to FP32 for generator)
    state_dict_fp16 = {}
    for k, v in model_state.items():
        if k in ['generator', 'discriminator']:
            state_dict_fp16[k] = {k2: (v2.half() if torch.is_floating_point(v2) else v2)
                                  for k2, v2 in v.items()}
        else:
            state_dict_fp16[k] = v

    torch.save(state_dict_fp16, path)
    print(f"\nSaved checkpoint: {path}")

    if HF_REPO_ID:
        hf_upload(path, HF_REPO_ID, subdir=HF_SUBDIR + "/checkpoints")
    return path


# ========================
# Models
# ========================
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x).view(-1, 3, 128, 128)  # Reshape to image


# ========================
# Dataset
# ========================
class CelebAHFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label


# ========================
# Training Functions
# ========================
def generate_faces_from_latest(
        latent_dim=LATENT_DIM,
        num_samples=16,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        samples_dir: Path = SAMPLES_DIR,
        prefix: str = FILENAME_PREFIX
) -> Path | None:
    latest = _find_latest_checkpoint(checkpoint_dir=checkpoint_dir, prefix=prefix)
    if latest is None:
        print("\nNo checkpoints found.")
        return None

    # Initialize generator
    generator = Generator(latent_dim, IMG_DIM).to(device)

    state = torch.load(latest, map_location=device)
    # Convert FP16 weights back to FP32 for inference
    generator_state = {k: v.float() for k, v in state['generator'].items()}
    generator.load_state_dict(generator_state)
    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = generator(z).cpu()
        grid = utils.make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))

        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0) * 0.5 + 0.5)  # Denormalize
        plt.axis("off")

        stem = latest.stem
        out_path = samples_dir / f"faces_{stem}_N{num_samples}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"\nGenerated faces from {latest.name} -> {out_path}")

        if HF_REPO_ID:
            hf_upload(out_path, HF_REPO_ID, subdir=HF_SUBDIR + "/samples")
        return out_path


def train_gan(
        dataset="flwrlabs/celeba",
        epochs=20,
        batch_size=BATCH_SIZE,
        short_run=False,
        checkpoint_freq=10,
        sampling_freq=5,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        checkpoint_prefix: str = FILENAME_PREFIX,
):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    print(f"\nLoading dataset from Hugging Face: {dataset}")
    celeba = load_dataset(dataset)

    full_train = celeba["train"]
    split = full_train.train_test_split(test_size=0.1, seed=42)
    train_dataset = CelebAHFDataset(split["train"], transform=transform)
    val_dataset = CelebAHFDataset(split["test"], transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == 'cuda',
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, NUM_WORKERS // 2),
        pin_memory=device.type == 'cuda',
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2
    )

    # Initialize models
    generator = Generator(LATENT_DIM, IMG_DIM).to(device)
    discriminator = Discriminator(IMG_DIM).to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=G_LR, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    run_idx = _next_run_index(checkpoint_dir=checkpoint_dir, prefix=checkpoint_prefix)

    # Fixed noise for sample generation
    fixed_noise = torch.randn(16, LATENT_DIM, device=device)

    print("\nStarting training…")
    t0_total = time.perf_counter()

    for epoch in range(epochs if not short_run else 1):
        t0_epoch = time.perf_counter()

        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        total_batches = 0

        generator.train()
        discriminator.train()

        for batch_idx, (real_imgs, _) in enumerate(tqdm(train_loader)):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device, non_blocking=True)

            # Create labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            d_optimizer.zero_grad()

            # Real images
            real_outputs = discriminator(real_imgs)
            d_loss_real = criterion(real_outputs, real_labels)

            # Fake images
            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_imgs = generator(noise)
            fake_outputs = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_imgs = generator(noise)
            fake_outputs = discriminator(fake_imgs)
            g_loss = -torch.log(fake_outputs + 1e-8).mean()  # Maximize log(D(G(z)))


            g_loss.backward()
            g_optimizer.step()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            total_batches += 1

            if short_run and batch_idx > 10:
                break

        # Calculate average losses
        avg_g_loss = g_loss_epoch / total_batches
        avg_d_loss = d_loss_epoch / total_batches

        # Validation
        generator.eval()
        discriminator.eval()
        val_g_loss = 0.0
        val_d_loss = 0.0
        val_total_batches = 0.0

        with torch.no_grad():
            for val_data, _ in val_loader:
                val_batch_size = val_data.size(0)
                val_data = val_data.to(device, non_blocking=True)

                # Validation discriminator loss
                real_val_outputs = discriminator(val_data)
                val_d_loss_real = criterion(real_val_outputs, torch.ones(val_batch_size, 1, device=device))

                val_noise = torch.randn(val_batch_size, LATENT_DIM, device=device)
                val_fake_imgs = generator(val_noise)
                val_fake_outputs = discriminator(val_fake_imgs)
                val_d_loss_fake = criterion(val_fake_outputs, torch.zeros(val_batch_size, 1, device=device))
                val_d_loss += (val_d_loss_real + val_d_loss_fake).item()

                # Validation generator loss
                val_g_loss += criterion(val_fake_outputs, torch.ones(val_batch_size, 1, device=device)).item()
                val_total_batches += 1

        avg_val_g_loss = val_g_loss / val_total_batches
        avg_val_d_loss = val_d_loss / val_total_batches

        elapsed_epoch = time.perf_counter() - t0_epoch

        # Print progress
        print(f"Epoch {epoch:03d} | G: {avg_g_loss:.4f} | D: {avg_d_loss:.4f} | "
              f"Val G: {avg_val_g_loss:.4f} | Val D: {avg_val_d_loss:.4f} | [TIME] {_fmt_hms(elapsed_epoch)}")

        # Generate and log sample images
        if epoch % sampling_freq == 0:
            with torch.no_grad():
                sample_imgs = generator(fixed_noise)
                img_grid = utils.make_grid(sample_imgs, nrow=4, normalize=True)

        # Save checkpoint and generate samples
        if (checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0) or epoch == epochs - 1:
            model_state = {
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
            }
            _save_checkpoint(
                model_state=model_state,
                epoch=epoch + 1,
                run_idx=run_idx,
                checkpoint_dir=checkpoint_dir,
                prefix=checkpoint_prefix
            )

        if (sampling_freq > 0 and (epoch + 1) % sampling_freq == 0) or epoch == epochs - 1:
            generate_faces_from_latest(
                num_samples=16,
                checkpoint_dir=checkpoint_dir,
                samples_dir=SAMPLES_DIR,
                prefix=checkpoint_prefix
            )

    elapsed_total = time.perf_counter() - t0_total
    print(f"\n[TIME] Full training took {_fmt_hms(elapsed_total)}")

    return {'generator': generator, 'discriminator': discriminator}


# ========================
# Main
# ========================
if __name__ == "__main__":
    # Get token from https://huggingface.co/settings/tokens
    login(token="token_goes_here")

    gan_model = train_gan(
        dataset="flwrlabs/celeba",
        epochs=100,
        short_run=False,
        checkpoint_freq=20,
        sampling_freq=20
    )
    # generate_faces_from_latest()