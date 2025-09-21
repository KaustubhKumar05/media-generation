# -*- coding: utf-8 -*-
"""
vae.py
VAE training script with optional Google Drive upload support.
Compatible with CPU, CUDA, and MPS.
"""

# pip install torch torchvision matplotlib datasets tqdm pydrive2

# ========================
# Imports & Utils
# ========================
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

import tqdm

CHECKPOINT_DIR = Path("checkpoints")
SAMPLES_DIR = Path("samples")
FILENAME_PREFIX = "vae"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


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
    """
    Return the most recently modified checkpoint file,
    regardless of its filename format.
    """
    candidates = list(checkpoint_dir.glob(f"{prefix}_E*_I*_D*.pt"))
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)



# ========================
# Google Drive Utils (optional)
# ========================
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def get_drive(service_account_json: str) -> GoogleDrive:
    gauth = GoogleAuth()
    gauth.LoadServiceConfigFile({
        "client_config_backend": "service",
        "service_config": {"client_json_file_path": service_account_json}
    })
    gauth.ServiceAuth()
    return GoogleDrive(gauth)


def upload_file(drive: GoogleDrive, local_path: str | Path, folder_id: str | None = None) -> str:
    file_path = Path(local_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found!")

    metadata = {"title": file_path.name}
    if folder_id:
        metadata["parents"] = [{"id": folder_id}]

    gfile = drive.CreateFile(metadata)
    gfile.SetContentFile(str(file_path))
    gfile.Upload()
    print(f"Uploaded {file_path} to Drive (id={gfile['id']})")
    return gfile["id"]


# ========================
# Dataset class
# ========================
from torch.utils.data import Dataset

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


# ========================
# Dataloaders
# ========================
import multiprocessing
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils

# Device auto-selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

BATCH_SIZE = 1024 if device == "cuda" else 128
LR = 5e-4
SEED = 0
LATENT_DIM = 128
WARMUP_EPOCHS = 10
MAX_BETA = 2.0
NUM_WORKERS = min(32, multiprocessing.cpu_count() - 1)

torch.manual_seed(SEED)

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(128),
    transforms.ToTensor(),
])

dataset = "flwrlabs/celeba"
print(f"\nLoading dataset from Hugging Face: {dataset}")
celeba = load_dataset(dataset)

full_train = celeba["train"]
split = full_train.train_test_split(test_size=0.1, seed=SEED)
train_dataset = CelebAHFDataset(split["train"], transform=transform)
val_dataset = CelebAHFDataset(split["test"], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          prefetch_factor=4, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=max(1, NUM_WORKERS // 2),
                        pin_memory=True, persistent_workers=True,
                        prefetch_factor=2)


# ========================
# VAE Model + Loss
# ========================
import torch.nn.functional as F
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()

        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.enc_conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.enc_conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU())

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_conv3(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 256, 16, 16)
        x = F.leaky_relu(self.dec_conv1(x))
        x = F.leaky_relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, epoch):
    recon_loss = F.mse_loss(recon_x.float(), x.float(), reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) / (x.size(0) * mu.size(1))
    beta = min(1.0, epoch / WARMUP_EPOCHS) * MAX_BETA
    return recon_loss + kld * beta, recon_loss, kld, beta


# ========================
# Training Loop
# ========================
def _save_checkpoint(model: nn.Module, epoch: int, run_idx: int,
                     checkpoint_dir: Path, prefix: str,
                     drive: GoogleDrive | None = None, drive_folder_id: str | None = None) -> Path:
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = checkpoint_dir / f"{prefix}_E{epoch:03d}_I{run_idx:03d}_D{datestr}.pt"

    # Converting for a smaller file size
    state_dict_fp16 = {k: (v.half() if torch.is_floating_point(v) else v)
                       for (k, v) in model.state_dict().items()}
    torch.save(state_dict_fp16, path)

    print(f"\nSaved checkpoint: {path}")

    if drive:
        upload_file(drive, path, folder_id=drive_folder_id)

    return path


def generate_faces_from_latest(latent_dim=LATENT_DIM, num_samples=16,
                               checkpoint_dir: Path = CHECKPOINT_DIR, samples_dir: Path = SAMPLES_DIR,
                               prefix: str = FILENAME_PREFIX,
                               drive: GoogleDrive | None = None, drive_folder_id: str | None = None) -> Path | None:
    latest = _find_latest_checkpoint(checkpoint_dir, prefix)
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

        if drive:
            upload_file(drive, out_path, folder_id=drive_folder_id)

        return out_path


def train_vae(epochs=20, latent_dim=LATENT_DIM, short_run=True,
              checkpoint_freq=10, sampling_freq=10,
              checkpoint_dir: Path = CHECKPOINT_DIR, checkpoint_prefix: str = FILENAME_PREFIX,
              service_account_json: str | None = None, drive_folder_id: str | None = None):
    """
    Train VAE model.
    If service_account_json + drive_folder_id are provided,
    checkpoints and sample images will be uploaded to Google Drive.
    """
    torch.backends.cudnn.benchmark = True
    model = VAE(latent_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_idx = _next_run_index(checkpoint_dir=checkpoint_dir, prefix=checkpoint_prefix)

    drive = None
    if service_account_json and drive_folder_id:
        drive = get_drive(service_account_json)
        print(f"Connected to Google Drive (folder id={drive_folder_id})")

    print("\nStarting training…")
    t0_total = time.perf_counter()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs if not short_run else 1):
        t0_epoch = time.perf_counter()

        model.train()
        total_recon, total_kld = 0.0, 0.0

        for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
            data = data.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(device.type != "cpu")):
                recon, mu, logvar = model(data)
                loss, recon_loss, kld, beta = vae_loss(recon, data, mu, logvar, epoch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_recon += recon_loss.item()
            total_kld += kld.item()

            if short_run and batch_idx > 10:
                break

        scheduler.step()

        avg_train_loss = (total_recon + beta * total_kld) / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kld = total_kld / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, _ in val_loader:
                val_data = val_data.to(device)
                recon, mu, logvar = model(val_data)
                current_val_loss, _, _, _ = vae_loss(recon, val_data, mu, logvar, epoch)
                val_loss += current_val_loss.item()
        avg_val_loss = val_loss / len(val_loader)

        elapsed_epoch = time.perf_counter() - t0_epoch

        print(f"Epoch {epoch:03d} | Train: {avg_train_loss:.4f} "
              f"(Recon: {avg_recon:.4f}, KL: {avg_kld:.4f}, β={beta:.2f}) "
              f"| Val: {avg_val_loss:.4f} | [TIME] {_fmt_hms(elapsed_epoch)}")

        if (checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0) or epoch == epochs - 1:
            _save_checkpoint(model, epoch + 1, run_idx, checkpoint_dir, checkpoint_prefix, drive, drive_folder_id)

        if (sampling_freq > 0 and (epoch + 1) % sampling_freq == 0) or epoch == epochs - 1:
            generate_faces_from_latest(latent_dim, 16, checkpoint_dir, SAMPLES_DIR, checkpoint_prefix,
                                       drive=drive, drive_folder_id=drive_folder_id)

    elapsed_total = time.perf_counter() - t0_total
    print(f"\n[TIME] Full training took {_fmt_hms(elapsed_total)}")

    return model


# ========================
# Main
# ========================
if __name__ == "__main__":
    vae_model = train_vae(
        epochs=20,
        checkpoint_freq=10,
        sampling_freq=10,
        short_run=False,
        # Optional:
        service_account_json="json file path here",
        drive_folder_id="folder id here"
    )
    # generate_faces_from_latest()
