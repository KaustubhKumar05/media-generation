import re
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

CHECKPOINT_DIR = Path("checkpoints")
SAMPLES_DIR = Path("samples")
FILENAME_PREFIX = "vae"


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _next_run_index(checkpoint_dir: Path = CHECKPOINT_DIR, prefix: str = FILENAME_PREFIX) -> int:
    """
    Scan existing checkpoints and return the next run index (1-based).
    Looks for files like: {prefix}_E###_I###_DYYYYMMDD-HHMMSS.pt
    """
    pattern = re.compile(rf"{re.escape(prefix)}_E\d+_I(\d+)_D\d{{8}}-\d{{6}}\.pt$")
    max_idx = 0
    for p in checkpoint_dir.glob(f"{prefix}_E*_I*_D*.pt"):
        m = pattern.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _save_checkpoint(model: nn.Module, epoch: int, run_idx: int,
                     checkpoint_dir: Path = CHECKPOINT_DIR,
                     prefix: str = FILENAME_PREFIX) -> Path:
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = checkpoint_dir / f"{prefix}_E{epoch:03d}_I{run_idx:03d}_D{datestr}.pt"

    state_dict_fp16 = {k: (v.half() if torch.is_floating_point(v) else v) for (k,v) in model.state_dict().items()}
    torch.save(state_dict_fp16, path)

    print(f"\nSaved checkpoint: {path}")
    return path


def _find_latest_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR,
                            prefix: str = FILENAME_PREFIX) -> Path | None:
    """
    Find the latest checkpoint file, preferring parsed (epoch, index, date),
    but falling back to modification time when needed.
    """
    candidates = list(checkpoint_dir.glob(f"{prefix}_E*_I*_D*.pt"))
    if not candidates:
        return None

    rx = re.compile(rf"{re.escape(prefix)}_E(\d+)_I(\d+)_D(\d{{8}}-\d{{6}})\.pt$")

    def key(p: Path):
        m = rx.match(p.name)
        if m:
            e, i, ds = int(m.group(1)), int(m.group(2)), m.group(3)
            # Include st_mtime as final tie-breaker
            return (e, i, ds, p.stat().st_mtime)
        else:
            # Unparseable → rank lowest, sorted by modified time
            return (0, 0, "00000000-000000", p.stat().st_mtime)

    return max(candidates, key=key)
