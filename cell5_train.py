"""
Cell 5 Standalone: SimCLR Training + Embeddings Extraction
Saves: ./output/simclr_model.pt, ./output/embeddings.npy, ./output/train_labels.npy
"""

import os
import random
import warnings
import sys
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18

warnings.filterwarnings("ignore")

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR = "./data"
OUTPUT_DIR = "./output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── SimCLR hyperparameters (paper Appendix F.1) ───────────────────────────────
SIMCLR_EPOCHS = 500
SIMCLR_BATCH_SIZE = 512
SIMCLR_LR = 0.4
SIMCLR_MOMENTUM = 0.9
SIMCLR_WD = 1e-4
SIMCLR_TEMP = 0.5
PROJ_DIM = 128  # projection head output dimension
EMBED_DIM = 512  # ResNet-18 penultimate layer dimension

# ── CIFAR-10 normalisation constants ─────────────────────────────────────────
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── SimCLR Model Architecture ─────────────────────────────────────────────────
class SimCLRProjectionHead(nn.Module):
    """2-layer MLP: EMBED_DIM → EMBED_DIM (ReLU) → PROJ_DIM."""
    def __init__(self, in_dim: int = EMBED_DIM, hidden_dim: int = EMBED_DIM,
                 out_dim: int = PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SimCLR(nn.Module):
    """ResNet-18 backbone + SimCLR projection head.
    forward() returns:
    h : (B, 512) penultimate-layer features (L2-normalised at extraction time)
    z : (B, 128) projection-head outputs (used only during contrastive training)
    """
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        # Drop the final linear classifier; keep everything up to and including avgpool
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        self.projector = SimCLRProjectionHead()

    def forward(self, x: torch.Tensor):
        h = self.encoder(x).flatten(1)  # (B, 512)
        z = self.projector(h)  # (B, 128)
        return h, z

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = SIMCLR_TEMP) -> torch.Tensor:
    """NT-Xent contrastive loss (Chen et al., 2020).
    Given N paired views (z1_i, z2_i), constructs a 2N * 2N similarity matrix
    and treats each pair as a positive while all 2(N-1) others are negatives.
    Args:
        z1, z2 : (N, D) raw (un-normalised) projection vectors
        temperature : τ scaling factor
    Returns:
        scalar loss
    """
    N = z1.size(0)
    # L2-normalise both views
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)

    # Full pairwise cosine-similarity matrix, scaled by temperature
    sim = torch.mm(z, z.T) / temperature  # (2N, 2N)

    # Mask out diagonal (self-similarity = trivial positive)
    diag_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(diag_mask, float("-inf"))

    # Positive-pair labels:
    # row i → positive at i+N
    # row i+N → positive at i
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N, device=z.device),
    ])  # (2N,)

    return F.cross_entropy(sim, labels)

# ── Data Augmentation ─────────────────────────────────────────────────────────
class TwoViewTransform:
    """Wraps a transform and applies it independently twice to produce two augmented views."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def get_simclr_transform() -> T.Compose:
    """SimCLR augmentation pipeline for CIFAR-10 (32 * 32 px).
    Matches paper Appendix F.1 exactly.
    """
    colour_jitter = T.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    )
    return T.Compose([
        T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([colour_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        # Gaussian blur: kernel_size = 0.1 * image_size ≈ 3 for CIFAR-10
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])

def get_eval_transform() -> T.Compose:
    """Plain deterministic transform for embedding extraction."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ])

# ── Training + Extraction ─────────────────────────────────────────────────────
def train_simclr() -> tuple[np.ndarray, np.ndarray]:
    """Train SimCLR on the full (unlabelled) CIFAR-10 training set.
    Saves:
    output/simclr_model.pt — trained model state dict
    output/embeddings.npy — (50000, 512) L2-normalised penultimate embeddings
    output/train_labels.npy — (50000,) ground-truth labels (oracle, used only for eval)
    Returns:
    embeddings : (N_TRAIN, EMBED_DIM) float32 array
    train_labels : (N_TRAIN,) int array
    """
    ckpt_path = os.path.join(OUTPUT_DIR, "simclr_model.pt")
    embeddings_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
    labels_path = os.path.join(OUTPUT_DIR, "train_labels.npy")

    # ── Load from disk if available ───────────────────────────────────────────
    if os.path.exists(embeddings_path) and os.path.exists(labels_path):
        print("[SimCLR] Checkpoint found — loading embeddings from disk.")
        return np.load(embeddings_path), np.load(labels_path)

    set_seed(42)

    # ── Dataset with two-view transform ───────────────────────────────────────
    train_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True,
        transform=TwoViewTransform(get_simclr_transform()),
    )
    loader = DataLoader(
        train_ds,
        batch_size=SIMCLR_BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # Windows-safe with spawn
        pin_memory=True,
        drop_last=True,  # keeps all batches the same size for NT-Xent
    )

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    model = SimCLR().to(DEVICE)
    optimiser = optim.SGD(
        model.parameters(),
        lr=SIMCLR_LR, momentum=SIMCLR_MOMENTUM, weight_decay=SIMCLR_WD,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=SIMCLR_EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    print(f"[SimCLR] Starting training for {SIMCLR_EPOCHS} epochs …")
    for epoch in range(1, SIMCLR_EPOCHS + 1):
        epoch_loss = 0.0
        n_batches = 0
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            optimiser.zero_grad()
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent_loss(z1, z2, SIMCLR_TEMP)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            avg = epoch_loss / n_batches
            lr = scheduler.get_last_lr()[0]
            print(f" Epoch [{epoch:3d}/{SIMCLR_EPOCHS}] avg_loss={avg:.4f} lr={lr:.6f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"[SimCLR] Model saved → {ckpt_path}")

    # ── Extract and save embeddings ───────────────────────────────────────────
    embeddings, train_labels = _extract_embeddings(model)
    np.save(embeddings_path, embeddings)
    np.save(labels_path, train_labels)
    print(f"[SimCLR] Embeddings saved → {embeddings_path} shape={embeddings.shape}")

    return embeddings, train_labels

def _extract_embeddings(model: SimCLR) -> tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalised 512-dim penultimate-layer representations from a
    trained SimCLR model for all 50,000 CIFAR-10 training images.
    """
    ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True,
        transform=get_eval_transform(),
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                       num_workers=8, pin_memory=True)
    model.eval()
    all_h, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            h, _ = model(x)
            h = F.normalize(h, dim=1)  # L2-normalise (paper Appendix F.1)
            all_h.append(h.cpu().numpy())
            all_y.append(y.numpy())
    return np.concatenate(all_h), np.concatenate(all_y)

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Windows multiprocessing fix
    if sys.platform == "win32":
        mp.set_start_method('spawn', force=True)
    
    print("=" * 60)
    print("Step 1 — SimCLR Representation Learning")
    print("=" * 60)
    
    embeddings, train_labels = train_simclr()
    
    print(f"Embeddings : {embeddings.shape} dtype={embeddings.dtype}")
    print(f"Labels : {train_labels.shape} classes={np.unique(train_labels)}")
    print(f"L2 norms : min={np.linalg.norm(embeddings, axis=1).min():.4f} "
          f"max={np.linalg.norm(embeddings, axis=1).max():.4f}")
    print(f"Class distribution: {np.bincount(train_labels)}")
    
    print("Cell 5 complete! Load in Jupyter:")
    print("embeddings = np.load('./output/embeddings.npy')")
    print("train_labels = np.load('./output/train_labels.npy')")
