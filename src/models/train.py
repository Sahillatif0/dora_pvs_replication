"""
Full PyTorch/MONAI Training Pipeline for DoRA-PVS (VICOROBIGR Replication).
Trains 3D DynUNet on synthetic multi-class data (0: Background, 1: PVS, 2: Pathology/WMH).
Uses White-Matter & Basal-Ganglia-constrained foreground/background patch sampling
to prevent false positives along the cortical ribbon and empty air.
"""

import os
import glob
import random
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np

from dynunet_model import PVSSegmentationModel, get_loss_function

class SyntheticPVSDataset(Dataset):
    def __init__(self, data_dir="data/synthetic", patch_size=(96, 96, 96), samples_per_volume=4):
        self.images_dir = os.path.join(data_dir, "imagesTr")
        self.labels_dir = os.path.join(data_dir, "labelsTr")
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, "*.nii.gz")))
        self.label_files = sorted(glob.glob(os.path.join(self.labels_dir, "*.nii.gz")))
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume

        # Pre-cache coordinates of PVS (Class 1) and Pathology (Class 2) for efficient ROI sampling
        self.cache = []
        print(f"[Dataset] Indexing {len(self.image_files)} training volumes for White-Matter-constrained sampling...")
        for img_path, lbl_path in zip(self.image_files, self.label_files):
            lbl = nib.load(lbl_path).get_fdata(dtype=np.uint8)
            pvs_coords = np.argwhere(lbl == 1)
            wmh_coords = np.argwhere(lbl == 2)
            # General brain tissue for negative sampling (avoiding air)
            tissue_coords = np.argwhere(lbl == 0)
            self.cache.append({
                "img_path": img_path,
                "lbl_path": lbl_path,
                "pvs_coords": pvs_coords,
                "wmh_coords": wmh_coords,
            })
        print(f"[Dataset] Successfully indexed {len(self.cache)} volumes.")

    def __len__(self):
        return len(self.image_files) * self.samples_per_volume

    def __getitem__(self, idx):
        vol_idx = idx % len(self.image_files)
        entry = self.cache[vol_idx]

        img = nib.load(entry["img_path"]).get_fdata(dtype=np.float32)
        lbl = nib.load(entry["lbl_path"]).get_fdata(dtype=np.uint8)

        # White-Matter Constrained Sampling Strategy:
        # 70% chance: crop centered on PVS (Class 1)
        # 20% chance: crop centered on Pathology/WMH (Class 2)
        # 10% chance: crop centered on deep white matter tissue
        rand_choice = np.random.rand()
        pvs_coords = entry["pvs_coords"]
        wmh_coords = entry["wmh_coords"]

        if rand_choice < 0.70 and len(pvs_coords) > 0:
            center = pvs_coords[np.random.randint(len(pvs_coords))]
        elif rand_choice < 0.90 and len(wmh_coords) > 0:
            center = wmh_coords[np.random.randint(len(wmh_coords))]
        elif len(pvs_coords) > 0:
            # Deep white matter neighborhood of PVS
            center = pvs_coords[np.random.randint(len(pvs_coords))]
        else:
            center = [s // 2 for s in img.shape]

        # Add random 3D spatial jitter to prevent center-bias
        jitter = np.random.randint(-16, 17, size=3)
        center = np.clip(center + jitter, 0, np.array(img.shape) - 1)

        img_patch, lbl_patch = self._crop_around_center(img, lbl, center)

        # 3D Random Flips for data augmentation (Axial, Coronal, Sagittal)
        if np.random.rand() > 0.5:
            img_patch = np.flip(img_patch, axis=0)
            lbl_patch = np.flip(lbl_patch, axis=0)
        if np.random.rand() > 0.5:
            img_patch = np.flip(img_patch, axis=1)
            lbl_patch = np.flip(lbl_patch, axis=1)
        if np.random.rand() > 0.5:
            img_patch = np.flip(img_patch, axis=2)
            lbl_patch = np.flip(lbl_patch, axis=2)

        # Add channel dimension: (1, D, H, W)
        img_tensor = torch.from_numpy(img_patch.copy()).unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_patch.copy()).unsqueeze(0).long()

        return img_tensor, lbl_tensor

    def _crop_around_center(self, img, lbl, center):
        pd, ph, pw = self.patch_size
        cz, cy, cx = center

        # Calculate bounding box
        z0 = max(0, cz - pd // 2)
        y0 = max(0, cy - ph // 2)
        x0 = max(0, cx - pw // 2)

        # Adjust bounds so crop stays inside volume if possible
        z0 = min(z0, max(0, img.shape[0] - pd))
        y0 = min(y0, max(0, img.shape[1] - ph))
        x0 = min(x0, max(0, img.shape[2] - pw))

        z1 = min(img.shape[0], z0 + pd)
        y1 = min(img.shape[1], y0 + ph)
        x1 = min(img.shape[2], x0 + pw)

        img_crop = img[z0:z1, y0:y1, x0:x1]
        lbl_crop = lbl[z0:z1, y0:y1, x0:x1]

        # Pad if volume dimension is smaller than patch size
        pad_d = max(0, pd - img_crop.shape[0])
        pad_h = max(0, ph - img_crop.shape[1])
        pad_w = max(0, pw - img_crop.shape[2])

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            img_crop = np.pad(img_crop, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            lbl_crop = np.pad(lbl_crop, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

        return img_crop, lbl_crop

def train_model(args):
    # Set seeds for reproducibility / ensemble diversification
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} (Seed: {args.seed})")

    dataset = SyntheticPVSDataset(
        data_dir=args.data_dir,
        patch_size=(args.patch_size, args.patch_size, args.patch_size),
        samples_per_volume=args.samples_per_volume
    )
    if len(dataset) == 0:
        print(f"[Error] No training files found in {args.data_dir}. Run generate_dataset.py first!")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    model = PVSSegmentationModel(in_channels=1, num_classes=3).to(device)
    loss_fn = get_loss_function()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)

    print(f"Starting training for {args.epochs} epochs ({len(dataset)} samples per epoch, batch size {args.batch_size})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best checkpoint to {ckpt_path} (Loss: {best_loss:.4f})")

    print(f"\n[Done] Training successfully completed! Checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Class DynUNet on Synthetic DoRA-PVS data.")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="best_dynunet_multiclass.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--samples_per_volume", type=int, default=4, help="Number of random crops sampled per volume each epoch")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_model(args)
