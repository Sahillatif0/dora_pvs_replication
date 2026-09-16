"""
Full PyTorch/MONAI Training Pipeline for DoRA-PVS (VICOROBIGR Replication).
Trains a 3D DynUNet on synthetic multi-class data (Background, PVS, Pathology).
"""

import os
import glob
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm

from dynunet_model import PVSSegmentationModel, get_loss_function

class SyntheticPVSDataset(Dataset):
    def __init__(self, data_dir="data/synthetic", patch_size=(96, 96, 96)):
        self.images_dir = os.path.join(data_dir, "imagesTr")
        self.labels_dir = os.path.join(data_dir, "labelsTr")
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, "*.nii.gz")))
        self.label_files = sorted(glob.glob(os.path.join(self.labels_dir, "*.nii.gz")))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_nii = nib.load(self.image_files[idx])
        lbl_nii = nib.load(self.label_files[idx])

        img = img_nii.get_fdata(dtype=np.float32)
        lbl = lbl_nii.get_fdata(dtype=np.float32)

        # Random crop to patch_size for memory-efficient 3D training
        img_patch, lbl_patch = self._random_crop(img, lbl)

        # Add channel dimension: (1, D, H, W)
        img_tensor = torch.from_numpy(img_patch).unsqueeze(0)
        lbl_tensor = torch.from_numpy(lbl_patch).unsqueeze(0).long()

        return img_tensor, lbl_tensor

    def _random_crop(self, img, lbl):
        d, h, w = img.shape
        pd, ph, pw = self.patch_size

        sd = np.random.randint(0, max(1, d - pd + 1)) if d > pd else 0
        sh = np.random.randint(0, max(1, h - ph + 1)) if h > ph else 0
        sw = np.random.randint(0, max(1, w - pw + 1)) if w > pw else 0

        img_crop = img[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        lbl_crop = lbl[sd:sd+pd, sh:sh+ph, sw:sw+pw]

        # Pad if volume is smaller than patch size
        pad_d = max(0, pd - img_crop.shape[0])
        pad_h = max(0, ph - img_crop.shape[1])
        pad_w = max(0, pw - img_crop.shape[2])

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            img_crop = np.pad(img_crop, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
            lbl_crop = np.pad(lbl_crop, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

        return img_crop, lbl_crop

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    dataset = SyntheticPVSDataset(data_dir=args.data_dir, patch_size=(args.patch_size, args.patch_size, args.patch_size))
    if len(dataset) == 0:
        print(f"[Error] No training files found in {args.data_dir}. Run generate_dataset.py first!")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = PVSSegmentationModel(in_channels=1, num_classes=3).to(device)
    loss_fn = get_loss_function()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    print(f"Starting training for {args.epochs} epochs on {len(dataset)} synthetic samples...")
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
            ckpt_path = os.path.join(args.checkpoint_dir, "best_dynunet_multiclass.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best checkpoint to {ckpt_path}")

    print("Training successfully completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Class DynUNet on Synthetic DoRA-PVS data.")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()
    train_model(args)
