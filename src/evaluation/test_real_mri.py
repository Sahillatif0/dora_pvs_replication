"""
Test Trained DynUNet on Real Human Brain MRI (MNI152 Nonlinear 2009c Asymmetric).
Demonstrates:
1. Raw Zero-shot prediction on real MRI.
2. Clinical Standard: Masking with Anatomical White Matter + Basal Ganglia ROI (CSO/BG).
"""

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import datasets

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dynunet_model import PVSSegmentationModel
from parcellation import get_brain_parcellation_mask

def test_on_real_mri():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Running real MRI inference on: {device}")

    # 1. Fetch real human MNI152 ICBM 2009c T1w MRI and tissue masks
    print("[1/4] Fetching real human MNI152 ICBM 2009c T1w MRI & tissue atlas...")
    mni = datasets.fetch_icbm152_2009()
    real_mri_nii = nib.load(mni.t1)
    real_mri_data = real_mri_nii.get_fdata(dtype=np.float32)

    # Tissue maps (WM probability)
    wm_map = nib.load(mni.wm).get_fdata(dtype=np.float32)
    # Binary White Matter & subcortical mask (threshold 0.5)
    wm_mask = (wm_map > 0.4).astype(np.float32)

    # 2. Intensity Normalization
    print("[2/4] Preprocessing & intensity normalization...")
    brain_mask = real_mri_data > 10.0
    mean_val = np.mean(real_mri_data[brain_mask])
    std_val = np.std(real_mri_data[brain_mask]) + 1e-6
    norm_mri = (real_mri_data - mean_val) / std_val

    # 3. Load Trained Model
    checkpoint_path = "checkpoints/best_dynunet_multiclass.pth"
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found at {checkpoint_path}")
        return

    print(f"[3/4] Loading trained DynUNet weights from: {checkpoint_path}")
    model = PVSSegmentationModel(in_channels=1, num_classes=3).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    wm_mask_vol = get_brain_parcellation_mask(real_mri_nii)

    # Crop an ROI covering Centrum Semiovale
    cz, cy, cx = 96, 120, 96
    pz, py, px = 96, 96, 96
    z0 = cz - pz // 2
    y0 = cy - py // 2
    x0 = cx - px // 2

    mri_roi = norm_mri[z0:z0+pz, y0:y0+py, x0:x0+px]
    raw_roi = real_mri_data[z0:z0+pz, y0:y0+py, x0:x0+px]
    wm_roi = wm_mask_vol[z0:z0+pz, y0:y0+py, x0:x0+px]

    tensor_input = torch.from_numpy(mri_roi).unsqueeze(0).unsqueeze(0).to(device)

    print("[4/4] Running 3D inference on real brain tissue...")
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pvs_prob = probs[1]  # Class 1: PVS

    decision_thr = 0.45
    raw_pvs_binary = (pvs_prob > decision_thr).astype(np.float32)
    # Apply automated parcellation mask (Centrum Semiovale / Basal Ganglia White Matter)
    masked_pvs_binary = raw_pvs_binary * wm_roi

    print(f"[Result] Total raw detections: {int(np.sum(raw_pvs_binary))} voxels")
    print(f"[Result] Detections inside White Matter (CSO/BG): {int(np.sum(masked_pvs_binary))} voxels")

    # Plot 4-column comparison
    slices = [36, 48, 60]
    fig, axes = plt.subplots(len(slices), 4, figsize=(18, 12), facecolor="#0d1117")

    for row_idx, z in enumerate(slices):
        img_slice = raw_roi[z]
        norm_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-6)

        # 1. Real MRI Slice
        axes[row_idx, 0].imshow(img_slice, cmap="gray", origin="lower")
        axes[row_idx, 0].set_title(f"Real Human T1w MRI (Z={z0+z})", color="white", fontsize=11)
        axes[row_idx, 0].axis("off")

        # 2. Probability Heatmap
        axes[row_idx, 1].imshow(pvs_prob[z], cmap="hot", vmin=0, vmax=1.0, origin="lower")
        axes[row_idx, 1].set_title(f"PVS Probability (Max: {np.max(pvs_prob[z]):.2f})", color="#ffaa00", fontsize=11)
        axes[row_idx, 1].axis("off")

        # 3. Raw Prediction Overlay (shows ventricular edge sensitivity)
        overlay_raw = np.zeros((*img_slice.shape, 3), dtype=float)
        overlay_raw[..., 0] = norm_slice * 0.8
        overlay_raw[..., 1] = norm_slice * 0.8
        overlay_raw[..., 2] = norm_slice * 0.8
        raw_mask = raw_pvs_binary[z] > 0
        overlay_raw[raw_mask, 0] = 1.0
        axes[row_idx, 2].imshow(overlay_raw, origin="lower")
        axes[row_idx, 2].set_title(f"Unmasked Detections\n({int(np.sum(raw_mask))} voxels)", color="#ff4444", fontsize=11)
        axes[row_idx, 2].axis("off")

        # 4. Clinically Masked Overlay (Centrum Semiovale / White Matter)
        overlay_clean = np.zeros((*img_slice.shape, 3), dtype=float)
        overlay_clean[..., 0] = norm_slice * 0.8
        overlay_clean[..., 1] = norm_slice * 0.8
        overlay_clean[..., 2] = norm_slice * 0.8
        clean_mask = masked_pvs_binary[z] > 0
        overlay_clean[clean_mask, 0] = 0.0  # Blue-cyan highlight
        overlay_clean[clean_mask, 1] = 1.0
        overlay_clean[clean_mask, 2] = 0.5
        axes[row_idx, 3].imshow(overlay_clean, origin="lower")
        axes[row_idx, 3].set_title(f"Centrum Semiovale PVS\n({int(np.sum(clean_mask))} voxels)", color="#00ff88", fontsize=11)
        axes[row_idx, 3].axis("off")

    plt.suptitle("Clinical PVS Evaluation on Real Human Brain MRI: Raw vs White Matter Masked", 
                 color="white", fontsize=15, y=0.98)
    plt.tight_layout()

    out_path = "data/real_mri_detection_visual.png"
    os.makedirs("data", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[Success] Updated real MRI visualization saved to: {out_path}")

if __name__ == "__main__":
    test_on_real_mri()
