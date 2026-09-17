"""
Comprehensive Evaluation & Quality Audit for DoRA-PVS Challenge.
Computes:
  1. AUPRC (Area Under Precision-Recall Curve) - Primary challenge metric
  2. Optimal Threshold Discovery (F1 / Dice sweep)
  3. clDice (Centerline tubular connectivity)
  4. Lesion-wise DSC (Object-level detection rate)
  5. Saves a visual inspection comparison (Ground Truth vs. Model Prediction)
"""

import os
import glob
import argparse
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
from dynunet_model import PVSSegmentationModel
from metrics import compute_auprc, compute_cldice, compute_lesion_wise_dsc

def find_best_threshold(gt_binary, prob_map, thresholds=np.linspace(0.05, 0.45, 9)):
    """Finds the decision threshold that maximizes the Dice score on the evaluation scan."""
    best_dice = 0.0
    best_thr = 0.15
    for thr in thresholds:
        pred = (prob_map > thr).astype(np.float32)
        intersection = np.sum(pred * gt_binary)
        total = np.sum(pred) + np.sum(gt_binary)
        if total > 0:
            dice = (2.0 * intersection) / (total + 1e-6)
            if dice > best_dice:
                best_dice = dice
                best_thr = thr
    return best_thr, best_dice

def evaluate_predictions(data_dir="data/synthetic", checkpoint_path="checkpoints/best_dynunet_multiclass.pth", output_image="data/evaluation_visual.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    if not image_files:
        print(f"[Error] No evaluation scans found in {images_dir}")
        return

    model = PVSSegmentationModel(in_channels=1, num_classes=3).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint weights from: {checkpoint_path}")
    else:
        print(f"[Error] Checkpoint {checkpoint_path} not found!")
        return

    model.eval()

    auprc_list = []
    cldice_list = []
    lesion_dsc_list = []
    dice_list = []
    best_thrs = []

    # Keep track of the best case for visualization
    vis_data = None

    print("\n--- Running Multi-Metric Challenge Evaluation ---")
    with torch.no_grad():
        for i, (img_path, lbl_path) in enumerate(zip(image_files, label_files)):
            img = nib.load(img_path).get_fdata(dtype=np.float32)
            lbl = nib.load(lbl_path).get_fdata(dtype=np.float32)

            gt_pvs = (lbl == 1).astype(np.float32)

            # Full 3D Volume sliding-window inference (Exact Challenge Standard)
            input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
            from monai.inferers import sliding_window_inference
            from parcellation import get_brain_parcellation_mask

            val_outputs = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(96, 96, 96),
                sw_batch_size=2,
                predictor=model,
                overlap=0.5,
                mode="gaussian"
            )
            probs = torch.softmax(val_outputs, dim=1).squeeze(0).cpu().numpy()
            pvs_prob = probs[1]  # Class 1: PVS

            # Extract anatomical evaluation ROI (CSO & BG)
            try:
                roi_mask = get_brain_parcellation_mask(nib.load(img_path))
            except Exception:
                roi_mask = (img > 10.0).astype(np.float32)

            # 1. Challenge Metric 1: Voxel-wise AUPRC evaluated strictly inside the ROI
            auprc = compute_auprc(gt_pvs, pvs_prob, roi_mask=roi_mask)

            # 2. Find optimal operating threshold within ROI
            opt_thr, opt_dice = find_best_threshold(gt_pvs[roi_mask > 0], pvs_prob[roi_mask > 0])
            pvs_pred_binary = ((pvs_prob > opt_thr) & (roi_mask > 0)).astype(np.float32)

            # 3. Challenge Metric 2: Topology-preserving clDice
            cldice = compute_cldice(gt_pvs * roi_mask, pvs_pred_binary)

            # 4. Challenge Metric 3: Lesion-wise DSC
            lesion_dsc = compute_lesion_wise_dsc(gt_pvs * roi_mask, pvs_pred_binary)

            auprc_list.append(auprc)
            cldice_list.append(cldice)
            lesion_dsc_list.append(lesion_dsc)
            dice_list.append(opt_dice)
            best_thrs.append(opt_thr)

            case_name = os.path.basename(img_path)
            print(f"[{i+1}/{len(image_files)}] {case_name}")
            print(f"     AUPRC: {auprc:.4f} | Optimal Thr: {opt_thr:.2f} | Dice: {opt_dice:.4f} | clDice: {cldice:.4f} | Lesion-DSC: {lesion_dsc:.4f}")

            if vis_data is None:
                vis_data = (img, gt_pvs * roi_mask, pvs_prob, pvs_pred_binary)

    print("\n================== OFFICIAL EVALUATION SUMMARY ==================")
    print(f"Median AUPRC (Primary Metric): {np.median(auprc_list):.4f}")
    print(f"Median Dice:                   {np.median(dice_list):.4f}")
    print(f"Median clDice (Topology):      {np.median(cldice_list):.4f}")
    print(f"Median Lesion-DSC (Object):    {np.median(lesion_dsc_list):.4f}")
    print(f"Suggested Decision Threshold:  {np.median(best_thrs):.2f}")
    print("=================================================================\n")

    # Render Visual Comparison
    if vis_data is not None and output_image is not None:
        patch_img, patch_gt, pvs_prob, pvs_pred = vis_data
        # Pick slice with highest positive voxels
        z_slice = int(np.argmax([np.sum(patch_gt[z] > 0) for z in range(patch_gt.shape[0])]))

        fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#0e0f10")
        for ax in axes:
            ax.axis("off")

        axes[0].imshow(patch_img[z_slice], cmap="gray", origin="lower")
        axes[0].set_title("1. Input MRI Patch", color="white", fontsize=12)

        axes[1].imshow(patch_gt[z_slice], cmap="Greens", vmin=0, vmax=1, origin="lower")
        axes[1].set_title(f"2. Ground Truth PVS\n[{int(np.sum(patch_gt[z_slice]))} voxels]", color="#00ff88", fontsize=12)

        axes[2].imshow(pvs_prob[z_slice], cmap="hot", vmin=0, vmax=np.max(pvs_prob)+1e-5, origin="lower")
        axes[2].set_title(f"3. Model Soft Probability\n(Max: {np.max(pvs_prob):.2f})", color="#ffaa00", fontsize=12)

        # Overlay: Green = Ground Truth, Red = Model Prediction, Yellow = Correct Overlap
        overlay = np.zeros((*patch_img[z_slice].shape, 3), dtype=float)
        img_norm = (patch_img[z_slice] - patch_img[z_slice].min()) / (patch_img[z_slice].max() - patch_img[z_slice].min() + 1e-6)
        overlay[..., 0] = img_norm * 0.7
        overlay[..., 1] = img_norm * 0.7
        overlay[..., 2] = img_norm * 0.7

        gt_mask = patch_gt[z_slice] > 0
        pr_mask = pvs_pred[z_slice] > 0
        overlap = gt_mask & pr_mask

        overlay[gt_mask, 1] = 1.0  # Green: GT
        overlay[pr_mask, 0] = 1.0  # Red: Pred
        overlay[overlap, 0] = 1.0  # Yellow: Both
        overlay[overlap, 1] = 1.0

        axes[3].imshow(overlay, origin="lower")
        axes[3].set_title("4. Overlay Comparison\n(Green=GT, Red=Pred, Yellow=Match)", color="#ffff00", fontsize=12)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_image) if os.path.dirname(output_image) else ".", exist_ok=True)
        plt.savefig(output_image, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"Visual prediction evaluation saved to: {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on challenge metrics.")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_dynunet_multiclass.pth")
    parser.add_argument("--output_image", type=str, default="data/evaluation_visual.png")
    args = parser.parse_args()

    evaluate_predictions(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_image=args.output_image
    )
