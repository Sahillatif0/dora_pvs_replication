"""
Evaluation Runner Script.
Runs trained DynUNet on synthetic/test scans and computes the 3 DoRA-PVS challenge metrics:
1. Voxel-wise AUPRC
2. Topology-preserving clDice
3. Lesion-wise DSC (one-inside criterion)
"""

import os
import glob
import argparse
import torch
import nibabel as nib
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
from dynunet_model import PVSSegmentationModel
from metrics import compute_auprc, compute_cldice, compute_lesion_wise_dsc

def evaluate_predictions(data_dir="data/synthetic", checkpoint_path="checkpoints/best_dynunet_multiclass.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    if not image_files:
        print("[Error] No files found for evaluation.")
        return

    model = PVSSegmentationModel(in_channels=1, num_classes=3).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"[Warning] Checkpoint {checkpoint_path} not found, running random weights evaluation.")

    model.eval()

    auprc_list, cldice_list, lesion_dsc_list = [], [], []

    print("\n--- Running Multi-Metric Challenge Evaluation ---")
    with torch.no_grad():
        for i, (img_path, lbl_path) in enumerate(zip(image_files, label_files)):
            img = nib.load(img_path).get_fdata(dtype=np.float32)
            lbl = nib.load(lbl_path).get_fdata(dtype=np.float32)

            # Extract PVS binary ground truth (Class 1)
            gt_pvs = (lbl == 1).astype(np.float32)

            # Model prediction: crop center 96x96x96 for test evaluation
            d, h, w = img.shape
            pd, ph, pw = 96, 96, 96
            sd, sh, sw = (d - pd) // 2, (h - ph) // 2, (w - pw) // 2
            patch_img = img[sd:sd+pd, sh:sh+ph, sw:sw+pw]
            patch_gt  = gt_pvs[sd:sd+pd, sh:sh+ph, sw:sw+pw]

            input_tensor = torch.from_numpy(patch_img).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            # Probabilities for Class 1 (PVS)
            pvs_prob = probs[1]
            pvs_pred_binary = (pvs_prob > 0.5).astype(np.float32)

            # Compute the three metrics
            auprc = compute_auprc(patch_gt, pvs_prob)
            cldice = compute_cldice(patch_gt, pvs_pred_binary)
            lesion_dsc = compute_lesion_wise_dsc(patch_gt, pvs_pred_binary)

            auprc_list.append(auprc)
            cldice_list.append(cldice)
            lesion_dsc_list.append(lesion_dsc)

            case_name = os.path.basename(img_path)
            print(f"[{i+1}/{len(image_files)}] {case_name} -> AUPRC: {auprc:.4f} | clDice: {cldice:.4f} | Lesion-DSC: {lesion_dsc:.4f}")

    print("\n================== EVALUATION SUMMARY ==================")
    print(f"Median AUPRC:       {np.median(auprc_list):.4f}")
    print(f"Median clDice:      {np.median(cldice_list):.4f}")
    print(f"Median Lesion-DSC:  {np.median(lesion_dsc_list):.4f}")
    print("========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on challenge metrics.")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_dynunet_multiclass.pth")
    args = parser.parse_args()
    evaluate_predictions(data_dir=args.data_dir, checkpoint_path=args.checkpoint)
