"""
Docker Inference Script for DoRA-PVS Challenge Submission.
Loads trained DynUNet model(s) (supports single checkpoint or 5-fold ensemble),
performs sliding window inference on test scans, and outputs predicted PVS binary masks.

Input:  /input/*.nii.gz
Output: /output/*.nii.gz (Class 1: PVS binary mask)
"""

import os
import glob
import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from src.models.dynunet_model import PVSSegmentationModel

def load_models(weights_dir="/workspace/checkpoints", device="cpu"):
    """
    Loads all available DynUNet checkpoint weights (.pth) for ensemble inference.
    If multiple weights exist (e.g. 5-fold CV), their predictions are averaged (VICOROBIGR strategy).
    """
    model_files = sorted(glob.glob(os.path.join(weights_dir, "*.pth")))
    if not model_files:
        # Fallback to local checkpoints folder if not running in container
        model_files = sorted(glob.glob(os.path.join("checkpoints", "*.pth")))

    models = []
    for mf in model_files:
        print(f"[Model] Loading weights: {mf}")
        model = PVSSegmentationModel(num_classes=3, in_channels=1).to(device)
        state = torch.load(mf, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    if not models:
        print(f"[Warning] No .pth weights found in {weights_dir}. Initializing baseline model.")
        model = PVSSegmentationModel(num_classes=3, in_channels=1).to(device)
        model.eval()
        models.append(model)

    print(f"[Model] Total models loaded for ensembling: {len(models)}")
    return models

def predict_single_scan(models, image_tensor, roi_size=(96, 96, 96), sw_batch_size=2, device="cpu"):
    """
    Runs sliding window inference across all ensemble models and averages their soft probabilities.
    """
    prob_accum = None
    with torch.no_grad():
        for model in models:
            val_outputs = sliding_window_inference(
                inputs=image_tensor.to(device),
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )
            # Softmax over 3 classes: [0: BG, 1: PVS, 2: Pathology]
            probs = torch.softmax(val_outputs, dim=1)
            if prob_accum is None:
                prob_accum = probs
            else:
                prob_accum += probs

    prob_avg = prob_accum / len(models)
    return prob_avg

def run_inference(input_dir="/input", output_dir="/output", weights_dir="/workspace/checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")) + glob.glob(os.path.join(input_dir, "*.nii")))
    
    if not image_paths:
        print(f"[Warning] No NIfTI images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} image(s) to process in {input_dir}.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    models = load_models(weights_dir=weights_dir, device=device)

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        out_path = os.path.join(output_dir, fname)
        print(f"Processing: {fname} -> {out_path}")
        
        nii = nib.load(img_path)
        data = nii.get_fdata().astype(np.float32)

        # Standard intensity normalization
        p1, p99 = np.percentile(data, 1), np.percentile(data, 99.5)
        norm_data = np.clip((data - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

        # Prepare tensor [B, C, H, W, D]
        tensor = torch.from_numpy(norm_data).unsqueeze(0).unsqueeze(0)

        # Run multi-class sliding-window prediction
        probs = predict_single_scan(models, tensor, roi_size=(96, 96, 96), device=device)
        
        # Extract Class 1 (PVS) probability map and binary decision
        pvs_prob = probs[0, 1].cpu().numpy()
        binary_pred = (pvs_prob > 0.50).astype(np.uint8)

        # Save binary prediction maintaining exact input header/affine
        out_nii = nib.Nifti1Image(binary_pred, nii.affine, nii.header)
        nib.save(out_nii, out_path)
        print(f"  Saved PVS segmentation: {int(np.sum(binary_pred))} positive voxels")

    print("[Done] All inference completed successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DoRA-PVS Container Inference Entrypoint.")
    parser.add_argument("--input_dir", type=str, default="/input")
    parser.add_argument("--output_dir", type=str, default="/output")
    parser.add_argument("--weights_dir", type=str, default="/workspace/checkpoints")
    args = parser.parse_args()

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
    )
