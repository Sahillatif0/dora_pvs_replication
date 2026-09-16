"""
Inference script executed inside the Docker container.
Mounts:
    /input  -> Directory containing input test MRI scans (e.g. *.nii.gz)
    /output -> Directory where predicted binary segmentations and probability maps are saved
"""

import os
import glob
import torch
import nibabel as nib
import numpy as np

def run_inference(input_dir="/input", output_dir="/output"):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")) + glob.glob(os.path.join(input_dir, "*.nii")))
    
    if not image_paths:
        print(f"[Warning] No NIfTI images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} image(s) to process.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # In full implementation: Load the 3 trained DynUNet weights from /workspace/weights/
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        out_path = os.path.join(output_dir, fname)
        print(f"Processing {fname} -> {out_path}")
        
        nii = nib.load(img_path)
        # Inference pipeline will run here
        
if __name__ == "__main__":
    run_inference()
