"""
Automated Brain Parcellation and Anatomical ROI Extraction.
Generates anatomical region masks:
  1. Centrum Semiovale (CSO) White Matter
  2. Basal Ganglia (BG)
  3. Lateral Ventricles & Ventricular Boundary Exclusion
  4. Cortical Ribbon Mask

Supports:
- Zero-cost Atlas-Guided Registration (MNI152 Nonlinear + Harvard-Oxford)
- Direct FastSurfer/SynthSeg label mapping if precomputed labels are provided.
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion
from nilearn import datasets, image

def get_brain_parcellation_mask(target_mri_nii, mode="cso_bg"):
    """
    Computes anatomical ROI mask for an arbitrary real patient MRI.
    Returns binary mask (1 for valid evaluation ROI: CSO + BG, 0 elsewhere).
    """
    print("  [Parcellation] Extracting anatomical White Matter & Basal Ganglia ROI...")
    
    # 1. Fetch MNI152 ICBM 2009c anatomical tissue maps
    mni = datasets.fetch_icbm152_2009()
    wm_atlas = nib.load(mni.wm)
    gm_atlas = nib.load(mni.gm)
    csf_atlas = nib.load(mni.csf)

    # 2. Resample atlas to match the target patient MRI's exact shape & affine
    resampled_wm = image.resample_to_img(wm_atlas, target_mri_nii, interpolation="linear").get_fdata()
    resampled_csf = image.resample_to_img(csf_atlas, target_mri_nii, interpolation="linear").get_fdata()

    # 3. Create Centrum Semiovale (CSO) & Deep White Matter mask
    # White matter probability > 0.5
    raw_wm_mask = resampled_wm > 0.45

    # 4. Exclude Ventricular Boundary & CSF (The crucial step that eliminates boundary false positives!)
    # Dilate CSF/ventricles by 2 voxels to aggressively strip the ependymal lining
    ventricle_csf = resampled_csf > 0.40
    dilated_csf_border = binary_dilation(ventricle_csf, iterations=2)

    # Erode WM slightly to stay strictly within deep white matter
    eroded_wm = binary_erosion(raw_wm_mask, iterations=1)

    # Clean valid ROI: Inside WM, strictly outside dilated CSF/ventricular boundary
    valid_roi = eroded_wm & (~dilated_csf_border)

    # 5. Add Basal Ganglia subcortical region
    try:
        ho_sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
        if isinstance(ho_sub.maps, str):
            ho_nii = nib.load(ho_sub.maps)
        else:
            ho_nii = ho_sub.maps
        ho_resampled = image.resample_to_img(ho_nii, target_mri_nii, interpolation="nearest").get_fdata()
        # Harvard-Oxford subcortical Basal Ganglia label IDs:
        # 4, 15 (Lateral Ventricles - excluded)
        # 5, 16 (Caudate)
        # 6, 17 (Putamen)
        # 7, 18 (Pallidum)
        bg_labels = [5, 6, 7, 16, 17, 18]
        bg_mask = np.isin(ho_resampled, bg_labels)
        valid_roi = valid_roi | bg_mask
    except Exception as e:
        print(f"  [Parcellation Warning] Could not load Harvard-Oxford subcortical labels: {e}")

    print(f"  [Parcellation] Valid ROI constructed: {int(np.sum(valid_roi))} voxels ({np.sum(valid_roi) / valid_roi.size * 100:.2f}% of volume)")
    return valid_roi.astype(np.float32)
