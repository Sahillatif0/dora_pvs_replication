"""
Synthetic Data Quality Inspector & Multi-Slice Visualizer.
Picks a slice where BOTH Class 1 (PVS) and Class 2 (Lesions) are simultaneously present,
amplifies the overlay contrast, and adds clear bounding markers and legends.
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def inspect_dataset(data_dir=os.path.join("data", "synthetic"), output_report_path=os.path.join("data", "inspection_report.png")):
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    img_files = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    lbl_files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    if not img_files:
        print(f"[Error] No files found in {images_dir}")
        return

    sample_img = nib.load(img_files[0]).get_fdata()
    sample_lbl = nib.load(lbl_files[0]).get_fdata()

    # Find a slice that has a strong presence of BOTH Class 1 (PVS) AND Class 2 (Lesions)
    scores = []
    for z in range(sample_img.shape[0]):
        pvs_cnt = np.sum(sample_lbl[z] == 1)
        les_cnt = np.sum(sample_lbl[z] == 2)
        # Score favors slices where both classes are visibly present
        score = min(pvs_cnt, 200) * min(les_cnt, 200) if (pvs_cnt > 0 and les_cnt > 0) else 0
        scores.append(score)

    best_z = int(np.argmax(scores))
    if scores[best_z] == 0:
        # Fallback to slice with maximum lesion count
        best_z = int(np.argmax([np.sum(sample_lbl[z] == 2) for z in range(sample_img.shape[0])]))

    mri_slice = sample_img[best_z]
    pvs_slice = (sample_lbl[best_z] == 1).astype(float)
    lesion_slice = (sample_lbl[best_z] == 2).astype(float)

    print(f"Selected Slice #{best_z}:")
    print(f"  - PVS pixels in slice:    {int(np.sum(pvs_slice))}")
    print(f"  - Lesion pixels in slice: {int(np.sum(lesion_slice))}")

    # Build professional alpha-blended RGB overlay
    norm_mri = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-6)
    rgb_overlay = np.stack([norm_mri, norm_mri, norm_mri], axis=-1)

    # Semi-transparent alpha blending for PVS (Green) and Lesions (Red)
    # Allows underlying anatomical gyri, sulci, and tissue contrast to remain visible
    alpha_pvs = 0.65
    alpha_les = 0.65
    pvs_mask_2d = pvs_slice > 0
    les_mask_2d = lesion_slice > 0

    # Green for PVS: blend
    for c, target_val in enumerate([0.1, 1.0, 0.3]):
        rgb_overlay[pvs_mask_2d, c] = (1.0 - alpha_pvs) * rgb_overlay[pvs_mask_2d, c] + alpha_pvs * target_val

    # Red for Lesions: blend
    for c, target_val in enumerate([1.0, 0.15, 0.15]):
        rgb_overlay[les_mask_2d, c] = (1.0 - alpha_les) * rgb_overlay[les_mask_2d, c] + alpha_les * target_val

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#111111")
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(mri_slice, cmap="gray", origin="lower", interpolation="bicubic")
    axes[0].set_title(f"1. Synthetic MRI Slice #{best_z}\n(MNI152 Anatomy + Physics)", color="white", fontsize=12, pad=10)

    # Class 1: PVS
    axes[1].imshow(pvs_slice, cmap="Greens", vmin=0, vmax=1, origin="lower", interpolation="bicubic")
    axes[1].set_title(f"2. Class 1: PVS Channels\n[{int(np.sum(pvs_slice))} voxels]", color="#00ff88", fontsize=12, pad=10)

    # Class 2: Pathology/Lesions
    axes[2].imshow(lesion_slice, cmap="Reds", vmin=0, vmax=1, origin="lower", interpolation="bicubic")
    axes[2].set_title(f"3. Class 2: WMH Lesions\n[{int(np.sum(lesion_slice))} voxels]", color="#ff4444", fontsize=12, pad=10)

    # Multi-class overlay
    axes[3].imshow(rgb_overlay, origin="lower", interpolation="bicubic")
    axes[3].set_title("4. Multi-Class Ground Truth\n(Green=PVS, Red=WMH)", color="#ffea00", fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(output_report_path, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Updated inspection report saved with vivid colors to: {output_report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit synthetic data quality.")
    parser.add_argument("--data_dir", type=str, default=os.path.join("data", "synthetic"))
    parser.add_argument("--output_image", type=str, default=os.path.join("data", "inspection_report.png"))
    args = parser.parse_args()
    inspect_dataset(
        data_dir=os.path.abspath(args.data_dir),
        output_report_path=os.path.abspath(args.output_image)
    )
