"""
Full Procedural Dataset Generator CLI.
Combines real MNI152 anatomical atlases (via build_atlases.py) or fallback ellipsoid templates,
seeds PVS and pathology, applies domain randomisation, and saves NIfTI training cases.

Run build_atlases.py FIRST to generate realistic brain canvases:
    python src/generator/build_atlases.py --num_atlases 10
"""

import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from structures import StructureGenerator
from mri_physics import MRIPhysicsSimulator

# Cache for loaded atlases
_atlas_cache = []

def load_real_atlases(atlas_dir=os.path.join("data", "atlases")):
    """Load MNI152-derived atlas label maps from disk."""
    global _atlas_cache
    if _atlas_cache:
        return _atlas_cache
    atlas_files = sorted(glob.glob(os.path.join(atlas_dir, "*.nii.gz")))
    for f in atlas_files:
        _atlas_cache.append(nib.load(f).get_fdata().astype(np.uint8))
    if _atlas_cache:
        print(f"[Atlas] Loaded {len(_atlas_cache)} real MNI152 brain canvases")
    return _atlas_cache

def get_brain_canvas(shape=(128, 128, 128), atlas_dir=os.path.join("data", "atlases")):
    """
    Returns an anatomical label map (real MNI152 atlas or fallback ellipsoid).
    Labels: 0=BG | 1=CSF | 2=Gray Matter | 3=White Matter | 4=Basal Ganglia
    """
    atlases = load_real_atlases(atlas_dir)
    if atlases:
        canvas = atlases[np.random.randint(len(atlases))].copy()
        if canvas.shape != tuple(shape):
            scale = [s / c for s, c in zip(shape, canvas.shape)]
            canvas = zoom(canvas, scale, order=0).astype(np.uint8)
        return canvas

    # Fallback: geometric ellipsoid brain phantom
    print("[Atlas] No real atlases found - using ellipsoid fallback. Run build_atlases.py first!")
    canvas = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    r_sq = ((z - cz)/52)**2 + ((y - cy)/48)**2 + ((x - cx)/40)**2
    brain = r_sq <= 1.0
    wm = r_sq <= 0.65
    bg = (((z - cz)/18)**2 + ((y - cy)/20)**2 + ((x - cx)/18)**2) <= 1.0
    ventricles = (((z - cz)/24)**2 + ((y - cy)/14)**2 + ((x - cx)/7)**2) <= 1.0
    canvas[brain] = 2
    canvas[wm] = 3
    canvas[bg] = 4
    canvas[ventricles] = 1
    return canvas

def generate_dataset(num_samples=10, output_dir="data/synthetic", shape=(128, 128, 128)):
    images_dir = os.path.join(output_dir, "imagesTr")
    labels_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Generating {num_samples} synthetic training cases into {output_dir}...")
    
    struct_gen = StructureGenerator(shape=shape)
    physics_sim = MRIPhysicsSimulator()

    for i in range(num_samples):
        # 1. Base anatomical canvas
        tissue_canvas = get_brain_canvas(shape=shape)
        brain_mask = (tissue_canvas > 0).astype(np.float32)
        wm_mask = (tissue_canvas == 3).astype(np.float32)
        bg_mask = (tissue_canvas == 4).astype(np.float32)

        # 2. Seed PVS structures (Class 1)
        num_pvs = np.random.randint(180, 380)
        pvs_mask = struct_gen.generate_curved_pvs(brain_mask, wm_mask, bg_mask, num_structures=num_pvs)

        # 3. Seed Pathology/WMH lesions (Class 2 - VICOROBIGR key innovation)
        num_lesions = np.random.randint(4, 18)
        pathology_mask = struct_gen.generate_pathology_lesions(brain_mask, wm_mask, num_lesions=num_lesions)

        # Avoid overlap (PVS priority over lesion, or lesion override)
        pvs_mask[pathology_mask > 0] = 0.0

        # 4. Synthesize MRI Volume via Domain Randomisation
        synthetic_mri, contrast_mode = physics_sim.synthesize(
            tissue_canvas, pvs_mask, pathology_mask, brain_mask
        )

        # 5. Build Multi-Class Target Label
        # 0: Background, 1: PVS, 2: Pathology
        target_label = np.zeros(shape, dtype=np.uint8)
        target_label[pvs_mask > 0] = 1
        target_label[pathology_mask > 0] = 2

        # 6. Save as standard NIfTI (.nii.gz)
        case_id = f"SYNTH_{i:04d}"
        img_path = os.path.join(images_dir, f"{case_id}_0000.nii.gz")
        lbl_path = os.path.join(labels_dir, f"{case_id}.nii.gz")

        affine = np.eye(4)
        nib.save(nib.Nifti1Image(synthetic_mri.astype(np.float32), affine), img_path)
        nib.save(nib.Nifti1Image(target_label, affine), lbl_path)

        print(f"[{i+1}/{num_samples}] Saved {case_id} (Contrast: {contrast_mode}, PVS voxels: {int(np.sum(pvs_mask))}, Lesion voxels: {int(np.sum(pathology_mask))})")

    print(f"Dataset generation complete! Files saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic DoRA-PVS training data.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of synthetic cases to generate")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Output directory")
    args = parser.parse_args()
    generate_dataset(num_samples=args.num_samples, output_dir=args.output_dir)
