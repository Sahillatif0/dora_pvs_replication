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
    """Load MNI152-derived atlas probability maps from disk, with label-map fallback."""
    global _atlas_cache
    if _atlas_cache:
        return _atlas_cache
    prob_files = sorted(glob.glob(os.path.join(atlas_dir, "*_prob.npz")))
    if prob_files:
        for f in prob_files:
            atlas = np.load(f)
            entry = {"labels": atlas["labels"].astype(np.uint8)}
            for k in atlas.files:
                if k != "labels":
                    entry[k] = atlas[k].astype(np.float32)
            _atlas_cache.append(entry)
    else:
        atlas_files = sorted(glob.glob(os.path.join(atlas_dir, "*.nii.gz")))
        for f in atlas_files:
            _atlas_cache.append(nib.load(f).get_fdata().astype(np.uint8))
    if _atlas_cache:
        print(f"[Atlas] Loaded {len(_atlas_cache)} real multi-structure brain canvases")
    return _atlas_cache

def get_brain_canvas(shape=(160, 192, 160), atlas_dir=os.path.join("data", "atlases"), allow_fallback=False):
    """
    Returns an anatomical label map (real multi-structure atlas or fallback).
    """
    atlases = load_real_atlases(atlas_dir)
    if atlases:
        canvas = atlases[np.random.randint(len(atlases))]
        if isinstance(canvas, dict):
            canvas = {key: value.copy() for key, value in canvas.items()}
            if canvas["labels"].shape != tuple(shape):
                scale = [s / c for s, c in zip(shape, canvas["labels"].shape)]
                canvas["labels"] = zoom(canvas["labels"], scale, order=0).astype(np.uint8)
                for key in canvas:
                    if key != "labels":
                        canvas[key] = zoom(canvas[key], scale, order=1).astype(np.float32)
            return canvas
        canvas = canvas.copy()
        if canvas.shape != tuple(shape):
            scale = [s / c for s, c in zip(shape, canvas.shape)]
            canvas = zoom(canvas, scale, order=0).astype(np.uint8)
        return canvas

    if not allow_fallback:
        raise FileNotFoundError(
            f"No atlas files found in {atlas_dir}. "
            "Run `python src/generator/build_atlases.py --num_atlases 10` first, "
            "or pass --allow_ellipsoid_fallback only for debugging."
        )

    # Fallback: geometric ellipsoid brain phantom. This is only for debugging.
    print("[Atlas] No real atlases found - using ellipsoid fallback for DEBUG only.")
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

def generate_dataset(
    num_samples=10,
    output_dir="data/synthetic",
    shape=(128, 128, 128),
    atlas_dir=os.path.join("data", "atlases"),
    allow_fallback=False,
):
    images_dir = os.path.join(output_dir, "imagesTr")
    labels_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Generating {num_samples} synthetic training cases into {output_dir}...")
    
    struct_gen = StructureGenerator(shape=shape)
    physics_sim = MRIPhysicsSimulator()

    for i in range(num_samples):
        # 1. Base anatomical canvas
        tissue_canvas = get_brain_canvas(shape=shape, atlas_dir=atlas_dir, allow_fallback=allow_fallback)
        tissue_labels = tissue_canvas["labels"] if isinstance(tissue_canvas, dict) else tissue_canvas
        brain_mask = (tissue_labels > 0).astype(np.float32)
        wm_mask = (tissue_labels == 3).astype(np.float32)
        bg_mask = (tissue_labels == 4).astype(np.float32)

        # 2. Seed PVS structures (Class 1) - smooth biological vascular channels
        num_pvs = np.random.randint(180, 380)
        pvs_mask, pvs_fraction = struct_gen.generate_curved_pvs(brain_mask, wm_mask, bg_mask, num_structures=num_pvs)

        # 3. Seed Pathology/WMH lesions (Class 2 - VICOROBIGR key innovation) - soft fuzzy margins
        num_lesions = np.random.randint(4, 18)
        pathology_mask, pathology_fraction = struct_gen.generate_pathology_lesions(brain_mask, wm_mask, num_lesions=num_lesions)

        # Avoid overlap (pathology overrides PVS)
        pvs_mask[pathology_mask > 0] = 0.0
        pvs_fraction[pathology_fraction > 0.2] = 0.0

        # 4. Synthesize MRI Volume via Domain Randomisation with Sub-voxel Physics
        synthetic_mri, contrast_mode = physics_sim.synthesize(
            tissue_canvas, (pvs_mask, pvs_fraction), (pathology_mask, pathology_fraction), brain_mask
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
    parser.add_argument("--atlas_dir", type=str, default=os.path.join("data", "atlases"), help="Directory containing MNI-derived atlas files")
    parser.add_argument(
        "--allow_ellipsoid_fallback",
        action="store_true",
        help="Use the old toy ellipsoid phantom if no atlas exists. Only recommended for code debugging.",
    )
    args = parser.parse_args()
    generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        atlas_dir=args.atlas_dir,
        allow_fallback=args.allow_ellipsoid_fallback,
    )
