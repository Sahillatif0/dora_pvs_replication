"""
Full Procedural Dataset Generator CLI.
Combines anatomical atlases (or procedural brain templates), seeds PVS and pathology,
applies domain randomisation, and saves training cases as NIfTI volumes.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from structures import StructureGenerator
from mri_physics import MRIPhysicsSimulator

def create_mock_brain_canvas(shape=(128, 128, 128)):
    """
    Creates an ellipsoidal synthetic brain template when real SynthSeg atlases
    are not yet available.
    Labels:
        0: Background
        1: CSF (ventricles + sulci)
        2: Gray Matter (cortex)
        3: White Matter (centrum semiovale)
        4: Basal Ganglia (deep gray/white nuclei)
    """
    canvas = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    
    # Brain outer boundary
    r_sq = ((z - cz)/52)**2 + ((y - cy)/48)**2 + ((x - cx)/40)**2
    brain = r_sq <= 1.0
    
    # White matter core
    wm = r_sq <= 0.65
    
    # Basal Ganglia inner region
    bg = (((z - cz)/18)**2 + ((y - cy)/20)**2 + ((x - cx)/18)**2) <= 1.0
    
    # Ventricles (CSF)
    ventricles = (((z - cz)/24)**2 + ((y - cy)/14)**2 + ((x - cx)/7)**2) <= 1.0
    
    # Assign labels
    canvas[brain] = 2  # Gray matter / cortex
    canvas[wm] = 3     # White matter
    canvas[bg] = 4     # Basal ganglia
    canvas[ventricles] = 1 # CSF
    
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
        tissue_canvas = create_mock_brain_canvas(shape=shape)
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
