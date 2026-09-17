"""
Full Procedural Dataset Generator CLI for DoRA-PVS Challenge Replication.
Uses authentic MNI152 ICBM 2009 human neuroimaging template + Harvard-Oxford subcortical parcellations
as anatomical substrates, with domain randomisation (T1/T2/FLAIR, B1 bias fields, slice anisotropy, Rician noise)
and physiologically accurate multi-class seeding (Class 1: PVS, Class 2: WMH Confounders).
"""

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from nilearn import datasets, image
from structures import StructureGenerator
from mri_physics import MRIPhysicsSimulator

# Global cache for base templates so they are loaded once per session
_template_cache = None

def get_authentic_anatomical_template():
    """
    Loads and caches the authentic MNI152 ICBM 2009 nonlinear template
    and Harvard-Oxford subcortical structures at 1mm isotropic resolution.
    """
    global _template_cache
    if _template_cache is not None:
        return _template_cache

    print("[Atlas] Loading MNI152 ICBM 2009 human anatomical templates...")
    mni = datasets.fetch_icbm152_2009()
    
    t1_img = image.load_img(mni['t1'])
    t2_img = image.load_img(mni['t2'])
    mask_img = image.load_img(mni['mask'])
    gm_img = image.load_img(mni['gm'])
    wm_img = image.load_img(mni['wm'])
    csf_img = image.load_img(mni['csf'])

    ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
    ho_img = image.resample_to_img(image.load_img(ho['maps']), gm_img, interpolation='nearest')

    t1_vol = t1_img.get_fdata().astype(np.float32)
    t2_vol = t2_img.get_fdata().astype(np.float32)
    mask_vol = (mask_img.get_fdata() > 0.5).astype(np.float32)
    gm_vol = gm_img.get_fdata().astype(np.float32)
    wm_vol = wm_img.get_fdata().astype(np.float32)
    csf_vol = csf_img.get_fdata().astype(np.float32)
    ho_data = ho_img.get_fdata().astype(np.int32)

    # Basal Ganglia union (caudate=5,16, putamen=6,17, pallidum=7,18)
    bg_vol = ((ho_data >= 5) & (ho_data <= 7) | (ho_data >= 16) & (ho_data <= 18)).astype(np.float32)
    lat_ventricles = ((ho_data == 3) | (ho_data == 14)).astype(np.float32)

    # Tight bounding box to eliminate empty air background while preserving 1mm resolution
    valid_coords = np.argwhere(t1_vol > 5.0)
    z_min, y_min, x_min = valid_coords.min(axis=0)
    z_max, y_max, x_max = valid_coords.max(axis=0)

    z_min, y_min, x_min = max(0, z_min - 4), max(0, y_min - 4), max(0, x_min - 4)
    z_max, y_max, x_max = min(t1_vol.shape[0], z_max + 4), min(t1_vol.shape[1], y_max + 4), min(t1_vol.shape[2], x_max + 4)

    _template_cache = {
        "t1": t1_vol[z_min:z_max, y_min:y_max, x_min:x_max],
        "t2": t2_vol[z_min:z_max, y_min:y_max, x_min:x_max],
        "brain_mask": mask_vol[z_min:z_max, y_min:y_max, x_min:x_max],
        "gm": gm_vol[z_min:z_max, y_min:y_max, x_min:x_max],
        "wm": wm_vol[z_min:z_max, y_min:y_max, x_min:x_max],
        "csf": csf_vol[z_min:z_max, y_min:y_max, x_min:x_max],
        "bg": bg_vol[z_min:z_max, y_min:y_max, x_min:x_max],
        "ventricles": lat_ventricles[z_min:z_max, y_min:y_max, x_min:x_max],
        "affine": gm_img.affine,
    }
    print(f"[Atlas] Authentic template cached. Native shape: {_template_cache['t1'].shape}")
    return _template_cache

def generate_dataset(
    num_samples=5,
    output_dir="data/synthetic",
    target_shape=None,
    prefix="SYNTH"
):
    images_dir = os.path.join(output_dir, "imagesTr")
    labels_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    template = get_authentic_anatomical_template()
    base_shape = template["t1"].shape
    shape = target_shape if target_shape is not None else base_shape

    print(f"Generating {num_samples} authentic synthetic training cases into {output_dir} (Volume shape: {shape})...")

    # Resample template if target_shape is explicitly requested
    if shape != base_shape:
        scale = [t / s for t, s in zip(shape, base_shape)]
        t1_base = zoom(template["t1"], scale, order=1)
        t2_base = zoom(template["t2"], scale, order=1)
        brain_mask = zoom(template["brain_mask"], scale, order=0)
        wm_mask = zoom(template["wm"], scale, order=1)
        bg_mask = zoom(template["bg"], scale, order=0)
    else:
        t1_base = template["t1"]
        t2_base = template["t2"]
        brain_mask = template["brain_mask"]
        wm_mask = template["wm"]
        bg_mask = template["bg"]

    struct_gen = StructureGenerator(shape=shape, voxel_size=(1.0, 1.0, 1.0))
    physics_sim = MRIPhysicsSimulator()

    for i in range(num_samples):
        # 1. Contrast Selection (T1-like, T2-like, FLAIR-like, or random GMM)
        contrast_mode = np.random.choice(["t1w", "t2w", "flair", "random"], p=[0.40, 0.35, 0.15, 0.10])
        
        if contrast_mode == "t1w":
            base_img = t1_base.copy()
            pvs_target_val = np.random.uniform(0.08, 0.22)   # hypointense
            wmh_target_val = np.random.uniform(0.40, 0.58)   # mildly hypointense
        elif contrast_mode == "t2w":
            base_img = t2_base.copy()
            pvs_target_val = np.random.uniform(0.80, 0.98)   # hyperintense
            wmh_target_val = np.random.uniform(0.70, 0.95)   # hyperintense
        elif contrast_mode == "flair":
            # FLAIR: T2-like brain tissue but CSF/PVS is suppressed dark, WMH is bright
            base_img = np.clip(t2_base * (1.0 - template["csf"] * 0.85), 0.0, None)
            pvs_target_val = np.random.uniform(0.05, 0.15)   # CSF suppressed
            wmh_target_val = np.random.uniform(0.80, 0.98)   # bright hyperintensity
        else: # Random domain randomisation
            alpha_mix = np.random.uniform(0.2, 0.8)
            base_img = alpha_mix * t1_base + (1.0 - alpha_mix) * t2_base
            pvs_target_val = np.random.uniform(0.1, 0.9)
            wmh_target_val = np.random.uniform(0.1, 0.9)

        # Normalize base image
        p1, p99 = np.percentile(base_img, 1), np.percentile(base_img, 99.5)
        base_norm = np.clip((base_img - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

        # 2. Seed PVS structures (Class 1) - smooth biological vascular channels
        num_pvs = np.random.randint(220, 450)
        pvs_mask, pvs_fraction = struct_gen.generate_curved_pvs(brain_mask, wm_mask, bg_mask, num_structures=num_pvs)

        # 3. Seed Pathology/WMH lesions (Class 2 - Confounder) - soft fuzzy margins
        num_lesions = np.random.randint(5, 18)
        pathology_mask, pathology_fraction = struct_gen.generate_pathology_lesions(brain_mask, wm_mask, num_lesions=num_lesions)

        # Pathology overrides PVS
        pvs_mask[pathology_mask > 0] = 0.0
        pvs_fraction[pathology_fraction > 0.2] = 0.0

        # 4. Sub-voxel physical rendering of PVS and WMH
        mri_volume = (1.0 - pvs_fraction) * base_norm + pvs_fraction * pvs_target_val
        mri_volume = (1.0 - pathology_fraction) * mri_volume + pathology_fraction * wmh_target_val

        # 5. Realistic scanner physics: B1 Bias Field + Rician Noise
        head_mask = (base_img > 5.0).astype(np.float32)
        if np.random.rand() > 0.15:
            mri_volume = physics_sim.apply_bias_field(mri_volume, head_mask)

        if np.random.rand() > 0.25:
            thick_factor = np.random.uniform(1.2, 2.2)
            mri_volume = physics_sim.apply_anisotropic_downsampling(mri_volume, thick_factor)

        noise_level = np.random.uniform(0.015, 0.035)
        mri_volume = physics_sim.add_rician_noise(mri_volume, snr_sigma=noise_level)
        mri_volume = np.clip(mri_volume, 0.0, 1.0)

        # 6. Build Multi-Class Target Label (0=BG, 1=PVS, 2=Pathology)
        target_label = np.zeros(shape, dtype=np.uint8)
        target_label[pvs_mask > 0] = 1
        target_label[pathology_mask > 0] = 2

        # 7. Save as standard NIfTI (.nii.gz)
        case_id = f"{prefix}_{i:04d}"
        img_path = os.path.join(images_dir, f"{case_id}_0000.nii.gz")
        lbl_path = os.path.join(labels_dir, f"{case_id}.nii.gz")

        affine = template["affine"]
        nib.save(nib.Nifti1Image(mri_volume.astype(np.float32), affine), img_path)
        nib.save(nib.Nifti1Image(target_label, affine), lbl_path)

        pvs_vox = int(np.sum(pvs_mask > 0))
        wmh_vox = int(np.sum(pathology_mask > 0))
        print(f"[{i+1}/{num_samples}] Saved {case_id} (Contrast: {contrast_mode}, PVS: {pvs_vox} voxels, WMH: {wmh_vox} voxels)")

    print(f"\nDataset generation complete! Files saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate authentic synthetic DoRA-PVS training/test data.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of synthetic cases to generate")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--prefix", type=str, default="SYNTH", help="Filename prefix (e.g. TEST or SYNTH)")
    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
