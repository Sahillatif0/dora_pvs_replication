"""
Anatomically Realistic Brain Atlas Generator using nilearn's MNI152 ICBM 2009 atlas.
Replaces the basic ellipsoid phantom with proper cortex, WM, CSF, Basal Ganglia, and ventricle anatomy.

This is equivalent to what VICOROBIGR described as "SynthSeg-derived label maps":
- Loads the standard MNI152 probabilistic tissue maps (WM, GM, CSF)
- Downloads the Harvard-Oxford subcortical atlas for Basal Ganglia
- Applies random 3D elastic deformations to produce diverse brain variants
- Saves ready-to-use label maps for the PVS + pathology generator

Label Convention (matches generate_dataset.py):
    0: Background / Air
    1: CSF (ventricles + sulci)
    2: Gray Matter (cortex)
    3: White Matter (centrum semiovale - PRIMARY PVS location)
    4: Basal Ganglia (PRIMARY PVS location)
"""

import os
import numpy as np
import nibabel as nib
import argparse
from scipy.ndimage import zoom, gaussian_filter, map_coordinates

def get_mni_tissue_maps():
    """Download and return MNI152 tissue probability maps via nilearn."""
    try:
        from nilearn import datasets, image
        print("Downloading MNI152 ICBM 2009 probabilistic brain atlas (one-time download)...")
        mni_data = datasets.fetch_icbm152_2009()
        
        # Load tissue probability maps
        gm_img  = image.load_img(mni_data['gm'])   # Gray Matter
        wm_img  = image.load_img(mni_data['wm'])   # White Matter
        csf_img = image.load_img(mni_data['csf'])  # CSF

        gm_data  = gm_img.get_fdata()
        wm_data  = wm_img.get_fdata()
        csf_data = csf_img.get_fdata()

        # Get basal ganglia from Harvard-Oxford subcortical atlas
        try:
            ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')
            ho_img = image.load_img(ho['maps'])
            ho_data = ho_img.get_fdata()
            
            # Caudate=11, Putamen=12, Pallidum=13, Thalamus=10
            bg_data = ((ho_data >= 10) & (ho_data <= 13)).astype(np.float32)
            bg_data = image.resample_to_img(
                nib.Nifti1Image(bg_data, ho_img.affine),
                gm_img
            ).get_fdata()
        except Exception:
            # Fallback: estimate BG from centeral WM region
            shape = wm_data.shape
            cz, cy, cx = shape[0]//2, shape[1]//2-5, shape[2]//2
            z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
            bg_data = (((z - cz)/18)**2 + ((y - cy)/22)**2 + ((x - cx)/18)**2 <= 1.0).astype(np.float32)

        affine = gm_img.affine
        return gm_data, wm_data, csf_data, bg_data, affine

    except Exception as e:
        print(f"nilearn download failed: {e}")
        return None, None, None, None, None

def probabilistic_to_labels(gm, wm, csf, bg, threshold=0.15):
    """
    Convert tissue probability maps to discrete label map.
    Labels: 0=BG, 1=CSF, 2=GM, 3=WM, 4=Basal Ganglia
    """
    label_map = np.zeros(gm.shape, dtype=np.uint8)
    
    # Combine all tissues; assign each voxel its most probable class
    stack = np.stack([
        np.zeros_like(gm),     # 0: Background
        csf,                    # 1: CSF
        gm,                     # 2: Gray Matter
        wm,                     # 3: White Matter
        np.clip(bg, 0, 1),      # 4: Basal Ganglia
    ], axis=-1)

    # Require at least threshold probability for a tissue to be labelled
    brain_mask = (gm + wm + csf) > threshold
    label_map[brain_mask] = np.argmax(stack[brain_mask], axis=-1).astype(np.uint8)

    return label_map

def apply_random_elastic_deformation(label_map, alpha_range=(15, 35), sigma=8.0):
    """
    Applies random smooth 3D elastic deformation to the label map.
    This creates diverse brain shape variants from a single atlas.
    alpha: deformation amplitude (mm); sigma: smoothness of deformation field
    """
    shape = label_map.shape
    alpha = np.random.uniform(*alpha_range)

    # Generate random displacement fields along each axis
    dx = gaussian_filter(np.random.randn(*shape), sigma=sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma=sigma) * alpha
    dz = gaussian_filter(np.random.randn(*shape), sigma=sigma) * alpha

    z, y, x = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing='ij'
    )

    coords = [
        np.clip(z + dz, 0, shape[0] - 1),
        np.clip(y + dy, 0, shape[1] - 1),
        np.clip(x + dx, 0, shape[2] - 1),
    ]

    # Nearest-neighbour interpolation to preserve discrete label integrity
    deformed = map_coordinates(label_map.astype(np.float32), coords, order=0, mode='nearest')
    return deformed.astype(np.uint8)

def downsample_to_target(label_map, target_shape=(128, 128, 128)):
    """Resample atlas to target generator size via nearest-neighbour."""
    scale = [t / s for t, s in zip(target_shape, label_map.shape)]
    resampled = zoom(label_map, scale, order=0, mode='nearest')
    # Ensure exact target shape
    slices = tuple(slice(0, t) for t in target_shape)
    out = np.zeros(target_shape, dtype=np.uint8)
    crop = resampled[slices]
    out[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
    return out

def generate_atlases(num_atlases=20, output_dir=os.path.join("data", "atlases"), target_shape=(128, 128, 128)):
    """Download MNI152 atlas and create N deformed variants as training canvases."""
    os.makedirs(output_dir, exist_ok=True)

    gm, wm, csf, bg, affine = get_mni_tissue_maps()

    if gm is None:
        print("[Error] Could not load MNI152 atlas. Check nilearn installation.")
        return

    print(f"MNI152 atlas loaded. Original shape: {gm.shape}")
    print(f"Generating {num_atlases} deformed anatomical canvases -> {output_dir}")

    base_labels = probabilistic_to_labels(gm, wm, csf, bg)

    unique, counts = np.unique(base_labels, return_counts=True)
    print("Base label distribution:", dict(zip(unique.tolist(), counts.tolist())))

    for i in range(num_atlases):
        # Apply elastic deformation for anatomical diversity
        if i == 0:
            # First one: straight atlas without deformation
            deformed_labels = base_labels
        else:
            alpha = np.random.uniform(10, 30)
            deformed_labels = apply_random_elastic_deformation(base_labels, alpha_range=(alpha, alpha + 10))

        # Downsample to target generator resolution
        small_labels = downsample_to_target(deformed_labels, target_shape=target_shape)

        out_path = os.path.join(output_dir, f"atlas_{i:04d}.nii.gz")
        nib.save(nib.Nifti1Image(small_labels, np.eye(4)), out_path)
        print(f"  [{i+1}/{num_atlases}] Saved atlas {os.path.basename(out_path)}: "
              f"WM={int(np.sum(small_labels==3))}, GM={int(np.sum(small_labels==2))}, "
              f"BG={int(np.sum(small_labels==4))}, CSF={int(np.sum(small_labels==1))}")

    print(f"\nAll {num_atlases} anatomically realistic brain canvases saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anatomically realistic brain atlases via MNI152 + elastic deformation.")
    parser.add_argument("--num_atlases", type=int, default=10, help="Number of diverse brain canvases to generate")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "atlases"))
    args = parser.parse_args()
    generate_atlases(num_atlases=args.num_atlases, output_dir=args.output_dir)
