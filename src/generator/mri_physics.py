"""
Domain Randomisation engine for MRI Physics Simulation.
Applies random contrast profiles (T1-like, T2-like, random), bias fields,
slice thickness / anisotropic blurring, and Rician noise.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

class MRIPhysicsSimulator:
    def __init__(self):
        pass

    def generate_contrast(self, tissue_data, pvs_mask, pathology_mask, mode="auto"):
        """
        Renders synthetic intensities based on tissue labels, PVS, and pathology.
        tissue_labels: 0: BG/air, 1: CSF, 2: Gray Matter, 3: White Matter, 4: Basal Ganglia
        """
        if isinstance(tissue_data, dict):
            tissue_labels = tissue_data["labels"]
            csf_prob = tissue_data.get("csf", (tissue_labels == 1).astype(np.float32))
            gm_prob = tissue_data.get("gm", (tissue_labels == 2).astype(np.float32))
            wm_prob = tissue_data.get("wm", ((tissue_labels == 3) | (tissue_labels == 4)).astype(np.float32))
            ventricles_prob = tissue_data.get("ventricles", None)
            caudate_prob = tissue_data.get("caudate", None)
            putamen_prob = tissue_data.get("putamen", None)
            pallidum_prob = tissue_data.get("pallidum", None)
            thalamus_prob = tissue_data.get("thalamus", None)
            brainstem_prob = tissue_data.get("brainstem", None)
            bg_prob = tissue_data.get("bg_union", tissue_data.get("bg", (tissue_labels == 4).astype(np.float32)))
        else:
            tissue_labels = tissue_data
            csf_prob = (tissue_labels == 1).astype(np.float32)
            gm_prob = (tissue_labels == 2).astype(np.float32)
            wm_prob = (tissue_labels == 3).astype(np.float32)
            bg_prob = (tissue_labels == 4).astype(np.float32)
            ventricles_prob = None
            caudate_prob = putamen_prob = pallidum_prob = thalamus_prob = brainstem_prob = None

        if mode == "auto":
            mode = np.random.choice(["t1w", "t2w", "flair", "random"], p=[0.34, 0.34, 0.22, 0.10])

        synthetic_image = np.zeros(tissue_labels.shape, dtype=np.float32)

        if mode == "t1w":
            # T1w: CSF dark, GM intermediate, WM bright
            # PVS (CSF filled) is hypointense (dark)
            # WMH is typically isointense to hypointense on T1
            csf_mean = np.random.uniform(0.15, 0.25)
            gm_mean  = np.random.uniform(0.55, 0.65)
            wm_mean  = np.random.uniform(0.85, 0.95)
            pvs_mean = np.random.uniform(0.10, 0.28) # hypointense
            pathology_mean = np.random.uniform(0.40, 0.60) # lower than normal WM

        elif mode == "t2w":
            # T2w: CSF bright, GM bright/intermediate, WM dark
            # PVS (CSF filled) is hyperintense (bright)
            # WMH is hyperintense (bright)
            csf_mean = np.random.uniform(0.85, 0.98)
            gm_mean  = np.random.uniform(0.55, 0.70)
            wm_mean  = np.random.uniform(0.25, 0.40)
            pvs_mean = np.random.uniform(0.80, 0.98) # hyperintense
            pathology_mean = np.random.uniform(0.70, 0.95) # hyperintense

        elif mode == "flair":
            # FLAIR-like: CSF/PVS suppressed dark, WMH/pathology bright.
            csf_mean = np.random.uniform(0.05, 0.18)
            gm_mean  = np.random.uniform(0.45, 0.62)
            wm_mean  = np.random.uniform(0.30, 0.48)
            pvs_mean = np.random.uniform(0.04, 0.16)
            pathology_mean = np.random.uniform(0.72, 0.98)

        else: # purely random contrast
            csf_mean = np.random.uniform(0.1, 0.9)
            gm_mean  = np.random.uniform(0.1, 0.9)
            wm_mean  = np.random.uniform(0.1, 0.9)
            pvs_mean = np.random.uniform(0.1, 0.9)
            pathology_mean = np.random.uniform(0.1, 0.9)

        brain_mask = tissue_labels > 0
        bg_mean = wm_mean * np.random.uniform(0.88, 1.08)

        # Probability-weighted continuous MNI tissue synthesis with realistic intra-tissue variation
        tissue_sum = csf_prob + gm_prob + wm_prob
        # Base continuous tissue synthesis
        synthetic_image = (
            csf_prob * csf_mean +
            gm_prob * gm_mean +
            wm_prob * wm_mean
        ).astype(np.float32)

        # Realistic anatomical sub-structure modulation (matches real neuroimaging contrast)
        if ventricles_prob is not None:
            # Ventricles: pure CSF signal
            synthetic_image = np.where(ventricles_prob > 0.3, csf_mean, synthetic_image)

        if pallidum_prob is not None and mode in ("t2w", "flair"):
            # Globus Pallidus: biological iron/hemosiderin causes profound T2/T2* signal drop
            iron_drop = np.random.uniform(0.65, 0.82)
            synthetic_image = np.where(pallidum_prob > 0.4, synthetic_image * iron_drop, synthetic_image)

        if putamen_prob is not None and caudate_prob is not None:
            # Striatum (Caudate & Putamen): deep gray matter signal
            striatum = np.clip(putamen_prob + caudate_prob, 0.0, 1.0)
            synthetic_image = np.where(striatum > 0.4, gm_mean * np.random.uniform(0.95, 1.05), synthetic_image)

        if thalamus_prob is not None:
            # Thalamus: mixed gray/white intensity
            thalamus_mean = (gm_mean + wm_mean) * 0.5
            synthetic_image = np.where(thalamus_prob > 0.4, thalamus_mean, synthetic_image)

        if brainstem_prob is not None:
            # Brainstem: compact descending white matter tracts
            synthetic_image = np.where(brainstem_prob > 0.4, wm_mean * np.random.uniform(0.92, 1.02), synthetic_image)

        # Multi-scale fractal / smooth spatial tissue texture (cellular & myelin density variance)
        fine_texture = np.random.normal(0.0, 1.0, size=tissue_labels.shape)
        fine_texture = gaussian_filter(fine_texture, sigma=np.random.uniform(2.0, 4.0))
        fine_texture /= (np.std(fine_texture[brain_mask]) + 1e-6)
        synthetic_image += np.random.uniform(0.02, 0.05) * fine_texture * brain_mask

        # Sub-voxel Partial-Volume Blending for PVS
        # I_rendered = (1 - f_pvs) * I_tissue + f_pvs * I_pvs
        # This replaces sharp pixelated blocks with continuous, smooth microvascular channels
        if isinstance(pvs_mask, tuple) or isinstance(pvs_mask, list):
            _, pvs_frac = pvs_mask
        else:
            pvs_frac = pvs_mask.astype(np.float32)
        
        synthetic_image = (1.0 - pvs_frac) * synthetic_image + pvs_frac * pvs_mean

        # Soft continuous blending for pathology lesions (fuzzy margins, core-to-rim gradient)
        if isinstance(pathology_mask, tuple) or isinstance(pathology_mask, list):
            _, path_frac = pathology_mask
        else:
            path_frac = pathology_mask.astype(np.float32)

        synthetic_image = (1.0 - path_frac) * synthetic_image + path_frac * pathology_mean

        # Realistic non-brain cranial anatomy (CSF subarachnoid space, skull bone, subcutaneous fat, scalp)
        # Real clinical MRI includes these head structures unless skull-stripped
        outside_dist = distance_transform_edt(~brain_mask)
        
        # 1. Subarachnoid space (CSF rim around cortex)
        csf_rim = (outside_dist > 0) & (outside_dist <= 1.8)
        synthetic_image[csf_rim] = csf_mean * np.random.uniform(0.9, 1.1)
        
        # 2. Inner & outer cortical bone (very low signal on both T1 and T2)
        skull_bone = (outside_dist > 1.8) & (outside_dist <= 4.0)
        bone_signal = np.random.uniform(0.03, 0.12)
        synthetic_image[skull_bone] = bone_signal
        
        # 3. Diploe (cancellous bone marrow) & Subcutaneous Fat
        # Fat is bright on T1, moderate on T2
        subcut_fat = (outside_dist > 4.0) & (outside_dist <= 7.5)
        fat_signal = np.random.uniform(0.75, 0.95) if mode == "t1w" else np.random.uniform(0.45, 0.65)
        synthetic_image[subcut_fat] = fat_signal
        
        # 4. Scalp skin layer
        scalp_skin = (outside_dist > 7.5) & (outside_dist <= 9.5)
        skin_signal = np.random.uniform(0.35, 0.55)
        synthetic_image[scalp_skin] = skin_signal

        # Scanner Point Spread Function (PSF) partial-volume blur
        head_mask = outside_dist <= 9.5
        synthetic_image = gaussian_filter(synthetic_image, sigma=np.random.uniform(0.3, 0.6))
        synthetic_image *= head_mask

        return synthetic_image, mode

    def apply_bias_field(self, image, head_mask):
        """Generates smooth multiplicative spatial B1 intensity inhomogeneities across the head."""
        grid_shape = [4, 4, 4]
        coarse_field = np.random.normal(1.0, 0.18, size=grid_shape)
        
        # Upsample coarse field to volume shape
        zoom_factors = [image.shape[i] / grid_shape[i] for i in range(3)]
        bias_field = zoom(coarse_field, zoom_factors, order=2)
        
        # Ensure dimensions match exactly
        bias_field = bias_field[:image.shape[0], :image.shape[1], :image.shape[2]]
        bias_field = gaussian_filter(bias_field, sigma=6.0)
        
        corrupted = image * bias_field
        return corrupted * (head_mask > 0)

    def apply_anisotropic_downsampling(self, image, target_thickness_factor=2.0):
        """Simulates thick 2D clinical acquisition slices (e.g. 2.0 mm vs 0.5 mm in-plane)."""
        if target_thickness_factor <= 1.1:
            return image
            
        # Blur along the slice select (axial) direction
        sigma_z = (target_thickness_factor - 1.0) * 0.8
        blurred = gaussian_filter(image, sigma=(sigma_z, 0.2, 0.2))
        return blurred

    def add_rician_noise(self, image, snr_sigma=0.03):
        """Simulates MRI Rician noise: sqrt((S + N1)^2 + N2^2)"""
        n1 = np.random.normal(0, snr_sigma, size=image.shape)
        n2 = np.random.normal(0, snr_sigma, size=image.shape)
        noisy = np.sqrt((image + n1)**2 + n2**2)
        return noisy

    def synthesize(self, tissue_labels, pvs_input, pathology_input, brain_mask):
        """Runs the complete domain randomisation synthesis pipeline on a volume."""
        img, mode = self.generate_contrast(tissue_labels, pvs_input, pathology_input)
        
        head_mask = (img > 0.01).astype(np.float32)
        
        # Multiplicative B1 RF coil bias field
        if np.random.rand() > 0.15:
            img = self.apply_bias_field(img, head_mask)
            
        # Slice thickness anisotropy simulation
        if np.random.rand() > 0.25:
            thick_factor = np.random.uniform(1.2, 2.2)
            img = self.apply_anisotropic_downsampling(img, thick_factor)
            
        # Add Rician noise across whole FOV (including background air)
        noise_level = np.random.uniform(0.015, 0.04)
        img = self.add_rician_noise(img, noise_level)
        
        # Dynamic range windowing and normalization [0, 1]
        mask_vox = img[head_mask > 0]
        if len(mask_vox) > 0:
            min_val, max_val = np.percentile(mask_vox, 0.5), np.percentile(mask_vox, 99.5)
            if max_val > min_val:
                img = np.clip((img - min_val) / (max_val - min_val), 0.0, 1.0)
                
        return img, mode
