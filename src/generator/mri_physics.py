"""
Domain Randomisation engine for MRI Physics Simulation.
Applies random contrast profiles (T1-like, T2-like, random), bias fields,
slice thickness / anisotropic blurring, and Rician noise.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

class MRIPhysicsSimulator:
    def __init__(self):
        pass

    def generate_contrast(self, tissue_labels, pvs_mask, pathology_mask, mode="auto"):
        """
        Renders synthetic intensities based on tissue labels, PVS, and pathology.
        tissue_labels: 0: BG/air, 1: CSF, 2: Gray Matter, 3: White Matter, 4: Basal Ganglia
        """
        if mode == "auto":
            mode = np.random.choice(["t1w", "t2w", "random"], p=[0.45, 0.45, 0.10])

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

        else: # purely random contrast
            csf_mean = np.random.uniform(0.1, 0.9)
            gm_mean  = np.random.uniform(0.1, 0.9)
            wm_mean  = np.random.uniform(0.1, 0.9)
            pvs_mean = np.random.uniform(0.1, 0.9)
            pathology_mean = np.random.uniform(0.1, 0.9)

        # Base tissue painting
        synthetic_image[tissue_labels == 1] = csf_mean
        synthetic_image[tissue_labels == 2] = gm_mean
        synthetic_image[(tissue_labels == 3) | (tissue_labels == 4)] = wm_mean

        # Add intra-tissue Gaussian variation
        noise_tissue = np.random.normal(0, 0.03, size=tissue_labels.shape)
        synthetic_image += noise_tissue * (tissue_labels > 0)

        # Render pathology lesions
        synthetic_image[pathology_mask > 0] = pathology_mean

        # Render PVS structures
        synthetic_image[pvs_mask > 0] = pvs_mean

        return synthetic_image, mode

    def apply_bias_field(self, image, brain_mask):
        """Generates smooth multiplicative spatial intensity inhomogeneities."""
        grid_shape = [4, 4, 4]
        coarse_field = np.random.normal(1.0, 0.18, size=grid_shape)
        
        # Upsample coarse field to volume shape
        zoom_factors = [image.shape[i] / grid_shape[i] for i in range(3)]
        bias_field = zoom(coarse_field, zoom_factors, order=2)
        
        # Ensure dimensions match exactly
        bias_field = bias_field[:image.shape[0], :image.shape[1], :image.shape[2]]
        bias_field = gaussian_filter(bias_field, sigma=5.0)
        
        corrupted = image * bias_field
        return corrupted * (brain_mask > 0)

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

    def synthesize(self, tissue_labels, pvs_mask, pathology_mask, brain_mask):
        """Runs the complete domain randomisation synthesis pipeline on a volume."""
        img, mode = self.generate_contrast(tissue_labels, pvs_mask, pathology_mask)
        
        # Bias field
        if np.random.rand() > 0.15:
            img = self.apply_bias_field(img, brain_mask)
            
        # Slice thickness anisotropy simulation
        if np.random.rand() > 0.25:
            thick_factor = np.random.uniform(1.2, 2.5)
            img = self.apply_anisotropic_downsampling(img, thick_factor)
            
        # Add Rician noise
        noise_level = np.random.uniform(0.01, 0.05)
        img = self.add_rician_noise(img, noise_level)
        
        # Normalization [0, 1]
        mask_vox = img[brain_mask > 0]
        if len(mask_vox) > 0:
            min_val, max_val = np.percentile(mask_vox, 1), np.percentile(mask_vox, 99)
            if max_val > min_val:
                img = np.clip((img - min_val) / (max_val - min_val), 0.0, 1.0)
                
        return img * (brain_mask > 0), mode
