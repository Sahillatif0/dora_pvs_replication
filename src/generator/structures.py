"""
Procedural 3D simulation of Perivascular Spaces (PVS) and Confounding Pathologies (WMH).
Draws realistic curved tubular structures for PVS and irregular ellipsoids for WMH lesions.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

class StructureGenerator:
    def __init__(self, shape=(128, 128, 128), voxel_size=(1.0, 1.0, 1.0)):
        self.shape = np.array(shape)
        self.voxel_size = np.array(voxel_size)

    def generate_curved_pvs(self, brain_mask, wm_mask, bg_mask, num_structures=350):
        """
        Seeds PVS adhering strictly to VICOROBIGR winner & STRIVE consensus:
        1. Basal Ganglia (BG): Vertical punctate dots & short cylindrical segments (1.5 - 4.0 mm, 1-2 voxels).
        2. Centrum Semiovale (CSO): Thin radial tubules radiating outward (3.0 - 7.0 mm, strictly sub-voxel 0.4 - 0.7 mm).
        3. Realistic Partial Volume: Continuous fraction [0.20, 0.65] avoiding fat oversaturated blobs.
        """
        pvs_density = np.zeros(self.shape, dtype=np.float32)
        bg_coords = np.argwhere(bg_mask > 0)
        wm_coords = np.argwhere((wm_mask > 0) & (bg_mask == 0))

        if len(bg_coords) == 0 and len(wm_coords) == 0:
            valid_coords = np.argwhere(brain_mask > 0)
            if len(valid_coords) == 0:
                return pvs_density, pvs_density
            wm_coords = valid_coords

        # Distribute PVS across both anatomical territories (approx 25% BG, 75% CSO)
        num_bg = int(num_structures * 0.25) if len(bg_coords) > 0 else 0
        num_cso = num_structures - num_bg

        selected_seeds = []
        if num_bg > 0:
            bg_idx = np.random.choice(len(bg_coords), size=min(num_bg, len(bg_coords)), replace=False)
            for idx in bg_idx:
                selected_seeds.append((bg_coords[idx].astype(np.float32), True))

        if len(wm_coords) > 0:
            cso_idx = np.random.choice(len(wm_coords), size=min(num_cso, len(wm_coords)), replace=False)
            for idx in cso_idx:
                selected_seeds.append((wm_coords[idx].astype(np.float32), False))

        cz, cy, cx = self.shape[0] / 2.0, self.shape[1] / 2.0, self.shape[2] / 2.0

        for start_pt, is_bg in selected_seeds:
            if is_bg:
                # 1. Basal Ganglia: Lenticulostriate perforators travel inferior-to-superior (axial Z)
                # Short punctate appearance: 1.5 - 4.0 mm
                length = np.random.uniform(1.5, 4.0)
                # Steep axial orientation (Z dominant)
                direction = np.array([np.random.choice([-1.0, 1.0]) * np.random.uniform(0.85, 1.0),
                                      np.random.uniform(-0.2, 0.2),
                                      np.random.uniform(-0.2, 0.2)])
                # Fractional intensity dip (subtle on MRI)
                intensity_factor = np.random.uniform(0.30, 0.65)
            else:
                # 2. Centrum Semiovale: Medullary arterioles radiating outward towards cortex
                # Realistic STRIVE length: 2.5 - 6.5 mm (never long 15mm snakes)
                length = np.random.uniform(2.5, 6.5)
                radial_vec = start_pt - np.array([start_pt[0], cy, cx])
                r_norm = np.linalg.norm(radial_vec) + 1e-5
                radial_dir = radial_vec / r_norm
                direction = np.array([np.random.uniform(-0.25, 0.25), radial_dir[1], radial_dir[2]])
                intensity_factor = np.random.uniform(0.25, 0.55)

            direction /= (np.linalg.norm(direction) + 1e-6)

            # High-resolution step integration for sub-voxel anti-aliasing
            step_size = 0.5
            num_steps = max(2, int(length / step_size))

            current_pt = start_pt.copy()
            points = [current_pt.copy()]
            momentum = direction.copy()

            for _ in range(num_steps):
                drift = np.random.randn(3) * 0.05
                momentum = momentum + drift
                momentum /= (np.linalg.norm(momentum) + 1e-6)
                current_pt += momentum * step_size
                if (current_pt < 1).any() or (current_pt >= self.shape - 1).any():
                    break
                points.append(current_pt.copy())

            # Trilinear sub-voxel splatting (1-2 voxel footprint strictly)
            for pt in points:
                iz, iy, ix = int(pt[0]), int(pt[1]), int(pt[2])
                fz, fy, fx = pt[0] - iz, pt[1] - iy, pt[2] - ix
                for dz, wz in ((0, 1 - fz), (1, fz)):
                    for dy, wy in ((0, 1 - fy), (1, fy)):
                        for dx, wx in ((0, 1 - fx), (1, fx)):
                            zz, yy, xx = iz + dz, iy + dy, ix + dx
                            if 0 <= zz < self.shape[0] and 0 <= yy < self.shape[1] and 0 <= xx < self.shape[2]:
                                if brain_mask[zz, yy, xx] > 0:
                                    pvs_density[zz, yy, xx] += wz * wy * wx * intensity_factor

        # Ground truth target: Any voxel containing significant microvessel fraction
        pvs_fraction = np.clip(pvs_density, 0.0, 0.70) * brain_mask
        binary_pvs = (pvs_fraction > 0.12).astype(np.float32) * brain_mask
        return binary_pvs, pvs_fraction

    def generate_pathology_lesions(self, brain_mask, wm_mask, num_lesions=12):
        """
        Seeds White Matter Hyperintensities (WMH) / small vessel disease lesions.
        Real WMH have soft, diffuse, infiltrating margins rather than hard geometric boundaries.
        Returns:
            binary_lesion: ground-truth target label (0 or 1)
            lesion_fraction: continuous intensity blend map [0, 1]
        """
        lesion_field = np.zeros(self.shape, dtype=np.float32)
        valid_coords = np.argwhere(wm_mask > 0) if np.sum(wm_mask) > 0 else np.argwhere(brain_mask > 0)
        
        if len(valid_coords) == 0:
            return lesion_field, lesion_field
            
        num_lesions = min(num_lesions, len(valid_coords))
        seed_indices = np.random.choice(len(valid_coords), size=num_lesions, replace=False)
        
        for idx in seed_indices:
            cz, cy, cx = valid_coords[idx]
            
            # Radii of lesion (2mm - 6mm)
            rx = np.random.uniform(2.0, 6.0) / self.voxel_size[2]
            ry = np.random.uniform(2.0, 6.0) / self.voxel_size[1]
            rz = np.random.uniform(1.8, 5.0) / self.voxel_size[0]
            
            z_min, z_max = max(0, int(cz - rz * 2.0)), min(self.shape[0], int(cz + rz * 2.0))
            y_min, y_max = max(0, int(cy - ry * 2.0)), min(self.shape[1], int(cy + ry * 2.0))
            x_min, x_max = max(0, int(cx - rx * 2.0)), min(self.shape[2], int(cx + rx * 2.0))
            
            sub_z, sub_y, sub_x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
            dist_sq = ((sub_z - cz)/rz)**2 + ((sub_y - cy)/ry)**2 + ((sub_x - cx)/rx)**2
            
            # Continuous Gaussian profile with irregular edge modulation
            edge_noise = np.random.normal(0, 0.2, size=dist_sq.shape)
            soft_blob = np.exp(-0.5 * (dist_sq + edge_noise))
            soft_blob[dist_sq > 2.2] = 0.0
            
            lesion_field[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
                lesion_field[z_min:z_max, y_min:y_max, x_min:x_max], soft_blob
            )

        # Smooth to eliminate any pixelation
        lesion_field = gaussian_filter(lesion_field, sigma=0.8)
        lesion_fraction = np.clip(lesion_field, 0.0, 1.0) * brain_mask
        binary_lesion = (lesion_fraction > 0.35).astype(np.float32) * brain_mask
        return binary_lesion, lesion_fraction
