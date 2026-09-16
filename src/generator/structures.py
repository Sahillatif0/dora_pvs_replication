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

    def generate_curved_pvs(self, brain_mask, wm_mask, bg_mask, num_structures=250):
        """
        Seeds curved tubular PVS channels along biologically realistic perforating vessel trajectories.
        Perivascular spaces follow the lenticulostriate arteries in the basal ganglia and medullary
        arteries radiating through the centrum semiovale (white matter).
        Returns:
            binary_pvs: binary target label (0 or 1) for ground truth segmentation.
            pvs_fraction: continuous sub-voxel volume fraction [0, 1] for MRI rendering.
        """
        pvs_density = np.zeros(self.shape, dtype=np.float32)
        valid_coords = np.argwhere((wm_mask > 0) | (bg_mask > 0))
        
        if len(valid_coords) == 0:
            valid_coords = np.argwhere(brain_mask > 0)
            
        if len(valid_coords) == 0:
            return pvs_density, pvs_density

        num_structures = min(num_structures, len(valid_coords))
        seed_indices = np.random.choice(len(valid_coords), size=num_structures, replace=False)
        
        # Center of basal ganglia / brain for radial orientation in centrum semiovale
        cz, cy, cx = self.shape[0] / 2.0, self.shape[1] / 2.0, self.shape[2] / 2.0

        for idx in seed_indices:
            start_pt = valid_coords[idx].astype(np.float32)
            is_bg = bg_mask[int(start_pt[0]), int(start_pt[1]), int(start_pt[2])] > 0
            
            # Trajectory length and radius in mm
            # Real PVS are typically 0.5mm - 2mm in diameter, 3mm - 15mm long
            length = np.random.uniform(4.0, 15.0)
            radius = np.random.uniform(0.4, 1.1)  # sub-voxel to ~1 voxel
            
            if is_bg:
                # In Basal Ganglia: lenticulostriate perforators travel predominantly inferior-superior (axial z)
                direction = np.array([np.random.uniform(0.7, 1.0), np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3)])
            else:
                # In CSO: medullary arteries radiate outwards towards cortex from ventricles
                radial_vec = start_pt - np.array([start_pt[0], cy, cx])
                r_norm = np.linalg.norm(radial_vec) + 1e-5
                radial_dir = radial_vec / r_norm
                direction = np.array([np.random.uniform(-0.3, 0.3), radial_dir[1], radial_dir[2]])
                
            direction /= (np.linalg.norm(direction) + 1e-6)
            
            # High-resolution step integration for sub-voxel anti-aliasing
            step_size = 0.5  # sub-voxel stepping
            num_steps = int(length / step_size)
            
            current_pt = start_pt.copy()
            points = [current_pt.copy()]
            
            # Smooth momentum curvature
            momentum = direction.copy()
            for _ in range(num_steps):
                drift = np.random.randn(3) * 0.08
                momentum = momentum + drift
                momentum /= (np.linalg.norm(momentum) + 1e-6)
                
                current_pt += momentum * step_size
                if (current_pt < 1).any() or (current_pt >= self.shape - 1).any():
                    break
                points.append(current_pt.copy())
                
            # Splat density along trajectory with sub-voxel trilinear weights
            for pt in points:
                iz, iy, ix = int(pt[0]), int(pt[1]), int(pt[2])
                fz, fy, fx = pt[0] - iz, pt[1] - iy, pt[2] - ix
                # 8-neighborhood trilinear splatting
                for dz, wz in ((0, 1 - fz), (1, fz)):
                    for dy, wy in ((0, 1 - fy), (1, fy)):
                        for dx, wx in ((0, 1 - fx), (1, fx)):
                            zz, yy, xx = iz + dz, iy + dy, ix + dx
                            if 0 <= zz < self.shape[0] and 0 <= yy < self.shape[1] and 0 <= xx < self.shape[2]:
                                if brain_mask[zz, yy, xx] > 0:
                                    pvs_density[zz, yy, xx] += wz * wy * wx

        # Filter to form smooth tubular profile
        pvs_density = np.clip(pvs_density, 0.0, 1.0)
        pvs_smooth = gaussian_filter(pvs_density, sigma=0.65)
        # Normalize to max 1.0
        max_val = np.max(pvs_smooth)
        if max_val > 0:
            pvs_fraction = np.clip(pvs_smooth / (max_val * 0.7), 0.0, 1.0) * brain_mask
        else:
            pvs_fraction = np.zeros_like(pvs_smooth)

        binary_pvs = (pvs_fraction > 0.25).astype(np.float32) * brain_mask
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
