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
        Seeds curved tubular PVS channels inside white matter (CSO) and basal ganglia (BG).
        Returns:
            pvs_mask: binary mask (0 or 1) indicating PVS channels
            pvs_distance: continuous distance map inside PVS with smooth cosine profile
        """
        pvs_mask = np.zeros(self.shape, dtype=np.float32)
        valid_coords = np.argwhere((wm_mask > 0) | (bg_mask > 0))
        
        if len(valid_coords) == 0:
            # Fallback if specific tissue masks are absent
            valid_coords = np.argwhere(brain_mask > 0)
            
        if len(valid_coords) == 0:
            return pvs_mask

        num_structures = min(num_structures, len(valid_coords))
        seed_indices = np.random.choice(len(valid_coords), size=num_structures, replace=False)

        for idx in seed_indices:
            start_pt = valid_coords[idx].astype(np.float32)
            
            # Sample length and radius
            length = np.random.uniform(4.0, 14.0) # in mm
            radius = np.random.uniform(0.5, 1.1)  # in mm
            
            # Trajectory: predominantly axial/coronal angle with random smooth drift
            direction = np.random.randn(3)
            direction[2] *= 1.5  # PVS often follow perforating vascular paths
            direction /= (np.linalg.norm(direction) + 1e-6)
            
            num_steps = int(length / (self.voxel_size.mean() * 0.5))
            step_size = length / max(num_steps, 1)
            
            current_pt = start_pt.copy()
            trajectory = [current_pt.copy()]
            
            for _ in range(num_steps):
                # Apply slight curvature perturbation
                drift = np.random.randn(3) * 0.12
                direction = direction + drift
                direction /= (np.linalg.norm(direction) + 1e-6)
                
                current_pt += direction * (step_size / self.voxel_size)
                
                # Check boundary
                if (current_pt < 1).any() or (current_pt >= self.shape - 1).any():
                    break
                    
                trajectory.append(current_pt.copy())
                
            # Rasterize points into volume
            for pt in trajectory:
                p_int = np.round(pt).astype(int)
                if brain_mask[p_int[0], p_int[1], p_int[2]] > 0:
                    pvs_mask[p_int[0], p_int[1], p_int[2]] = 1.0

        # Dilate along Euclidean distance to generate target radius with smooth profile
        dist_map = distance_transform_edt(1.0 - pvs_mask, sampling=self.voxel_size)
        target_radius = np.random.uniform(0.6, 1.2)
        binary_pvs = (dist_map <= target_radius).astype(np.float32) * brain_mask
        
        return binary_pvs

    def generate_pathology_lesions(self, brain_mask, wm_mask, num_lesions=12):
        """
        Seeds irregular ellipsoidal White Matter Hyperintensities (WMH) / pathology blobs.
        This provides the essential multi-class confounder signal (VICOROBIGR strategy).
        """
        pathology_mask = np.zeros(self.shape, dtype=np.float32)
        valid_coords = np.argwhere(wm_mask > 0) if np.sum(wm_mask) > 0 else np.argwhere(brain_mask > 0)
        
        if len(valid_coords) == 0:
            return pathology_mask
            
        num_lesions = min(num_lesions, len(valid_coords))
        seed_indices = np.random.choice(len(valid_coords), size=num_lesions, replace=False)
        
        # Coordinate grid for ellipsoid computation
        z, y, x = np.ogrid[:self.shape[0], :self.shape[1], :self.shape[2]]
        
        for idx in seed_indices:
            cz, cy, cx = valid_coords[idx]
            
            # Radii of lesion (much larger than PVS, e.g. 2mm - 6mm)
            rx = np.random.uniform(2.0, 5.5) / self.voxel_size[2]
            ry = np.random.uniform(2.0, 5.5) / self.voxel_size[1]
            rz = np.random.uniform(2.0, 5.5) / self.voxel_size[0]
            
            # Bounding box slice for efficiency
            z_min, z_max = max(0, int(cz - rz - 2)), min(self.shape[0], int(cz + rz + 2))
            y_min, y_max = max(0, int(cy - ry - 2)), min(self.shape[1], int(cy + ry + 2))
            x_min, x_max = max(0, int(cx - rx - 2)), min(self.shape[2], int(cx + rx + 2))
            
            sub_z, sub_y, sub_x = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
            dist_sq = ((sub_z - cz)/rz)**2 + ((sub_y - cy)/ry)**2 + ((sub_x - cx)/rx)**2
            
            # Add stochastic perturbation to make edges irregular
            noise = np.random.normal(0, 0.15, size=dist_sq.shape)
            lesion_blob = (dist_sq + noise) <= 1.0
            
            pathology_mask[z_min:z_max, y_min:y_max, x_min:x_max] = np.logical_or(
                pathology_mask[z_min:z_max, y_min:y_max, x_min:x_max], lesion_blob
            )

        pathology_mask = pathology_mask.astype(np.float32) * brain_mask
        return pathology_mask
