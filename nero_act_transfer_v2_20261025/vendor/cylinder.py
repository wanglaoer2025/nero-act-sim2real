"""Read-only extraction of existing nominal RGB cylinder geometry; unqualified real accuracy."""
import numpy as np
import cv2
from scipy.optimize import least_squares

class NominalImageEstimator:
    """Per-image upright cylinder bbox fit; no pose cache or truth fallback."""
    def __init__(self, profile, image_size):
        self.profile = profile
        self.height, self.width = image_size
        camera = profile['camera']
        self.transform = np.asarray(camera['transform_base_from_raw_optical_m'], float)
        self.K = np.asarray(camera['intrinsic_matrix'], float)
        self.size = np.asarray(camera['resolution_wh'], float)
        self.scale = np.array([self.width, self.height])/self.size
        self.radius = float(profile['cup']['outer_diameter_m'])/2
        self.cup_height = float(profile['cup']['height_m'])
        angle = np.linspace(0, 2*np.pi, 64, endpoint=False)
        circle = np.c_[self.radius*np.cos(angle), self.radius*np.sin(angle), np.zeros_like(angle)]
        self.surface = np.r_[circle, circle+[0, 0, self.cup_height]]

    def project(self, points):
        camera = (np.asarray(points)-self.transform[:3, 3])@self.transform[:3, :3]
        if np.any(camera[:, 2] <= 0):
            raise ValueError('Cylinder behind camera')
        pixels = camera@self.K.T
        pixels = pixels[:, :2]/pixels[:, 2:]
        return ((self.size-1-pixels)+.5)*self.scale-.5

    def ray_at_table(self, pixel):
        raw = self.size-1-((np.asarray(pixel)+.5)/self.scale-.5)
        direction = self.transform[:3, :3]@np.linalg.solve(self.K, [*raw, 1.])
        depth = (self.profile['table_top_z_m']-self.transform[2, 3])/direction[2]
        if not np.isfinite(depth) or depth <= 0:
            raise ValueError('Invalid table intersection')
        return self.transform[:3, 3]+depth*direction

    def cup_hull(self, bottom):
        return cv2.convexHull(self.project(self.surface+bottom).astype(np.float32)).reshape(-1, 2)

    @staticmethod
    def component(mask, minimum):
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        indices = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum]
        if len(indices) != 1:
            raise ValueError('Unique visible cup and target required')
        return labels == indices[0], centers[indices[0]]

    def measure(self, rgb):
        if rgb.dtype != np.uint8 or rgb.shape != (self.height, self.width, 3):
            raise ValueError('Expected current RGB at the declared resolution')
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        cup, _ = self.component((hsv[:, :, 0] >= 15)&(hsv[:, :, 0] <= 45)&
                               (hsv[:, :, 1] > 85)&(hsv[:, :, 2] > 30), 15)
        card, center = self.component((hsv[:, :, 0] >= 95)&(hsv[:, :, 0] <= 130)&
                                      (hsv[:, :, 1] > 90)&(hsv[:, :, 2] > 25), 12)
        y, x = np.nonzero(cup)
        if min(x) <= 0 or max(x) >= self.width-1 or min(y) <= 0 or max(y) >= self.height-1:
            raise ValueError('Cup clipped by image boundary')
        bounds = np.array([min(x)-.5, min(y)-.5, max(x)+.5, max(y)+.5])
        initial = self.ray_at_table([(min(x)+max(x))/2, max(y)])
        initial[2] = self.profile['table_top_z_m']

        def residual(bottom):
            pixels = self.project(self.surface+bottom)
            return np.r_[pixels.min(0), pixels.max(0)]-bounds

        fit = least_squares(residual, np.clip(initial, [-.9, -.6, -.03], [-.15, .4, .42]),
            bounds=([-.9, -.6, -.03], [-.15, .4, .42]), max_nfev=30, diff_step=1e-4)
        error = float(np.sqrt(np.mean(fit.fun**2)))
        if not fit.success or error > 2.5:
            raise ValueError('Visible silhouette inconsistent with nominal upright cup')
        # Fit the visible outer edges in the nominal axis-aligned card plane;
        # the centroid of blue pixels moves when the cup covers the card centre.
        card_y, card_x = np.nonzero(card)
        raw = self.size-1-((np.c_[card_x, card_y]+.5)/self.scale-.5)
        rays = np.c_[raw, np.ones(len(raw))]@np.linalg.inv(self.K).T@self.transform[:3, :3].T
        depths = (self.profile['table_top_z_m']-self.transform[2, 3])/rays[:, 2]
        points = self.transform[:3, 3]+rays*depths[:, None]
        target = (points.min(0)+points.max(0))/2
        return fit.x, target, dict(bbox_fit_rmse_px=error,
            runtime_qualified=False, error_bound_m=None, method='current_RGB_nominal_cylinder_bbox')
