"""Image-derived observations for OFFLINE training with retained nominal geometry.

This estimator fits a known upright cylinder silhouette, without simulator truth
or robot state. It is deliberately separate from the qualified live observer:
nominal dimensions and colour segmentation do not qualify real cup/handle safety.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import h5py
import mujoco
import numpy as np
from scipy.optimize import least_squares

from simulation.scene import build_scene
from training.generalized import validate_manifest
from training.contact_sampling import visual_sampling_windows,PROFILE as CONTACT_SAMPLING_PROFILE
from training.visual_dataset import audit_visual_episode

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_profile(scene):
    profile = dict(schema='retained_nominal_offline_training_v1', verified=False,
        hardware_deployment_allowed=False, camera=scene['camera_calibration'],
        tcp_link7_xyz_m=scene['tcp_link7_xyz_m'], cup=scene['cup'],
        table_top_z_m=scene['table_top_z_m'],
        limitations=['TCP is model nominal, not newly measured',
            'Camera fit is historical, not an independent task-space accuracy bound',
            'Cup is upright cylinder and card sides are base-axis aligned; real handle/rim frontend unqualified',
            'No permission or qualification for physical execution'])
    if profile['camera']['image_rotation_degrees'] != 180:
        raise ValueError('Expected existing upright180-degree camera convention')
    profile['sha256'] = hashlib.sha256(json.dumps(profile, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return profile


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


def derive_episode(source, target, *, expected_sha, profile, image_size=(180, 320)):
    source, target = Path(source), Path(target)
    if sha(source) != expected_sha:
        raise ValueError('Source physical demonstration changed')
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    code_hash = sha(__file__)
    requested = sorted({t for _, t in visual_sampling_windows([source],20)})
    valid, rejected, errors, stages_valid = [], Counter(), [], Counter()
    with h5py.File(source, 'r') as old, h5py.File(target, 'r+') as f:
        scene = json.loads(old.attrs['scene_config'])
        if training_profile(scene)['sha256'] != profile['sha256']:
            raise ValueError('Episode camera/TCP/nominal geometry differs from retained profile')
        model = build_scene(scene); data = mujoco.MjData(model)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
        estimator = NominalImageEstimator(profile, image_size)
        n = len(f['action']); times = old['timestamp'][:]
        stages = old['simulation/stage'].asstr()[:]
        del f['observations/images/camera_0']
        images = f.create_dataset('observations/images/camera_0', (n, *image_size, 3), dtype='u1',
                                  chunks=(1, *image_size, 3), compression='lzf')
        features = np.full((n, 6), np.nan, np.float32)
        valid_mask = np.zeros(n, bool)
        # This is a synthetic simulator timeline, never evidence of a real camera clock.
        f['action_time_monotonic_s'] = times+.001
        f['observations/camera_monotonic_s'] = times
        f['observations/feedback_monotonic_s'] = times
        f['observations/frame_id'] = np.arange(n)
        with mujoco.Renderer(model, height=image_size[0], width=image_size[1]) as renderer:
            for t in requested:
                data.qpos[:] = old['simulation/qpos'][t]
                data.qvel[:] = old['simulation/qvel'][t]
                data.time = float(times[t]); mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='calibrated_nero_camera')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                rgb = renderer.render().copy()
                try:
                    cup, card, diagnostic = estimator.measure(rgb)
                except ValueError as exc:
                    rejected[str(exc)] += 1
                    continue
                # Truth is used ONLY for offline diagnostic arrays, never feature creation/filtering.
                truth = old['observations/object_xyz'][t]-[0, 0, scene['cup']['height_m']/2]
                errors.append(float(np.linalg.norm(cup-truth)))
                features[t] = np.r_[cup-data.site_xpos[site], card-data.site_xpos[site]]
                images[t] = rgb; valid_mask[t] = True
                valid.append(t); stages_valid[stages[t]] += 1
        if not valid or any(stages_valid[s] == 0 for s in set(stages)):
            raise ValueError('Image estimator lacks valid inputs for a complete task stage: '+str(stages_valid))
        f['observations/visual_relative_m'] = features
        f['observations/visual_valid'] = valid_mask
        # Unknown real bound remains NaN; finite nominal training tolerance is NOT a measured bound.
        f['observations/visual_error_bound_m'] = np.where(valid_mask, .03, np.nan)
        f['simulation/image_estimator_truth_error_m'] = np.array(errors)
        f.attrs.update(visual_observation_profile='visual_relative_base_v1',
            visual_sampling_profile=CONTACT_SAMPLING_PROFILE,
            source_group=scene['generalization']['source_group'], split=scene['generalization']['split'],
            action_semantics='submitted_absolute_target', initial_previous_gripper_target_m=float(old['observations/gripper_state'][0, 0]),
            calibration_sha256=profile['sha256'], observation_code_sha256=code_hash,
            clock_domain='synthetic_simulation_timeline', visual_runtime_qualified=False,
            visual_bound_kind='unverified_nominal_training_tolerance_not_real_error_bound',
            hardware_deployment_allowed=False, source_path=str(source.resolve()), source_sha256=expected_sha,
            observation_derivation='fresh_preaction_physics_render_then_RGB_only_cylinder_estimation')
        for key in ['action', 'observations/qpos', 'observations/gripper_state', 'simulation/qpos', 'simulation/qvel']:
            if not np.array_equal(f[key][:], old[key][:]):
                raise ValueError('Physical source labels or states changed: '+key)
    audit_visual_episode(target)
    return dict(path=str(target.resolve()), sha256=sha(target), requested=len(requested), valid=len(valid),
        rejected=dict(rejected), stages_valid=dict(stages_valid),
        image_estimator_error_m=dict(median=float(np.median(errors)), p95=float(np.percentile(errors, 95)), max=float(max(errors))),
        truth_used_as_policy_input=False, physical_labels_unchanged=True)


def prepare_row(arguments):
    row, output, profile = arguments
    report = derive_episode(row['path'], Path(output)/Path(row['path']).name,
                            expected_sha=row['sha256'], profile=profile)
    return dict(row, **report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    manifest = json.loads((args.source/'manifest.json').read_text())
    split = json.loads((args.source/'split.json').read_text())
    validate_manifest(manifest, split)
    profile = json.loads(args.profile.read_text())
    args.output.mkdir(exist_ok=False, parents=True)
    rows = []
    if not 1 <= args.workers <= 2: raise ValueError('Use one or two offline rendering workers')
    jobs = [(row, str(args.output), profile) for row in manifest['episodes']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for report in pool.map(prepare_row, jobs):
            rows.append(report)
            print(json.dumps(report), flush=True)
            (args.output/'preparation_status.json').write_text(json.dumps(dict(completed=len(rows), total=48, latest=report), indent=2))
    derived = dict(manifest, episodes=rows, calibration_sha256=profile['sha256'],
        complete=True, visual_profile='visual_relative_base_v1', hardware_deployment_allowed=False,
        visual_sampling_profile=CONTACT_SAMPLING_PROFILE,
        scope='retained_nominal_calibration_offline_training_not_measured_geometry_dataset')
    (args.output/'manifest.json').write_text(json.dumps(derived, indent=2))
    shutil.copy2(args.source/'split.json', args.output/'split.json')
    shutil.copy2(args.source/'scene_config.json', args.output/'scene_config.json')
    shutil.copy2(args.profile, args.output/'calibration_profile.json')


if __name__ == '__main__':
    main()
