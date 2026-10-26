"""Independent complete-task scene. No ROS, SDK or hardware connections."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simulation.nero_scene_camera import add_calibrated_camera

ROOT = Path(__file__).resolve().parent


def build_scene(config):
    pick = np.asarray(config['pick_xy_m'], float)
    place = np.asarray(config['place_xy_m'], float)
    tcp = np.asarray(config['tcp_link7_xyz_m'], float)
    rotation = np.asarray(config['ik_goal_rotation_link7'], float)
    z = float(config['table_top_z_m'])
    force = float(config['finger_force_limit_n'])
    cup = config['cup']
    if (pick.shape != (2,) or place.shape != (2,) or tcp.shape != (3,)
            or not np.isfinite(np.r_[pick, place, tcp, z, force]).all()
            or not 0 < force <= 1):
        raise ValueError('Invalid finite geometry or low-force contact limit')
    if np.linalg.norm(place-pick) < config['minimum_transfer_m']:
        raise ValueError('Full transfer requires distinct source and destination')
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-8)
            or np.linalg.det(rotation) < .999):
        raise ValueError('Invalid proper grasp rotation')
    # The retained meshes and cup dimensions are a fixed physical source model.
    if not np.allclose([cup['height_m'], cup['outer_diameter_m']], [.095, .075]):
        raise ValueError('Changing cup dimensions requires regenerating its physical geometry')
    tree = ET.parse(ROOT/'scene_template.xml')
    root = tree.getroot()
    # Reduce regularized friction creep without increasing friction or grip force.
    option = root.find('option')
    option.set('cone', 'elliptic')
    option.set('impratio', str(config.get('solver_impratio', 10)))
    option.set('tolerance', '1e-10')
    for mesh in root.findall('asset/mesh'):
        mesh.set('file', str(ROOT/mesh.get('file')))
    world = root.find('worldbody')
    table = world.find("geom[@name='table']")
    table.set('pos', ' '.join(map(str, [*config['table_center_xy_m'], z-.025])))
    table.set('size', ' '.join(map(str, [*config['table_half_size_xy_m'], .025])))
    marker = world.find("geom[@name='target_marker']")
    marker.set('pos', ' '.join(map(str, [*place, z+.0005])))
    body = world.find("body[@name='cup']")
    for geom, color in [(table, config['appearance'].get('table_rgb'))]+[(g,config['appearance'].get('cup_rgb')) for g in body.findall('.//geom')]:
        if color is not None:
            if len(color)!=3 or not np.isfinite(color).all() or np.any(np.asarray(color)<0) or np.any(np.asarray(color)>1):
                raise ValueError('Appearance colors must be finite RGB in0..1')
            geom.set('rgba',' '.join(map(str,[*color,1.])))
    for texture in root.findall("asset/texture[@type='skybox']"):
        color=config['appearance'].get('background_rgb')
        if color is not None:
            texture.set('rgb1',' '.join(map(str,color)));texture.set('rgb2',' '.join(map(str,color)))
    body.set('pos', ' '.join(map(str, [*pick, z+cup['height_m']/2+.001])))
    world.remove(world.find("camera[@name='calibrated_nero_camera']"))
    add_calibrated_camera(world, config)
    baseline_light = .92
    for light in world.findall('light'):
        values = np.fromstring(light.get('diffuse'), sep=' ')
        light.set('diffuse', ' '.join(map(str, values*config['appearance']['light_scale']/baseline_light)))
    # Set the logical site to the selected link7 TCP; retain all rigid geometry.
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    link = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    parent = model.site_bodyid[site]
    desired = data.xpos[link]+data.xmat[link].reshape(3, 3)@tcp
    local = data.xmat[parent].reshape(3, 3).T@(desired-data.xpos[parent])
    root.find(".//site[@name='grasp_tcp']").set('pos', ' '.join(map(str, local)))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


def tcp_in_link7(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    link = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'grasp_tcp')
    rotation = data.xmat[link].reshape(3, 3)
    return (rotation.T@(data.site_xpos[site]-data.xpos[link]),
            rotation.T@data.site_xmat[site].reshape(3, 3))
