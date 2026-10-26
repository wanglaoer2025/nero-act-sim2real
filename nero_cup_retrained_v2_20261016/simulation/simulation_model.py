"""Portable unchanged robot/contact model; only per-scene object pose and lighting vary."""
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
ROOT=Path(__file__).resolve().parent

def scene(config):
 tree=ET.parse(ROOT/'scene_template.xml');root=tree.getroot()
 for mesh in root.findall('asset/mesh'):mesh.set('file',str(ROOT/mesh.get('file')))
 cup=root.find("worldbody/body[@name='cup']")
 cup.set('pos',' '.join(map(str,[*config['pick_xy_m'],config['table_top_z_m']+config['cup']['height_m']/2+.001])))
 template=__import__('json').loads((ROOT/'template_config.json').read_text())
 scale=config['appearance']['light_scale']/template['appearance']['light_scale']
 for light in root.findall('worldbody/light'):light.set('diffuse',' '.join(str(float(x)*scale) for x in light.get('diffuse').split()))
 return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
