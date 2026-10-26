"""Episode-wise reproducible experimental domains and shared delayed servo."""
import copy
import numpy as np


def sample_domain(base, ranges, seed):
    config=copy.deepcopy(base)
    rng=np.random.default_rng(seed)
    def uniform(key):
        low,high=ranges[key]
        if not np.isfinite([low,high]).all() or low>high:
            raise ValueError(f'Invalid randomization range: {key}')
        return float(rng.uniform(low,high))
    offset=float(ranges['pick_offset_m'])
    if not 0<=offset<=0.05:raise ValueError('pick_offset_m must be in [0, .05]')
    config['pick_xy_m']=(np.array(base['pick_xy_m'])+rng.uniform(-offset,offset,2)).tolist()
    config['cup']['mass_kg']*=uniform('cup_mass_scale')
    # Optional geometry ranges are actual metres, not visual-only scaling.
    if 'cup_outer_diameter_m' in ranges:config['cup']['outer_diameter_m']=uniform('cup_outer_diameter_m')
    if 'cup_height_m' in ranges:config['cup']['height_m']=uniform('cup_height_m')
    diameter=config['cup']['outer_diameter_m'];height=config['cup']['height_m']
    if not 0<diameter<.1 or ('cup_outer_diameter_m' in ranges and diameter>.09) or height<=2*config['cup']['wall_thickness_m']:
        raise ValueError('Randomized cup geometry requires diameter <= .09 m and positive wall clearance')
    config['contact_friction'][0]=uniform('sliding_friction')
    scale=uniform('servo_gain_scale')
    if scale<=0:raise ValueError('Servo gain scale must be positive')
    config['arm_kp']=(np.array(base['arm_kp'])*scale).tolist()
    config['arm_kd']=(np.array(base['arm_kd'])*np.sqrt(scale)).tolist()
    config['finger_force_limit_n']=uniform('finger_force_n')
    low,high=ranges['action_delay_steps']
    if not (isinstance(low,int) and isinstance(high,int) and 0<=low<=high<=10):
        raise ValueError('action_delay_steps must contain integers in [0,10]')
    config['action_delay_steps']=int(rng.integers(low,high+1))
    config['appearance']={'table_rgb':rng.uniform(.25,.8,3).tolist(),
                          'cup_rgb':rng.uniform(.15,.85,3).tolist(),
                          'arm_rgb':rng.uniform(.2,.55,3).tolist(),
                          'background_rgb':rng.uniform(.05,.45,3).tolist(),
                          'light_scale':uniform('light_scale')}
    config['camera']={'lookat':[.3,0,.2],'distance':1.1*uniform('camera_distance_scale'),
                      'azimuth':90+uniform('camera_azimuth_offset_deg'),
                      'elevation':-45+uniform('camera_elevation_offset_deg')}
    config['image_noise_std_uint8']=uniform('image_noise_std_uint8')
    jitter=float(ranges.get('initial_joint_jitter_rad',0))
    if not 0<=jitter<=.05:raise ValueError('initial_joint_jitter_rad must be in [0,.05]')
    if jitter:config['initial_joint_offset_rad']=rng.uniform(-jitter,jitter,7).tolist()
    if 'arm_disturbance_torque_nm' in ranges:
        magnitude=uniform('arm_disturbance_torque_nm')
        if not 0<=magnitude<=2:raise ValueError('Experimental torque pulses must be in [0,2] Nm')
        config['arm_disturbance']={'seed':int(seed)+2000000,'max_torque_nm':magnitude,
                                   'period_steps':40,'duration_steps':3,'start_step':15}
    config['domain_randomization']={'seed':int(seed),'ranges':copy.deepcopy(ranges),
                                    'real_parameters_identified':False}
    if config['cup']['mass_kg']<=0 or config['contact_friction'][0]<=0 or scale<=0 or config['finger_force_limit_n']<=0:
        raise ValueError('Randomized physical parameters must be positive')
    return config


def disturbance_torque(config,control_step):
    """Deterministic physical force pulse. Never changes qpos or expert labels."""
    settings=config.get('arm_disturbance')
    if not settings:return np.zeros(7)
    start=int(settings['start_step']);period=int(settings['period_steps']);duration=int(settings['duration_steps'])
    magnitude=float(settings['max_torque_nm'])
    if start<0 or not 0<duration<=period or not 0<=magnitude<=2:
        raise ValueError('Invalid torque disturbance settings')
    if control_step<start or (control_step-start)%period>=duration:return np.zeros(7)
    pulse=(control_step-start)//period
    rng=np.random.default_rng(np.random.SeedSequence([int(settings['seed']),int(pulse)]))
    return rng.uniform(-magnitude,magnitude,7)


class NeroServo:
    """Implicit model damping plus position commands delayed at control boundaries."""
    def __init__(self,model,data,config):
        self.model,self.data,self.config=model,data,config
        model.dof_damping[:7]=config['arm_kd'];model.dof_damping[7:9]=4
        self.delay=int(config.get('action_delay_steps',0))
        if not 0<=self.delay<=10:raise ValueError('Invalid action delay')
        initial=np.r_[data.qpos[:7],data.qpos[7]-data.qpos[8]]
        self.queue=[initial.copy() for _ in range(self.delay)]
        self.active=initial.copy()

    def command(self,target,width):
        self.queue.append(np.r_[target,width].astype(float))
        self.active=self.queue.pop(0)

    def apply(self):
        m,d,c=self.model,self.data,self.config
        d.ctrl[:7]=np.clip(np.array(c['arm_kp'])*(self.active[:7]-d.qpos[:7])+d.qfrc_bias[:7],
                            -m.actuator_ctrlrange[:7,1],m.actuator_ctrlrange[:7,1])
        desired=np.array([self.active[7]/2,-self.active[7]/2])
        d.ctrl[7:9]=np.clip(400*(desired-d.qpos[7:9]),-c['finger_force_limit_n'],c['finger_force_limit_n'])
