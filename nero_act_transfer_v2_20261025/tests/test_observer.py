import cv2
import numpy as np
import pytest


def picture():
    rng = np.random.default_rng(42)
    im = np.repeat(rng.integers(110, 210, (180, 320, 1), dtype=np.uint8), 3, axis=2)
    im[100:151, 190:241] = [0, 60, 210]
    return im


def profile():
    return dict(table_top_z_m=0., cup={'outer_diameter_m':.075,'height_m':.095},
                camera={'resolution_wh':[320,180], 'intrinsic_matrix':[[500,0,160],[0,500,90],[0,0,1]],
                        'transform_base_from_raw_optical_m':[[1,0,0,0],[0,-1,0,0],[0,0,-1,1],[0,0,0,1]]})


def test_visible_corner_and_two_edges_support_partial_card_without_centroid_shift():
    from observer import RegisteredCardObserver
    im = picture()
    ob = RegisteredCardObserver(profile(), im.shape[:2])
    ob.register([im, im.copy(), im.copy()])
    before = ob.measure(im, 1., 1)
    partial = im.copy()
    partial[112:145, 203:241] = [220, 190, 0]
    after = ob.measure(partial, 1.05, 2)
    np.testing.assert_allclose(after['target_base_m'], before['target_base_m'], atol=.002)
    assert after['source'] == 'registered_card_constrained_by_current_geometry'
    assert after['current_supported_edges'] >= 2 and after['current_visible_corners'] >= 1
    assert after['real_geometry_qualified'] is False


def test_full_occlusion_and_single_edge_cannot_use_registered_target():
    from observer import RegisteredCardObserver
    im = picture()
    ob = RegisteredCardObserver(profile(), im.shape[:2])
    ob.register([im]*3)
    hidden = im.copy(); hidden[98:154, 188:244] = [220,190,0]
    with pytest.raises(ValueError): ob.measure(hidden, 1., 1)
    edge = im.copy(); edge[103:154, 188:244] = [220,190,0]
    with pytest.raises(ValueError): ob.measure(edge, 1.05, 2)


def test_card_motion_with_static_background_is_rejected():
    from observer import RegisteredCardObserver
    im = picture()
    ob = RegisteredCardObserver(profile(), im.shape[:2])
    ob.register([im]*3)
    shifted = im.copy(); shifted[100:151,190:241]=[160]*3
    shifted[100:151,200:251]=[0,60,210]
    with pytest.raises(ValueError): ob.measure(shifted, 1., 1)


def test_background_motion_and_repeated_frame_are_rejected():
    from observer import RegisteredCardObserver
    im=picture(); ob=RegisteredCardObserver(profile(),im.shape[:2]);ob.register([im]*3)
    moved=np.roll(im, 3, axis=1)
    with pytest.raises(ValueError): ob.measure(moved, 1., 1)
    ob.measure(im, 1.05, 2)
    with pytest.raises(ValueError): ob.measure(im, 1.1, 2)


def test_registration_requires_temporal_agreement_not_one_image():
    from observer import RegisteredCardObserver
    im=picture();ob=RegisteredCardObserver(profile(),im.shape[:2])
    with pytest.raises(ValueError):ob.register([im])


def test_unified_observer_uses_current_rgb_and_only_joint_based_tcp(tmp_path):
    from observer import UnifiedObserver
    # Project a nominal cylinder using the same optical geometry, no task xyz input.
    im=picture()
    calibration=profile()
    calibration['camera']['transform_base_from_raw_optical_m'][0][3]=-.4
    theta=np.linspace(0,2*np.pi,64,endpoint=False)
    ring=np.c_[.0375*np.cos(theta),.0375*np.sin(theta),np.zeros(64)]
    points=np.r_[ring,ring+[0,0,.095]]+[-.33,-.02,0]
    T=np.array(calibration['camera']['transform_base_from_raw_optical_m'])
    camera=(points-T[:3,3])@T[:3,:3]
    uv=camera@np.array(calibration['camera']['intrinsic_matrix']).T
    pixels=np.array([319,179])-uv[:,:2]/uv[:,2:]
    hull=cv2.convexHull(np.rint(pixels).astype('int32'))
    cv2.fillConvexPoly(im,hull,(220,190,0))
    observed=[]
    def tcp(q):
        observed.append(q.copy())
        return np.array([.1,.2,.3])
    ob=UnifiedObserver(calibration,im.shape[:2],tcp_from_q=tcp)
    ob.register([im]*3)
    state=np.r_[np.arange(7)*.01,.095]
    result=ob.measure(im,1.,1,state)
    assert result['relative_m'].shape==(6,)
    np.testing.assert_array_equal(observed[0],state[:7])
    np.testing.assert_allclose(result['relative_m'][3:],np.asarray(result['card']['target_base_m'])-[.1,.2,.3])


def test_occluded_top_edge_does_not_discard_three_visible_card_edges():
    from pathlib import Path
    import json
    from observer import RegisteredCardObserver
    fixture=Path(__file__).parent/'fixtures'
    initial=cv2.cvtColor(cv2.imread(str(fixture/'card_initial.png')),cv2.COLOR_BGR2RGB)
    partial=cv2.cvtColor(cv2.imread(str(fixture/'card_partial.png')),cv2.COLOR_BGR2RGB)
    ob=RegisteredCardObserver(json.loads((fixture/'card_profile.json').read_text()),initial.shape[:2])
    ob.register([initial]*3)
    a=ob.measure(initial,1.,1)
    b=ob.measure(partial,1.05,2)
    assert b['current_supported_edges']>=2
    np.testing.assert_allclose(a['target_base_m'],b['target_base_m'],atol=.003)
def test_real_failed_sim_frame_fragment_can_join_but_separate_targets_still_reject():
    import cv2,numpy as np,pytest
    from pathlib import Path
    from observer import current_cup_component
    p=Path(__file__).resolve().parents[1]/'evidence/visibility_failure_diagnosis/yellow_mask.png'
    mask=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)>0
    joined=current_cup_component(mask)
    assert joined.sum()>13529
    multiple=np.zeros((100,160),np.uint8)
    multiple[20:70,20:60]=1;multiple[20:70,90:130]=1
    with pytest.raises(ValueError):current_cup_component(multiple)
    with pytest.raises(ValueError):current_cup_component(np.zeros_like(multiple))


def test_enclosed_cup_colour_islands_are_one_observed_body():
    from pathlib import Path
    from observer import current_cup_component
    root=Path(__file__).resolve().parents[1]/'evidence/edge_failure_diagnosis'
    for folder,frame in [('train_02_00',1400),('train_02_01',1313)]:
        mask=cv2.imread(str(root/folder/f'mask_{frame}.png'),0)>0
        body=current_cup_component(mask)
        assert np.count_nonzero(body)==np.count_nonzero(mask)


def test_exterior_colour_candidate_is_not_absorbed_into_cup():
    from observer import current_cup_component
    mask=np.zeros((100,160),np.uint8)
    mask[20:80,20:100]=1
    mask[40:46,103:109]=1
    with pytest.raises(ValueError):current_cup_component(mask)


def test_previous_valid_component_outputs_are_bit_identical():
    from observer import current_cup_component
    from vendor.cylinder import NominalImageEstimator
    rng=np.random.default_rng(42)
    for _ in range(100):
        mask=np.zeros((100,160),np.uint8)
        centre=tuple(int(v) for v in rng.integers([40,30],[120,70]))
        cv2.ellipse(mask,centre,(20,25),float(rng.integers(0,180)),0,360,1,-1)
        old=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        expected=NominalImageEstimator.component(old,15)[0]
        np.testing.assert_array_equal(current_cup_component(mask),expected)

