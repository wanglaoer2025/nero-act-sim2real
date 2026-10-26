import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from observer import UnifiedObserver


FIXTURE = Path(__file__).parent / 'fixtures'


def observer_and_image():
    rgb = cv2.cvtColor(cv2.imread(str(FIXTURE / 'cup_split_by_finger.png')), cv2.COLOR_BGR2RGB)
    profile = json.loads((FIXTURE / 'cup_split_profile.json').read_text())
    ob = UnifiedObserver(profile, rgb.shape[:2], tcp_from_q=lambda q: np.zeros(3))
    ob.register([rgb] * 3)
    return ob, rgb


def test_finger_split_cup_uses_current_projected_geometry():
    ob, rgb = observer_and_image()
    result = ob.measure(rgb, 1., 1, np.r_[np.zeros(7), .075])
    # Independent simulator diagnostic is an assertion only, never an input.
    np.testing.assert_allclose(result['cup_bottom_base_m'], [-.54651, .08453, .16763], atol=.002)
    assert result['truth_used_as_input'] is False
    assert result['real_geometry_qualified'] is False


def test_carry_transition_uses_contour_constraints_when_bbox_fit_is_biased():
    ob, _ = observer_and_image()
    rgb = cv2.cvtColor(cv2.imread(str(FIXTURE / 'cup_split_carry_transition.png')), cv2.COLOR_BGR2RGB)
    ob = UnifiedObserver(ob.cylinder.profile, rgb.shape[:2], tcp_from_q=lambda q: np.zeros(3))
    ob.register([rgb] * 3)
    result = ob.measure(rgb, 1., 1, np.r_[np.zeros(7), .075])
    # Existing bbox readout remains nominal: this is NOT a real accuracy bound.
    assert np.linalg.norm(result['cup_bottom_base_m'] - [-.54654, .08449, .19424]) < .005
    assert result['truth_used_as_input'] is False


def test_separate_second_cup_is_not_hidden_by_geometry_association():
    ob, rgb = observer_and_image()
    second = rgb.copy()
    hull = ob.cylinder.cup_hull(np.array([-.60, -.07, .16]))
    cv2.fillConvexPoly(second, np.rint(hull).astype('int32'), (220, 190, 0))
    with pytest.raises(ValueError):
        ob.measure(second, 1., 1, np.r_[np.zeros(7), .075])


def test_fragment_outside_projected_body_is_not_discarded():
    ob, rgb = observer_and_image()
    wrong = rgb.copy()
    wrong[296:303, 462:469] = [220, 190, 0]
    with pytest.raises(ValueError):
        ob.measure(wrong, 1., 1, np.r_[np.zeros(7), .075])


def test_missing_and_image_clipped_cup_still_stop():
    ob, rgb = observer_and_image()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45) & (hsv[:, :, 1] > 85)
    hidden = rgb.copy(); hidden[yellow] = 120
    with pytest.raises(ValueError):
        ob.measure(hidden, 1., 1, np.r_[np.zeros(7), .075])
    ob, rgb = observer_and_image()
    rgb[:450, :20] = [220, 190, 0]
    with pytest.raises(ValueError):
        ob.measure(rgb, 1., 1, np.r_[np.zeros(7), .075])
