import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, '/home/user/wq/nero_pick_place_20261016')
from perception import unique_blue_box
from table_blue import select_blue

HERE = Path(__file__).resolve().parent
SCENE = Path('/home/user/wq/nero_act_transfer_resume_20261021/results/unqualified_nominal_full_development/scene_configuration.json')


class TableBlueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(SCENE.read_text())
        cls.rgb = cv2.cvtColor(cv2.imread(str(HERE/'live_preview/camera_000.png')), cv2.COLOR_BGR2RGB)

    def test_actual_monitor_popup_does_not_hide_complete_table_card(self):
        with self.assertRaisesRegex(ValueError, 'convex quadrilateral'):
            unique_blue_box(self.rgb)
        box = select_blue(self.rgb, self.config, unique_blue_box)
        np.testing.assert_allclose(box, [385, 270, 452, 319], atol=4)

    def test_two_table_cards_are_still_rejected(self):
        rgb = self.rgb.copy()
        cv2.rectangle(rgb, (125, 275), (165, 310), (0, 0, 255), -1)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            select_blue(rgb, self.config, unique_blue_box)

    def test_blue_above_table_horizon_is_not_a_target(self):
        rgb = np.zeros_like(self.rgb)
        cv2.rectangle(rgb, (230, 5), (275, 30), (0, 0, 255), -1)
        with self.assertRaises(ValueError):
            select_blue(rgb, self.config, unique_blue_box)

    def test_target_crossing_projected_table_boundary_is_rejected(self):
        rgb = np.zeros_like(self.rgb)
        cv2.rectangle(rgb, (240, 30), (300, 160), (0, 0, 255), -1)
        with self.assertRaises(ValueError):
            select_blue(rgb, self.config, unique_blue_box)


if __name__ == '__main__':
    unittest.main()
