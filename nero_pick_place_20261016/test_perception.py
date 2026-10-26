import unittest
import cv2
import numpy as np

from perception import unique_blue_box, require_current_targets, revalidated_targets, yellow_cup_box


class PerceptionTests(unittest.TestCase):
    def test_cup_outline_does_not_follow_yolo_shadow_padding(self):
        rgb = np.zeros((360, 640, 3), np.uint8)
        cv2.rectangle(rgb, (252, 182), (301, 255), (255, 255, 0), -1)
        first = yellow_cup_box(rgb, [[250, 180, 305, 262]])
        cv2.rectangle(rgb, (252, 220), (301, 255), (130, 130, 0), -1)
        second = yellow_cup_box(rgb, [[250, 181, 309, 257]])
        np.testing.assert_equal(first, [252, 182, 301, 255])
        np.testing.assert_equal(second, first)

    def test_yellow_refinement_preserves_unique_cup_and_rejects_ambiguity(self):
        rgb = np.zeros((360, 640, 3), np.uint8)
        for boxes in [[], [[100, 100, 200, 200], [300, 100, 400, 200]]]:
            with self.assertRaises(ValueError):
                yellow_cup_box(rgb, boxes)
        with self.assertRaises(ValueError):
            yellow_cup_box(rgb, [[100, 100, 200, 200]])
        cv2.rectangle(rgb, (110, 110), (145, 190), (255, 255, 0), -1)
        cv2.rectangle(rgb, (155, 110), (190, 190), (255, 255, 0), -1)
        with self.assertRaises(ValueError):
            yellow_cup_box(rgb, [[100, 100, 200, 200]])
    def test_reacquisition_must_be_the_same_targets(self):
        original = {'plan': {'source_bottom_mm': [-500, 0, 0],
                             'destination_mm': [-500, -160, 0]}, 'capture_wall_s': 1}
        observed = {'source_bottom_mm': [-499, 0, 0], 'destination_mm': [-500, -161, 0]}
        refreshed = revalidated_targets(original, observed, 100)
        self.assertEqual(refreshed['capture_wall_s'], 100)
        self.assertEqual(original['capture_wall_s'], 1)
        self.assertEqual(refreshed['plan'], original['plan'])
        observed['source_bottom_mm'] = [-490, 0, 0]
        with self.assertRaises(ValueError):
            revalidated_targets(original, observed, 100)
    def test_only_one_complete_blue_quadrilateral(self):
        image = np.zeros((360, 640, 3), np.uint8)
        cv2.rectangle(image, (200, 120), (250, 160), (0, 100, 255), -1)
        np.testing.assert_equal(unique_blue_box(image), [200, 120, 250, 160])
        cv2.rectangle(image, (400, 120), (450, 160), (0, 100, 255), -1)
        with self.assertRaises(ValueError):
            unique_blue_box(image)

    def test_missing_or_truncated_card_rejected(self):
        image = np.zeros((360, 640, 3), np.uint8)
        with self.assertRaises(ValueError):
            unique_blue_box(image)
        cv2.rectangle(image, (0, 120), (50, 160), (0, 100, 255), -1)
        with self.assertRaises(ValueError):
            unique_blue_box(image)

    def test_execution_geometry_must_be_fresh_and_explicit(self):
        data = dict(capture_wall_s=100, geometry_fit_passed=True,
                    table_z_base_mm=0, tcp_approximate=True,
                    source_method='YOLO cup', destination_method='unique HSV blue quadrilateral')
        require_current_targets(data, 110)
        for now in [99, 191, float('nan')]:
            with self.assertRaises(ValueError):
                require_current_targets(data, now)
        data['geometry_fit_passed'] = False
        with self.assertRaises(ValueError):
            require_current_targets(data, 110)


if __name__ == '__main__':
    unittest.main()
