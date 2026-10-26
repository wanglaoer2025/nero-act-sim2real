import importlib.util
import json
import os
import subprocess
import sys
import unittest


class DriverCliTests(unittest.TestCase):
    def test_motion_profile_cannot_hide_in_another_ros_domain(self):
        env = dict(os.environ, ROS_DOMAIN_ID='43')
        result = subprocess.run([sys.executable, '-m', 'executor.driver', '--execute',
            '--model', '/missing', '--evaluation', '/missing', '--receipt', '/missing'],
            capture_output=True, text=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ROS_DOMAIN_ID=42', result.stderr)

    def test_default_never_connects_and_execute_needs_current_artifacts(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.driver'))
        default = subprocess.run([sys.executable, '-m', 'executor.driver'], capture_output=True, text=True)
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertFalse(json.loads(default.stdout)['hardware_connected'])
        denied = subprocess.run([sys.executable, '-m', 'executor.driver', '--execute'], capture_output=True, text=True)
        self.assertNotEqual(denied.returncode, 0)
        self.assertIn('model, evaluation, receipt', denied.stderr)


if __name__ == '__main__':unittest.main()
