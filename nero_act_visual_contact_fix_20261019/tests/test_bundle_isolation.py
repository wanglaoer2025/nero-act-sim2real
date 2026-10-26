import importlib.util
import subprocess
import sys
import unittest


class BundleIsolationTests(unittest.TestCase):
    def test_permission_verifier_does_not_load_policy_or_ros_namespaces(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.bundle'))
        result = subprocess.run([sys.executable, '-c',
            "import sys; from executor.bundle import verify_model_bundle; assert 'src' not in sys.modules; assert 'rclpy' not in sys.modules; assert 'pyAgxArm' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':unittest.main()
