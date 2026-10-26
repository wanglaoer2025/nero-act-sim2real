from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


class ShadowCliTests(unittest.TestCase):
    def test_help_works_without_ros_setup_and_execute_is_not_an_option(self):
        env = dict(os.environ, ROS_DOMAIN_ID='101', PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-m', 'executor.shadow', '--help'],
                                env=env, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'not_created'
            result = subprocess.run([sys.executable, '-m', 'executor.shadow', '--model', directory,
                                     '--output', str(target), '--execute'], env=env,
                                    capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
