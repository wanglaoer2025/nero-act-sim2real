import contextlib
import io
import unittest
from unittest.mock import patch
from src.act import test_joint_stream


class OfflineJointStreamTests(unittest.TestCase):
    def test_plain_dry_run_does_not_construct_robot_or_import_sdk(self):
        with patch('sys.argv',['test_joint_stream','--dry-run']), \
             patch.object(test_joint_stream,'NeroRobotWrapper',side_effect=AssertionError('SDK must stay unused')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(test_joint_stream.main(),0)
        self.assertIn('未连接硬件',output.getvalue())


if __name__=='__main__':unittest.main()
