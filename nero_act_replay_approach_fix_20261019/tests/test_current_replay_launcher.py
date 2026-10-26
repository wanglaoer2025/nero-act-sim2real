import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]

class CurrentReplayLauncherTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('executor.current_replay_run'),
                             'current receive-only driver must have a scoped replay handoff')
        from executor import current_replay_run
        return current_replay_run

    def test_default_has_no_hardware_or_attempt_side_effect(self):
        self.module();output=ROOT/'evidence/real_attempt_01';before=output.exists()
        result=subprocess.run([sys.executable,'-m','executor.current_replay_run'],cwd=ROOT,
                              text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)['hardware_connected'])
        self.assertEqual(output.exists(),before)

    def test_ros_environment_accepts_existing_domain_and_preserves_paths(self):
        module=self.module();env=module.trial_environment(dict(ROS_DOMAIN_ID='42',PYTHONPATH='/ros/path'))
        self.assertEqual(env['ROS_DOMAIN_ID'],'42');self.assertEqual(env['PYTHONPATH'],'/ros/path')

    def test_owned_child_is_stopped(self):
        module=self.module()
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            module.stop_owned(child)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:child.kill();child.wait()

if __name__=='__main__':unittest.main()
