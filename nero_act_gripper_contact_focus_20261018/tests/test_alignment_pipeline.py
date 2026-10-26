import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class AlignmentPipelineTests(unittest.TestCase):
    def test_reusing_a_pipeline_never_overwrites_its_existing_status(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp)/'evidence'
            evidence.mkdir()
            script = evidence/'run18_pipeline.py'
            shutil.copy2(source/'evidence/run18_pipeline.py', script)
            status = evidence/'run18_pipeline_status.json'
            original = b'{"stage":"prior_completed_run","evidence":"keep me"}\n'
            status.write_bytes(original)
            env = dict(os.environ,PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1')
            run = subprocess.run([sys.executable,str(script)],env=env,capture_output=True,timeout=30)
            self.assertNotEqual(run.returncode,0)
            self.assertEqual(status.read_bytes(),original)
