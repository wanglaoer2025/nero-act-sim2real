import unittest
from executor.gate import GateLatch
class StopLatchTests(unittest.TestCase):
    def test_stop_blocks_all_future_open_requests_even_after_successful_close(self):
        latch=GateLatch();latch.acknowledge(latch.request(False),True)
        latch.acknowledge(latch.request(True),True)
        latch.stop();latch.acknowledge(latch.request(False),True)
        self.assertFalse(latch.open);self.assertIsNone(latch.request(True))
    def test_inflight_open_acknowledgement_cannot_reopen_stopped_latch(self):
        latch=GateLatch();latch.acknowledge(latch.request(False),True)
        sequence=latch.request(True);latch.stop();latch.acknowledge(sequence,True)
        self.assertFalse(latch.open);self.assertIsNone(latch.request(True))
    def test_readonly_receiver_survives_launcher_exit_in_its_own_session(self):
        import os,signal,subprocess,sys,tempfile
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            pidpath=Path(directory)/'pid'
            command="from executor.supervision import start_detached; import sys; p=start_detached([sys.executable,'-c','import time;time.sleep(20)'],cwd=sys.argv[1],log=open(sys.argv[2]+'.log','w'));open(sys.argv[2],'w').write(str(p.pid))"
            parent=subprocess.run([sys.executable,'-c',command,str(root),str(pidpath)],cwd=root,capture_output=True,text=True,timeout=3)
            self.assertEqual(parent.returncode,0,parent.stderr);pid=int(pidpath.read_text())
            try:
                self.assertTrue(Path(f'/proc/{pid}/cmdline').read_bytes());self.assertEqual(os.getsid(pid),pid)
            finally:os.kill(pid,signal.SIGTERM)
