import os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
class SupervisorSignalTests(unittest.TestCase):
    def test_termination_and_hangup_run_cleanup_and_leave_detached_receiver_alive(self):
        root=Path(__file__).resolve().parents[1]
        for signum in [signal.SIGHUP,signal.SIGTERM]:
            with self.subTest(signum=signum),tempfile.TemporaryDirectory() as directory:
                p=Path(directory)
                code="""from executor.supervision import install_cleanup_signals,ignore_cleanup_signals,start_detached
import sys,time
from pathlib import Path
p=Path(sys.argv[1]);install_cleanup_signals()
try:
 (p/'ready').write_text('ready')
 while True:time.sleep(.02)
finally:
 ignore_cleanup_signals()
 with (p/'receiver.log').open('w') as log:
  receiver=start_detached([sys.executable,'-c','import time;time.sleep(20)'],cwd=sys.argv[2],log=log)
 (p/'receiver_pid').write_text(str(receiver.pid));(p/'cleaned').write_text('closed then restored')
"""
                parent=subprocess.Popen([sys.executable,'-c',code,str(p),str(root)],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                receiver_pid=None
                try:
                    end=time.monotonic()+3
                    while not (p/'ready').exists() and parent.poll() is None and time.monotonic()<end:time.sleep(.01)
                    self.assertTrue((p/'ready').exists(),'Cleanup signal installation is missing')
                    parent.send_signal(signum);parent.wait(timeout=3)
                    self.assertTrue((p/'cleaned').exists())
                    receiver_pid=int((p/'receiver_pid').read_text())
                    self.assertTrue(Path(f'/proc/{receiver_pid}/cmdline').read_bytes())
                    self.assertEqual(os.getsid(receiver_pid),receiver_pid)
                finally:
                    if parent.poll() is None:parent.kill();parent.wait()
                    if receiver_pid is not None:os.kill(receiver_pid,signal.SIGTERM)
