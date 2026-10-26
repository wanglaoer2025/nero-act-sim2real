"""Receiver lifetime is independent of the interactive task's terminal."""
import subprocess
import signal


def install_cleanup_signals():
    def cleanup(signum, _frame):
        raise KeyboardInterrupt(f'Supervisor cleanup requested by signal {signum}')
    for signum in [signal.SIGHUP, signal.SIGTERM]:
        signal.signal(signum, cleanup)


def ignore_cleanup_signals():
    for signum in [signal.SIGINT, signal.SIGHUP, signal.SIGTERM]:
        signal.signal(signum, signal.SIG_IGN)


def start_detached(command, *, cwd, log):
    return subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
