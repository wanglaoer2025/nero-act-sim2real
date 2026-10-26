"""Fixed 0.5-second joint-stream probe using the existing NERO control entry."""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='offline preview (default)')
    mode.add_argument('--execute', action='store_true', help='enable and move real arm')
    parser.add_argument('--joint-index', type=int, choices=range(7), default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.execute and not Path('/sys/class/net/can0').exists():
        parser.error('can0 is absent; run this package on the real control computer')
    if args.execute:
        print('真实短测试：先停止 ROS CAN 驱动及其他控制程序，确保 CAN 只有一个控制者。', flush=True)
        print('空夹爪悬空，所选关节扫掠范围无人和障碍物；退出不会失能。', flush=True)
        print('20 Hz / 0.5 秒 / ±0.005 rad / 速度5%；随后回到测试起点。', flush=True)
        print('回位最长等待10秒；异常或Ctrl-C触发电子急停，需准备机械支撑。', flush=True)
        print('这不是v9策略测试。控制器速度设置5%不会自动恢复。', flush=True)
        if input('确认独占CAN并检查现场后输入 SINGLE_CAN：').strip() != 'SINGLE_CAN':
            return 0
    command = [sys.executable, '-m', 'src.act.test_joint_stream',
               '--execute' if args.execute else '--dry-run',
               '--joint-index', str(args.joint_index), '--frequency', '20',
               '--duration', '0.5', '--amplitude-rad', '0.005',
               '--speed-percent', '5', '--log-dir', str(root / 'logs')]
    # Keep the existing entry's signal handling and hardware confirmations.
    import os
    os.chdir(root / 'nero_act_source')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    raise SystemExit(main())
