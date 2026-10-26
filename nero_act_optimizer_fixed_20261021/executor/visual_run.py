"""Offline visual15D readiness report. This implementation stage has no motion adapter."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--receipt',type=Path)
    args = parser.parse_args()
    if args.execute or args.receipt is not None:
        parser.error('This stage is offline only: measured calibration, qualified live frontend, complete15D evaluation and ROS adapter are pending')
    report = dict(mode='offline_visual15d_implementation_readiness',
                  model_exists=args.model.is_dir(),profile_exists=args.profile.is_file(),
                  motion_sent=False,hardware_connected=False,action_clients_created=0,
                  execution_ready=False,formal_training_started=False,
                  pending=['independent_real_calibration','qualified_live_frontend_and_exposure_clock',
                           'measured_geometry_dataset','frozen15d_complete_evaluation','ROS_visual_adapter'])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
