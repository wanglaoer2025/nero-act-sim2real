"""Plan the historical six poses on ROS feedback; never submit an action goal."""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/home/user/zyh/data/nero-teacher/runtime-v4')
from src.utils.moveit import (
    MoveItSession, beaker_collision_cylinders, joint_state_after_trajectory,
)


def main():
    data = json.loads((ROOT / 'historical_parameter_plan.json').read_text())
    plan = data['cartesian_plan']
    result = dict(started_wall_s=time.time(), motion_sent=False, stages=[])
    session = MoveItSession(link_name='link7', timeout_s=60)
    collision = []
    try:
        start = session.joint_state()
        result['actual_start'] = dict(zip(start.name, start.position))
        session.add_table_collision(np.array([[-650, 0], [-300, 300], [0, 500]]), 0)
        collision = beaker_collision_cylinders(
            [('teacher_source', np.asarray(plan['source_bottom_mm']))], 0, 95, 75, 5,
        )
        session.update_beaker_collisions(collision, add=True)
        for name in ('pregrasp_high_pose', 'grasp_pose', 'lift_pose',
                     'carry_pose', 'place_pose', 'retreat_pose'):
            print('PLANNING', name, flush=True)
            trajectory = session.plan_to_pose(
                plan['flange'][name], .01 if name in ('grasp_pose', 'place_pose') else .05,
                start_state=start, description=name,
            )
            duration = trajectory.points[-1].time_from_start
            result['stages'].append(dict(
                name=name, points=len(trajectory.points),
                duration_s=duration.sec + duration.nanosec / 1e9,
                joint_names=trajectory.joint_names,
                endpoint=list(trajectory.points[-1].positions),
            ))
            print(json.dumps(result['stages'][-1]), flush=True)
            start = joint_state_after_trajectory(start, trajectory)
            if name == 'pregrasp_high_pose':
                session.update_beaker_collisions(collision, add=False)
        result['all_stages_planned'] = True
    except Exception as exc:
        result.update(all_stages_planned=False, error=f'{type(exc).__name__}: {exc}')
        print(result['error'], flush=True)
    finally:
        if collision:
            try:
                session.update_beaker_collisions(collision, add=False)
            except Exception as exc:
                result['scene_cleanup_error'] = str(exc)
        session.close()
        (ROOT / 'moveit_preview.json').write_text(json.dumps(result, indent=2))
    return 0 if result.get('all_stages_planned') else 1


if __name__ == '__main__':
    raise SystemExit(main())
