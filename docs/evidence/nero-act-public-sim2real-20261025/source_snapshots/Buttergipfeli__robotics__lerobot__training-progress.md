# Training progress

| | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- | --- |
| Data | 50 expert episodes, fixed scene | 300 expert episodes, domain randomisation (mat offset, ball friction and mass, camera pose and fov, lighting, mat and paper shade) | 300 expert episodes, run 2 randomisation plus appearance: ball colour and markings, speckled cork, floor shade, camera exposure, white balance, noise and blur; colours matched to a real frame | 1000 expert episodes, run 3 randomisation |
| Training | ACT, 20k steps, batch 8, Mac (MPS) | ACT, 20k of 50k steps, batch 32, RunPod RTX 5090, loss 5.36 → 0.20 | ACT, 50k steps, batch 32, RunPod RTX 5090; evaluated at the 20k checkpoint while training | ACT, 50k steps, batch 32, RunPod; evaluated at the 20k checkpoint while training |
| Sim eval, seed 100, 20 episodes | 11/20 (5 timeouts, 4 ball lost), no randomisation | 11/20 (7 timeouts, 2 ball lost), with randomisation | 8/20 at 20k (9 timeouts, 3 ball lost), with full randomisation | 11/20 at 20k (7 timeouts, 2 ball lost), with full randomisation |
| Real rollout | lifts, opens gripper, then oscillates ±1 cm | no coherent behaviour | pending | pending |

## Findings

- Same rate on a harder test. Run 2 holds the ball more reliably and fails by not
  finding it; all three balls near the far mat edge (x ≥ 0.34 m) timed out.
- Run 3 at 20k drops to 8/20 under the much harder appearance randomisation,
  again mostly timeouts. Its seed-100 scenes differ from runs 1 and 2 (more
  random draws per reset), so episodes are not comparable one to one.
- Run 4 at 20k is back to 11/20 on the same hard test: 1000 instead of 300
  episodes recover what the appearance randomisation cost at equal steps.
- Both real rollouts failed for reasons outside the policy:
  1. Joint units differed: sim normalises over MJCF ranges, the follower over its
     calibration ranges (wrist roll 78° off). Fixed by `real/joint_mapping.py`.
  2. The real ball is light blue with print and a barcode, the sim ball plain
     dark blue. Open.

## Next

- Match the ball colour to the real one, randomise colour and markings, add
  camera exposure and noise variation, re-record 300 episodes, retrain.
- Evaluate every checkpoint with 50 episodes at seed 100 so runs compare per
  episode.
