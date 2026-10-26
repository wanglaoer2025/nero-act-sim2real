What this change does: It prepares fresh image-derived 15D observations from the retained 48 full-physics demonstrations and trains the full ACT/CVAE policy with ResNet layer4 and the gripper branch. It explicitly treats the existing camera/TCP calibration as an unverified offline baseline and leaves hardware disabled.

Review scope: new formal offline continuation named in /tmp/nero_visual_review_package.txt, its dataset, initializer, action loader, policy loss, manifest checks, camera projection and integrity dependencies. Expected workload: two 20-epoch candidates over 36 training and 12 disjoint validation episodes.

Critical: none found.

Important:

1. Success summary precedes the integrity gate (training/visual.py:250–253).
   The code writes artifacts/training_summary.json with complete=true before finalize_training calls require_unchanged(protected). If a protected source, dataset, or protocol file changes during optimization, finalize_training raises as intended, but the durable summary still claims successful completion. The CLI also writes failure.json, leaving contradictory results. This does not bypass require_finalized export checks, but violates the requested truthful completion reporting.
   Smallest fix: call finalize_training successfully before writing the complete=true summary; keep the final status write after both. Add a regression that a failed final integrity check does not produce a successful summary.

Minor: none found.

Checks: 29 scoped tests passed with PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MPLCONFIGDIR=/tmp/nero_visual_mpl, .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_existing_calibration_training.py tests/test_visual_dataset.py tests/test_visual_policy.py. One existing matplotlib Axes3D import warning. Source tracing confirms fresh render at sampled pre-action states, image-only cup/card estimation, simulator object truth restricted to diagnostics, causal previous gripper target lookup, unchanged physical action arrays, train-only normalization, full posterior training, zero-latent validation, explicit optimizer groups, and completion epoch count checking. No additional concrete data leak or normalization/alignment bug found in these paths.

Not checked: GPU/full training execution; full regenerated 48-episode dataset acceptance and estimator error distribution (preparation was not complete at review time); learned quality or held-out task performance; real camera/TCP accuracy; real cup handle perception; live frontend timing; ROS/hardware execution; deployment qualification. These remain separate work and are not prerequisites for the explicitly authorized unverified offline baseline. No hardware accessed and no source edited.

Verdict: fix 1 before formal run completion; offline baseline may proceed after that small reporting correction. This review does not certify training completion or deployment readiness.
