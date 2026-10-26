What this change does: Loads a finalized visual15 ACT checkpoint with its paired normalization, then optimizes only its existing nonlinear gripper adapter. Retains the repaired image pool and sampler, and verifies that every non-gripper state tensor remains exactly equal after each epoch.

Verdict: Proceed with the controlled experiment. No Critical or Important correctness/provenance issues found in the focused change.

Reviewed: training/visual_gripper_repair.py; the adapter-only gradient_evidence branch in training/visual.py; tests/test_visual_gripper_repair.py; connected FullTaskPolicy, BoundedActPolicy, normalization, checkpoint finalization, dataset, and frozen-backbone implementations. No training, hardware, or physical rollout launched. No source modified.

Evidence:
- Adapter construction freezes every parameter except gripper_adapter.*, and verifies that rule by name. FullTaskPolicy replaces the optimizer with AdamW over only that adapter, with no decay. The old action-head rows are frozen.
- Source state loads strictly without resetting the action head. The adapter hook changes only output dimension 7; seven arm output dimensions remain identical for the same inference observation and image. The focused test verifies this numerically after an adapter update.
- Training uses the deployed zero-latent prior for adapter-only optimization, with positive unbounded gripper supervision; it does not retrain the CVAE posterior. Gradient evidence requires nonzero finite adapter gradients and no frozen parameter gradients.
- Frozen state snapshots include registered state buffers as well as parameters, and exact comparisons run before saving each epoch's checkpoints. The visual backbone uses FrozenBatchNorm2d, so train mode does not update running batch statistics. Training mode still enables existing dropout; validation/inference disable it. This is stochastic adapter supervision, not frozen-weight drift.
- Paired source statistics are reused and verified against both visual normalization and physical feedback-readout configuration. Calibration hash must match the repaired dataset. Source finalization is checked before loading, and initializer, source statistics, source finalization proof, new code, data, protocols, and output config/statistics enter the protected snapshot. Finalization checks these inputs remain unchanged.
- The actual repaired manifest is hash-checked, with train/validation source-group separation retained. This new entry point relies on the supplied completed pool and source split rather than independently reasserting the earlier 48/36/12 cardinalities; no concrete mismatch was found or reported in the supplied run context.

Verification run from correction root:
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nero_visual_mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_visual_gripper_repair.py tests/test_visual_contact_sampling.py
Result: 7 passed in 3.46s, one existing Matplotlib Axes3D warning. This is my independently executed focused subset, not a claim to have rerun the parent's nine-test set or full suite.

Not checked / declined to judge: Learned recovery, optimizer convergence, completed-run frozen-weights evidence, actual approach/contact/lift success, real geometry, or hardware readiness. Frozen arm parameters guarantee unchanged arm outputs only at the same observation. Changing gripper commands can alter future observations and therefore the closed-loop arm trajectory; nominal and full physical assessments remain necessary. The original baseline's own task failures are not resolved merely by this review.
