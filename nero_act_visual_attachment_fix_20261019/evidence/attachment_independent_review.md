What this change does: Reconstructs the cup-to-flange attachment from current simulation safety geometry while table-supported, then freezes that transform after observing an airborne cup. Keeps the predicted-bottom rejection threshold at -1 micrometre and does not modify policy features or generated action values.

Must fix:

1. **Important — rollout can miss lift-off and reanchor after landing** (`/home/user/wq/nero_act_visual_attachment_fix_20261019/simulation/supervised_rollout.py:188`; connected execution at lines 94–103).
   - What this is: The helper correctly becomes permanently frozen when observe(...supported=False) is called. Rollout invokes that helper only in the policy-query loop, before executing the prefix.
   - Problem: Each query can execute two ACT points, and confirmation/hold paths execute many more points through advance(). Those advances observe current support but never feed it to the attachment helper. A supported cup can therefore become unsupported at one completed advance and return to table support at the next. The following query sees supported=True and replaces the transform, despite an already-observed lift and landing. This contradicts the stated permanent first-airborne freeze and can absorb subsequent slip into the attachment model.
   - Fix: Feed current paired flange/cup/support observations to an initialized, not-yet-airborne attachment during advance(), at least at each completed feedback point, including holds/pauses. If the intended guarantee is the first physics-step loss of support, track that at physics-step cadence too. Retain the helper's permanent latch. Add an integration regression in which airborne then supported observations occur between policy queries and assert the first-airborne transform remains fixed.
   - If skipped: Safety geometry may reanchor after a brief lift/landing, even though the helper-only tests pass and the report claims a permanent freeze.

No Critical issues found. The coordinate conversion and oriented-cylinder bottom formula are consistent with the existing flange-pose convention and cup dimensions. Current truth poses flow into the safety guard only; the reviewed delta does not route them into policy observations or targets. The threshold is unchanged. The evaluation protocol symlink resolves to the existing attempt2 protocol file, as described; runtime source changes remain a separate post-training change.

Verification: From the correction root, PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nero_visual_mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_supported_attachment.py: 4 passed in 0.04s. Reviewed helper, rollout delta, connected advance/pause/support observations, and first-lift integration. No training, evaluation rollout, hardware, or source edits performed.

Verdict: Fix finding 1 before the controlled runtime assessment.

Not checked / declined to judge: Whether the corrected guard enables a successful lift or full task; physical accuracy of simulator contact/support or nominal cylindrical bounds; real metrology/hardware readiness; completed-run provenance and runtime artifact hashes. A guard correction changes the evaluated controller despite frozen policy parameters, so results should retain the explicit post-training-runtime-correction designation.
