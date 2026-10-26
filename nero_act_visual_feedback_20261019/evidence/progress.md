# SDD ledger — plan: /home/user/wq/docs/superpowers/plans/2026-10-19-nero-act-live-visual-feedback.md

User approved sequential execution; user confirms no fresh calibration samples. No physical motion authorized.

Ruling: The approved plan explicitly uses a non-Git isolated directory; Git-only skill scripts cannot run here. Keep per-task briefs, initial/final hashes, RED/GREEN logs and this persistent ledger instead. Cost: no Git commit-based recovery.

Pre-flight: Task2→3/4/6: calibration v1 schema, units and separate measured verification/accuracy goals must agree.
Pre-flight: Task3/4→5/7/10: one frame, paired8D state,6D geometry and causal history produce15D observation. No valid observation on missing calibration or complete occlusion.
Pre-flight: Task5→7/8: only first8D enters physical readout; CVAE and augmentation must preserve15D.
Pre-flight: Task6→9/10: identical compound geometry and conservative clearance budget in simulation and runtime.
Pre-flight: Task10 no-hardware core→8→9→10 shadow: freeze execution code before training/evaluation.

Task1: in progress.

Task1: Ruling: baseline fixture/config and sibling-test import failures resolved by copying small source configs and adding pytest pythonpath. Three historical fixed-trial permission tests explicitly skip without old receipt; no authorization copied. run18 pipeline test script comes from its original alignment package because retained source also lacked it. Cost: historical fixed-trial integration is not revalidated here; new-profile rejection will be tested separately.

Task1: complete — compact health RED1→GREEN; stop/ACK/freshness RED8→GREEN; causal context preserved. task1_green.log:15 passed. Baseline fixture-only failures resolved (baseline_final_dependencies.log:7 passed). Historical receipt tests3 skipped;2 pre-existing optional integration tests skipped.765 sealed source hashes verified unchanged.
Task2: offline tools complete — task2_red.log:6 failed on missing approved interfaces; task2_green.log:6 passed. Real calibration samples unavailable as explicitly confirmed by user; no measured accuracy or deployment claim.

Task3: geometry/synchronization core RED5→GREEN5. Ruling: inherited bounding boxes do not certify cup rim/body/handle keypoints; do not fabricate a qualified frontend. Implement explicit invalid observations and annotated-frame diagnostic CLI. Live semantic frontend, camera exposure mapping and visibility qualification remain pending real calibration/annotations. Cost: no live15D execution until those are verified.
Task4: complete pure feature/error-budget core — RED4→GREEN4;9mm conservative budget fixture checked.
Task5:15D model/initializer/paired-normalization/CVAE/worker core RED6→GREEN6 (plus6 existing CVAE checks). Ruling: extend the local NeroActPolicy adapter with a separate explicit visual-relative profile; its inherited legacy15D profile means a different feature order. Do not enable legacy absolute-coordinate flags. Cost: requires regression coverage for both adapter profiles.
Task7: independent loader implementation proceeds while measured geometry/collection is pending, as explicitly allowed in plan; no new dataset or training launched.

Task7: loader/audit core RED5→GREEN5, no new physical dataset.
Task8: optimizer/gradient probe RED2→GREEN2, no formal optimization pipeline or checkpoint.
Task10: pure protocol/disabled CLI RED7→GREEN7. Live worker provenance, contact/ROS completion/cancel/restore remain pending; no claim of deployment readiness.
Final offline review:4 Important issues,0 Critical/Minor; reproduced all4 plus intrinsics-hash gap in review_fixes_red.log (5 failures). Fixed in one pass: dataset7/7,protocol7/7,camera/calibration/features15/15. Full suite offline_suite_final.log:250 passed,5 skipped,1 preexisting warning.
Final: Ruling: reviewer set aside physical metrology, live semantic/exposure mapping, formal data/training/evaluation/deployment, ROS completion/cancel/restore and live prediction provenance. All remain explicitly pending, not silently passed; entry stays offline-only. Cost: tools cannot complete physical migration until these stages are implemented and validated.
Saved-real-input initializer probe:3 actual historical images,source run27 SHA97f329a4...,all maximum prediction differences0; no checkpoint saved, geometry columns zero, not learned generalization.
Offline phase verified; overall plan remains incomplete because user has no new calibration samples. No source-package modifications or hardware commands;765 sealed source dependencies unchanged.

User steering 2026-10-19: start formal training using the existing calibration version.
Ruling: user explicitly permits retained approximate calibration for OFFLINE formal training; override the earlier fresh-metrology precondition only for this baseline. Preserve strict verified-calibration/live frontend/deployment gates. Cost: training cannot establish corrected real TCP or millimetre accuracy.
Ruling: derive current-image cylinder/card observations from 48 existing full-physics demonstrations; no simulator truth enters policy inputs. Nominal upright cylinder/axis-aligned card estimator is training-only and not qualified for real handles/occlusion. Cost: complete-task robustness and real perception remain separate validation work.
Task8 continuation: formal CLI and retained-calibration preparation RED8→GREEN8; optimizer status and card-occlusion regressions RED→GREEN; selected gradients remain required. Declare20 epochs seed42 batch32, both160x90 and320x180 candidates; zero-latent validation only. Final16 protocol frozen before optimization.
Formal continuation final review:1 Important,0 Critical/Minor; summary previously preceded integrity gate. Failed-input-summary regression RED→GREEN; full suite formal_offline_suite_final.log:261 passed,5 skipped,1 preexisting Matplotlib warning.
Final: fixed premature success summary — test_changed_training_input_never_writes_successful_summary RED→GREEN; success files now written only after final integrity succeeds.
Final: Ruling: reviewer declined GPU run/full48 acceptance/learned quality/real accuracy/live frontend/ROS qualification; accept only offline pipeline review, verify actual GPU training separately below. Cost: no success or deployment inference from unit tests. No minor findings.
Operational note: sandbox-detached preparation PID7 could not survive namespace teardown; discarded start record. Foreground supervised CPU preparation worked but used software GL; retained partial outputs/log as cpu_attempt and stopped only owned preparation PIDs. Host EGL benchmark30frames0.141s; restarted exact source preparation on GPU, no hardware involved.
Actual-data CPU probe:one batch of4 current RGB frames from first complete demonstration optimized successfully, loss0.016779, CVAE/geometry/layer4 gradient proof retained. Explicit probe only; no checkpoint saved, not counted as formal training.
Task7 retained baseline data accepted:48/48 physics-complete source demonstrations,36train/12val disjoint groups,16104/16104 fresh-render image-only estimates valid across all task stages; physical states/targets unchanged. Worst observed simulated cylinder-estimator error19.814mm; this is diagnostic, not a real error bound or accuracy acceptance. Source truth remains diagnostics only.
Formal training ACTUALLY started on host RTX4090,PID2374658;320x180 candidate first,160x90 queued. First complete epoch checkpoint verified input15/output20x8,active CVAE and nonzero newly trained geometry columns; gradients posterior/geometry/layer4 nonzero and earlier visual frozen. See formal_training_start_verified.json and per-candidate training.csv/status.json. No deployable checkpoint, no motion, task9/full-task acceptance pending.
Live post-start integrity audit:all 168 protected inputs unchanged. GPU process2374658 confirmed active using1290MiB; prior independent processes left untouched. New source manifest saved. Job status and each training.csv are live authoritative progress; root pipeline records start only.
Formal training completion verified:320x180 and160x90 each20/20epochs; both selected-checkpoint integrity checks pass against current inputs/outputs. Best zero-latent val320=0.0076308827@17,160=0.0078920197@18. Task9 complete-task/position generalization and real frontend/metrology/ROS remain pending; no deployment success or motion claim.
