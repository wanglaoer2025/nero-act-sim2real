# SDD ledger — plan: /home/user/wq/docs/superpowers/plans/2026-10-18-nero-act-position-generalization-training.md

User authorized new training under the approved position-generalization design; preserved native execution. No hardware action authorized or performed.

Ruling: no Git repository exists; use isolated copied source directory plus hash snapshots, preserve all old sources/data/model/receipts; no commits.
Pre-flight: collector -> manifest/split -> fresh training statistics -> paired context model -> frozen evaluation; same-cycle action and causal previous target shared.

Task1: baseline reproduced — run13/v7 fails with nonexpected place contact (maxlift189.58mm/transfer180.19mm), run11/v6 complete nominal passes (lift197.26mm/transfer178.32mm). Different model and protocols explicitly retained.
Ruling: successful run11/v6 establishes that the full physical task is possible, not that the new9D candidate already passes; generalized teacher data may be collected while the failed9D candidate is corrected by new supervision. Its new complete baseline must pass before hardware eligibility.
Task2/3 TDD: four new tests failed before implementation, then all4 passed. Test proves re-normalized9D initializer preserves physical actions within2e-6,48-scene geometry covers±30mm and has no exact train/val scene overlap; tested IK errors within0.2mm/rotation0.002.

Task2 pilot: scene0 at source(-30,-30)mm completed2772frames, full physical success,278prefix windows, exact replay error0;34.24seconds. Started remaining47 scenes with2workers, persistent per-candidate failure files and manifest.
Task1 real audit:9793frames over489.6s, median50ms samples, real image age median17.35ms; action has13.72mrad arm and95mm gripper jumps. Ruling: real record stays diagnostic-only until event alignment is verified; do not interpolate through contact or feed these jumps into the new causal training labels. Simulation training may proceed independently; real-adaptation is not claimed.
Task2 initial full suite:105 tests,1failure/1error due to missing copied run.py and historical run6 config fixture,2opt-in physics skips. Copied original entry and config into isolated directory; no behavior change or relaxed tests.

Task2/3 full existing+new suite105 tests passed,2 explicitly opt-in physics tests skipped (pilot full physical execution/replay separately verified).
Task1 real audit confirmed legacy action_semantics=future_measured_qpos_plus_gripper_target with zero lead; arm actions equal same-frame observed joints exactly. Ruling: reject this record as dense ACT action supervision because it teaches stationary arm and contains unslewed0/95mm event labels. Keep raw record and use RGB only for diagnostics; no real fine-tuning claimed in this training round. New properly aligned real demonstrations will be needed for supervised real adaptation.

Final independent code review:3Important findings,0Minor. One author fix pass: interrupted candidate/canonical recovery, filename/manifest/HDF ID agreement, and pre/post optimization source+input integrity with export-finalization requirement. Five targeted tests failed first, then5/5passed. Preserved active collector's old code in evidence/active_collection_sources_before_review_fix; activeworkers finish using that archived version, same physical generator.
Postfix full suite exposed existing nondeterministic permission test: CASES is a set and last item could already be nominal, so mutation sometimes made no duplicate. Ruling: mutate last name to first actual name, guaranteeing duplicated case; preserve actual permission rules. Red reproduced in all_tests_after_review.log, deterministic assertion now verified in final suite.
Explicit teacher physics suite13/13passed, including both previously skipped integration cases, no hardware.

Final review fixes:5regressions RED→GREEN; whole suite112 tests,110passed+2optin skips, then explicit teacher physics13/13passed. No deferred minors or reviewer findings left. Pipeline is waiting for accepted48 before CUDA smoke and independent20epoch run14.

Task3 actual-weight integration: copied trainedrun13 + fresh36train-only statistics +4real simulated observations gave max physical output difference2.38418579e-7; no validation data in normalization. Report actual_initializer_rebase_audit.json.
All reviewer Important items fixed in one pass; no deferred minors. Original hardware receivers, source packages and receipts untouched throughout.

Task2 complete:48/48 full physical demonstrations,36train/12val,0rejected candidates, all prefix/replay audits pass. See dataset_acceptance.json. Background pipeline enteredCUDA1epochsmoke; formal20epoch run14 will start from originalrun13 initializer independently.

Task3 started: CUDA1epochsmoke completed0.00500443validationL1; formalindependent20epochrun14 active, latestepoch2, pipelinePID1686580, childPID1704244. Detached process identity/session verified. Training request has entered actual optimization, not only data preparation. Subsequent export and16caseevaluation queued.
