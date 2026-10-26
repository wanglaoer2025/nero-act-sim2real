# SDD ledger — plan: /home/user/wq/docs/superpowers/plans/2026-10-19-nero-act-short-history.md

Approval: 用户“计划符合，顺序实施”；2026-10-19。No real robot commands authorized by this plan.

Ruling: use isolated copy and SHA256 ledger, not Git helpers — approved plan specifies no Git repository — cost if wrong: history tracked by hashes rather than commits. Preserve ledger permanently.

Pre-flight: T1→T3/4/5 HistoryBatch and causal receipts agree; T2→T5/7 complete offline checkpoint agrees; T3→T4/6 explicit history dict and zero-z agree; T5→T6 shared samples/stats agree; T6→T7/8 separate updates/epochs budgets agree; T7/8/9→T10 all source precedes freeze; T10→T11 gate agrees; T9/10/11→T12 evidence and no-motion status agree.

Task 1: complete (baseline 292 passed/5 skipped; RED 16 failed; GREEN/full suite 308 passed/5 skipped; 1 inherited Axes3D warning). See task1_{red,green,suite}.log.
Task 2: complete (6 new tests RED→GREEN; full suite 314 passed/5 skipped; frozen original CUDA trace exactly identical, 1792 points, same stop reason; see task2_suite.log and original_failure_control_capture.json).
Task 3: complete (5 new tests; legacy empty-keyword compatibility regression reproduced and fixed; final suite 319 passed/5 skipped; task3_suite_fixed.log).
Task 4: complete (10 new inference/dispatch/physics tests RED→GREEN; full suite 329 passed/5 skipped; task4_suite.log; common runtime contract config/history_runtime_contract.json).
Task 5: complete (18 verified partial recoveries;66 derived physical files,45188 unique current views,118895 actual rendered frames;37/12 source groups; all stages covered; see task5_data_final_audit.json).
Task 6: complete (paired initialization/optimizer,strict1000-update vs20-epoch budgets,export metadata and protocol-seal API;9 targeted tests and regression suite348 passed/5 skipped; actual optimization still gated).
Task 7: complete (mechanism evaluator and explicit offline policy transition;13 targeted tests,full suite352 passed/5 skipped; no actual trained model evaluated yet).
Task 8: complete; see final regression evidence below
Task 9: complete; see source seal evidence below
Task 10: complete; A/B each1000 updates,each2/6 mechanism checks; gate stop; see final evidence below
Task 11: not executed — approved gate stop; no formal epochs/development/independent16
Task 12: complete; final integrity and result documents below

Task 1: Ruling: use the source project virtualenv link, not /home/user/nero — that environment lacks pytest; no dependencies installed — cost if wrong: environment compatibility, checked by baseline.

Task 4: Ruling: isolate completed-prefix image sampling in simulation/history_observation.py — legacy exposure deque drops eligible hold frames while a two-point prefix runs; preserve actual eligible snapshots and delay without admitting mid-prefix frames — cost if wrong: image/runtime latency; common A/B contract and physics tests bind it.
Task 7 pre-flight note: new policy mechanism resume must explicitly declare an offline policy transition; original exact-reproduction restore deliberately binds original weights/config. Never infer real permission from either.

Task 5: Ruling: reject and preserve recovery batch v1; teacher interruption context omitted airborne attachment tracking — regenerate as v2 after RED→GREEN latch test — cost if wrong: extra compute, not false deployment qualification. CUDA RNG from original diagnostic contexts also requires a matching initialized host CUDA runtime.

Ruling: while Task5 physical generation runs, implement Task6 against the already tested data interface in this same sequential implementer — independent code preparation does not launch pilot/formal optimization; Task5 acceptance and Task9 source seal remain mandatory gates — cost if wrong: rework of the unsealed trainer, no hardware effect.

Task 5: Ruling: v2 contains16 verified teacher-parent recoveries but nominal2016 replay fails; diagnosis isolates7.9936e-15 float64→float32 warmup-command loss (original replay max3.767e-4, exact warmup replay max0). Preserve actual action precision only for new partial records; legacy save default unchanged. Assemble v3 referencing the16 unchanged verified files and newly regenerate2 nominal contexts, not the rejected2016 — cost if wrong: continued strict1e-7 replay gate rejects it.

Task 5:18/18 recovery batch v3 verified, including nominal pre-close and holding continuations with replay maximum0.0.16 parent rows reference unchanged v2 files; v2 rejected nominal2016 is excluded. Full66 visual-history derivation running.
Task 6: trainer/budget/export9 targeted tests passed, actual optimization not started; completion waits regression suite and downstream source freeze.

Ruling: enforce the already approved pilot/formal/independent gates in CLI entry points, not only in orchestration; add history_qualification.py and bind qualifying model/checkpoint/summary artifacts — cost if wrong: a valid experiment is stopped with an explicit qualification error, rather than consuming unqualified training/test budgets.
Ruling: perform the one fresh whole-code review after implementation and before the immutable source seal, then verify actual experimental outcomes separately — avoids invalidating expensive training on source fixes; cost if wrong: later result-only issues require an explicit new experiment, not silent source changes.

Final review: four Important findings accepted (mid-prefix delayed frames, shared RGB mutation, direct-development schema mismatch, missing development prerequisites); one fix pass started with five reproducing RED tests. No Critical or Minor findings.
Final: Ruling: actual pilot/formal/independent performance and hardware qualification were deliberately unrun at review — verify experimental artifacts after immutable source seal; no calibration or hardware qualification inferred — cost if wrong: experiment fails and remains explicitly unqualified.
Final: Ruling: add isolated saved-input history masking/order diagnostic before source freeze — Task10 requires interventions but lawful worker rejects invalid histories; direct offline tensor diagnostics retain current observation, never enter control or gate results — cost if wrong: diagnostics measure sensitivity only, not causal task benefit.

Final: fixed delayed mid-prefix exposure admission — test_delayed_midprefix_exposure_is_excluded_without_losing_past RED→GREEN; actual completion timestamp passed by dispatcher; old valid observations retained. Full suite373 passed/5 skipped.
Final: fixed inference buffer mutation binding — test_prediction_detaches_camera_and_geometry_before_inference RED→GREEN; detached immutable RGB and geometry, digest established before model call and verified afterward. Full suite373 passed/5 skipped.
Final: fixed direct development schema incompatibility — test_direct_development_summary_uses_unique_bound_case_results RED→GREEN; direct/suite checkpoint binding and16 unique expected cases validated. Full suite373 passed/5 skipped.
Final: fixed missing pilot/development prerequisites — test_development_requires_successful_bound_prerequisite RED→GREEN, plus absent/failed/mismatched nominal and failed pilot regression cases; both CLIs gate before model initialization. Full suite373 passed/5 skipped.
Final: minor (deferred): none.
Task 8: complete (full-task predicates, explicit protocol binding, candidate and stage gates; post_review_tests.log373 passed/5 skipped).
Task 9: verification373 passed/5 skipped and10 CLI help checks passed; immutable source sealing started. History intervention test RED→GREEN; actual diagnostics pending trained pilot.
Task 9: complete (source seal160 files,data seal170 files,zero optimizer steps before seal; source_seal.log and pre_optimization_freeze.json). Actual optimizer launch follows this boundary.

Task 10: complete (paired_pilot_training_verified.json binds actual1000+1000 updates and identical sample/order hashes; mechanisms_A/B each6 completed,each2 passed; pilot_decision.json gate stop; isolated two-capture interventions and saved-input A/B predictions recorded; full telemetry preserves simulated exposure/consumption and inference, real camera capture timing remains unmeasured).
Task 11: conditional branch not executed (neither model passed all mechanisms; no budget expansion and no fresh16 consumed). This is the approved stop condition, not complete-task qualification.
Task 12: complete (final_verification.json confirms160 source/assets,170 sealed data inputs,214 original files unchanged;373 passed/5 skipped and10 CLI proofs cover unchanged source; both pilot proofs finalized; documents updated; no real shadow or hardware).
Finish: copy/SHA-ledger experiment retained; no functional Git repository/branch exists, so no merge/PR/worktree cleanup applies. Runtime source remains frozen; all failed and rejected data/results retained. No second reviewer and no post-seal source changes.
