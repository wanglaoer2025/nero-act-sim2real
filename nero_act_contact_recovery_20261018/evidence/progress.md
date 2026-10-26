# Contact recovery ledger 2026-10-18

User requested next step after frozenrun14 finished4/16. Offline diagnosis and bounded correction work only; no hardware authorization. Preserve originalrun14 source, data, evaluation and all prior receipts.

Initial offline reconstruction found small downward first targets and wrist tilt near first lift, but did not reconstruct persisted attachment state; no conclusion of a false guard yet. Need exact instrumented rerun.

Exact instrumented nominal rerun has identical sent_targets and measured states to originalrun14. Atstep1054 currentattachedmodelbottom+0.0528mm, ACT firstthree targetTCP deltas−0.3055/−0.4083/−0.1222mm; predictedcupbottom−0.2911/−0.4057/−0.0928mm. Guard correctly rejects this downward prefix; do not relaxfloor/timing/contact limits.
Next: recover exact qpos/qvel/servoqueue/previouscommand snapshot, retain measuredgrasp orientation, offlineexpert upward/carry/support/release/retreat correction. NoIK or teacher takeover in future policy execution.

Ruling: use isolated contact_recovery root rather than edit original run14 — preserve run14's source-bound training proof and frozen failures — cost if wrong: extra source copy, no loss of original evidence.
Ruling: bounded first-lift correction only,6 exact failed-state continuations, preserve48full episodes — diagnose and correct observed downward prefix rather than change guard — cost if insufficient: next frozen evaluation can still fail; no guaranteed task success.
Ruling: skip repeated close dwell in already-held correction, use new exact-snapshot lift_to_retreat — an already stable grip cannot observe elapsed waiting time — cost if wrong: physical contact checks reject pilot;6/6 pilots passed.
Data:6/6 corrections passed physical outcome, original prefix contract, exact replay(maxerror0); 4train/2val, historicalsite0/1/2 samegrouptrain. Diagnostic1000 notincluded. All first10 targetTCP deltas positive,0.099..3.648mm.
Adapter: verified partial role/snapshot/provenance/grouping, optional combined54episode40/14 split, train-only statistics and frozenrun14 initializer; unit RED→GREEN logged. No runtime teacher fallback.
Ruling: original16 cases now development regression because6 failures entered correctiondata — declare genuinely different16 inputs before next training — cost if wrong: cannot call reused16 independent, must report both separately.
Fresh16 predeclared definitionSHA aff2bea7da167361422edad32e1056a289f430f350a6e8de3d42654acf9cc714; starts/appearance/opening/delay actually changed, no hardware readiness claim.
Verification: default suite117ran,115passed/2optin skipped; explicit NERO_PHYSICS_TEST=1 tests.test_teacher13/13passed. Default suite logs contain existing Matplotlib Axes3D warning; no robot access.
Review: one fresh whole-change reviewer first_lift_recovery_review dispatched per executing-plans; waiting before actual optimization so any Important findings can be corrected without invalidating training source hashes.
Final review: first_lift_recovery_review found3Important,0Critical,0Minor. One author fix pass (no rereview):
- Fixed predeclaredcasegap: require --independent-cases for correction runs, hash and protect file before training, fresh evaluation reads exactbound artifact (not regenerated current site). Regression test RED→GREEN.
- Fixed assessmentrolegap: recoverycheckpoint config forces all old16/default/nominal paths to development status regardless optionalCLI. Regression test RED→GREEN.
- Fixed physicalscenegap: compare original physicalscene with HDF, only exactdeclared start/teacher/continuation overrides allowed; table/force/servo changes rejected evenifHDFSHAupdated. Regression test RED→GREEN.
Final: Ruling: reviewer declined optimization/learnedbehavior/hardware judgments because jobs notyetstarted — these remain measured later, hardware untested — cost if skipped: no valid success claim, so do not skip frozenevaluation.
No deferred minors. Fullsuite with opt-in physics running beforetraining. No protectedcode edits permitted once optimization begins.
Final fixes verified:119/119 fullsuite including opt-inphysics passed, all54data accepted40/14. Originalrun14 protectedinputs/output unchanged. See final_pretrain_acceptance.json.
Formal run15 pipelinePID1744448, trainPID1744449 launched; CUDA20epoch training banner and actualinitializerload confirmed. Bound independentcasefile enters config/protectedinputs. No further edits to protectedsource/data during optimization. Pipeline will finalizedexport validationbest_l1 then development16 and independent16; process exit alone never means taskpass.

User requested operation after pipeline finished. Read actual results: run15 20/20 epochs complete, valbest17; development2/16, predeclared independent0/16. No hardware sent; deployment remains false. Preserved all failed tests; did not select a different checkpoint using these tests.
Operation performed offline: one repeated known successful development case last_site_geometry_prepared_open_0, current frozen ACT weights, same guarded simulation protocol. Full physical task passed with maxlift0.1988386m, transfer0.1554843m,104released-stable frames, no contract/unexpected-contact failure, no attachmentconstraint. Every state and submittedtarget bit-identical to previous successfulcase. This repetition is NOT a new independent success or real robot operation. Evidence: run15_simulation_operation_01.
No new training launched and no protected source/data/weight changed. Failure classification recorded in run15_operational_readiness_review.json; complete result still unfit for generalization deployment. Rendering exact query snapshots to video, not generating or substituting any new task targets.

Operation video done:415 saved query snapshots,171.3s physics-time span, H.264 decoded415/415; visual inspection of frame200 passed. No hardware action. See video_verification.json.
