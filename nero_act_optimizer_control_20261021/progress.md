# SDD ledger — plan: approved_scope.md

Pre-flight: approved scope is the in-chat bounded design; no separate architectural spec is needed. Diagnostic outputs feed the pre-optimization seal; both optimizer schedules feed the same loss loop and budget verifier; fixed/decay results feed one declared gate.
Ruling: workspace is not a Git repository; use two isolated source copies and SHA seals instead of Git worktrees/commits. Cost if wrong: no Git merge history; all source and output provenance remains in manifests.
Ruling: unspecified decay is linear, using initial rate at update1 and final rate at update1000; gripper rate stays unchanged. Cost if wrong: this experiment cannot characterize other decay families.
Task1: in_progress; diagnostic collection and zero-update gradient probe.

Task1 finding: exact-current-RGB/15D inputs conflict at4 release→retreat windows (max5.45mrad first2label difference).
Ruling: remove only the4 earlier ambiguous wait starts from both new training views, retain later retreat starts and all raw files/labels/stats. Cost if wrong: comparison against older runs now also includes this24-weighted-row sampling correction; only the two new arms isolate decay. No runtime phase truth is added.

Task1: complete — exact conflict scan4→0 after excluding4starts (36375→36367 unique view rows,68049→68025 weighted samples). Read-only CUDA diagnostics8records over2model states,zero optimizer updates,all423 baseline replay prefixes exactly unchanged. Frozen failed model arm preservation gradients12588–274430 vs ordinary9.86–10.81; global clip multipliers3.64e-7–7.94e-6. No evidence of opposing arm gradients in these4batches (cos0.359–0.553); one nonzero gripper preservation comparison opposes ordinary (cos−0.491). This is limited batch evidence, not proof of a universal cause.
Task2: new tests8/8 green; full suites and one fresh final source review underway.
Task2: complete — bothfullsuites421passed/5skipped,8new tests RED→GREEN; onefreshreview0findings. Reviewdeclined futurepilot/evaluationintegrity/hardware, to be verified byfinal artifactcheck; nohardware planned. Sourcecomplete beforesealing.
Task3 progress: fixedpilot finalized1000updates/0epochs,421tests/5skipped; replay RMS7.78890656834847e-6rad,max2.1517276763916016e-5rad,gripmax.00023562461137771606m. Decaypilot running, no scope/budgetchange. Bothpreoptimizationseals164source/369datainputs.
Task3: complete — fixed/decay each1000updates,0epochs,total2000. Both423replaygatespass; both6mechanisms4/6. Nominalpreclose reachesbilateralcontact butmaxexecutedmargin2.06186/2.35330mm<original3mm; holding recoveryrejects at2/4newpoints. Noeligiblecandidate,so conditionalfullnominalnotrun. Noformal/newindependent16/hardware.
Task4: reports/plot/masterdocumentupdated; finalsource/outputverificationpending.
Final: Ruling: preserveisolatedworkspacesandledger because noGit historyexists; finishing-a-development-branch integration/cleanup menus do notapply. Costifwrong: diskspace retained; nohistory destroyed.
Task4: complete — final_verify.py exit0; allfinalartifact hashes, sealedtestedsource/data/outputs unchanged;6caseJSON/NPZ consistencychecked; bothcore runtimebyte-identical toparent; reportlinksvalid andfigurevisuallyinspected. Bothpilots finalized2000totalupdates,4/6each,noeligiblefullnominalcandidate. NoGitmerge/push/cleanup performed; workspaces preserved. No deferred minors.
