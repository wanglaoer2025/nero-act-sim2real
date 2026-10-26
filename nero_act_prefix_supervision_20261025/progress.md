# SDD ledger — plan: docs/superpowers/plans/2026-10-25-nero-act-prefix-supervision.md

User approved execution of the proposed five-step plan, including A/B1000 updates each. Original20K training remains untouched; this is offline only.

Pre-flight: loss implementation → training wrapper → frozen policy evaluation share the same state_dict and15D/20x8 contracts. Preserve original A loss exactly; B replaces only arm loss. Evaluation must distinguish bounded diagnosis from full-task acceptance.

Ruling: use a new experiment directory with read-only imports of sealed v2, because editing v2 would invalidate the running original20K's hash checks — cost is a small wrapper, no historical/code overwrite.
Ruling: checkpoint stores AdamW but not RNG; restore identical moments and use one recorded new RNG seed for both arms — matched A/B is valid, exact continuation of the original RNG stream is not claimed.

Task1: in progress; tests first, then implementation and zero-update5K diagnosis.

Task1: complete — prefix supervision RED ModuleNotFoundError then GREEN5/5;5K read-only diagnosis reproduces raw predictions and guarded submissions exactly. Approach error0.00385035rad vs teacher motion0.00268085rad(1.436x), actual closed loop remainsZ481.059mm. No optimization in diagnosis.
Task2: training controls RED then GREEN8/8; matching model/optimizer restoration and Python/NumPy/Torch RNG tested. Evaluation-scope tests RED then GREEN; whole experiment suite11/11 in2.23s. Ready to launch exactlyA1000+B1000.

Initial launch: failed before first forward completed due unsupported strict CUDA cumsum;0optimizer updates, all outputs archived attempts/strict_determinism_failed.
Final whole-code review: one fresh read-only reviewer found3Important,0Critical,0Minor. Fixed strict-runtime configuration(RED→GREEN), AdamW CPUstep alias(source immutability test RED→GREEN), and initial data-byte provenance verification(episode+dependency mutation tests RED→GREEN). Full suite14/14 in2.18s. No second review; corrected findings tested by author.
Ruling: strict bitwise GPU determinism is not claimed because ACT mask cumsum has no strict CUDA certificate; both arms use identical RNG/CUBLAS and deterministic cuDNN, and first stochastic prediction/gripper/KL must match exactly before B updates. Cost: later kernel-level bitwise identity is not certified.
Ruling: rehash all150episodes and declared VDS dependencies once before optimization, then monitor snapshots and bound proof; cost is offline disk-read time, prevents accepting prelaunch source mutation.

Task2 launch verified: complete150episodes plus dependencies177paths, 105075269932bytes rehashed against sealed manifest in1033.21s, no mismatch. A first actual update5001 completed; original20K untouched, B not yet started. Active child PID723084/session63202.

Task2: complete — A1000+B1000, identical initial model/AdamW/RNG/first-forward and batch sequence. Actual checkpointsA e9dd13950b5f79f6dfa7cc0b20b47db347ca53c065684e2dcd32482920c7de74;B4ba2f801348345f2e3fc3119be2f954dcc05e5c48b106df4ed3354eb74545390.
Task3: complete within conditional scope — same668windows, source5K/A/B3short cases each, all0/3reach. BapproachMAE0.00328852rad vsA0.00440829rad(25.40%better) vs source0.00385035rad(14.59%better), but TCPstaysZ480.14mm with last100framesextent0.140mm. No full nominal/44case gate entered; no full-task success claim.
Final verification:14/14 in2.18s;131bound source/checkpoint/proof hashes unchanged,177data snapshots unchanged; no extra updates, final40test, ROS/CAN or hardware commands. Results savedresults.json/results.md.
Ruling: stop conditional escalation when all three short probes fail, not relabel short600point caps as complete failure or consume hours on44identical stalls — cost is that this round has no full-task score and cannot prove eventual3900point behavior; report that limit explicitly.
