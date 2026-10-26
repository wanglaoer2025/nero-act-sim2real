# SDD ledger — plan: docs/superpowers/plans/2026-10-26-nero-act-arm-increment.md

User requested latest progress and continued completion of the proposed incremental-arm path. Original20K is now complete;2K/5K/10K/20K each0/44 with no task events. This round is offline A1000+B1000 only, no hardware.

Pre-flight: train-only delta stats→143D readout→absolute20x8 output→sealed original guard. Keep15D observation and causal previous gripper unchanged. Base eval/z0/no gradients; only new7-axis head trains.

Ruling: use completed20K rather than obsolete5K as the common source — full20K evidence is now available and no approach skill was recovered; cost is a new explicitly identified source checkpoint.
Ruling: freeze core/vision/CVAE/gripper and fit same143→7 head in both coordinates for this first mechanism experiment — avoids simultaneously changing learned sensory/gripper behavior; cost is that failure cannot rule out a deeper trainable-policy change.
Ruling: do not load old AdamW state into differently parameterized readouts; both use fresh same optimizer — old moments are not coordinate-invariant; cost is this is a bounded readout adaptation, not exact20K continuation.
Ruling: reuse prior full-byte verification only if its bound hash, manifest and all177dev/inode/size/mtime/ctime snapshots remain unchanged — sufficient for this non-adversarial local source immutability check without rereading97.86GiB; cost is trusting filesystem metadata continuity, not claiming an adversarial fresh byte audit.

Task1: in progress, TDD before implementation. Preserve all historical source/evidence.

Task1: complete — encoding/equivalence/freezing/source/control/evaluation tests RED→GREEN11/11 in2.21s. GPUzero-update668windows: physicalA/Bvs20Kmax6.02484e-7rad,grip0; actual savedcontexts0/975/1949 replayoldrawexact and physicalnewmax4.51863e-7rad. Train-only702950prefixlabels fit7scales0.00103..0.00521rad. Priorbyteproof/manifest/177metadata snapshots unchanged.
Task2: ready — one fresh read-only whole-implementation reviewer found0Critical/Important/Minor and independently11/11CPUtests in2.16s; declinednone. Author actual source-binding gate136files and177snapshots passes. Launch exactlyA1000+B1000, no hardware.

Task2 launched: actualGPUchild743337/session98290;A800/1000 verified, frozen20Kcore unchanged at every100checks,177data snapshots unchanged. Sourcebinding136filessealed. No ROS/CAN or final40test.

Task2: complete — A1000+B1000, matched batchsequence74f4851ec1a3b8c4f4cd9b72696a979dcc1ab2612d6594e94d893f25b5646477; both frozen cores identical to20K15c4b4a2ff02b611103940af91e49d568a864c925c5afb961256277f644b2557.
Task3: complete within conditional scope — same668windows and3shortcases each; source20K/A/B all0/3. Aguardrejectat14points(rawpoint10J7gap0.030486>0.03rad);B600pointsstagnatesZ476.26mm,last100extent0.110mm,goal-distancegain3.81mm. No full nominal/44gate entered, final40untouched.
Ruling: add one zero-update teacher recovery from B's exact saved integration state to distinguish physical infeasibility from policy failure — all8stages succeed2920points,maxlift194.31mm,transfer186.57mm,stable135,nounexpectedcontact/contracts; cost is one development-only diagnostic, excluded fromtrain/ACTsuccess.
Final verification:11/11 in2.15s,136source/checkpoint/proof hashes unchanged,177metadata snapshots unchanged, checkpoints frozen-core equal. Complete readout experiment, ACTfulltask stillunresolved; report results.md/results.json/final_verification.json/expert_recovery_diagnosis.json.
