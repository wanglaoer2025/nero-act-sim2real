# SDD ledger — plan: approved_scope.md

Ruling: 当前无Git，沿用用户已接受的隔离副本/SHA封存，不更改历史实验，保留本目录记录；代价是无Git合并历史。
Pre-flight: 原68物理数据/统计/423参考/批序列→四类训练标签→分组损失→只读方向门槛→训练封存→1000更新→原六快照。分组只进入训练loss，不进入policyforward推理或模型观测；七轴保持原梯度和预算。
Task1: isolated workspace created; old finalized outputs remain immutable. Baseline suite running.
Task1: complete —baseline432passed/5skipped beforeimplementation. Task2: groupclassification/exactlossdecomposition RED3missingmodule→GREEN3 forbothinverseprofiles; minimum-normprimitive REDmissingfunction→GREEN4. Existingtraining/runtimeforwardunchanged.
Readonly gate (before running): compare legacy,equal-group,gradient-norm-equalized and minimum-norm convex group combinations on4oldfixedbatches0/24/99/499 plus4newcheckbatches149/249/749/999, bothB252andpreviousfailed1000. Includeunchanged423referencegripgradient. A candidate passes only if every testednonzero nominalclose/holdinggradient cosine≥0.01, opening/release physicalerrorgradientcosine≥−1e-5 whennonzero, and ordinarycombinedgradientnorm/mean-groupnorm≥1e-3 (no zero-gradient trick). ApproacharmgradientmustremainunchangedinlaterTDD. Candidateorder simplestpassing equal→norm→minnorm; no optimizerupdates until actualgatepass. These are training-state mechanism checks,not independentgeneralizationacceptance.
