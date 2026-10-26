# Independent final review — 2026-10-19

What this change does: It evaluates frozen visual15D ACT policies in short-prefix physics rollouts and aggregates sixteen nominal-domain cases. It also adds offline-tested inference and ROS transport components; it does not supply a qualified live executor.

Scope: fresh-context review of simulation/visual_evaluation.py, simulation/visual_suite.py, simulation/supervised_rollout.py, executor/visual_dispatch.py, executor/visual_inference.py and connected control, contract, physics outcome, scene, training-integrity and ROS-session code. Assumed load: one controller and up to four independent physics processes. No hardware access or commands. No source edits.

## Must fix

1. **Important — observation-delay case supplies current observations** (`simulation/visual_evaluation.py:92`, connected `simulation/supervised_rollout.py:120` and `:140`).
   - What this is: The sixteenth case is named heldout_observation_delay and declares a 100 ms observation age.
   - Problem: It only sets servo_delay_steps=1 and changes the scalar age supplied to the guard. The rollout renders current simulator state and passes that current state/image to the policy each time. The scalar does not change the inferred action. Thus this case tests an actuator delay, while reporting an observation-delay scenario.
   - Fix: Actually deliver a timestamped older image with its paired historical feedback, advancing physics under the previous command during delivery delay; record exposure and consumption times. Alternatively explicitly rename/reclassify the case as actuator delay and acknowledge the observation-delay assessment remains absent (the retained required ID makes implementing the intended delay preferable).
   - If skipped: Sixteen finished IDs can misleadingly imply the declared observation-delay condition was assessed.

2. **Important — full-suite freeze omits physical geometry dependencies** (`simulation/visual_suite.py:24`, aggregation at `:49`).
   - What this is: Each subprocess hashes its own scene template and mesh files, but the parent binds only Python files, protocol and checkpoint.
   - Problem: If scene_template.xml or a mesh changes after one wave of workers finishes and before the next starts, each child can truthfully pass its own before/after check with a different geometry digest. The parent never checks the shared geometry or compares child manifests, yet emits source_and_model_unchanged=True for the full suite. simulation/scene.py:34 and :41 load these physical dependencies directly.
   - Fix: Freeze the same complete dependency set once in the parent (including scene template, meshes and statistics); require every child dependency manifest to match that frozen set, then verify it again at suite finalization.
   - If skipped: A finalized sixteen-case artifact can combine different physical scene versions.

3. **Important — already-completed controller results bypass deadline and health checks** (`executor/visual_dispatch.py:118`).
   - What this is: RosVisualTransport.complete checks safety health and the completion deadline only while at least one future is pending.
   - Problem: If all futures are done on entry, or become done during the last spin that overruns the deadline, it returns success without checking current health or the deadline. Reproduced without ROS: two successful precompleted Future objects, clock=11.0, deadline=10.0; complete returns normally and calls guard zero times.
   - Fix: Check health and deadline before accepting completed results as well as while waiting. Add regressions for precompleted late results and completion during a deadline-crossing spin.
   - If skipped: The component's documented observation-tail completion deadline is not enforced on every successful return.

## Verification

Ran the requested four-file scope with PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nero_visual_mpl .venv/bin/python -m pytest -q tests/test_visual_evaluation.py tests/test_visual_inference.py tests/test_visual_dispatch.py tests/test_visual_runtime.py: **23 passed**, one existing Matplotlib Axes3D availability warning, 3.76 s. The focused no-ROS deadline reproduction above returned success beyond deadline.

The two-point core commits only the acknowledged submitted prefix. Image feature extraction fails on missing visual detections rather than substituting scene cup coordinates. Final pass criteria reject GRASP-only outcomes and require all four supervision events, physical transfer/lift, release stability and zero contract failures. Nominal qualification flags remain false.

Declined to judge: real TCP/camera/gripper metrology; live semantic frontend; real ROS/CAN timing, stop and readonly-restoration effectiveness; one-shot receipt consumption or assembled full-phase launcher (explicitly deferred); actual full16 physics outcome and visual causal task-success evidence (not run in this review); consolidated issue document and artifact packaging (parent's separate verification); whether the trained policy can achieve the task after the reported nominal failures. Offline unit checks establish component behavior only.

Verdict: fix 1–3 before freezing the final assessment/components. Hardware remains unqualified regardless of these fixes.
