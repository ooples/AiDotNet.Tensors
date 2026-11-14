# Compiled Closure Objectives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every AiDotNet closure-objective model (TrainWithCustomLoss, EagerObjectiveStep, EagerSgdObjectiveStep) trains
on a compiled plan, CUDA graph included, without ever replaying a previous step's data or scalars.

**Architecture:** Tensors gains a storage-free probe trace, a structural signature over lazy nodes (closure fields
included) and a `ClosureObjectiveGuard<T>` that attaches to an ordinary compiled plan. Before each replay the guard
re-probes the closure, compares signatures, and copies the new step's leaf data into the plan's leaves. AiDotNet's
`CompiledTapeTrainingStep` gains a closure-objective mode, and the three eager entry points route through one
`ObjectiveStep` that tries it first.

**Tech Stack:** C# (net10.0/net8.0/net471/net462/netstandard2.0 for Tensors), xUnit, GitHub Actions CI.

**Spec:** `docs/superpowers/specs/2026-11-14-compiled-closure-objectives-design.md`

## Global Constraints

- No local `dotnet test`. Tests run in GitHub CI. Locally, build only, one build at a time:
  `dotnet build <proj> -c Release -f net10.0 -m:1 -p:UseSharedCompilation=false`. The Tensors library must also
  build net471.
- No null-forgiving `!`. Closed sets are enums, never strings. No `KeyValuePair` deconstruction in `foreach`
  (net471).
- Branches: Tensors `feat/compiled-closure-objectives` off `wip/release-at-last-use-arena`. AiDotNet
  `feat/compiled-closure-objectives` off `wip/backward-sources-trainable-params`, built with `-p:UseLocalTensors=true`.
  Both are draft PRs. Commit as Franklin Moormann <cheatcountry@gmail.com>.
- Dynamic rule (spec): an owner that aborts or misses the cache on more than 2 of its first 8 steps is eager from then
  on. Plan cache: at most 4 signatures per owner, LRU.
- Every fallback emits one diagnostic per owner and reason. `AIDOTNET_FUSED_DIAG=1` prints it.

## Review Focus

1. A closure that captures a per-step scalar (a sampled timestep's coefficient) must never replay stale → Task 3 test
   `Captured_scalar_change_is_a_mismatch_and_never_replays`.
2. A closure that reads an intermediate on the host (`loss.ToArray()` for logging, a data-dependent branch) must fall
   back, not freeze → Task 1 test `Host_read_of_a_probe_tensor_aborts_the_probe`.
3. A closure that returns the same tensor objects every step (persistent buffers mutated in place) must still match and
   rebind as a no-op → Task 3 test `Same_leaf_objects_rebind_as_a_no_op`.
4. A closure whose batch shape changes on the last partial batch must compile a second plan, not corrupt the first →
   Task 4 test `Partial_last_batch_compiles_a_second_signature`.
5. A genuinely dynamic objective (a new shape every step) must go eager after 8 steps, not grow the cache → Task 4 test
   `Dynamic_objective_goes_eager_and_caps_the_cache`.

---

### Task 1: Probe trace scope (Tensors)

**Files:**
- Modify: `src/AiDotNet.Tensors/Engines/Compilation/GraphMode.cs` (add `IsProbe`, `EnableProbe()`)
- Modify: `src/AiDotNet.Tensors/Engines/Compilation/LazyTensorScope.cs` (probe flag; the `Record*` output allocation
  uses a shape-only tensor when probing)
- Modify: `src/AiDotNet.Tensors/LinearAlgebra/TensorBase.cs:160-175, 2918-2937` (realizing a lazy node, or
  materializing deferred storage, while `GraphMode.IsProbe` throws `ObjectiveProbeAbortedException`)
- Create: `src/AiDotNet.Tensors/Engines/Compilation/ObjectiveProbeAbortedException.cs`
- Test: `tests/AiDotNet.Tensors.Tests/Engines/Compilation/ObjectiveProbeScopeTests.cs`

**Interfaces:**
- Produces:
  - `internal static LazyTensorScope GraphMode.EnableProbe()`
  - `internal static bool GraphMode.IsProbe`
  - `internal sealed class ObjectiveProbeAbortedException : InvalidOperationException { internal string OpName { get; } }`
  - `LazyTensorScope.Nodes` (exists) lists the probe's nodes. Probe outputs come from `Tensor<T>.CreateDeferred(shape)`
    and are never materialized.

- [ ] **Step 1: Write the tests**
  - `Probe_records_every_op_without_tensor_storage`: CPU engine, x [256,256], closure
    `ReduceSum(Tanh(MatMul(x, w)))`. Warm up one probe, then measure `GC.GetTotalAllocatedBytes(true)` around a
    second. Assert `scope.Nodes.Count == 3` and delta `< 3 * 256 * 256 * 4 / 4` (under a quarter of one activation's
    bytes).
  - `Host_read_of_a_probe_tensor_aborts_the_probe`: closure calls `.ToArray()` on the Tanh output. Assert
    `ObjectiveProbeAbortedException` with `OpName` naming the Tanh node.
  - `Probe_does_not_change_ordinary_graph_mode`: `GraphMode.Enable()` traced the same closure; the output is
    materialized and readable as before.
- [ ] **Step 2: Implement.** In probe mode, `Record*` methods allocate output via `Tensor<T>.CreateDeferred(outputShape)`
  in place of `TensorAllocator.RentUninitialized`. The existing realization sites check `GraphMode.IsProbe` before
  `node.Realize` and before `MaterializeDeferredStorage`, and throw.
- [ ] **Step 3: Build** (Tensors net10.0 + net471, tests net10.0). Expected `0 Error(s)`.
- [ ] **Step 4: Commit** `feat(compile): storage-free probe trace scope`.

### Task 2: Objective signature (Tensors)

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/Compilation/ObjectiveSignature.cs`
- Test: `tests/AiDotNet.Tensors.Tests/Engines/Compilation/ObjectiveSignatureTests.cs`

**Interfaces:**
- Consumes: `LazyTensorScope.Nodes`, `LazyNode<T>` (`OpName`, `OpType`, `OutputShape`, inputs, `Execute`,
  `BackwardFn`, `SavedState`).
- Produces:
  - `internal sealed class ObjectiveSignature : IEquatable<ObjectiveSignature>`, with
    `static ObjectiveSignature Build<T>(IReadOnlyList<ILazyNode> nodes, Tensor<T>[] parameters, out Tensor<T>[] leaves)`.
  - `bool Signable`, `string? UnsignableReason` (names op and field), `int NodeCount`.
  - Equality compares the full token sequence, not just the hash.

Token rules (spec §Signature):
- Per node: `OpName`, `OpType`, `OutputShape`, element type, and the `Execute.Method`/`BackwardFn.Method` handles.
- Each input as a reference class: node index, leaf position, parameter index.
- Each field of `Execute.Target` by value: primitives and enums; primitive arrays by content; `int[]`; tensors as a
  reference class; `IEngine` ignored. Any other reference type makes the node unsignable.
- `SavedState` entries by the same rules.
- Leaves are listed in first-use order and contribute shape + dtype. Field lists are cached per closure `Type` in a
  `ConcurrentDictionary<Type, FieldInfo[]>`.

- [ ] **Step 1: Write the tests**
  - `Same_closure_twice_is_equal_and_lists_leaves_in_first_use_order`: two probes of a closure over a new `x` each
    time. Equal; `leaves[0]` is that probe's `x`.
  - `Different_leaf_data_same_shapes_is_equal`.
  - `Captured_scalar_difference_is_not_equal`: `TensorMultiplyScalar(x, a)` with `a = 0.5f` vs `0.25f`.
  - `Shape_change_is_not_equal`: [4,8] vs [3,8].
  - `Unknown_reference_field_is_unsignable_and_named`: a test op recorded through `RecordUnary` with a closure
    capturing a `List<int>`. `Signable == false` and `UnsignableReason` contains the op name and the field name.
- [ ] **Step 2: Implement** `Build`.
- [ ] **Step 3: Build.** Expected `0 Error(s)`.
- [ ] **Step 4: Commit** `feat(compile): structural objective signature over lazy nodes`.

### Task 3: ClosureObjectiveGuard (Tensors)

**Files:**
- Create: `src/AiDotNet.Tensors/Engines/Compilation/ClosureObjectiveGuard.cs`
- Create: `src/AiDotNet.Tensors/Engines/Compilation/ClosureGuardResult.cs` (enum)
- Modify: `src/AiDotNet.Tensors/Engines/Compilation/LazyTensorScope.cs:520-540` (`CompileTraining` hands nodes and
  parameters to an active recording)
- Test: `tests/AiDotNet.Tensors.Tests/Engines/Compilation/ClosureObjectiveGuardTests.cs`

**Interfaces:**
- Consumes: Task 1 `GraphMode.EnableProbe`, Task 2 `ObjectiveSignature.Build`.
- Produces:
  - `internal enum ClosureGuardResult { Matched, Mismatched, Aborted, Unsignable }`
  - `internal sealed class ClosureObjectiveGuard<T>`:
    - `static IDisposable BeginRecording()`
    - `static ClosureObjectiveGuard<T>? TakeRecorded()`: the guard for the plan compiled inside the recording, or null
      when it was unsignable.
    - `ObjectiveSignature Signature { get; }`
    - `ClosureGuardResult TryRebind(Func<Tensor<T>> objective, Tensor<T>[] parameters, out ObjectiveSignature probed)`.
      On `Matched` it has copied every probe leaf's data into the compiled leaf at the same position (skipping
      same-object leaves, and calling `IncrementVersion()` on each written leaf so the plan's `RefreshGraphInputInPlace`
      uploads it). On `Aborted`, `probed` carries the abort reason.
    - `string? LastReason { get; }`

- [ ] **Step 1: Write the tests** (CPU engine; GPU variants `[SkippableFact]` for CI's CUDA lane)
  - `Rebound_plan_matches_eager_on_fresh_data_each_step`: the closure builds `x_t` new each step from seed `t`, loss
    `ReduceSum(Square(MatMul(x_t, w)))`, SGD lr 0. For t = 0..4: `TryRebind == Matched`, then `plan.Step()` loss
    equals the eager loss on `x_t` within `1e-5` relative.
  - `Captured_scalar_change_is_a_mismatch_and_never_replays`: the closure multiplies by `a_t = 1 + t`. Step 1 returns
    `Mismatched`, and the caller doesn't step the plan. Falsify: with closure fields excluded from the signature, this
    test fails.
  - `Same_leaf_objects_rebind_as_a_no_op`: the closure reuses one persistent `x` mutated in place; `Matched` every
    step, and losses match eager.
  - `Host_read_closure_returns_aborted`.
- [ ] **Step 2: Implement.** `CompileTraining` checks a thread-static recording and, if present, builds the signature
  from `_nodes` and `parameters` and stores the leaves. `TryRebind` runs the objective under `GraphMode.EnableProbe()`,
  catches `ObjectiveProbeAbortedException` (returns `Aborted`), builds the signature, compares, and copies leaves.
- [ ] **Step 3: Build.** Expected `0 Error(s)`.
- [ ] **Step 4: Commit and open the Tensors draft PR** (base `wip/release-at-last-use-arena`). If PR CI doesn't run
  for that base, dispatch `ci.yml` on the branch. Expected: CI build + Compilation tests PASS.

### Task 4: Closure-objective mode in CompiledTapeTrainingStep (AiDotNet)

**Files:**
- Modify: `src/Training/FusedTrainingSession.cs:35-70` (`FusedTrainingStepRequest<T>.ClosureObjective`)
- Modify: `src/Training/CompiledTapeTrainingStep.cs:1130-1700` (`TryStepWithFusedOptimizerCore` closure mode)
- Create: `src/Training/ClosureObjectiveState.cs` (per-owner plan cache, step counts, dynamic flag)
- Test: `tests/AiDotNet.Tests/IntegrationTests/Training/ClosureObjectiveFusedTrainingTests.cs`

**Interfaces:**
- Consumes: Task 3 `ClosureObjectiveGuard<T>`, `ClosureGuardResult`.
- Produces:
  - `public Func<Tensor<T>>? ClosureObjective { get; init; }` on `FusedTrainingStepRequest<T>`. When set, `Forward`
    and `ComputeLoss` are ignored and `Input`/`Target` may be 1-element placeholders.
  - `internal sealed class ClosureObjectiveState<T>`: a ≤4-entry LRU of
    `(ObjectiveSignature, ICompiledTrainingPlan<T>, ClosureObjectiveGuard<T>)`, the first-8-step miss counter and
    `bool IsDynamic`.
  - `internal enum ClosureFallbackReason { Aborted, Unsignable, Dynamic }`, reported through
    `CompiledTapeTrainingStep<T>.GetLastFallbackException()`'s diagnostic path.

- [ ] **Step 1: Write the tests**
  - `Closure_objective_trains_fused_and_matches_eager`: a 2-layer net, with the closure drawing fresh noise per step
    inside itself. 6 steps fused vs 6 eager from cloned weights; parameters within `1e-5` relative;
    `FusedSteps == 6`.
  - `Partial_last_batch_compiles_a_second_signature`: batches 8, 8, 5, 8; all fused, parity holds, 2 cached plans.
  - `Dynamic_objective_goes_eager_and_caps_the_cache`: batch size `t + 1` for 12 steps. After step 8 every step is
    eager, cached plans ≤ 4, one diagnostic.
- [ ] **Step 2: Implement.** In closure mode, compile inside `ClosureObjectiveGuard<T>.BeginRecording()` tracing
  `ClosureObjective()`. Before each replay, call `guard.TryRebind`:
  - `Matched`: replay.
  - `Mismatched`: LRU lookup, else compile.
  - `Aborted`/`Unsignable`: return false with the reason.
  - Apply the dynamic rule.
- [ ] **Step 3: Build** AiDotNet with `-p:UseLocalTensors=true`. Expected `0 Error(s)`.
- [ ] **Step 4: Commit** `feat(training): compiled closure objectives in the fused session`.

### Task 5: Route the eager objective entry points (AiDotNet)

**Files:**
- Modify: `src/Training/TapeTrainingStepper.cs:240-330` (`EagerObjectiveStep` ×2, `EagerSgdObjectiveStep` call a new
  `ObjectiveStep`)
- Modify: `src/NeuralNetworks/NeuralNetworkBase.cs:14862-14890` (`TrainWithCustomLoss` routes through it)
- Test: `tests/AiDotNet.Tests/IntegrationTests/Training/ClosureObjectiveFamilyParityTests.cs`

**Interfaces:**
- Consumes: Task 4 `FusedTrainingStepRequest<T>.ClosureObjective`.
- Produces: `internal static T ObjectiveStep(object owner, Func<IReadOnlyList<Tensor<T>>> parameters,
  Func<Tensor<T>> objective, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer, EagerStepHooks hooks)`.
  It tries the fused closure path and otherwise runs the existing `EagerTapeStep`.
  - Stays eager when `hooks.BeforeUpdate` is set, the optimizer is a line-search one, or
    `TensorCodecOptions.Current.EnableCompilation` is false.
  - The SGD variant maps to the fused SGD optimizer type.

- [ ] **Step 1: Write the tests** (theory over families, each 5 steps with fresh data, compiled vs eager from cloned
  weights, plus `FusedSteps > 0`): DDPM via `StepWithCustomLoss`, OCTGAN generator, a `SelfSupervisedLearningMethodBase`
  method, `FTTransformerClassifier`, a `TensorModelTrainer` model, `ChronosT5`.
- [ ] **Step 2: Implement** the routing. `TapeTrainingLoopRatchetTests` must stay green.
- [ ] **Step 3: Build.** Expected `0 Error(s)`.
- [ ] **Step 4: Commit and open the AiDotNet draft PR.** Mark it ready only when CI evidence is needed (AiDotNet draft
  PRs skip tests).

### Task 6: Measure and report

- [ ] Parity-harness A/B (compiled vs `EnableCompilation=false`, median-of-9 across 3 launches) on a closure model per
  family available in the harness. Post the per-model ratio, probe time and any model where compiled isn't faster to
  #2561 and both PRs.
