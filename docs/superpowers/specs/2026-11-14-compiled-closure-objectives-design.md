# Compiled plans for closure objectives (base-class level)

Date: 2026-11-14. Repos: AiDotNet.Tensors (compiled-plan machinery) and AiDotNet (one shared entry point).

## Intent

Every AiDotNet model should train on a compiled plan by default, because eager is always slower and the library's
job is to pick the best settings for a beginner. About 130 model files currently never try one:

- `NeuralNetworkBase.TrainWithCustomLoss`: 93 callers.
- `TapeTrainingStepper.EagerObjectiveStep` / `EagerSgdObjectiveStep`: about 17 direct callers. These include
  `SelfSupervisedLearningMethodBase` (8 methods), `TensorModelTrainer` (7 vision models), the synthetic-data GANs, the
  tabular transformers, Chronos, SAM2 and `DiffusionModelBase.StepWithCustomLoss`.

They stay eager because their loss is a closure. A closure creates its per-step data (noise, labels, generated
samples, a sampled timestep) as new tensors and scalars inside itself, and a compiled plan traced once would replay
the first step's values forever.

The fix belongs in shared code: model authors write only their loss, and new models need no extra work. Eager stays
correct and selectable.

## Goals

- Closure objectives run on a compiled plan, CUDA graph included, with no per-model code.
- It is correct by construction: a step whose closure differs from the compiled one in any way the plan would freeze
  never replays that plan.
- Per-step overhead stays well below an eager step. A model where it doesn't falls back to eager, chosen by
  measurement rather than guessed.
- The eager path is unchanged and selectable (`TensorCodecOptions.EnableCompilation = false`, or an explicit eager
  choice).

## Non-goals

- Migrating call sites to `TrainObjectiveFused` with explicit tensor packing. Hot models may still do that later to
  drop the probe cost.
- Changing any model's loss or optimizer semantics.

## Anti-goals (a failure even if it "works")

- Ever training on a previous step's data or scalar.
- Being slower than eager on a model, unnoticed.
- A plan cache that grows without bound on a genuinely dynamic objective.

## Approaches considered

1. **Re-trace and rebind (chosen).** Each step, re-run the closure as a storage-free probe trace. Compare its
   signature with the compiled plan's. If they match, copy the new step's leaf data into the plan's leaves and replay
   the plan. Pros: generic, base-class level, correct by construction. Cons: host tracing cost every step.
2. **Migrate call sites.** Move each of the ~130 files to `TrainObjectiveFused` with per-step tensors packed into
   one input. Rejected as the primary fix: it's per-model work with a freeze-the-data bug risk at every site, and
   new models must remember to do it.
3. **Auto-compile the eager tape step.** `AutoTrainingCompiler` already hashes eager tape structure and compiles the
   backward pass. It doesn't compile the forward or give a whole-step CUDA graph, and it keys on structure only, so it
   can't see a changed captured scalar. It's reused below only for its structure-hash idea.

## Design

### Tensors: `ClosureObjectiveGuard<T>` attached to an ordinary compiled plan (internal, visible to AiDotNet)

The guard attaches to a plan the way `CompiledStepRandom` does: a recording scope wraps the compile, and `Attach` binds
the trace's signature and leaves to the plan. AiDotNet's `CompiledTapeTrainingStep` already composes the plan with the
optimizer, schedules, clip and gradient surface, so it calls the guard before each replay. A separate plan type would
duplicate that composition. The steps below describe the plan + guard pair.

Inputs: `Func<Tensor<T>> objective` and `Func<IReadOnlyList<Tensor<T>>> parameters`. Optimizer configuration is the
same as `CompiledTrainingPlan`'s.

1. **First step: compile.** Trace `objective` under `GraphMode` and `CompileTraining(parameters)`, exactly as
   compiled plans are built today. Record the trace's **signature** and its **external leaves in record order**
   (inputs no node produces that aren't parameters).
2. **Later steps: probe.** Re-run `objective` under a probe scope (below). The probe records nodes with storage-free
   outputs and never executes a kernel. Compute the probe's signature.
   - **Match:** for each leaf position `i`, make plan leaf `i` hold the probe's leaf `i` data. Skip it if it's the
     same object; copy host data if the new leaf is on the host; device-to-device copy if it's resident. The existing
     `RefreshGraphInputInPlace` then uploads into the captured buffers, and the plan steps (graph replay).
   - **Mismatch:** look the signature up in a small per-owner cache (up to 4 plans, LRU) and compile on a miss.
3. **Signature.** For every node in record order:
   - `OpName`, `OpType`, `OutputShape` and element type.
   - Each input as a reference class: node index, leaf position or parameter index.
   - `Execute.Method` and `BackwardFn.Method` identity.
   - Every field of the `Execute` closure (`Execute.Target`) by value: primitives, enums, primitive arrays by content,
     and tensors as a reference class (as inputs). Any other field type makes the node **unsignable**.
   - `SavedState` entries by the same rules.

   Field lists are cached per closure type, so steady state is field reads plus a hash. Leaves contribute
   shape + dtype. This is what catches a per-step scalar (a sampled timestep's `alpha_t` passed to
   `TensorMultiplyScalar`): it's a closure field, so a new value changes the signature.
4. **Probe scope.** A `LazyTensorScope` mode that:
   - gives each recorded output a shape-only tensor (no storage rent);
   - aborts the probe on any realization or host read of a probe tensor (`.Item`, `ToArray`, data-dependent shape
     logic). An aborted probe means the objective reads its own intermediate values on the host and can't replay a
     fixed graph; that step runs eagerly.
5. **Dynamic objectives.** If an owner aborts, or misses the cache, on more than 2 of its first 8 steps, it's marked
   dynamic. It trains eagerly from then on and emits one diagnostic (`AIDOTNET_FUSED_DIAG` names the node that
   differed or the read that aborted).
6. **Unsignable nodes** (a closure field of an unknown reference type) make the whole objective eager, again with one
   diagnostic naming the op and field. These are fixed op by op, which is a finite, testable list rather than a
   per-model one.

### AiDotNet: one shared entry point

- `TapeTrainingStepper.EagerObjectiveStep`, `EagerSgdObjectiveStep` and `NeuralNetworkBase.TrainWithCustomLoss` route
  through a new `ObjectiveStep`. It tries the owner's `CompiledObjectivePlan` on the shared `FusedTrainingSession`
  (the same optimizer mapping, schedules, clip, regularization and gradient surface `TrainCustomTapeStep` already
  uses) and falls back to the existing eager step. Plans are keyed per owner via the existing `ByOwner` table.
- Stays eager by rule, with a diagnostic:
  - a `beforeUpdate` hook (QAT shadow restore) until the plan can run a hook between backward and update;
  - line-search optimizers that re-evaluate the objective;
  - `EnableCompilation = false`.
- `TapeTrainingLoopRatchetTests` keeps new copies of the loop from appearing.

## Verification

- **Parity:** for each closure-model family (diffusion custom loss, a GAN, an SSL method, a tabular transformer, a
  vision trainer model, Chronos), the compiled path's weights after N steps with fresh per-step data match the eager
  path's (the `CustomObjectiveFusedTrainingTests` pattern).
- **Guard falsification:**
  - a closure that captures a per-step scalar must never replay stale; the test fails with the signature's closure
    fields removed;
  - a closure with a host read must fall back.
- **Probe cost:** the probe allocates no tensor storage (checked with `GC.GetTotalAllocatedBytes` per step after
  warm-up), and probe time is reported per model.
- **Speed:** a parity-harness A/B (compiled vs eager, median-of-9 across launches) on DDPM's custom loss, a GAN and an
  SSL method. A model where compiled isn't faster is listed, not hidden.

## Left to the builder

- Where the probe flag lives in `LazyTensorScope`, and how a shape-only tensor is represented.
- The exact field-type allowlist for signing.
- Whether leaf copies use the existing upload paths or a new device-to-device copy.
