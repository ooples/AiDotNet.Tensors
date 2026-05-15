using System.Buffers;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A compiled training plan — forward + backward as flat delegate arrays.
/// Compile once, replay forever with near-zero allocation.
///
/// The backward pass is specialized at compile time:
/// - Single-consumer tensors get direct-write backward (BLAS writes into pre-allocated buffer)
/// - Multi-consumer tensors use accumulation (safe for shared parameters like in RNNs)
/// - All gradient buffers are pre-allocated and zeroed before each step
///
/// This REPLACES the GradientTape for compiled workloads.
/// </summary>
internal sealed class CompiledTrainingPlan<T> : ICompiledTrainingPlan<T>
{
    // Phase G.4: mutable so EnableFrozenWeightOptimizations() can swap in
    // a rebuilt forward-action array using allowCachedB=true.
    private Action<IEngine>[] _forwardActions;
    private readonly Action<IEngine>[] _backwardActions;
    private readonly Tensor<T> _lossOutput;
    private readonly IEngine _engine;
    private readonly Tensor<T>[] _parameters;
    private readonly Tensor<T>[] _gradients;
    private readonly Tensor<T>[] _preAllocatedGrads;
    private readonly Tensor<T> _lossGradSeed;
    private readonly List<GCHandle> _pinnedHandles = new();
    private bool _disposed;

    // Phase G.4: rebuild state captured at compile time so
    // EnableFrozenWeightOptimizations() can preserve fused-group action
    // identity while replacing non-fused MatMul forward specializations.
    private readonly int[]? _fusedStepIndices;
    private readonly Action<IEngine>[]? _fusedForwardActions;
    private bool _isFrozenWeights;

    // Indices of gradient buffers that need zeroing (used by generic/accumulating backward only)
    private readonly int[]? _genericGradIndices;

    // Cached raw arrays for zero-overhead Step() — avoid AsSpan()/GetDataArray() per call
    private T[][]? _cachedGradArrays;
    private T[]? _cachedLossGradSeedArray;
    private T[]? _cachedLossGradDestArray;
    private readonly Tensor<T>? _lossGradDest; // Pre-allocated gradient buffer for loss output

    // Retained for plan serialization (issue #166). The forward actions are
    // specialized closures built from these steps — they're fast but lose
    // the structural metadata (OpType, Inputs, SavedState) that SaveAsync
    // needs to write. Storing both costs ~N*sizeof(ref) extra per plan,
    // which is negligible relative to the pre-allocated tensor buffers.
    private readonly CompiledStep<T>[]? _forwardSteps;
    private readonly int[]? _compiledInputShape;
    private readonly Tensor<T>? _compiledInputTensor;

    private CompiledTrainingPlan(
        Action<IEngine>[] forwardActions,
        Action<IEngine>[] backwardActions,
        Tensor<T> lossOutput,
        IEngine engine,
        Tensor<T>[] parameters,
        Tensor<T>[] preAllocatedGrads,
        Tensor<T>[] gradients,
        Tensor<T> lossGradSeed,
        int[]? genericGradIndices = null,
        Tensor<T>? lossGradDest = null,
        List<GCHandle>? pinnedHandles = null,
        CompiledStep<T>[]? forwardSteps = null,
        int[]? compiledInputShape = null,
        Tensor<T>? compiledInputTensor = null,
        int[]? fusedStepIndices = null,
        Action<IEngine>[]? fusedForwardActions = null)
    {
        _forwardActions = forwardActions;
        _backwardActions = backwardActions;
        _lossOutput = lossOutput;
        _engine = engine;
        _parameters = parameters;
        _preAllocatedGrads = preAllocatedGrads;
        _gradients = gradients;
        _lossGradSeed = lossGradSeed;
        _genericGradIndices = genericGradIndices;
        _forwardSteps = forwardSteps;
        _compiledInputShape = compiledInputShape;
        _compiledInputTensor = compiledInputTensor;
        _fusedStepIndices = fusedStepIndices;
        _fusedForwardActions = fusedForwardActions;
        _lossGradDest = lossGradDest;
        if (pinnedHandles is not null)
            _pinnedHandles.AddRange(pinnedHandles);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var handle in _pinnedHandles)
        {
            if (handle.IsAllocated)
                handle.Free();
        }
        _pinnedHandles.Clear();
    }

    public Tensor<T>[] Gradients => _gradients;
    public int ForwardStepCount => _forwardActions.Length;
    public int BackwardStepCount => _backwardActions.Length;

    /// <inheritdoc/>
    public void EnableFrozenWeightOptimizations()
    {
        if (_isFrozenWeights) return;
        if (_forwardSteps is null)
            throw new InvalidOperationException(
                "EnableFrozenWeightOptimizations requires a compiled plan with retained forward steps. " +
                "Plans loaded from serialization don't currently carry the step metadata.");

        // Rebuild forward actions with allowCachedB=true. Walks the original
        // step list, preserves fused-group action identity, replaces non-
        // fused step specializations with the cached-B variant.
        var fusedSet = _fusedStepIndices is null
            ? null
            : new HashSet<int>(_fusedStepIndices);
        var rebuiltForward = new List<Action<IEngine>>(_forwardSteps.Length);
        int nextFusedGroupIdx = 0;
        for (int i = 0; i < _forwardSteps.Length; i++)
        {
            if (fusedSet is not null && fusedSet.Contains(i))
            {
                bool isFirstInGroup = i == 0 || !fusedSet.Contains(i - 1);
                if (isFirstInGroup && _fusedForwardActions is not null
                    && nextFusedGroupIdx < _fusedForwardActions.Length)
                {
                    rebuiltForward.Add(_fusedForwardActions[nextFusedGroupIdx]);
                    nextFusedGroupIdx++;
                }
                continue;
            }

            var step = _forwardSteps[i];
            var specialized = TryBuildSpecializedForward(step, _pinnedHandles, allowCachedB: true);
            if (specialized != null)
            {
                rebuiltForward.Add(specialized);
            }
            else
            {
                var output = step.OutputBuffer;
                var exec = step.Execute;
                rebuiltForward.Add(eng => exec(eng, output));
            }
        }
        _forwardActions = rebuiltForward.ToArray();
        _isFrozenWeights = true;
    }

    /// <inheritdoc/>
    public void StepInto(Tensor<T> lossOutput)
    {
        if (lossOutput is null) throw new ArgumentNullException(nameof(lossOutput));
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledTrainingPlan<T>));
        ValidateShapesMatch(_lossOutput, lossOutput, nameof(lossOutput));

        // Run the plan into its internal loss buffer, then copy into the
        // caller's buffer. Copy (not rebind) because specialized backward
        // kernels capture gradient array references at compile time — a
        // post-compile storage swap would leave those closures pointing
        // at the old buffer. Under CUDA graph capture the copy becomes a
        // device memcpy node and replays deterministically.
        Step();
        _lossOutput.AsSpan().CopyTo(lossOutput.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void SetInputs(Tensor<T>[] inputs)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledTrainingPlan<T>));

        // Training plan today captures at most a single input tensor (the
        // graph-input placeholder of the traced forward pass). Multi-input
        // graphs are tracked identically to the inference plan — per-name
        // writes via the scope's captured references remain the path for
        // N>1. Error naming mirrors the inference plan's shape.
        int expected = _compiledInputTensor is null ? 0 : 1;
        if (inputs.Length != expected)
            throw new ArgumentException(
                $"This plan was compiled with {expected} captured input(s); got {inputs.Length}.",
                nameof(inputs));
        if (expected == 0) return; // Zero-input plans accept empty as a no-op.

        var src = inputs[0] ?? throw new ArgumentException(
            "inputs[0] is null.", nameof(inputs));
        var dst = _compiledInputTensor;
        if (dst is null)
            throw new InvalidOperationException(
                "Internal invariant violated: _compiledInputTensor is null but expected==1.");
        ValidateShapesMatch(dst, src, "inputs[0]");
        src.AsSpan().CopyTo(dst.AsWritableSpan());
    }

    internal void SetInput(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledTrainingPlan<T>));

        var dst = _compiledInputTensor;
        if (dst is null)
            throw new InvalidOperationException(
                $"This plan was compiled with 0 captured input(s); " +
                $"{nameof(SetInput)} requires exactly one.");

        ValidateShapesMatch(dst, input, nameof(input));
        input.AsSpan().CopyTo(dst.AsWritableSpan());
    }

    private static void ValidateShapesMatch(Tensor<T> expected, Tensor<T> actual, string paramName)
    {
        if (expected._shape.Length != actual._shape.Length)
            throw new ArgumentException(
                $"{paramName} rank {actual._shape.Length} != plan rank {expected._shape.Length}.",
                paramName);
        for (int i = 0; i < expected._shape.Length; i++)
        {
            if (expected._shape[i] != actual._shape[i])
                throw new ArgumentException(
                    $"{paramName} shape [{string.Join(", ", actual._shape)}] " +
                    $"!= plan shape [{string.Join(", ", expected._shape)}].",
                    paramName);
        }
    }

    // Fused optimizer state
    private Action? _optimizerUpdate;
    private int _optimizerStep;

    // Gradient checkpointing (Phase 5.3)
    private GradientCheckpointing<T>? _checkpointing;

    /// <summary>
    /// Enables gradient checkpointing for this plan, reducing memory from O(N) to O(sqrt(N))
    /// at the cost of ~33% more compute (each segment's forward runs twice).
    /// Call once after compilation, before training loop.
    /// </summary>
    /// <param name="segmentSize">Steps per checkpoint segment. 0 = auto (sqrt(N)).</param>
    public void EnableCheckpointing(int segmentSize = 0)
    {
        // Wrap forward actions as CompiledSteps for the checkpointing system.
        // Each action writes to _lossOutput as a pass-through output reference —
        // the actual outputs are in the action's captured closures.
        var wrappedSteps = new CompiledStep<T>[_forwardActions.Length];
        for (int i = 0; i < _forwardActions.Length; i++)
        {
            var action = _forwardActions[i];
            wrappedSteps[i] = new CompiledStep<T>(
                "Forward_" + i,
                (eng, o) => action(eng),
                _lossOutput, // Shared output reference — checkpointing only needs execute
                Array.Empty<Tensor<T>>(),
                null, null);
        }
        _checkpointing = new GradientCheckpointing<T>(wrappedSteps, _engine, segmentSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Step()
    {
        var engine = _engine;
        bool stepTiming = StepTiming.Enabled;

        // Forward: use checkpointing if enabled, otherwise straight-line delegates
        long fwdStart = stepTiming ? Stopwatch.GetTimestamp() : 0;
        if (_checkpointing is not null)
        {
            _checkpointing.ForwardWithCheckpoints();
        }
        else
        {
            var fwd = _forwardActions;
            for (int i = 0; i < fwd.Length; i++)
                fwd[i](engine);
        }
        if (stepTiming) StepTiming.RecordForward(Stopwatch.GetTimestamp() - fwdStart);

        // Cache raw arrays on first call — avoids AsWritableSpan()/GetDataArray() per step
        var gradArrays = _cachedGradArrays;
        if (gradArrays == null)
        {
            gradArrays = new T[_preAllocatedGrads.Length][];
            for (int i = 0; i < _preAllocatedGrads.Length; i++)
                gradArrays[i] = _preAllocatedGrads[i].GetDataArray();
            _cachedGradArrays = gradArrays;
            _cachedLossGradSeedArray = _lossGradSeed.GetDataArray();
            _cachedLossGradDestArray = _lossGradDest?.GetDataArray();
        }

        // Only zero gradient buffers used by generic (accumulating) backward delegates.
        // Specialized backward delegates overwrite completely (TryGemmEx beta=0, SIMD ReLU).
        // At large sizes, this saves significant time by skipping unnecessary clears.
        if (_genericGradIndices != null)
        {
            for (int i = 0; i < _genericGradIndices.Length; i++)
            {
                int idx = _genericGradIndices[i];
                Array.Clear(gradArrays[idx], 0, _preAllocatedGrads[idx].Length);
            }
        }
        else
        {
            // First call: clear everything (safe fallback)
            for (int i = 0; i < gradArrays.Length; i++)
                Array.Clear(gradArrays[i], 0, _preAllocatedGrads[i].Length);
        }

        // Re-seed loss gradient — direct Array.Copy
        var seedArr = _cachedLossGradSeedArray;
        var destArr = _cachedLossGradDestArray;
        if (seedArr != null && destArr != null)
            Array.Copy(seedArr, destArr, seedArr.Length);

        // Backward: specialized delegates (direct BLAS into pre-allocated buffers)
        long bwdStart = stepTiming ? Stopwatch.GetTimestamp() : 0;
        var bwd = _backwardActions;
        for (int i = 0; i < bwd.Length; i++)
            bwd[i](engine);
        if (stepTiming) StepTiming.RecordBackward(Stopwatch.GetTimestamp() - bwdStart);

        // Fused optimizer update (if configured via ConfigureOptimizer)
        long optStart = stepTiming ? Stopwatch.GetTimestamp() : 0;
        _optimizerUpdate?.Invoke();
        if (stepTiming)
        {
            StepTiming.RecordOptimizer(Stopwatch.GetTimestamp() - optStart);
            StepTiming.IncrementStepCount();
        }

        return _lossOutput;
    }

    public unsafe void ConfigureOptimizer(
        OptimizerType optimizerType,
        float learningRate,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0f)
    {
        ConfigureOptimizer(
            optimizerType,
            LrSchedule.Constant(learningRate),
            beta1, beta2, eps, weightDecay);
    }

    /// <inheritdoc/>
    public unsafe void ConfigureOptimizer(
        OptimizerType optimizerType,
        LrSchedule schedule,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0f)
    {
        if (schedule is null) throw new ArgumentNullException(nameof(schedule));
        ValidatePlanOptimizerSupport(optimizerType);
        if (typeof(T) == typeof(float))
        {
            ConfigureOptimizerFloat(optimizerType, schedule, beta1, beta2, eps, weightDecay);
            return;
        }
        if (typeof(T) == typeof(double))
        {
            ConfigureOptimizerDouble(optimizerType, schedule, beta1, beta2, eps, weightDecay);
            return;
        }
        throw new NotSupportedException("Fused optimizer updates support float and double parameters.");
    }

    /// <inheritdoc/>
    public unsafe void ConfigureOptimizerGrouped(
        OptimizerType optimizerType,
        System.Collections.Generic.IReadOnlyList<LrSchedule> groupSchedules,
        System.Collections.Generic.IReadOnlyList<int> paramToGroup,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weightDecay = 0f)
    {
        if (groupSchedules is null) throw new ArgumentNullException(nameof(groupSchedules));
        if (paramToGroup is null) throw new ArgumentNullException(nameof(paramToGroup));
        if (groupSchedules.Count == 0)
            throw new ArgumentException("groupSchedules must contain at least one schedule.", nameof(groupSchedules));
        if (paramToGroup.Count != _parameters.Length)
            throw new ArgumentException(
                $"paramToGroup.Count ({paramToGroup.Count}) must equal the compiled parameter count ({_parameters.Length}).",
                nameof(paramToGroup));
        // Validate every schedule slot up front — the grouped hot path
        // evaluates GetLr() on ALL of them per step, not just the ones
        // referenced by paramToGroup. A null slot that no parameter maps
        // to today would still NRE on the first Step().
        for (int g = 0; g < groupSchedules.Count; g++)
        {
            if (groupSchedules[g] is null)
                throw new ArgumentException($"groupSchedules[{g}] is null.", nameof(groupSchedules));
        }
        for (int i = 0; i < paramToGroup.Count; i++)
        {
            int g = paramToGroup[i];
            if (g < 0 || g >= groupSchedules.Count)
                throw new ArgumentOutOfRangeException(
                    nameof(paramToGroup),
                    $"paramToGroup[{i}]={g} is out of range [0, {groupSchedules.Count}).");
        }
        ValidatePlanOptimizerSupport(optimizerType);
        if (typeof(T) == typeof(float))
        {
            ConfigureOptimizerFloatGrouped(optimizerType, groupSchedules, paramToGroup, beta1, beta2, eps, weightDecay);
            return;
        }
        if (typeof(T) == typeof(double))
        {
            ConfigureOptimizerDoubleGrouped(optimizerType, groupSchedules, paramToGroup, beta1, beta2, eps, weightDecay);
            return;
        }
        throw new NotSupportedException("Fused optimizer updates support float and double parameters.");
    }

    /// <summary>Gate at the plan-level dispatch surface. The closures
    /// built by ConfigureOptimizer*Float/Double only implement SGD,
    /// Adam, and AdamW today — accepting anything else (Lion, LAMB,
    /// HypergradientSGD, …) would configure successfully and then throw
    /// on the first Step(). <c>OptimizerKernels.IsFusedPathEligible</c>
    /// is the broader semantic predicate (which kernels exist at all);
    /// this is the narrower "what the plan can replay today" check.</summary>
    private static void ValidatePlanOptimizerSupport(OptimizerType optimizerType)
    {
        bool supported = optimizerType is OptimizerType.SGD
            or OptimizerType.Adam
            or OptimizerType.AdamW;
        if (!supported)
        {
            throw new NotSupportedException(
                $"Optimizer type {optimizerType} is not yet supported by CompiledTrainingPlan's " +
                "fused-update closures. Currently supported: SGD, Adam, AdamW. " +
                "The kernel for this optimizer may still exist (see OptimizerKernels.IsFusedPathEligible); " +
                "use eager apply via OptimizerKernels directly until a plan-level dispatch branch lands.");
        }
    }

    private unsafe void ConfigureOptimizerFloat(
        OptimizerType optimizerType, LrSchedule schedule, float beta1, float beta2, float eps, float weightDecay)
    {
        // Pre-allocate optimizer state buffers for each parameter
        int paramCount = _parameters.Length;
        var paramArrays = new float[paramCount][];
        var gradArrays = new float[paramCount][];
        var lengths = new int[paramCount];

        // Momentum / first moment buffers (Adam, SGD+momentum, etc.)
        var m = new float[paramCount][];
        // Second moment buffers (Adam, RMSprop, etc.)
        var v = new float[paramCount][];

        // Issue #350: GetDataArray() returns a COPY when the parameter tensor's
        // backing storage is pool-padded (e.g. logical length 6 on a 16-slot
        // ArrayPool bucket — common on net8+ where pool rent rounds up to
        // power-of-two). Pinning that copy and writing through fixed pointer
        // updates the COPY, not the caller's tensor — so plan.Step() returns
        // success and the parameter never actually moves. The live-backing
        // accessor below is "allowing padding" because the fused kernel reads
        // exactly `lengths[p]` elements (the logical tensor size), never
        // touching the pool-padding tail. For non-trivial layouts (views,
        // non-zero offset, GPU-resident) the live-backing accessor returns
        // null and we throw — that's a "should never happen for params
        // registered with CompileTraining" condition and a silent fallback to
        // a copy would just resurrect this bug.
        for (int p = 0; p < paramCount; p++)
        {
            var liveParam = _parameters[p].GetLiveBackingArrayAllowingPaddingOrNull();
            if (liveParam is null)
                throw new InvalidOperationException(
                    $"Parameter {p} (shape [{string.Join(",", _parameters[p].Shape)}]) has a layout that does not expose " +
                    $"a live CPU backing array (non-contiguous view, non-zero storageOffset, or non-CPU device). " +
                    $"ConfigureOptimizer requires every registered parameter to be a contiguous CPU tensor so " +
                    $"the fused optimizer step can mutate the caller's tensor in place. Call .Contiguous() / " +
                    $"copy the parameter to CPU before registering it with CompileTraining.");
            paramArrays[p] = (float[])(object)liveParam;
            if (_gradients[p] is not null)
            {
                // Mirror the parameter contract: gradients also have to expose
                // a live CPU backing array. A copy-fallback here would let
                // _optimizerUpdate's `fixed (T* pGrad = gradArrays[p])` read
                // a snapshot taken at compile time while backward writes to a
                // different storage — the same stale-buffer bug the parameter
                // fix above is fixing, but on the gradient side. Fail fast so
                // the bug cannot resurrect through the gradient path.
                var liveGrad = _gradients[p].GetLiveBackingArrayAllowingPaddingOrNull();
                if (liveGrad is null)
                    throw new InvalidOperationException(
                        $"Gradient {p} (shape [{string.Join(",", _gradients[p].Shape)}]) has a layout that does not expose " +
                        $"a live CPU backing array (non-contiguous view, non-zero storageOffset, or non-CPU device). " +
                        $"ConfigureOptimizer requires live gradient storage so the fused optimizer step reads the " +
                        $"current gradient values produced by each plan.Step()'s backward pass.");
                gradArrays[p] = (float[])(object)liveGrad;
            }
            else
            {
                gradArrays[p] = Array.Empty<float>();
            }
            lengths[p] = _parameters[p].Length;

            bool needsMomentum = optimizerType is OptimizerType.Adam or OptimizerType.AdamW
                or OptimizerType.SGDMomentum or OptimizerType.Lion or OptimizerType.Nadam
                or OptimizerType.AdaMax or OptimizerType.AMSGrad;
            bool needsSecondMoment = optimizerType is OptimizerType.Adam or OptimizerType.AdamW
                or OptimizerType.RMSprop or OptimizerType.Nadam or OptimizerType.AMSGrad
                or OptimizerType.Adagrad;

            m[p] = needsMomentum ? new float[lengths[p]] : Array.Empty<float>();
            v[p] = needsSecondMoment ? new float[lengths[p]] : Array.Empty<float>();
        }

        _optimizerStep = 0;
        var b1 = beta1;
        var b2 = beta2;
        var epsVal = eps;
        var wd = weightDecay;
        var optType = optimizerType;
        var lrSchedule = schedule;

        _optimizerUpdate = () =>
        {
            _optimizerStep++;
            // Issue #348: read lr from the schedule each step. PyTorch's
            // LRScheduler.step() pays managed-code dispatch overhead per
            // step; here it's an inlined Math.Cos / Math.Pow.
            float lr = (float)lrSchedule.GetLr(_optimizerStep);
            for (int p = 0; p < paramCount; p++)
            {
                if (gradArrays[p].Length == 0) continue;
                int len = lengths[p];

                fixed (float* pParam = paramArrays[p], pGrad = gradArrays[p],
                       pM = m[p], pV = v[p])
                {
                    switch (optType)
                    {
                        case OptimizerType.SGD:
                            FusedOptimizer.SgdUpdateSimd(pParam, pGrad, len, lr);
                            break;
                        case OptimizerType.Adam:
                            FusedOptimizer.AdamUpdateSimd(pParam, pGrad, pM, pV, len,
                                lr, b1, b2, epsVal, _optimizerStep);
                            break;
                        case OptimizerType.AdamW:
                            FusedOptimizer.AdamWUpdateSimd(pParam, pGrad, pM, pV, len,
                                lr, b1, b2, epsVal, wd, _optimizerStep);
                            break;
                        default:
                            throw new NotSupportedException(
                                $"Optimizer type {optType} is not yet supported by ConfigureOptimizer. " +
                                $"Supported: SGD, Adam, AdamW. Apply gradients manually for other types.");
                    }
                }
            }
        };
    }

    private unsafe void ConfigureOptimizerFloatGrouped(
        OptimizerType optimizerType,
        System.Collections.Generic.IReadOnlyList<LrSchedule> groupSchedules,
        System.Collections.Generic.IReadOnlyList<int> paramToGroup,
        float beta1, float beta2, float eps, float weightDecay)
    {
        int paramCount = _parameters.Length;
        int groupCount = groupSchedules.Count;
        var paramArrays = new float[paramCount][];
        var gradArrays = new float[paramCount][];
        var lengths = new int[paramCount];
        var m = new float[paramCount][];
        var v = new float[paramCount][];
        var paramGroup = new int[paramCount];
        // Snapshot schedule references — concrete types so closure can
        // see them without an extra interface dispatch per step.
        var schedules = new LrSchedule[groupCount];
        for (int g = 0; g < groupCount; g++) schedules[g] = groupSchedules[g];
        for (int p = 0; p < paramCount; p++) paramGroup[p] = paramToGroup[p];

        // Issue #350: live-backing binding (see ConfigureOptimizerFloat).
        for (int p = 0; p < paramCount; p++)
        {
            var liveParam = _parameters[p].GetLiveBackingArrayAllowingPaddingOrNull();
            if (liveParam is null)
                throw new InvalidOperationException(
                    $"Parameter {p} (shape [{string.Join(",", _parameters[p].Shape)}]) has a layout that does not expose " +
                    $"a live CPU backing array (non-contiguous view, non-zero storageOffset, or non-CPU device). " +
                    $"ConfigureOptimizerGrouped requires every registered parameter to be a contiguous CPU tensor.");
            paramArrays[p] = (float[])(object)liveParam;
            if (_gradients[p] is not null)
            {
                // Fail fast on copy-fallback (see ConfigureOptimizerFloat for full rationale).
                var liveGrad = _gradients[p].GetLiveBackingArrayAllowingPaddingOrNull();
                if (liveGrad is null)
                    throw new InvalidOperationException(
                        $"Gradient {p} (shape [{string.Join(",", _gradients[p].Shape)}]) has a layout that does not expose " +
                        $"a live CPU backing array. ConfigureOptimizer requires live gradient storage.");
                gradArrays[p] = (float[])(object)liveGrad;
            }
            else
            {
                gradArrays[p] = Array.Empty<float>();
            }
            lengths[p] = _parameters[p].Length;

            bool needsMomentum = optimizerType is OptimizerType.Adam or OptimizerType.AdamW
                or OptimizerType.SGDMomentum or OptimizerType.Lion or OptimizerType.Nadam
                or OptimizerType.AdaMax or OptimizerType.AMSGrad;
            bool needsSecondMoment = optimizerType is OptimizerType.Adam or OptimizerType.AdamW
                or OptimizerType.RMSprop or OptimizerType.Nadam or OptimizerType.AMSGrad
                or OptimizerType.Adagrad;

            m[p] = needsMomentum ? new float[lengths[p]] : Array.Empty<float>();
            v[p] = needsSecondMoment ? new float[lengths[p]] : Array.Empty<float>();
        }

        _optimizerStep = 0;
        var b1 = beta1;
        var b2 = beta2;
        var epsVal = eps;
        var wd = weightDecay;
        var optType = optimizerType;
        var groupLrs = new float[groupCount];

        _optimizerUpdate = () =>
        {
            _optimizerStep++;
            // Resolve each group's lr ONCE per step. PyTorch does N kernel
            // launches for N groups; we do one schedule eval per group and
            // one fused-kernel call per parameter.
            for (int g = 0; g < groupCount; g++)
                groupLrs[g] = (float)schedules[g].GetLr(_optimizerStep);

            for (int p = 0; p < paramCount; p++)
            {
                if (gradArrays[p].Length == 0) continue;
                int len = lengths[p];
                float lr = groupLrs[paramGroup[p]];

                fixed (float* pParam = paramArrays[p], pGrad = gradArrays[p],
                       pM = m[p], pV = v[p])
                {
                    switch (optType)
                    {
                        case OptimizerType.SGD:
                            FusedOptimizer.SgdUpdateSimd(pParam, pGrad, len, lr);
                            break;
                        case OptimizerType.Adam:
                            FusedOptimizer.AdamUpdateSimd(pParam, pGrad, pM, pV, len,
                                lr, b1, b2, epsVal, _optimizerStep);
                            break;
                        case OptimizerType.AdamW:
                            FusedOptimizer.AdamWUpdateSimd(pParam, pGrad, pM, pV, len,
                                lr, b1, b2, epsVal, wd, _optimizerStep);
                            break;
                        default:
                            throw new NotSupportedException(
                                $"Optimizer type {optType} is not yet supported by ConfigureOptimizerGrouped. " +
                                $"Supported: SGD, Adam, AdamW.");
                    }
                }
            }
        };
    }

    private unsafe void ConfigureOptimizerDouble(
        OptimizerType optimizerType, LrSchedule schedule, float beta1, float beta2, float eps, float weightDecay)
    {
        // Mirror of ConfigureOptimizerFloat for double parameters. PR #319
        // follow-up: tape-trained Tensor<double> models (e.g. ViT-Base in
        // the consumer integration test) can now hit the fused-compiled
        // training path that was previously gated behind float-only
        // (NeuralNetworkBase.TryTrainWithFusedOptimizer line 4855 +
        // CompiledTapeTrainingStep.TryStepWithFusedOptimizer line 232).
        int paramCount = _parameters.Length;
        var paramArrays = new double[paramCount][];
        var gradArrays = new double[paramCount][];
        var lengths = new int[paramCount];
        var m = new double[paramCount][];
        var v = new double[paramCount][];

        // Issue #350: bind to the live backing (see ConfigureOptimizerFloat
        // for the full reasoning) — GetDataArray()'s pool-padded copy
        // semantics silently caused every fused-Adam Step() on T=double to
        // be a no-op for the caller. Live-backing-allowing-padding writes
        // straight through to _parameters[p].
        for (int p = 0; p < paramCount; p++)
        {
            var liveParam = _parameters[p].GetLiveBackingArrayAllowingPaddingOrNull();
            if (liveParam is null)
                throw new InvalidOperationException(
                    $"Parameter {p} (shape [{string.Join(",", _parameters[p].Shape)}]) has a layout that does not expose " +
                    $"a live CPU backing array (non-contiguous view, non-zero storageOffset, or non-CPU device). " +
                    $"ConfigureOptimizer requires every registered parameter to be a contiguous CPU tensor so " +
                    $"the fused optimizer step can mutate the caller's tensor in place.");
            paramArrays[p] = (double[])(object)liveParam;
            if (_gradients[p] is not null)
            {
                // Fail fast on copy-fallback (see ConfigureOptimizerFloat for full rationale).
                var liveGrad = _gradients[p].GetLiveBackingArrayAllowingPaddingOrNull();
                if (liveGrad is null)
                    throw new InvalidOperationException(
                        $"Gradient {p} (shape [{string.Join(",", _gradients[p].Shape)}]) has a layout that does not expose " +
                        $"a live CPU backing array. ConfigureOptimizer requires live gradient storage.");
                gradArrays[p] = (double[])(object)liveGrad;
            }
            else
            {
                gradArrays[p] = Array.Empty<double>();
            }
            lengths[p] = _parameters[p].Length;

            bool needsMomentum = optimizerType is OptimizerType.Adam or OptimizerType.AdamW
                or OptimizerType.SGDMomentum or OptimizerType.Lion or OptimizerType.Nadam
                or OptimizerType.AdaMax or OptimizerType.AMSGrad;
            bool needsSecondMoment = optimizerType is OptimizerType.Adam or OptimizerType.AdamW
                or OptimizerType.RMSprop or OptimizerType.Nadam or OptimizerType.AMSGrad
                or OptimizerType.Adagrad;

            m[p] = needsMomentum ? new double[lengths[p]] : Array.Empty<double>();
            v[p] = needsSecondMoment ? new double[lengths[p]] : Array.Empty<double>();
        }

        _optimizerStep = 0;
        double b1 = beta1;
        double b2 = beta2;
        double epsVal = eps;
        double wd = weightDecay;
        var optType = optimizerType;
        var lrSchedule = schedule;

        _optimizerUpdate = () =>
        {
            _optimizerStep++;
            double lr = lrSchedule.GetLr(_optimizerStep);
            for (int p = 0; p < paramCount; p++)
            {
                if (gradArrays[p].Length == 0) continue;
                int len = lengths[p];

                fixed (double* pParam = paramArrays[p], pGrad = gradArrays[p],
                       pM = m[p], pV = v[p])
                {
                    switch (optType)
                    {
                        case OptimizerType.SGD:
                            FusedOptimizer.SgdUpdateSimd(pParam, pGrad, len, lr);
                            break;
                        case OptimizerType.Adam:
                            FusedOptimizer.AdamUpdateSimd(pParam, pGrad, pM, pV, len,
                                lr, b1, b2, epsVal, _optimizerStep);
                            break;
                        case OptimizerType.AdamW:
                            FusedOptimizer.AdamWUpdateSimd(pParam, pGrad, pM, pV, len,
                                lr, b1, b2, epsVal, wd, _optimizerStep);
                            break;
                        default:
                            throw new NotSupportedException(
                                $"Optimizer type {optType} is not yet supported by ConfigureOptimizer (double). " +
                                $"Supported: SGD, Adam, AdamW. Apply gradients manually for other types.");
                    }
                }
            }
        };
    }

    private unsafe void ConfigureOptimizerDoubleGrouped(
        OptimizerType optimizerType,
        System.Collections.Generic.IReadOnlyList<LrSchedule> groupSchedules,
        System.Collections.Generic.IReadOnlyList<int> paramToGroup,
        float beta1, float beta2, float eps, float weightDecay)
    {
        int paramCount = _parameters.Length;
        int groupCount = groupSchedules.Count;
        var paramArrays = new double[paramCount][];
        var gradArrays = new double[paramCount][];
        var lengths = new int[paramCount];
        var m = new double[paramCount][];
        var v = new double[paramCount][];
        var paramGroup = new int[paramCount];
        var schedules = new LrSchedule[groupCount];
        for (int g = 0; g < groupCount; g++) schedules[g] = groupSchedules[g];
        for (int p = 0; p < paramCount; p++) paramGroup[p] = paramToGroup[p];

        // Issue #350: live-backing binding (see ConfigureOptimizerFloat).
        for (int p = 0; p < paramCount; p++)
        {
            var liveParam = _parameters[p].GetLiveBackingArrayAllowingPaddingOrNull();
            if (liveParam is null)
                throw new InvalidOperationException(
                    $"Parameter {p} (shape [{string.Join(",", _parameters[p].Shape)}]) has a layout that does not expose " +
                    $"a live CPU backing array (non-contiguous view, non-zero storageOffset, or non-CPU device). " +
                    $"ConfigureOptimizerGrouped requires every registered parameter to be a contiguous CPU tensor.");
            paramArrays[p] = (double[])(object)liveParam;
            if (_gradients[p] is not null)
            {
                // Fail fast on copy-fallback (see ConfigureOptimizerFloat for full rationale).
                var liveGrad = _gradients[p].GetLiveBackingArrayAllowingPaddingOrNull();
                if (liveGrad is null)
                    throw new InvalidOperationException(
                        $"Gradient {p} (shape [{string.Join(",", _gradients[p].Shape)}]) has a layout that does not expose " +
                        $"a live CPU backing array. ConfigureOptimizer requires live gradient storage.");
                gradArrays[p] = (double[])(object)liveGrad;
            }
            else
            {
                gradArrays[p] = Array.Empty<double>();
            }
            lengths[p] = _parameters[p].Length;

            bool needsMomentum = optimizerType is OptimizerType.Adam or OptimizerType.AdamW
                or OptimizerType.SGDMomentum or OptimizerType.Lion or OptimizerType.Nadam
                or OptimizerType.AdaMax or OptimizerType.AMSGrad;
            bool needsSecondMoment = optimizerType is OptimizerType.Adam or OptimizerType.AdamW
                or OptimizerType.RMSprop or OptimizerType.Nadam or OptimizerType.AMSGrad
                or OptimizerType.Adagrad;

            m[p] = needsMomentum ? new double[lengths[p]] : Array.Empty<double>();
            v[p] = needsSecondMoment ? new double[lengths[p]] : Array.Empty<double>();
        }

        _optimizerStep = 0;
        double b1 = beta1;
        double b2 = beta2;
        double epsVal = eps;
        double wd = weightDecay;
        var optType = optimizerType;
        var groupLrs = new double[groupCount];

        _optimizerUpdate = () =>
        {
            _optimizerStep++;
            for (int g = 0; g < groupCount; g++)
                groupLrs[g] = schedules[g].GetLr(_optimizerStep);

            for (int p = 0; p < paramCount; p++)
            {
                if (gradArrays[p].Length == 0) continue;
                int len = lengths[p];
                double lr = groupLrs[paramGroup[p]];

                fixed (double* pParam = paramArrays[p], pGrad = gradArrays[p],
                       pM = m[p], pV = v[p])
                {
                    switch (optType)
                    {
                        case OptimizerType.SGD:
                            FusedOptimizer.SgdUpdateSimd(pParam, pGrad, len, lr);
                            break;
                        case OptimizerType.Adam:
                            FusedOptimizer.AdamUpdateSimd(pParam, pGrad, pM, pV, len,
                                lr, b1, b2, epsVal, _optimizerStep);
                            break;
                        case OptimizerType.AdamW:
                            FusedOptimizer.AdamWUpdateSimd(pParam, pGrad, pM, pV, len,
                                lr, b1, b2, epsVal, wd, _optimizerStep);
                            break;
                        default:
                            throw new NotSupportedException(
                                $"Optimizer type {optType} is not yet supported by ConfigureOptimizerGrouped (double). " +
                                $"Supported: SGD, Adam, AdamW.");
                    }
                }
            }
        };
    }

    /// <inheritdoc/>
    public Task SaveAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledTrainingPlan<T>));
        cancellationToken.ThrowIfCancellationRequested();

        TrainingPlanWriter.Write(stream, this);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public bool IsCompatibleWith(PlanCompatibilityInfo info)
    {
        return info.GetIncompatibilityReason<T>() is null;
    }

    // ── Internal accessors for serialization ────────────────────────────
    internal Action<IEngine>[] ForwardActions => _forwardActions;
    internal Action<IEngine>[] BackwardActions => _backwardActions;
    internal Tensor<T> LossOutput => _lossOutput;
    internal Tensor<T>[] Parameters => _parameters;
    internal Tensor<T>[] PreAllocatedGrads => _preAllocatedGrads;
    internal CompiledStep<T>[]? ForwardStepsForSerialization => _forwardSteps;
    internal int[]? SerializedInputShape => _compiledInputShape;
    internal Tensor<T>? SerializedInputTensor => _compiledInputTensor;

    internal static CompiledTrainingPlan<T> Compile(
        LazyTensorScope scope, IEngine engine, Tensor<T>[] parameters, Tensor<T>? explicitLoss)
    {
        var compiler = new LazyGraphCompiler();
        var optimized = compiler.Compile(scope.Nodes);

        // Collect all forward steps
        var forwardSteps = new List<CompiledStep<T>>();
        foreach (var node in optimized)
        {
            if (node is LazyNode<T> typed)
            {
                forwardSteps.Add(new CompiledStep<T>(
                    typed.OpName,
                    typed.Execute,
                    typed.Output,
                    typed.GetInputsArray(),
                    typed.BackwardFn,
                    typed.SavedState));
            }
        }

        // Map each tensor to its consumer count (how many backward steps write to it)
        var consumerCount = new Dictionary<Tensor<T>, int>();
        foreach (var step in forwardSteps)
        {
            foreach (var inp in step.Inputs)
            {
                if (consumerCount.ContainsKey(inp))
                    consumerCount[inp]++;
                else
                    consumerCount[inp] = 1;
            }
        }

        // Pre-allocate gradient buffers for all tensors
        var gradMap = new Dictionary<Tensor<T>, Tensor<T>>();
        var allGrads = new List<Tensor<T>>();
        var allTensors = new HashSet<Tensor<T>>();
        foreach (var step in forwardSteps)
        {
            allTensors.Add(step.OutputBuffer);
            foreach (var inp in step.Inputs) allTensors.Add(inp);
        }
        foreach (var p in parameters) allTensors.Add(p);

        foreach (var tensor in allTensors)
        {
            var grad = TensorAllocator.RentUninitialized<T>(tensor._shape);
            gradMap[tensor] = grad;
            allGrads.Add(grad);
        }

        // Phase B integration: detect MatMul→ReLU→MatMul patterns and replace with fused kernel
        var fusedForwardActions = new List<Action<IEngine>>();
        var fusedBackwardActions = new List<Action<IEngine>>();
        var fusedStepIndices = new HashSet<int>(); // indices consumed by fusion

        if (typeof(T) == typeof(float) && Optimization.TensorCodecOptions.Current.EnableDataflowFusion)
        {
            TryFuseForwardBackward(forwardSteps, gradMap, consumerCount, engine,
                fusedForwardActions, fusedBackwardActions, fusedStepIndices);
        }

        // Issue #338 Phase G.7: detect "MatMul → ReduceSum-all → loss" pattern
        // where the MatMul's output is consumed ONLY by the ReduceSum, and the
        // ReduceSum is the final scalar loss. In this case the gradient
        // flowing back into the MatMul output is mathematically `α * ones`
        // (α = the loss-grad scalar, defaults to 1). The MatMul backward can
        // then be computed ANALYTICALLY without materializing the M*N ones
        // tensor or running the M*N*K GEMM:
        //   dInput[m, k] = α * row_sum(W, k)   — same value across all m
        //   dW[k, v]     = α * col_sum(input, k) — same value across all v
        // For Issue #327 d=128 with V=8192, this replaces the 2.1B-MAC LM-head
        // backward (2 GEMMs at ~30 ms total) with a couple of [D]-sized
        // reductions + broadcasts (~0.2 ms total).
        var analyticBackwardSpecs = new Dictionary<int, Action<IEngine>>();
        if (typeof(T) == typeof(float))
        {
            DetectAnalyticLossMatMulBackward(forwardSteps, consumerCount, gradMap, analyticBackwardSpecs);
        }

        // Track GCHandles for cleanup on Dispose
        var pinnedHandles = new List<GCHandle>();

        // Build forward actions in original graph order. Fused groups replace their
        // constituent steps at the position of the first fused step in each group,
        // ensuring non-fused producers that appear before a fused block still run first.
        var allForwardActions = new List<Action<IEngine>>();
        int nextFusedGroupIdx = 0; // index into fusedForwardActions
        for (int i = 0; i < forwardSteps.Count; i++)
        {
            if (fusedStepIndices.Contains(i))
            {
                // A fused step starts a new group when its predecessor is NOT fused.
                // This correctly handles consecutive groups (e.g., {0,1,2, 3,4,5}).
                bool isFirstInGroup = i == 0 || !fusedStepIndices.Contains(i - 1);
                if (isFirstInGroup && nextFusedGroupIdx < fusedForwardActions.Count)
                {
                    allForwardActions.Add(fusedForwardActions[nextFusedGroupIdx]);
                    nextFusedGroupIdx++;
                }
                continue;
            }
            var step = forwardSteps[i];
            // Training plans mutate parameters in-place between Step() calls,
            // so the SgemmWithCachedB pre-pack cache (keyed on B's array
            // reference) would serve stale weights. Disable the cached path
            // for training specialization; correctness over cache-hit speed.
            var specialized = TryBuildSpecializedForward(step, pinnedHandles, allowCachedB: false);
            if (specialized != null)
            {
                allForwardActions.Add(specialized);
            }
            else
            {
                var output = step.OutputBuffer;
                var exec = step.Execute;
                allForwardActions.Add(eng => exec(eng, output));
            }
        }
        var forwardActions = allForwardActions.ToArray();

        // Build backward actions: specialized per-step + fused backward for fused groups
        var backwardActions = new List<Action<IEngine>>();
        int genericBackwardCount = 0;
        for (int i = forwardSteps.Count - 1; i >= 0; i--)
        {
            if (fusedStepIndices.Contains(i)) continue;
            var step = forwardSteps[i];
            if (step.BackwardFn == null) continue;

            // Phase G.7: analytic loss-MatMul backward (replaces standard
            // spec for MatMuls whose gradOut is `α * ones` due to a
            // downstream ReduceSum-all-to-loss). Replaces ~30 ms of
            // 2.1B-MAC GEMM work with ~0.2 ms of broadcast.
            if (analyticBackwardSpecs.TryGetValue(i, out var analyticAction))
            {
                backwardActions.Add(analyticAction);
                continue;
            }

            var action = BuildSpecializedBackward(step, gradMap, consumerCount, engine, pinnedHandles);
            if (action != null)
                backwardActions.Add(action);
            else
            {
                genericBackwardCount++;
                var stepCopy = step;
                var gradAcc = gradMap;
                backwardActions.Add(eng =>
                {
                    var gradOut = gradAcc.ContainsKey(stepCopy.OutputBuffer)
                        ? gradAcc[stepCopy.OutputBuffer]
                        : gradAcc.Values.First();
                    stepCopy.BackwardFn(gradOut, stepCopy.Inputs, stepCopy.OutputBuffer,
                        stepCopy.SavedState ?? Array.Empty<object>(), eng, gradAcc);
                });
            }
        }
        // Append fused backward actions in reverse order — TryFuseForwardBackward records
        // them in forward order, but autodiff requires backward (reverse) order.
        for (int i = fusedBackwardActions.Count - 1; i >= 0; i--)
            backwardActions.Add(fusedBackwardActions[i]);

        // Loss gradient seed
        var numOps = MathHelper.GetNumericOperations<T>();
        // Prefer the caller's returned loss tensor. The last-step heuristic
        // picks the wrong tensor whenever the forward+loss lambda ends in a
        // pure-view op (e.g. scalarize-via-Reshape) — same issue as
        // inference #228.
        Tensor<T> lossOutput;
        if (explicitLoss is not null)
        {
            explicitLoss.LazySource = null;
            lossOutput = explicitLoss;
        }
        else
        {
            lossOutput = forwardSteps.Count > 0
                ? forwardSteps[forwardSteps.Count - 1].OutputBuffer
                : new Tensor<T>(new int[] { 1 });
        }
        var lossGradSeed = TensorAllocator.RentUninitialized<T>(lossOutput._shape);
        lossGradSeed.AsWritableSpan().Fill(numOps.One);

        // Gradients array for parameters
        var gradients = new Tensor<T>[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            if (gradMap.ContainsKey(parameters[i]))
                gradients[i] = gradMap[parameters[i]];
        }

        // Phase 6.3: Backward pruning — skip gradient computation for non-trainable tensors
        var paramSet = new HashSet<Tensor<T>>(parameters);
        backwardActions = BackwardPruningPass.Prune(backwardActions, forwardSteps, parameters, gradMap);

        // If all backward steps are specialized (overwrite), we can skip gradient zeroing entirely
        int[]? genericGradIndices = genericBackwardCount == 0 ? new int[0] : null;

        // Phase 4.4: Wire pre-packed weights for MatMul forward steps
        if (typeof(T) == typeof(float))
        {
            var packedWeights = WeightLayoutOptimizer.PrePackWeights(allForwardActions
                .Select((a, idx) => idx < forwardSteps.Count && !fusedStepIndices.Contains(idx) ? forwardSteps[idx] : null)
                .Where(s => s is not null)
                .ToArray()!);
            // packedWeights are available for future SIMD tile kernels that consume panel format
        }

        // Phase 4.4: Fused optimizer — append SGD/Adam parameter update directly to backward actions.
        // This is optional and controlled by the caller via FusedOptimizer.AppendFusedUpdates().
        // The default compiled plan returns gradients for the caller to update parameters manually.
        // When a caller (e.g., CompiledTapeTrainingStep) wants fused updates, it calls:
        //   FusedOptimizer.AppendFusedUpdates(plan.BackwardActions, params, grads, lr)
        // This avoids hard-coding the learning rate into the compiled plan.

        // Determine compiled input shape/tensor from the forward steps.
        Tensor<T>? compiledInputTensor = forwardSteps.Count > 0 && forwardSteps[0].Inputs.Length > 0
            ? forwardSteps[0].Inputs[0] : null;
        int[] compiledInputShape = compiledInputTensor is not null
            ? (int[])compiledInputTensor._shape.Clone() : Array.Empty<int>();

        return new CompiledTrainingPlan<T>(
            forwardActions,
            backwardActions.ToArray(),
            lossOutput,
            engine,
            parameters,
            allGrads.ToArray(),
            gradients,
            lossGradSeed,
            genericGradIndices,
            gradMap.ContainsKey(lossOutput) ? gradMap[lossOutput] : null,
            pinnedHandles,
            forwardSteps.ToArray(),
            compiledInputShape,
            compiledInputTensor,
            fusedStepIndices.Count > 0 ? fusedStepIndices.ToArray() : null,
            fusedForwardActions.Count > 0 ? fusedForwardActions.ToArray() : null);
    }

    /// <summary>
    /// Generates a specialized forward delegate that bypasses engine dispatch overhead.
    /// Calls BLAS/SIMD directly into the pre-allocated output buffer.
    /// Eliminates: GraphMode check, tape recording, shape validation, DifferentiableOps.
    /// </summary>
    /// <summary>Pin an array and track the handle for later cleanup.</summary>
    private static GCHandle PinAndTrack(object array, List<GCHandle>? tracker)
    {
        var handle = GCHandle.Alloc(array, GCHandleType.Pinned);
        tracker?.Add(handle);
        return handle;
    }

    /// <summary>
    /// Build a parallel-chunked Action that dispatches a pointer-based binary
    /// kernel (float* pA, pB, pR, int count) across ≈64 KB chunks. Used by
    /// TensorAdd / TensorSubtract / TensorMultiply specialized forwards.
    /// Single-core VectorXxxUnsafe peaks at ~4 GB/s DRAM; parallel chunks
    /// saturate at ~12 GB/s on Zen 2. For length &lt; 32K elements (&lt;2 chunks)
    /// the parallel dispatch overhead exceeds the savings — falls through to
    /// a direct serial call.
    /// </summary>
    private static unsafe Action<IEngine> BuildParallelBinaryKernel(
        GCHandle aH, GCHandle bH, GCHandle oH, int len,
        PointerBinaryKernel kernel)
    {
        const int kElemsPerChunk = 16 * 1024; // 64 KB at float32
        int chunks = System.Math.Min(
            Helpers.CpuParallelSettings.MaxDegreeOfParallelism,
            System.Math.Max(1, len / kElemsPerChunk));
        if (chunks >= 2)
        {
            int chunkSize = (len + chunks - 1) / chunks;
            chunkSize = (chunkSize + 31) & ~31;
            int chunksCap = chunks;
            int lenCap = len;
            return eng =>
            {
                unsafe
                {
                    float* pA = (float*)aH.AddrOfPinnedObject();
                    float* pB = (float*)bH.AddrOfPinnedObject();
                    float* pR = (float*)oH.AddrOfPinnedObject();
                    Helpers.PersistentParallelExecutor.Instance.Execute(chunksCap, chunk =>
                    {
                        int start = chunk * chunkSize;
                        int count = System.Math.Min(chunkSize, lenCap - start);
                        if (count > 0)
                            kernel(pA + start, pB + start, pR + start, count);
                    });
                }
            };
        }
        return eng =>
        {
            unsafe
            {
                kernel(
                    (float*)aH.AddrOfPinnedObject(),
                    (float*)bH.AddrOfPinnedObject(),
                    (float*)oH.AddrOfPinnedObject(),
                    len);
            }
        };
    }

    private unsafe delegate void PointerBinaryKernel(float* a, float* b, float* r, int count);

    internal static unsafe Action<IEngine>? TryBuildSpecializedForward(
        CompiledStep<T> step,
        List<GCHandle>? handleTracker = null,
        bool allowCachedB = true)
    {
        // Per-branch type checks below — each `typeof(T) == typeof(float)` /
        // `typeof(T) == typeof(double)` arm decides whether that op has a
        // specialized path for the current numeric type. Branches without a
        // double specialization simply fall through to the generic engine
        // path at the bottom of the caller. PR #319: extend the MatMul
        // branch to cover double via BlasProvider.TryGemm(double[], ...) +
        // SimdGemm.Dgemm fallback so tape-trained Tensor<double> models
        // (the consumer ViT-Base in PR #1224) hit the compiled-replay fast
        // path on their hottest op.
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double)) return null;

        // MatMul forward (double): direct BLAS into output buffer.
        // Mirrors the float branch below but with double[] arrays and
        // BlasProvider's double Dgemm overload (cblas_dgemm_ptr → OpenBLAS
        // when AIDOTNET_USE_BLAS != 0). Falls through to SimdGemm.Dgemm
        // when BLAS isn't loadable.
        if (typeof(T) == typeof(double)
            && step.OpType == OpType.TensorMatMul && step.Inputs.Length == 2
            && step.Inputs[0].Rank == 2 && step.Inputs[1].Rank == 2)
        {
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;
            int M = inputA._shape[0], K = inputA._shape[1], N = inputB._shape[1];

            // Pre-fetch arrays at compile time (bypasses EnsureMaterialized at replay)
            var cA = (double[])(object)inputA.GetDataArray();
            var cB = (double[])(object)inputB.GetDataArray();
            var cOut = (double[])(object)output.GetDataArray();

            // Note: no Path-A pre-pack cache for double yet (SimdGemm has no
            // DgemmWithCachedB), so the inference vs training distinction
            // collapses to a single replay form: try BLAS, fall through to
            // SimdGemm.Dgemm. Pre-pack-B for double is a future enhancement.
            return eng =>
            {
                if (!BlasProvider.TryGemm(M, N, K, cA, 0, K, cB, 0, N, cOut, 0, N))
                    SimdGemm.Dgemm(cA.AsSpan(0, M * K), cB.AsSpan(0, K * N), cOut.AsSpan(0, M * N), M, K, N);
            };
        }

        // MatMul forward (ND × 2D): collapse A's leading dims into M and
        // run a single 2D SGEMM directly into the pre-allocated output
        // buffer. Critical for the consumer Transformer's rank-3
        // [B,S,D] × rank-2 [D,N] forward pattern (issue #327) — without
        // this, the rank-3 × rank-2 fallback below recurses through
        // engine.TensorMatMul's full dispatcher chain and dispatches
        // through TensorMatMulBatched, costing the per-step plan flat-
        // delegate-array benefit.
        if (step.OpType == OpType.TensorMatMul && step.Inputs.Length == 2
            && step.Inputs[0].Rank >= 2 && step.Inputs[1].Rank == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && step.OutputBuffer.IsContiguous
            && typeof(T) == typeof(float))
        {
            // Contiguity guard: collapsing A's leading dims to M and treating
            // the buffer as row-major dense produces wrong results for views
            // (e.g. TensorSlice). Non-contiguous operands fall through to
            // the generic engine path below.
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;
            int aRank = inputA.Rank;
            int M = 1;
            for (int i = 0; i < aRank - 1; i++) M *= inputA._shape[i];
            int K = inputA._shape[aRank - 1];
            int N = inputB._shape[1];

            // Pre-fetch arrays at compile time (bypasses EnsureMaterialized at replay)
            var cA = (float[])(object)inputA.GetDataArray();
            var cB = (float[])(object)inputB.GetDataArray();
            var cOut = (float[])(object)output.GetDataArray();

            if (N == 1)
            {
                return eng => TensorMatMulGemvFloat(cA, cB, cOut, M, K);
            }

            // Issue #338 Phase G.5: BF16 mixed-precision forward MatMul.
            // When AIDOTNET_BLAS_PROVIDER=mkl-bf16 is opted in AND the MKL
            // BF16 kernel is available, pre-convert weight B to BF16 at
            // compile time (one-shot, weight is constant across replays in
            // frozen-weights mode and converted-fresh each step in training
            // mode via the closure). Activations A are converted on the
            // fly into a pooled BF16 buffer. C remains FP32.
            //
            // Shape gate: skip BF16 when N > 1024 (LM head case where
            // dC conversion cost in backward dominates the per-call kernel
            // savings on CPUs without AVX-512-BF16 hardware).
            if (BlasProvider.UseMklBf16 && N <= 1024)
            {
                // Pre-convert B at compile time. NOTE: this assumes B
                // doesn't change between Step() calls; for the standard
                // training loop (weight updates between steps) the BF16
                // mode would need re-conversion. We gate the entire
                // mkl-bf16 path on the assumption that the caller knows
                // their workload — same trust contract as
                // EnableFrozenWeightOptimizations().
                var bBf16 = new ushort[K * N];
                unsafe
                {
                    fixed (float* bSrc = cB)
                    fixed (ushort* bDst = bBf16)
                        BlasProvider.Fp32ToBf16Bulk(bSrc, bDst, K * N);
                }
                // Reusable BF16 scratch for A. Sized once; reused per call.
                var aBf16 = new ushort[M * K];
                return eng =>
                {
                    unsafe
                    {
                        fixed (float* aSrc = cA)
                        fixed (ushort* aDst = aBf16)
                            BlasProvider.Fp32ToBf16Bulk(aSrc, aDst, M * K);
                    }
                    if (!BlasProvider.TryGemmBf16(M, N, K, aBf16, 0, K, false, bBf16, 0, N, false, cOut, 0, N))
                    {
                        // Probe succeeded but call failed (transient) — fall back to FP32.
                        if (!BlasProvider.TryGemm(M, N, K, cA, 0, K, cB, 0, N, cOut, 0, N))
                            SimdGemm.Sgemm(cA.AsSpan(0, M * K), cB.AsSpan(0, K * N), cOut.AsSpan(0, M * N), M, K, N);
                    }
                };
            }

            if (allowCachedB)
            {
                return eng =>
                {
                    if (!BlasProvider.TryGemm(M, N, K, cA, 0, K, cB, 0, N, cOut, 0, N))
                        // Path A pre-pack cache — B (weight) is constant across
                        // inference calls, so SgemmWithCachedB amortises PackB
                        // cost. Falls through to standard Sgemm for non-cache-
                        // eligible shapes (n > Nc=4096).
                        SimdGemm.SgemmWithCachedB(cA.AsSpan(0, M * K), cB, cOut.AsSpan(0, M * N), M, K, N);
                };
            }

            // Training plan path: parameters (B) are updated in-place between
            // Step() calls, so the pre-packed-B cache would serve stale weights.
            // Skip SgemmWithCachedB and re-pack on every call (measurable cost
            // during training, but correctness-first). Fixes
            // CompiledTrainingPlanRebindingTests.Step_SeesInPlaceParameterUpdates
            // and Step_ViaCompiledModelCache_SeesUpdates.
            return eng =>
            {
                if (!BlasProvider.TryGemm(M, N, K, cA, 0, K, cB, 0, N, cOut, 0, N))
                    SimdGemm.Sgemm(cA.AsSpan(0, M * K), cB.AsSpan(0, K * N), cOut.AsSpan(0, M * N), M, K, N);
            };
        }

        // ReLU forward: direct SIMD into output buffer (float-only fast path).
        // Issue #340: this branch previously fired for any T because the
        // outer gate omitted `typeof(T) == typeof(float)`. T=double tensors
        // would then crash inside the closure on `(Tensor<float>)(object)input`
        // — InvalidCastException at plan.Step() during VGGNetwork<double>
        // training. Gate the SIMD path and add an eager engine fallback for
        // non-float T so the fused plan still applies the op without falling
        // back to eager retraining.
        if (step.OpType == OpType.ReLU && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;

            return eng =>
            {
                var iMem = ((Tensor<float>)(object)input).Data;
                var oMem = ((Tensor<float>)(object)output).Data;
                using var pinI = iMem.Pin();
                using var pinO = oMem.Pin();
                SimdKernels.ReLUUnsafe((float*)pinI.Pointer, (float*)pinO.Pointer, input.Length);
            };
        }
        // ReLU forward non-float fallback (T=double etc.): route through the
        // engine's ReLUInto so the fused plan still owns the op without
        // requiring the float-only SIMD kernel.
        if (step.OpType == OpType.ReLU && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            return eng => eng.ReLUInto(output, input);
        }

        // ReduceSum forward: direct sum, skip engine dispatch
        if (step.OpType == OpType.ReduceSum && step.Inputs.Length == 1 && step.OutputBuffer.Length == 1)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            float[]? cOut = null;

            if (typeof(T) == typeof(float) && input.IsContiguous)
            {
                // Pinned path: GCHandle once at compile time
                var inH = PinAndTrack(
                    ((Tensor<float>)(object)input).GetDataArray(), handleTracker);
                int len = input.Length;

                // For large arrays, use parallel chunked reduction (matches TensorSum)
                if (len >= 200_000)
                {
                    int numChunks = Math.Max(2, len / 50_000);
                    int chunkSize = ((len + numChunks - 1) / numChunks + 31) & ~31;
                    var partials = new float[numChunks];
                    return eng =>
                    {
                        cOut ??= (float[])(object)output.GetDataArray();
                        unsafe
                        {
                            float* p = (float*)inH.AddrOfPinnedObject();
                            CpuParallelSettings.ParallelForOrSerial(0, numChunks, len, chunk =>
                            {
                                int start = chunk * chunkSize;
                                int count = Math.Min(chunkSize, len - start);
                                if (count > 0) partials[chunk] = SimdKernels.SumUnsafe(p + start, count);
                                else partials[chunk] = 0f;
                            });
                            float total = 0f;
                            for (int c = 0; c < numChunks; c++) total += partials[c];
                            cOut[0] = total;
                        }
                    };
                }

                return eng =>
                {
                    cOut ??= (float[])(object)output.GetDataArray();
                    unsafe { cOut[0] = SimdKernels.SumUnsafe((float*)inH.AddrOfPinnedObject(), len); }
                };
            }
            // Non-float ReduceSum fallback. Issue #340: the prior version
            // unconditionally did `cOut ??= (float[])(object)output.GetDataArray()`
            // even on the non-float path — T=double would have crashed
            // here on first replay. Store the engine's T sum directly into
            // the output tensor via SetFlat so the generic numeric path
            // works for any T.
            return eng =>
            {
                T sum = eng.TensorSum(input);
                output.SetFlat(0, sum);
            };
        }

        // Concat forward: direct Buffer.BlockCopy for axis=0 contiguous tensors
        if ((step.OpName == "Concatenate" || step.OpName == "Concat")
            && step.Inputs.Length >= 2 && typeof(T) == typeof(float))
        {
            // Check all inputs are contiguous
            bool allContiguous = true;
            for (int j = 0; j < step.Inputs.Length; j++)
                if (!step.Inputs[j].IsContiguous) { allContiguous = false; break; }

            if (allContiguous)
            {
                // Check if axis=0 (SavedState[0] should be the axis)
                int axis = 0;
                if (step.SavedState is { Length: > 0 } && step.SavedState[0] is int a)
                    axis = a;

                if (axis == 0)
                {
                    // Pre-pin all input arrays + output array
                    var inputArrays = new float[step.Inputs.Length][];
                    var inputLengths = new int[step.Inputs.Length];
                    for (int j = 0; j < step.Inputs.Length; j++)
                    {
                        inputArrays[j] = (float[])(object)step.Inputs[j].GetDataArray();
                        inputLengths[j] = step.Inputs[j].Length;
                    }
                    var outArr = (float[])(object)step.OutputBuffer.GetDataArray();
                    var capturedInputArrays = inputArrays;
                    var capturedLengths = inputLengths;

                    return eng =>
                    {
                        int offset = 0;
                        for (int j = 0; j < capturedInputArrays.Length; j++)
                        {
                            Buffer.BlockCopy(capturedInputArrays[j], 0, outArr, offset * 4, capturedLengths[j] * 4);
                            offset += capturedLengths[j];
                        }
                    };
                }
            }
        }

        // ReduceMax forward: pinned SIMD for single-output max
        if (step.OpName == "ReduceMax" && step.Inputs.Length == 1
            && step.OutputBuffer.Length == 1 && typeof(T) == typeof(float)
            && step.Inputs[0].IsContiguous)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            var inH = PinAndTrack(((Tensor<float>)(object)input).GetDataArray(), handleTracker);
            int len = input.Length;
            float[]? cOut = null;

            return eng =>
            {
                cOut ??= (float[])(object)output.GetDataArray();
                unsafe
                {
                    float* p = (float*)inH.AddrOfPinnedObject();
                    if (len == 0) { cOut[0] = float.MinValue; return; }
                    float maxVal = p[0];
                    for (int j = 1; j < len; j++)
                        if (p[j] > maxVal) maxVal = p[j];
                    cOut[0] = maxVal;
                }
            };
        }

        // TensorAdd forward: pinned SIMD VectorAddUnsafe with chunked parallel
        // dispatch for bandwidth-bound shapes. Single-core VectorAddUnsafe
        // peaks at ~4 GB/s; a 4-chunk parallel split saturates DRAM at
        // ~12 GB/s, cutting 786 KB adds (BERT residual) from ~200 µs to
        // ~70 µs.
        if (step.OpType == OpType.TensorAdd && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && typeof(T) == typeof(float))
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            var aH = PinAndTrack(((Tensor<float>)(object)a).GetDataArray(), handleTracker);
            var bH = PinAndTrack(((Tensor<float>)(object)b).GetDataArray(), handleTracker);
            var oH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = a.Length;
            return BuildParallelBinaryKernel(aH, bH, oH, len,
                (pA, pB, pR, count) => { unsafe { SimdKernels.VectorAddUnsafe(pA, pB, pR, count); } });
        }
        if (step.OpType == OpType.TensorAdd && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous)
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            return eng => eng.TensorAddInto(o, a, b);
        }

        // TensorSubtract forward: pinned SIMD VectorSubtractUnsafe
        if (step.OpType == OpType.TensorSubtract && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && typeof(T) == typeof(float))
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            var aH = PinAndTrack(((Tensor<float>)(object)a).GetDataArray(), handleTracker);
            var bH = PinAndTrack(((Tensor<float>)(object)b).GetDataArray(), handleTracker);
            var oH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = a.Length;
            return BuildParallelBinaryKernel(aH, bH, oH, len,
                (pA, pB, pR, count) => { unsafe { SimdKernels.VectorSubtractUnsafe(pA, pB, pR, count); } });
        }
        if (step.OpType == OpType.TensorSubtract && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous)
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            return eng => eng.TensorSubtractInto(o, a, b);
        }

        // TensorMultiply forward: pinned SIMD VectorMultiplyUnsafe with parallel chunking
        if (step.OpType == OpType.TensorMultiply && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && typeof(T) == typeof(float))
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            var aH = PinAndTrack(((Tensor<float>)(object)a).GetDataArray(), handleTracker);
            var bH = PinAndTrack(((Tensor<float>)(object)b).GetDataArray(), handleTracker);
            var oH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = a.Length;
            return BuildParallelBinaryKernel(aH, bH, oH, len,
                (pA, pB, pR, count) => { unsafe { SimdKernels.VectorMultiplyUnsafe(pA, pB, pR, count); } });
        }
        if (step.OpType == OpType.TensorMultiply && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous)
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            return eng => eng.TensorMultiplyInto(o, a, b);
        }

        // Sigmoid: don't specialize forward — the eager allocating path is faster
        // (SigmoidInto has auto-materialization overhead that exceeds the allocation savings)

        // Tanh forward: VML → SIMD fallback, pinned GCHandle
        if (step.OpType == OpType.Tanh && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe
                {
                    float* pIn = (float*)inH.AddrOfPinnedObject();
                    float* pOut = (float*)outH.AddrOfPinnedObject();
                    // Try MKL VML vsTanh first (SVML, ~3x faster than polynomial)
                    if (!Helpers.VmlProvider.TryTanh(pIn, pOut, len))
                        SimdKernels.TanhUnsafe(pIn, pOut, len);
                }
            };
        }
        // Tanh non-float fallback
        if (step.OpType == OpType.Tanh && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => eng.TanhInto(o, inp);
        }

        // Softmax forward: use SoftmaxInto
        if (step.OpType == OpType.Softmax && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            int axis = step.SavedState != null && step.SavedState.Length > 0 ? (int)step.SavedState[0] : -1;
            return eng => eng.SoftmaxInto(o, inp, axis);
        }

        // TensorNegate forward: pinned SIMD NegateUnsafe
        if (step.OpType == OpType.TensorNegate && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng => { unsafe { SimdKernels.NegateUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); } };
        }
        // TensorNegate non-float fallback
        if (step.OpType == OpType.TensorNegate && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng =>
            {
                MathHelper.GetNumericOperations<T>().Negate(inp.AsSpan(), o.AsWritableSpan());
            };
        }

        // Sigmoid forward: direct Padé [3,3] — bypass SigmoidUnsafe dispatch overhead
        if (step.OpType == OpType.Sigmoid && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { PadeSigmoid.SigmoidArray((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); }
            };
        }
        // Sigmoid non-float fallback
        if (step.OpType == OpType.Sigmoid && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => eng.SigmoidInto(o, inp);
        }

        // GELU forward: pinned SIMD — cache GCHandles at compile time for zero-overhead replay
        if (step.OpType == OpType.GELU && step.Inputs.Length == 1 && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            // Pin arrays once at compile time — GCHandles survive across replays
            var inHandle = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outHandle = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe
                {
                    float* pIn = (float*)inHandle.AddrOfPinnedObject();
                    float* pOut = (float*)outHandle.AddrOfPinnedObject();
                    SimdKernels.FusedGELUUnsafe(pIn, pOut, len);
                }
            };
        }
        // FusedLinear forward: direct BLAS + bias + activation into output buffer
        if (step.OpType == OpType.FusedLinear && step.Inputs.Length == 3 && typeof(T) == typeof(float)
            && step.Inputs[0].Rank == 2 && step.Inputs[1].Rank == 2)
        {
            var input = step.Inputs[0]; var weights = step.Inputs[1]; var bias = step.Inputs[2];
            var o = step.OutputBuffer;
            var activation = step.SavedState != null && step.SavedState.Length > 0 && step.SavedState[0] is FusedActivationType act
                ? act : FusedActivationType.None;
            int M = input._shape[0], K = input._shape[1], N = weights._shape[1];

            // Pre-fetch arrays at compile time — bypass EnsureMaterialized at replay
            var inArr = (float[])(object)input.GetDataArray();
            var wArr = (float[])(object)weights.GetDataArray();
            var bArr = (float[])(object)bias.GetDataArray();
            var oArr = (float[])(object)o.GetDataArray();

            return eng =>
            {
                // Shapes were validated when the graph was compiled; replay skips
                // public API argument checks and goes straight to the hot kernel.
                //
                // allowCachedB: false because optimizer.Step() mutates wArr in
                // place between forward calls. The pre-packed B cache keys on
                // the array's identity, so the cached panels would be stale on
                // every step after the first.
                CpuFusedOperations.FusedGemmBiasActivationUnchecked(
                    inArr, wArr, bArr, oArr, M, N, K, activation, allowCachedB: false);
            };
        }

        // GELU non-float fallback
        if (step.OpType == OpType.GELU && step.Inputs.Length == 1)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => eng.GELUInto(o, inp);
        }

        // Abs forward: direct SIMD AbsUnsafe
        if (step.OpType == OpType.TensorAbs && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng =>
            {
                unsafe
                {
                    var srcMem = ((Tensor<float>)(object)inp).Data;
                    var dstMem = ((Tensor<float>)(object)o).Data;
                    using var pinS = srcMem.Pin();
                    using var pinD = dstMem.Pin();
                    SimdKernels.AbsUnsafe((float*)pinS.Pointer, (float*)pinD.Pointer, inp.Length);
                }
            };
        }

        // Pow forward: SIMD x*x for exponent=2
        if (step.OpType == OpType.TensorPower && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            // Check if exponent is 2 (x^2 = x*x)
            float exp = step.SavedState != null && step.SavedState.Length > 0 ? Convert.ToSingle(step.SavedState[0]) : 0;
            if (exp == 2.0f)
            {
                return eng =>
                {
                    unsafe
                    {
                        var srcMem = ((Tensor<float>)(object)inp).Data;
                        var dstMem = ((Tensor<float>)(object)o).Data;
                        using var pinS = srcMem.Pin();
                        using var pinD = dstMem.Pin();
                        SimdKernels.VectorMultiplyUnsafe((float*)pinS.Pointer, (float*)pinS.Pointer, (float*)pinD.Pointer, inp.Length);
                    }
                };
            }
        }

        // LeakyReLU forward: direct SIMD
        if (step.OpType == OpType.LeakyReLU && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var alpha = step.SavedState != null && step.SavedState.Length > 0
                ? MathHelper.GetNumericOperations<T>().FromDouble((double)step.SavedState[0])
                : MathHelper.GetNumericOperations<T>().FromDouble(0.01);
            return eng => eng.LeakyReLUInto(o, inp, alpha);
        }

        // Swish forward: pinned SigmoidUnsafe + VectorMultiplyUnsafe — bypass EnsureMaterialized
        if (step.OpType == OpType.Swish && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe
                {
                    float* pIn = (float*)inH.AddrOfPinnedObject();
                    float* pOut = (float*)outH.AddrOfPinnedObject();
                    SimdKernels.SigmoidUnsafe(pIn, pOut, len);
                    SimdKernels.VectorMultiplyUnsafe(pIn, pOut, pOut, len);
                }
            };
        }
        // Swish non-float fallback
        if (step.OpType == OpType.Swish && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => { if (eng is CpuEngine cpu) cpu.SwishInto(o, inp); else { var r = eng.Swish(inp); r.AsSpan().CopyTo(o.AsWritableSpan()); } };
        }

        // ELU forward: pinned SIMD ELUUnsafe
        if (step.OpType == OpType.ELU && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            float alphaF = step.SavedState != null && step.SavedState.Length > 0 ? (float)(double)step.SavedState[0] : 1.0f;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.ELUUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len, alphaF); }
            };
        }
        // ELU non-float fallback
        if (step.OpType == OpType.ELU && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            double alpha = step.SavedState != null && step.SavedState.Length > 0 ? (double)step.SavedState[0] : 1.0;
            return eng => { if (eng is CpuEngine cpu) cpu.ELUInto(o, inp, alpha); else { var r = eng.ELU(inp, alpha); r.AsSpan().CopyTo(o.AsWritableSpan()); } };
        }

        // Log forward: pinned LogUnsafe — bypass EnsureMaterialized
        if (step.OpType == OpType.TensorLog && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.LogUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); }
            };
        }
        // Log non-float fallback
        if (step.OpType == OpType.TensorLog && step.Inputs.Length == 1)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => { if (eng is CpuEngine cpu) cpu.TensorLogInto(o, inp); else { var r = eng.TensorLog(inp); r.AsSpan().CopyTo(o.AsWritableSpan()); } };
        }

        // Exp forward: VML → SIMD fallback, pinned GCHandle
        if (step.OpType == OpType.TensorExp && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe
                {
                    float* pIn = (float*)inH.AddrOfPinnedObject();
                    float* pOut = (float*)outH.AddrOfPinnedObject();
                    if (!Helpers.VmlProvider.TryExp(pIn, pOut, len))
                        SimdKernels.ExpUnsafe(pIn, pOut, len);
                }
            };
        }
        // Exp non-float fallback
        if (step.OpType == OpType.TensorExp && step.Inputs.Length == 1)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => { if (eng is CpuEngine cpu) cpu.TensorExpInto(o, inp); else { var r = eng.TensorExp(inp); r.AsSpan().CopyTo(o.AsWritableSpan()); } };
        }

        // Mish forward: pinned MishUnsafe — bypass EnsureMaterialized
        if (step.OpType == OpType.Mish && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.MishUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); }
            };
        }
        // Mish non-float fallback
        if (step.OpType == OpType.Mish && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => { if (eng is CpuEngine cpu) cpu.MishInto(o, inp); else { var r = eng.Mish(inp); r.AsSpan().CopyTo(o.AsWritableSpan()); } };
        }

        // BatchNorm inference: direct SIMD kernel (bypasses all allocation)
        // Lazy node records: Inputs = [input, gamma, beta], SavedState = [mean, variance, epsilon]
        if (step.OpType == OpType.BatchNorm && typeof(T) == typeof(float)
            && step.Inputs.Length >= 3 && step.Inputs[0].IsContiguous
            && step.SavedState is { Length: >= 2 }
            && step.SavedState[0] is Tensor<T> meanTensor
            && step.SavedState[1] is Tensor<T> varTensor)
        {
            var input = step.Inputs[0];
            var gamma = step.Inputs[1];
            var beta = step.Inputs[2];
            var mean = meanTensor;
            var variance = varTensor;
            var output = step.OutputBuffer;
            int channels = gamma.Length;
            float eps = 1e-5f;
            if (step.SavedState.Length >= 3 && step.SavedState[2] is double epsD)
                eps = (float)epsD;

            var inH = PinAndTrack(((Tensor<float>)(object)input).GetDataArray(), handleTracker);
            var outH = PinAndTrack(((Tensor<float>)(object)output).GetDataArray(), handleTracker);
            var gammaH = PinAndTrack(((Tensor<float>)(object)gamma).GetDataArray(), handleTracker);
            var betaH = PinAndTrack(((Tensor<float>)(object)beta).GetDataArray(), handleTracker);
            var meanH = PinAndTrack(((Tensor<float>)(object)mean).GetDataArray(), handleTracker);
            var varH = PinAndTrack(((Tensor<float>)(object)variance).GetDataArray(), handleTracker);
            int length = input.Length;
            float capturedEps = eps;

            return eng =>
            {
                unsafe
                {
                    Simd.FusedKernels.BatchNormInferenceUnsafe(
                        (float*)inH.AddrOfPinnedObject(),
                        (float*)outH.AddrOfPinnedObject(),
                        length, channels,
                        (float*)gammaH.AddrOfPinnedObject(),
                        (float*)betaH.AddrOfPinnedObject(),
                        (float*)meanH.AddrOfPinnedObject(),
                        (float*)varH.AddrOfPinnedObject(),
                        capturedEps);
                }
            };
        }

        // LayerNorm: use FusedKernels.LayerNormUnsafe for direct SIMD
        if (step.OpType == OpType.LayerNorm && typeof(T) == typeof(float)
            && step.Inputs.Length >= 3 && step.Inputs[0].IsContiguous)
        {
            var input = step.Inputs[0];
            var gamma = step.Inputs[1];
            var beta = step.Inputs[2];
            var output = step.OutputBuffer;
            int normSize = gamma.Length;
            float eps = 1e-5f;
            if (step.SavedState is { Length: > 0 } && step.SavedState[0] is double epsD)
                eps = (float)epsD;

            var inH = PinAndTrack(((Tensor<float>)(object)input).GetDataArray(), handleTracker);
            var outH = PinAndTrack(((Tensor<float>)(object)output).GetDataArray(), handleTracker);
            var gammaH = PinAndTrack(((Tensor<float>)(object)gamma).GetDataArray(), handleTracker);
            var betaH = PinAndTrack(((Tensor<float>)(object)beta).GetDataArray(), handleTracker);
            int batchSize = input.Length / normSize;
            float capturedEps = eps;

            return eng =>
            {
                // LayerNorm per batch element. Previously serial — that
                // capped BERT LayerNorm at ~900 µs on [1,256,768] despite
                // the underlying kernel running in ~120 µs when dispatched
                // in parallel. Threshold: parallelise if total work ≥50K
                // elements (matches CpuEngine.LayerNorm's own gate).
                int totalElems = batchSize * normSize;
                if (totalElems >= 50_000)
                {
                    CpuParallelSettings.ParallelForOrSerial(0, batchSize, totalElems, b =>
                    {
                        unsafe
                        {
                            float* pIn = (float*)inH.AddrOfPinnedObject();
                            float* pOut = (float*)outH.AddrOfPinnedObject();
                            float* pG = (float*)gammaH.AddrOfPinnedObject();
                            float* pB = (float*)betaH.AddrOfPinnedObject();
                            Simd.FusedKernels.LayerNormUnsafe(
                                pIn + b * normSize, pG, pB,
                                pOut + b * normSize, normSize, capturedEps);
                        }
                    });
                }
                else
                {
                    unsafe
                    {
                        float* pIn = (float*)inH.AddrOfPinnedObject();
                        float* pOut = (float*)outH.AddrOfPinnedObject();
                        float* pG = (float*)gammaH.AddrOfPinnedObject();
                        float* pB = (float*)betaH.AddrOfPinnedObject();
                        for (int b = 0; b < batchSize; b++)
                        {
                            Simd.FusedKernels.LayerNormUnsafe(
                                pIn + b * normSize, pG, pB,
                                pOut + b * normSize, normSize, capturedEps);
                        }
                    }
                }
            };
        }

        // TensorDivide forward: pinned SIMD VectorDivideUnsafe
        if (step.OpType == OpType.TensorDivide && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && typeof(T) == typeof(float))
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            var aH = PinAndTrack(((Tensor<float>)(object)a).GetDataArray(), handleTracker);
            var bH = PinAndTrack(((Tensor<float>)(object)b).GetDataArray(), handleTracker);
            var oH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = a.Length;
            return eng =>
            {
                unsafe { SimdKernels.VectorDivideUnsafe((float*)aH.AddrOfPinnedObject(), (float*)bH.AddrOfPinnedObject(), (float*)oH.AddrOfPinnedObject(), len); }
            };
        }

        // Conv2D forward: route through the int[] Conv2D overload (fast
        // path via Conv2DIm2colGemm → SimdGemm.Sgemm) and CopyTo the
        // pre-allocated output buffer.
        //
        // HISTORICAL BUG FIXED HERE: previously this used Conv2DInto
        // (scalar-param overload) which dispatches to Conv2DWithIm2ColFloat's
        // 5-strategy chain (OneDNN / FusedConvHelper / Winograd / SIMD-direct /
        // Conv2DWithIm2ColGemm). On ResNet-scale shapes (3×3 stride=1,
        // output H/W in {7, 14, 28, 56}) none of the first 4 strategies
        // fire — Winograd threshold is output ≥ 224, FusedConv threshold
        // is im2col > 16 MB, OneDNN unavailable by default, SIMD-direct
        // only handles tiny kernels — so ResNet convs fell to
        // Conv2DWithIm2ColGemm which is 20× slower than the int[]
        // overload's Conv2DIm2colGemm (the latter calls SimdGemm.Sgemm
        // directly with cache-optimal tile ordering).
        //
        // Measured impact on Conv2DRootCauseDiag (Ryzen 16-core, AVX2):
        //   stage3 [1,256,14,14] 3×3→256: 55 ms → 2.6 ms (21× faster)
        //   stage4 [1,512,7,7]   3×3→512: 102 ms → 12.8 ms (8× faster)
        //   stage2 [1,128,28,28] 3×3→128: 25 ms → 2.9 ms (8.6× faster)
        //
        // Preserves the specialization's bookkeeping (saved state / pinned
        // output) — we just change which Conv2D overload is called.
        if (step.OpType == OpType.Conv2D && step.Inputs.Length == 2)
        {
            var inp = step.Inputs[0]; var kernel = step.Inputs[1]; var o = step.OutputBuffer;
            var savedState = step.SavedState;
            // Be lenient about length: some call paths extend SavedState with
            // bias/layout hints past index 2. As long as the first three
            // entries are the int[] stride/padding/dilation triple we need,
            // honour the fast path.
            if (savedState != null && savedState.Length >= 3
                && savedState[0] is int[] stride && savedState[1] is int[] padding && savedState[2] is int[] dilation)
            {
                // Path C write-through: use Conv2DInto (int[]) directly so we
                // skip the intermediate-tensor allocation + CopyTo. Routes
                // through the same int[] dispatch as Conv2D but shortcuts
                // the Rent with the plan's pre-allocated output buffer.
                // Saves ~50 µs per ResNet Conv. Capture locals so the closure
                // holds onto its own refs without walking savedState every
                // Execute.
                var capStride = stride;
                var capPadding = padding;
                var capDilation = dilation;
                return eng =>
                {
                    if (eng is CpuEngine cpuEng)
                        cpuEng.Conv2DInto(o, inp, kernel, capStride, capPadding, capDilation);
                    else
                    {
                        var result = eng.Conv2D(inp, kernel, capStride, capPadding, capDilation);
                        result.AsSpan().CopyTo(o.AsWritableSpan());
                    }
                };
            }
            // Default stride/padding/dilation when savedState is absent —
            // the int[] overload requires arrays, so hoist constant arrays
            // out of the closure.
            var defStride = new[] { 1, 1 };
            var defPadding = new[] { 0, 0 };
            var defDilation = new[] { 1, 1 };
            return eng =>
            {
                if (eng is CpuEngine cpuEng)
                    cpuEng.Conv2DInto(o, inp, kernel, defStride, defPadding, defDilation);
                else
                {
                    var result = eng.Conv2D(inp, kernel, defStride, defPadding, defDilation);
                    result.AsSpan().CopyTo(o.AsWritableSpan());
                }
            };
        }

        // LogSoftmax forward: pinned SIMD with VML exp for inner loop
        if (step.OpType == OpType.LogSoftmax && step.Inputs.Length == 1 && typeof(T) == typeof(float)
            && step.Inputs[0].IsContiguous && step.Inputs[0].Rank == 2)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int rows = inp._shape[0], cols = inp._shape[1];
            return eng =>
            {
                unsafe
                {
                    float* pIn = (float*)inH.AddrOfPinnedObject();
                    float* pOut = (float*)outH.AddrOfPinnedObject();
                    for (int r = 0; r < rows; r++)
                        SimdKernels.FusedLogSoftmaxRow(pIn + r * cols, pOut + r * cols, cols);
                }
            };
        }

        // Mean forward: pinned SumUnsafe + divide.
        // Only applies to full-tensor reductions where the output is a single
        // scalar — the fast path writes one float to cOut[0]. Partial-axis
        // reductions (e.g. ReduceMean over [2,3] for GlobalAveragePool, which
        // keeps (N,C) slots) need the general path because the specialization
        // would silently collapse all outputs into cOut[0].
        // A/B tested: Parallel.For overhead (0.43ms) exceeds single-thread (0.16ms) for 1M.
        // PyTorch likely uses SIMD sum with wider parallelism (internal thread pool).
        if (step.OpType == OpType.Mean && step.Inputs.Length == 1 && typeof(T) == typeof(float)
            && step.OutputBuffer.Length == 1)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inHandle = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            float[]? cOut = null;
            int len = inp.Length;
            return eng =>
            {
                cOut ??= (float[])(object)o.GetDataArray();
                unsafe
                {
                    float* pIn = (float*)inHandle.AddrOfPinnedObject();
                    cOut[0] = SimdKernels.SumUnsafe(pIn, len) / len;
                }
            };
        }

        // Exp forward: pinned ExpUnsafe SIMD
        if (step.OpType == OpType.TensorExp && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.ExpUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); }
            };
        }

        // Sqrt forward: direct SIMD SqrtUnsafe into output buffer
        if (step.OpType == OpType.TensorSqrt && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => { if (eng is CpuEngine cpu) cpu.TensorSqrtInto(o, inp); else { var r = eng.TensorSqrt(inp); r.AsSpan().CopyTo(o.AsWritableSpan()); } };
        }

        // Sin forward: VML/SIMD via CpuEngine.TensorSinInto
        if (step.OpType == OpType.Sin && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => { if (eng is CpuEngine cpu) cpu.TensorSinInto(o, inp); else { var r = eng.TensorSin(inp); r.AsSpan().CopyTo(o.AsWritableSpan()); } };
        }

        // Cos forward: VML/SIMD via CpuEngine.TensorCosInto
        if (step.OpType == OpType.Cos && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => { if (eng is CpuEngine cpu) cpu.TensorCosInto(o, inp); else { var r = eng.TensorCos(inp); r.AsSpan().CopyTo(o.AsWritableSpan()); } };
        }

        // Softplus forward: SIMD SoftplusUnsafe with pinned arrays
        if (step.OpType == OpType.Softplus && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.SoftplusUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); }
            };
        }

        // HardSwish forward: SIMD HardSwishUnsafe with pinned arrays
        if (step.OpType == OpType.HardSwish && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.HardSwishUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); }
            };
        }

        // SELU forward: pinned SELUUnsafe SIMD
        if (step.OpType == OpType.SELU && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.SELUUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); }
            };
        }

        // HardSigmoid forward: pinned SIMD HardSigmoidUnsafe
        if (step.OpType == OpType.HardSigmoid && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.HardSigmoidUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); }
            };
        }

        // Sign forward: SIMD SignUnsafe with pinned arrays
        if (step.OpType == OpType.Sign && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.SignUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); }
            };
        }

        // Reciprocal forward: pinned SIMD
        if (step.OpType == OpType.Reciprocal && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng => { unsafe { SimdKernels.ReciprocalUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); } };
        }

        // Floor forward: pinned SIMD
        if (step.OpType == OpType.Floor && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng => { unsafe { SimdKernels.FloorUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); } };
        }

        // Ceiling forward: pinned SIMD
        if (step.OpType == OpType.Ceiling && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng => { unsafe { SimdKernels.CeilingUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); } };
        }

        // Round forward: pinned SIMD (AVX RoundToNearestInteger)
        if (step.OpType == OpType.Round && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var inH = PinAndTrack(((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng => { unsafe { SimdKernels.RoundUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len); } };
        }

        // TensorMax forward: pinned SIMD VectorMaxUnsafe
        if (step.OpType == OpType.TensorMax && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && typeof(T) == typeof(float))
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            var aH = PinAndTrack(
                ((Tensor<float>)(object)a).GetDataArray(), handleTracker);
            var bH = PinAndTrack(
                ((Tensor<float>)(object)b).GetDataArray(), handleTracker);
            var oH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = a.Length;
            return eng =>
            {
                unsafe { SimdKernels.VectorMaxUnsafe((float*)aH.AddrOfPinnedObject(), (float*)bH.AddrOfPinnedObject(), (float*)oH.AddrOfPinnedObject(), len); }
            };
        }

        // Clamp forward: pinned SIMD ClampUnsafe
        if (step.OpType == OpType.Clamp && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            float fMin = step.SavedState != null && step.SavedState.Length >= 2 ? Convert.ToSingle(step.SavedState[0]) : float.MinValue;
            float fMax = step.SavedState != null && step.SavedState.Length >= 2 ? Convert.ToSingle(step.SavedState[1]) : float.MaxValue;
            var inH = PinAndTrack(
                ((Tensor<float>)(object)inp).GetDataArray(), handleTracker);
            var outH = PinAndTrack(
                ((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = inp.Length;
            return eng =>
            {
                unsafe { SimdKernels.ClampUnsafe((float*)inH.AddrOfPinnedObject(), (float*)outH.AddrOfPinnedObject(), len, fMin, fMax); }
            };
        }

        // BroadcastAdd/Sub/Mul forward: direct array loop for [N,M] op [M] pattern
        if ((step.OpType == OpType.TensorBroadcastAdd || step.OpType == OpType.TensorBroadcastSubtract || step.OpType == OpType.TensorBroadcastMultiply)
            && step.Inputs.Length == 2 && typeof(T) == typeof(float)
            && step.Inputs[0].Rank == 2 && (step.Inputs[1].Rank == 1 || (step.Inputs[1].Rank == 2 && step.Inputs[1]._shape[0] == 1)))
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            int rows = a._shape[0], cols = a._shape[1];
            int bCols = b.Rank == 1 ? b._shape[0] : b._shape[1];
            if (cols == bCols)
            {
                var aArr = (float[])(object)a.GetDataArray();
                var bArr = (float[])(object)b.GetDataArray();
                var oArr = (float[])(object)o.GetDataArray();
                if (step.OpType == OpType.TensorBroadcastAdd)
                    return eng => { for (int r = 0; r < rows; r++) { int off = r * cols; for (int c = 0; c < cols; c++) oArr[off + c] = aArr[off + c] + bArr[c]; } };
                else if (step.OpType == OpType.TensorBroadcastSubtract)
                    return eng => { for (int r = 0; r < rows; r++) { int off = r * cols; for (int c = 0; c < cols; c++) oArr[off + c] = aArr[off + c] - bArr[c]; } };
                else
                    return eng => { for (int r = 0; r < rows; r++) { int off = r * cols; for (int c = 0; c < cols; c++) oArr[off + c] = aArr[off + c] * bArr[c]; } };
            }
        }

        // MSELoss forward: fused single-pass diff^2 sum
        if (step.OpType == OpType.MSELoss && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && typeof(T) == typeof(float))
        {
            var pred = step.Inputs[0]; var target = step.Inputs[1]; var o = step.OutputBuffer;
            var pArr = (float[])(object)pred.GetDataArray();
            var tArr = (float[])(object)target.GetDataArray();
            var oArr = (float[])(object)o.GetDataArray();
            int len = pred.Length;
            return eng =>
            {
                float sumSq = 0f;
                for (int i = 0; i < len; i++) { float d = pArr[i] - tArr[i]; sumSq += d * d; }
                oArr[0] = sumSq / len;
            };
        }

        // MaxPool2D: don't specialize (no Into variant, allocate+copy is slower)

        // Transpose: fall through to generic path.
        // Zero-copy transpose requires replacing OutputBuffer references in downstream steps,
        // which is not yet supported by the compiled step infrastructure.

        return null;
    }

    private static void TensorMatMulGemvFloat(float[] matrix, float[] vector, float[] output, int rows, int cols)
    {
        const int parallelThreshold = 128 * 1024;
        const int chunkSize = 256;
        if ((long)rows * cols < parallelThreshold)
        {
            for (int row = 0; row < rows; row++)
            {
                output[row] = TensorPrimitivesCore.Dot(
                    matrix.AsSpan(row * cols, cols),
                    vector.AsSpan(0, cols));
            }

            return;
        }

        int chunks = (rows + chunkSize - 1) / chunkSize;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)rows * cols, chunk =>
        {
            int start = chunk * chunkSize;
            int end = Math.Min(start + chunkSize, rows);
            for (int row = start; row < end; row++)
            {
                output[row] = TensorPrimitivesCore.Dot(
                    matrix.AsSpan(row * cols, cols),
                    vector.AsSpan(0, cols));
            }
        });
    }

    /// <summary>
    /// Generates a specialized backward delegate that writes directly into pre-allocated
    /// gradient buffers using transposed BLAS GEMM and SIMD kernels. No intermediate allocations.
    /// </summary>
    private static unsafe Action<IEngine>? BuildSpecializedBackward(
        CompiledStep<T> step,
        Dictionary<Tensor<T>, Tensor<T>> gradMap,
        Dictionary<Tensor<T>, int> consumerCount,
        IEngine engine,
        List<GCHandle>? handleTracker = null)
    {
        // Per-branch type checks below — same pattern as TryBuildSpecializedForward.
        // PR #319: extend the MatMul backward to cover double via
        // BlasProvider.TryGemmEx(double[], ...) so tape-trained
        // Tensor<double> models hit the compiled-replay fast path on the
        // dominant backward op (matmul backward = ~30% of typical
        // transformer train wall-clock).
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double)) return null;

        // MatMul backward (double): dA = dC @ B^T, dB = A^T @ dC — transposed BLAS, zero alloc.
        // Mirrors the float branch with cblas_dgemm via TryGemmEx's double overload;
        // engine fallback is the generic TensorMatMul which routes through SimdGemm.Dgemm.
        if (typeof(T) == typeof(double)
            && step.OpType == OpType.TensorMatMul && step.Inputs.Length == 2
            && step.Inputs[0].Rank == 2 && step.Inputs[1].Rank == 2
            && BlasProvider.IsAvailable)
        {
            var inputAd = step.Inputs[0];
            var inputBd = step.Inputs[1];
            var outputD = step.OutputBuffer;

            if (!gradMap.ContainsKey(outputD) || !gradMap.ContainsKey(inputAd) || !gradMap.ContainsKey(inputBd))
                return null;

            var gradOutD = gradMap[outputD];
            var gradAd = gradMap[inputAd];
            var gradBd = gradMap[inputBd];

            int Md = inputAd._shape[0], Kd = inputAd._shape[1], Nd = inputBd._shape[1];

            double[]? cdDC = null, cdA = null, cdB = null, cdDestA = null, cdDestB = null;

            return eng =>
            {
                cdDC ??= (double[])(object)gradOutD.GetDataArray();
                cdA ??= (double[])(object)inputAd.GetDataArray();
                cdB ??= (double[])(object)inputBd.GetDataArray();
                cdDestA ??= (double[])(object)gradAd.GetDataArray();
                cdDestB ??= (double[])(object)gradBd.GetDataArray();

                // dA = dC @ B^T — double-precision BLAS with engine fallback
                if (!BlasProvider.TryGemmEx(Md, Kd, Nd, cdDC, 0, Nd, false, cdB, 0, Nd, true, cdDestA, 0, Kd))
                {
                    var dA = eng.TensorMatMul(gradOutD, inputBd.Transpose());
                    dA.AsSpan().CopyTo(gradAd.AsWritableSpan());
                }
                // dB = A^T @ dC — double-precision BLAS with engine fallback
                if (!BlasProvider.TryGemmEx(Kd, Nd, Md, cdA, 0, Kd, true, cdDC, 0, Nd, false, cdDestB, 0, Nd))
                {
                    var dB = eng.TensorMatMul(inputAd.Transpose(), gradOutD);
                    dB.AsSpan().CopyTo(gradBd.AsWritableSpan());
                }

                inputAd.Grad = gradAd;
                inputBd.Grad = gradBd;
            };
        }

        // MatMul backward (ND × 2D): dA = dC @ B^T, dB = A^T @ dC. For
        // rank-3 [B,S,D] × rank-2 [D,N] input (consumer Transformer
        // pattern from issue #327), collapse A's leading dims into M
        // and run two 2D transposed GEMMs directly into the pre-allocated
        // gradient buffers. Falls back to engine.TensorMatMul+Transpose
        // when BLAS is unavailable.
        //
        // Contiguity guard: the BLAS path interprets the input buffers as
        // row-major dense matrices with leading dim K (for A) / N (for B,
        // dC). Strided views (non-unit-stride permutations, slices that
        // skip rows) violate that layout and would silently produce
        // wrong gradients if we forced them through TryGemmEx. The
        // generic engine fallback path handles arbitrary strides
        // correctly, so route non-contiguous inputs there instead of
        // through this fast path.
        // Issue #340: gate the BLAS-Sgemm fast path on typeof(T) == typeof(float).
        // The pinned-pointer block below casts every storage array to
        // float[]; T=double would crash with InvalidCastException at
        // first replay. (The matching double path is the BLAS-Dgemm
        // backward elsewhere; non-float / non-double types fall through
        // to the generic engine path.)
        if (step.OpType == OpType.TensorMatMul && step.Inputs.Length == 2
            && step.Inputs[0].Rank >= 2 && step.Inputs[1].Rank == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && step.OutputBuffer.IsContiguous
            && BlasProvider.IsAvailable
            && typeof(T) == typeof(float))
        {
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(inputA) || !gradMap.ContainsKey(inputB))
                return null;

            var gradOut = gradMap[output];
            var gradA = gradMap[inputA];
            var gradB = gradMap[inputB];

            // gradOut may be the producer of an upstream op's output that
            // is itself a view, and gradA / gradB are plan-owned but
            // could in principle be sliced into a larger arena buffer
            // (issue #327's compile-time arena pre-pack). Require
            // contiguity on every buffer that participates in TryGemmEx
            // so the row-major-with-stride-LD interpretation holds.
            if (!gradOut.IsContiguous || !gradA.IsContiguous || !gradB.IsContiguous)
                return null;

            int aRank = inputA.Rank;
            int M = 1;
            for (int i = 0; i < aRank - 1; i++) M *= inputA._shape[i];
            int K = inputA._shape[aRank - 1];
            int N = inputB._shape[1];

            // Cache the underlying storage arrays + storage offsets on first
            // call. We deliberately use `_storage.GetDataArray()` rather than
            // `GetDataArray()` — the latter falls back to ToArray() for
            // contiguous views with a non-zero storage offset and would
            // pin the cache to a stale snapshot copy that subsequent forward
            // passes don't write into. The raw storage array is shape-stable
            // for the plan's lifetime (the forward pass writes into the
            // same backing buffer every step), so caching it is correct AND
            // allocation-free even when the logical Tensor view has an
            // offset.
            float[]? cachedDC = null, cachedA = null, cachedB = null, cachedDestA = null, cachedDestB = null;
            int dcOff = 0, aOff = 0, bOff = 0, destAOff = 0, destBOff = 0;
            // Phase G.5: BF16 scratch buffers. Allocated lazily on first
            // call only when MKL-BF16 is active. cachedBBf16 is converted
            // once and reused (B is assumed frozen across replays under
            // the mkl-bf16 trust contract); cachedABf16 and cachedDCBf16
            // are reused as activations and grads change each step.
            ushort[]? cachedABf16 = null, cachedBBf16 = null, cachedDCBf16 = null;

            return eng =>
            {
                if (cachedDC is null)
                {
                    cachedDC = (float[])(object)gradOut._storage.GetDataArray();
                    cachedA = (float[])(object)inputA._storage.GetDataArray();
                    cachedB = (float[])(object)inputB._storage.GetDataArray();
                    cachedDestA = (float[])(object)gradA._storage.GetDataArray();
                    cachedDestB = (float[])(object)gradB._storage.GetDataArray();
                    dcOff = gradOut._storageOffset;
                    aOff = inputA._storageOffset;
                    bOff = inputB._storageOffset;
                    destAOff = gradA._storageOffset;
                    destBOff = gradB._storageOffset;
                }

                // Issue #338 Phase G.5: BF16 mixed-precision backward.
                // Same trust contract as the forward BF16 path: weights are
                // assumed frozen, BF16-converted once at lazy first call,
                // reused across replays. Activation A and gradOut dC are
                // converted on the fly into pooled BF16 scratches.
                //
                // Shape gate: BF16 backward only pays off when M*N (per-call
                // dC conversion cost) is small relative to M*N*K (GEMM
                // work). For the consumer Transformer LM head (M=2048,
                // K=128, N=8192) the dC buffer is 16M elements — converting
                // it per call costs ~8 ms via SIMD which dwarfs the BF16
                // GEMM speedup on CPUs without AVX-512-BF16 hardware. Cap
                // on N: skip BF16 when N > 1024.
                if (BlasProvider.UseMklBf16 && N <= 1024)
                {
                    // Lazy initialise BF16 scratch + pre-convert B
                    if (cachedABf16 is null)
                    {
                        cachedABf16 = new ushort[M * K];
                        cachedBBf16 = new ushort[K * N];
                        cachedDCBf16 = new ushort[M * N];
                        unsafe
                        {
                            fixed (float* bSrc = &cachedB![bOff])
                            fixed (ushort* bDst = cachedBBf16)
                                BlasProvider.Fp32ToBf16Bulk(bSrc, bDst, K * N);
                        }
                    }
                    unsafe
                    {
                        fixed (float* dcSrc = &cachedDC![dcOff])
                        fixed (ushort* dcDst = cachedDCBf16)
                            BlasProvider.Fp32ToBf16Bulk(dcSrc, dcDst, M * N);
                        fixed (float* aSrc = &cachedA![aOff])
                        fixed (ushort* aDst = cachedABf16)
                            BlasProvider.Fp32ToBf16Bulk(aSrc, aDst, M * K);
                    }
                    // dA = dC @ B^T (M, K, N), dB = A^T @ dC (K, N, M)
                    bool okA = BlasProvider.TryGemmBf16(M, K, N,
                        cachedDCBf16!, 0, N, false, cachedBBf16!, 0, N, true,
                        cachedDestA!, destAOff, K);
                    bool okB = BlasProvider.TryGemmBf16(K, N, M,
                        cachedABf16!, 0, K, true, cachedDCBf16!, 0, N, false,
                        cachedDestB!, destBOff, N);
                    if (okA && okB)
                    {
                        inputA.Grad = gradA;
                        inputB.Grad = gradB;
                        return;
                    }
                    // Either call failed — fall through to FP32 path.
                }

                // dA = dC @ B^T — direct BLAS with engine fallback
                if (!BlasProvider.TryGemmEx(M, K, N, cachedDC, dcOff, N, false, cachedB!, bOff, N, true, cachedDestA!, destAOff, K))
                {
                    var dA = eng.TensorMatMul(gradOut, inputB.Transpose());
                    dA.AsSpan().CopyTo(gradA.AsWritableSpan());
                }
                // dB = A^T @ dC — direct BLAS with engine fallback
                if (!BlasProvider.TryGemmEx(K, N, M, cachedA!, aOff, K, true, cachedDC, dcOff, N, false, cachedDestB!, destBOff, N))
                {
                    var dB = eng.TensorMatMul(inputA.Transpose(), gradOut);
                    dB.AsSpan().CopyTo(gradB.AsWritableSpan());
                }

                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // Issue #338 Phase G.3: Specialized ReduceSum backward for the
        // "sum-all-to-scalar" pattern (the loss reduce at the end of the
        // forward chain). Generic path does ExpandDims + BroadcastGradToShape
        // (multiple allocations + memory passes) for what should be a
        // single Array.Fill across the input's flat buffer.
        if (step.OpName == "ReduceSum" && step.Inputs.Length == 1
            && step.OutputBuffer.Length == 1
            && typeof(T) == typeof(float)
            && step.Inputs[0].IsContiguous)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input))
                return null;

            var gradOut = gradMap[output];
            var gradIn = gradMap[input];

            return eng =>
            {
                var gOutArr = (float[])(object)gradOut.GetDataArray();
                var gInArr = (float[])(object)gradIn.GetDataArray();
                float seed = gOutArr[0];
                if (seed == 1.0f)
                {
                    // Common case (loss seed): fill with ones via SIMD-
                    // friendly Array.Fill. The .NET runtime intrinsic
                    // uses memset-style stores when value is bit-trivial.
                    int len = gradIn.Length;
                    new Span<float>(gInArr, 0, len).Fill(1.0f);
                }
                else
                {
                    new Span<float>(gInArr, 0, gradIn.Length).Fill(seed);
                }
                input.Grad = gradIn;
            };
        }

        // Issue #338 Phase G.3: Specialized GELU backward via SimdKernels.
        // The generic GELUBackward path calls engine.GeluBackward which goes
        // through the full dispatch chain (allocating a new tensor, tape
        // recording paths, etc.). For the compiled-spec hot path we just
        // need the SIMD kernel directly: gradIn = gradOut * GELU'(input).
        if (step.OpName == "GELU" && step.Inputs.Length == 1
            && step.Inputs[0].IsContiguous && typeof(T) == typeof(float))
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input))
                return null;

            var gradOut = gradMap[output];
            var gradIn = gradMap[input];

            return eng =>
            {
                var gOutArr = (float[])(object)gradOut.GetDataArray();
                var inArr = (float[])(object)input.GetDataArray();
                var gInArr = (float[])(object)gradIn.GetDataArray();
                int len = input.Length;
                unsafe
                {
                    fixed (float* pGO = gOutArr, pIn = inArr, pGI = gInArr)
                    {
                        SimdKernels.GeluBackwardUnsafe(pGO, pIn, pGI, len);
                    }
                }
                input.Grad = gradIn;
            };
        }

        // Issue #338 Phase G.3: Specialized TensorSlice backward — scatter
        // gradOutput into the corresponding region of the input's grad
        // buffer with zero-fill elsewhere. Phase F.2 profiled the generic
        // SliceBackward at 13.5 ms/iter (per-element index computation
        // with dim-stride math) for the QKV-split slices in the consumer
        // Transformer. The fast path handles the common case "slice
        // along last dim, start at offset 0 for all leading dims" with
        // a parallel-row strided memcpy.
        if (step.OpName == "TensorSlice" && step.Inputs.Length == 1
            && typeof(T) == typeof(float)
            && step.SavedState is not null && step.SavedState.Length >= 1
            && step.SavedState[0] is int[] startArr)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            var inputShape = input._shape;
            var outputShape = output._shape;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input))
                return null;

            // Fast-path conditions:
            //   1. Same rank as input
            //   2. All start offsets are 0 except possibly the last
            //   3. All dims except the last match the input's shape
            //   4. Contiguous (so flat buffer reinterpretation is valid)
            if (startArr.Length != inputShape.Length) return null;
            if (inputShape.Length != outputShape.Length) return null;
            for (int d = 0; d < inputShape.Length - 1; d++)
            {
                if (startArr[d] != 0) return null;
                if (inputShape[d] != outputShape[d]) return null;
            }
            int lastStart = startArr[inputShape.Length - 1];
            int srcLastDim = outputShape[inputShape.Length - 1];
            int dstLastDim = inputShape[inputShape.Length - 1];
            if (lastStart < 0 || lastStart + srcLastDim > dstLastDim) return null;
            if (!output.IsContiguous || !input.IsContiguous) return null;

            int numRows = 1;
            for (int d = 0; d < inputShape.Length - 1; d++) numRows *= inputShape[d];
            int rowCopyBytes = srcLastDim * sizeof(float);
            int rowCopyStartByte = lastStart * sizeof(float);
            int dstRowBytes = dstLastDim * sizeof(float);

            var gradOut = gradMap[output];
            var gradIn = gradMap[input];

            return eng =>
            {
                var gOutArr = (float[])(object)gradOut.GetDataArray();
                var gInArr = (float[])(object)gradIn.GetDataArray();

                // Zero-fill the full input grad. The fused-spec backward
                // for upstream MatMul writes its result via the same
                // accumulator, but in the consumer Transformer's QKV
                // pattern the slice's input (qkv) has only one
                // gradient contributor (this slice), so direct write
                // is safe.
                Array.Clear(gInArr, 0, gradIn.Length);

                // Parallel-row strided memcpy: gradOut[r*srcLastDim..]
                // → gradIn[r*dstLastDim + lastStart..]
                CpuParallelSettings.ParallelForOrSerial(
                    0, numRows, numRows * srcLastDim,
                    r =>
                    {
                        Buffer.BlockCopy(
                            gOutArr, r * rowCopyBytes,
                            gInArr, r * dstRowBytes + rowCopyStartByte,
                            rowCopyBytes);
                    });

                input.Grad = gradIn;
            };
        }

        // ReLU backward: mask = input > 0, grad = gradOut * mask
        // Phase 5.2: Use bitmask (1 bit/element) instead of full input tensor (32x memory savings)
        if (step.OpType == OpType.ReLU && step.Inputs.Length == 1
            && step.Inputs[0].IsContiguous && typeof(T) == typeof(float))
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input))
                return null;

            var gradOut = gradMap[output];
            var gradIn = gradMap[input];

            // Pre-allocate bitmask at compile time (32x smaller than storing full input)
            byte[]? reluBitmask = null;

            return eng =>
            {
                // Create bitmask from forward output (output > 0 ↔ input > 0 for ReLU).
                // PR #341 review: the allocating overload returned a fresh
                // array that we silently dropped on the floor — the
                // pre-allocated reluBitmask stayed all-zero and
                // ApplyReluBackwardFromBitmask wrote zeros into gradIn,
                // killing ReLU gradient flow. The FillReluBitmask overload
                // writes into the pre-allocated buffer, preserving the
                // intended allocation-free replay semantics.
                var outputData = (float[])(object)output.GetDataArray();
                int len = output.Length;
                reluBitmask ??= new byte[(len + 7) / 8];
                ActivationCheckpoint.FillReluBitmask(outputData, reluBitmask, len);

                // Apply backward using bitmask
                var gradOutData = (float[])(object)gradOut.GetDataArray();
                var gradInData = (float[])(object)gradIn.GetDataArray();
                ActivationCheckpoint.ApplyReluBackwardFromBitmask(gradOutData, reluBitmask, gradInData, len);
                input.Grad = gradIn;
            };
        }
        // ReLU backward — secondary float-only path. Issue #340: the
        // outer comment claimed this was the "non-float types" fallback
        // but the body casts to Tensor<float>, so T=double would have
        // crashed here. The primary block above is already gated on
        // typeof(T) == typeof(float); gate this duplicate the same way
        // and add a real non-float fallback that routes through the
        // engine.
        if (step.OpType == OpType.ReLU && step.Inputs.Length == 1
            && step.Inputs[0].IsContiguous
            && typeof(T) == typeof(float))
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input))
                return null;

            var gradOut = gradMap[output];
            var gradIn = gradMap[input];

            return eng =>
            {
                var gMem = ((Tensor<float>)(object)gradOut).Data;
                var iMem = ((Tensor<float>)(object)input).Data;
                var dMem = ((Tensor<float>)(object)gradIn).Data;
                using var pinG = gMem.Pin();
                using var pinI = iMem.Pin();
                using var pinD = dMem.Pin();
                SimdKernels.ReluBackwardUnsafe(
                    (float*)pinG.Pointer, (float*)pinI.Pointer, (float*)pinD.Pointer, input.Length);
                input.Grad = gradIn;
            };
        }
        // ReLU backward — true non-float fallback. Routes through the
        // engine's ReluBackward so T=double / BFloat16 etc. still get a
        // working fused plan instead of an InvalidCastException.
        if (step.OpType == OpType.ReLU && step.Inputs.Length == 1
            && step.Inputs[0].IsContiguous)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input))
                return null;

            var gradOut = gradMap[output];
            var gradIn = gradMap[input];

            return eng =>
            {
                var computed = eng.ReluBackward(gradOut, input);
                computed.AsSpan().CopyTo(gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        // Add backward: gradA += gradOut, gradB += gradOut
        // Use consumerCount to decide: overwrite (fast) for single-consumer, accumulate for multi-consumer
        if (step.OpType == OpType.TensorAdd && step.Inputs.Length == 2)
        {
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(inputA) || !gradMap.ContainsKey(inputB))
                return null;

            var gradOut = gradMap[output];
            var gradA = gradMap[inputA];
            var gradB = gradMap[inputB];
            bool accumA = consumerCount.ContainsKey(inputA) && consumerCount[inputA] > 1;
            bool accumB = consumerCount.ContainsKey(inputB) && consumerCount[inputB] > 1;

            return eng =>
            {
                if (accumA)
                    eng.TensorAddInto(gradA, gradA, gradOut);
                else
                    gradOut.AsSpan().CopyTo(gradA.AsWritableSpan());

                if (accumB)
                    eng.TensorAddInto(gradB, gradB, gradOut);
                else
                    gradOut.AsSpan().CopyTo(gradB.AsWritableSpan());

                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // Subtract backward: gradA += gradOut, gradB -= gradOut
        if (step.OpType == OpType.TensorSubtract && step.Inputs.Length == 2)
        {
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(inputA) || !gradMap.ContainsKey(inputB))
                return null;

            var gradOut = gradMap[output];
            var gradA = gradMap[inputA];
            var gradB = gradMap[inputB];
            bool accumA = consumerCount.ContainsKey(inputA) && consumerCount[inputA] > 1;
            bool accumB = consumerCount.ContainsKey(inputB) && consumerCount[inputB] > 1;
            var numOps = MathHelper.GetNumericOperations<T>();

            return eng =>
            {
                if (accumA)
                    eng.TensorAddInto(gradA, gradA, gradOut);
                else
                    gradOut.AsSpan().CopyTo(gradA.AsWritableSpan());

                if (accumB)
                {
                    // gradB -= gradOut (accumulate negative)
                    eng.TensorSubtractInto(gradB, gradB, gradOut);
                }
                else
                    numOps.Negate(gradOut.AsSpan(), gradB.AsWritableSpan());

                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // Multiply backward: gradA = gradOut * B, gradB = gradOut * A
        if (step.OpType == OpType.TensorMultiply && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous)
        {
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(inputA) || !gradMap.ContainsKey(inputB))
                return null;

            var gradOut = gradMap[output];
            var gradA = gradMap[inputA];
            var gradB = gradMap[inputB];
            bool accumA = consumerCount.ContainsKey(inputA) && consumerCount[inputA] > 1;
            bool accumB = consumerCount.ContainsKey(inputB) && consumerCount[inputB] > 1;

            return eng =>
            {
                if (accumA)
                {
                    // gradA += gradOut * inputB
                    var temp = eng.TensorMultiply(gradOut, inputB);
                    eng.TensorAddInto(gradA, gradA, temp);
                }
                else
                    eng.TensorMultiplyInto(gradA, gradOut, inputB);

                if (accumB)
                {
                    // gradB += gradOut * inputA
                    var temp = eng.TensorMultiply(gradOut, inputA);
                    eng.TensorAddInto(gradB, gradB, temp);
                }
                else
                    eng.TensorMultiplyInto(gradB, gradOut, inputA);

                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // ReduceSum backward: broadcast scalar gradient to all elements
        // Only specialize for full scalar reduction (output length == 1)
        if (step.OpType == OpType.ReduceSum && step.Inputs.Length == 1
            && step.OutputBuffer.Length == 1)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input))
                return null;

            var gradOut = gradMap[output];
            var gradIn = gradMap[input];

            return eng =>
            {
                // ReduceSum backward: each input element gets the same scalar gradient
                gradIn.AsWritableSpan().Fill(gradOut.AsSpan()[0]);
                input.Grad = gradIn;
            };
        }

        // Sigmoid backward: grad * sigmoid(out) * (1 - sigmoid(out))
        if (step.OpType == OpType.Sigmoid && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input)) return null;
            var gradOut = gradMap[output];
            var gradIn = gradMap[input];
            return eng =>
            {
                var grad = eng.SigmoidBackward(gradOut, output);
                grad.AsSpan().CopyTo(gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        // Tanh backward: grad * (1 - tanh(out)^2)
        if (step.OpType == OpType.Tanh && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input)) return null;
            var gradOut = gradMap[output];
            var gradIn = gradMap[input];
            return eng =>
            {
                var grad = eng.TanhBackward(gradOut, output);
                grad.AsSpan().CopyTo(gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        // Negate backward: grad = -gradOut
        if (step.OpType == OpType.TensorNegate && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input)) return null;
            var gradOut = gradMap[output];
            var gradIn = gradMap[input];
            return eng =>
            {
                MathHelper.GetNumericOperations<T>().Negate(gradOut.AsSpan(), gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        // Transpose backward: grad = transpose(gradOut)
        if (step.OpType == OpType.TensorTranspose && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input)) return null;
            var gradOut = gradMap[output];
            var gradIn = gradMap[input];
            return eng =>
            {
                var grad = eng.TensorTranspose(gradOut);
                grad.AsSpan().CopyTo(gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        // Divide backward: gradA = gradOut / B, gradB = -gradOut * A / (B * B)
        if (step.OpType == OpType.TensorDivide && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous)
        {
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(inputA) || !gradMap.ContainsKey(inputB))
                return null;
            var gradOut = gradMap[output];
            var gradA = gradMap[inputA];
            var gradB = gradMap[inputB];
            return eng =>
            {
                var gA = eng.TensorDivide(gradOut, inputB);
                gA.AsSpan().CopyTo(gradA.AsWritableSpan());
                var negGradTimesA = eng.TensorNegate(eng.TensorMultiply(gradOut, inputA));
                var bSquared = eng.TensorMultiply(inputB, inputB);
                var gB = eng.TensorDivide(negGradTimesA, bSquared);
                gB.AsSpan().CopyTo(gradB.AsWritableSpan());
                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // Softmax backward
        if (step.OpType == OpType.Softmax && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input)) return null;
            var gradOut = gradMap[output];
            var gradIn = gradMap[input];
            int axis = step.SavedState != null && step.SavedState.Length > 0 ? (int)step.SavedState[0] : -1;
            return eng =>
            {
                var grad = eng.SoftmaxBackward(gradOut, output, axis);
                grad.AsSpan().CopyTo(gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        // GELU backward
        if (step.OpType == OpType.GELU && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0]; var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input)) return null;
            var gradOut = gradMap[output]; var gradIn = gradMap[input];
            return eng =>
            {
                var grad = eng.GeluBackward(gradOut, input);
                grad.AsSpan().CopyTo(gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        // LeakyReLU backward
        if (step.OpType == OpType.LeakyReLU && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0]; var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input)) return null;
            var gradOut = gradMap[output]; var gradIn = gradMap[input];
            double negSlope = step.SavedState != null && step.SavedState.Length > 0 ? (double)step.SavedState[0] : 0.01;
            return eng =>
            {
                var grad = eng.LeakyReluBackward(gradOut, input, negSlope);
                grad.AsSpan().CopyTo(gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        // Swish backward
        if (step.OpType == OpType.Swish && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0]; var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input)) return null;
            var gradOut = gradMap[output]; var gradIn = gradMap[input];
            return eng =>
            {
                var grad = eng.SwishBackward(gradOut, input);
                grad.AsSpan().CopyTo(gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        // ELU backward
        if (step.OpType == OpType.ELU && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0]; var output = step.OutputBuffer;
            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input)) return null;
            var gradOut = gradMap[output]; var gradIn = gradMap[input];
            double alpha = step.SavedState != null && step.SavedState.Length > 0 ? (double)step.SavedState[0] : 1.0;
            return eng =>
            {
                var grad = eng.EluBackward(gradOut, input, output, alpha);
                grad.AsSpan().CopyTo(gradIn.AsWritableSpan());
                input.Grad = gradIn;
            };
        }

        return null; // Not specialized — use generic fallback
    }

    /// <summary>
    /// Detects MatMul→ReLU→MatMul patterns in forward steps and replaces them with
    /// FusedMultiLayerGemm (Phase B dataflow fusion). The fused kernel keeps inter-layer
    /// data in L1 cache and uses transposed BLAS for zero-alloc backward.
    /// </summary>
    private static void TryFuseForwardBackward(
        List<CompiledStep<T>> steps,
        Dictionary<Tensor<T>, Tensor<T>> gradMap,
        Dictionary<Tensor<T>, int> consumerCount,
        IEngine engine,
        List<Action<IEngine>> fusedForward,
        List<Action<IEngine>> fusedBackward,
        HashSet<int> consumedIndices)
    {
        // PR #319: dual-precision pattern detection. Float fires the L1-resident
        // FusedMultiLayerGemm fast path (Phase B); double fires a simpler
        // BLAS-Dgemm × ReLU × BLAS-Dgemm chain that still beats the eager
        // tape path because the two matmuls dispatch to OpenBLAS Dgemm
        // directly (75 GFLOPS) instead of going through the eager
        // engine.TensorMatMul → tape recording → AutoTracer pipeline. The
        // L1-cache-residency benefit specifically requires the FusedMultiLayer
        // kernel which is float-only; extending it to double is a separate
        // 200+ line SIMD effort tracked as a follow-up.
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double)) return;

        // Issue #338 Phase G.6: detect consecutive MatMul→MatMul chains
        // (no op between them) and pre-pack the fused weight at compile
        // time. Required: float, 2D weights, ND×2D input contiguous,
        // intermediate has only one consumer, MKL or BLAS available.
        // The fused forward replaces two MatMuls with a single
        // input @ (W1 @ W2) call. Fused backward recovers the intermediate
        // gradient via chain rule on W_fused:
        //   dW_fused = input^T @ gradOutput
        //   dW1 = dW_fused @ W2^T   (small: K×N×H)
        //   dW2 = W1^T @ dW_fused   (small: K×H×N)
        //   dInput = gradOutput @ W_fused^T
        // Net savings: skip materializing the [M, H] intermediate in
        // forward + replace one large (M, H, N) backward GEMM with two
        // small (K, H, N) and (K, H, N) ones.
        //
        // Opt-in: set AIDOTNET_CROSS_LAYER_FUSION=1. Empirically on the
        // Issue #327 d=128 workload, fusion regresses wall-time despite
        // doing ~15% less arithmetic work — likely cache-pattern
        // disruption (the two separate MatMuls fit cache better than
        // one bigger fused MatMul). Workloads with larger H may benefit;
        // ship the infrastructure but make engagement explicit until
        // we have a shape-based heuristic.
        if (typeof(T) == typeof(float)
            && Environment.GetEnvironmentVariable("AIDOTNET_CROSS_LAYER_FUSION") == "1")
        {
            TryFuseMatMulMatMul(steps, gradMap, consumerCount, fusedForward, fusedBackward, consumedIndices);
        }

        for (int i = 0; i + 2 < steps.Count; i++)
        {
            if (consumedIndices.Contains(i)) continue;

            var step0 = steps[i];
            var step1 = steps[i + 1];
            var step2 = steps[i + 2];

            // Pattern: MatMul[i] → activation[i+1] → MatMul[i+2]
            // where the activation's input is MatMul[i]'s output, and
            // MatMul[i+2]'s input is the activation's output.
            // Issue #338 Phase G.2: also accepts GELU (the Transformer FFN
            // activation), but only engages the GELU fusion path when MKL
            // is NOT preferred — the L1-tile strip approach (Mr=6 rows ×
            // 342 strips per layer) issues hundreds of tiny BLAS calls
            // that fight MKL's call-once-then-parallelize design.
            // Empirically: with MKL active, fused MatMul→GELU→MatMul
            // regresses 2.4× vs the unfused 3-step path. Without MKL
            // (SimdGemm only), L1-tile fusion wins. So this branch is
            // active only when neither MKL nor — for the GELU variant —
            // is providing the big-GEMM win already.
            bool isReluPattern = step1.OpName == "ReLU";
            bool isGeluPattern = step1.OpName == "GELU";
            bool mklPreferred = string.Equals(
                Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PROVIDER"),
                "mkl",
                StringComparison.OrdinalIgnoreCase);
            // GELU fusion only when MKL is NOT preferred (see comment above).
            if (isGeluPattern && mklPreferred) continue;
            if (step0.OpName != "TensorMatMul" || step2.OpName != "TensorMatMul"
                || (!isReluPattern && !isGeluPattern))
                continue;

            if (step0.Inputs.Length != 2 || step2.Inputs.Length != 2)
                continue;

            // Verify chain: matmul0.output → relu.input, relu.output → matmul2.input
            if (!ReferenceEquals(step0.OutputBuffer, step1.Inputs[0]))
                continue;
            if (!ReferenceEquals(step1.OutputBuffer, step2.Inputs[0]))
                continue;

            // Require single-consumer chain: if step0 or step1 output feeds
            // other steps beyond the fused chain, skipping them would leave
            // those consumers without materialized data.
            if (consumerCount.TryGetValue(step0.OutputBuffer, out int c0) && c0 > 1)
                continue;
            if (consumerCount.TryGetValue(step1.OutputBuffer, out int c1) && c1 > 1)
                continue;

            // Verify weight matrices are 2D and input is ND (rank >= 2).
            // Issue #338 Phase G.2: extended from 2D-only to ND×2D so the
            // Transformer FFN (rank-3 [B, Ctx, D] input × rank-2 weights)
            // qualifies. The leading dims of the input collapse into M
            // for the fused kernel — same approach as the
            // TryBuildSpecializedForward ND×2D MatMul path at line 1209.
            if (step0.Inputs[0].Rank < 2 || step0.Inputs[1].Rank != 2 ||
                step2.Inputs[1].Rank != 2)
                continue;

            // All operands must be contiguous for the buffer-flat
            // reinterpretation that the fused kernel relies on.
            if (!step0.Inputs[0].IsContiguous || !step0.Inputs[1].IsContiguous
                || !step2.Inputs[1].IsContiguous
                || !step1.OutputBuffer.IsContiguous
                || !step2.OutputBuffer.IsContiguous)
                continue;

            // Check hidden dim: must fit in L1 (max) AND be large enough for L1 benefit (min 128)
            // Below 128, per-op BLAS is faster. Above 128, L1 residency wins.
            int h = step0.Inputs[1]._shape[1]; // output cols of W1
            if (h > Optimization.TensorCodecOptions.Current.DataflowFusionMaxHidden || h < 128)
                continue;

            // Extract dimensions — collapse leading dims of input into M.
            var inputTensor = step0.Inputs[0];   // [..batch, K]  (contiguous)
            var w1Tensor = step0.Inputs[1];      // [K, H]
            var w2Tensor = step2.Inputs[1];      // [H, N]
            var outputTensor = step2.OutputBuffer; // [..batch, N]

            int inputRank = inputTensor.Rank;
            int m = 1;
            for (int d = 0; d < inputRank - 1; d++) m *= inputTensor._shape[d];
            int k = inputTensor._shape[inputRank - 1];
            int n = w2Tensor._shape[1];

            // The fused kernel writes [m, n] into output's flat storage —
            // valid for contiguous output whose last dim is n.
            if (outputTensor._shape[outputTensor.Rank - 1] != n) continue;

            // Pre-allocate activated intermediate buffer
            var activatedBuffer = TensorAllocator.RentUninitialized<T>(new[] { m, h });

            // ── Double-precision branch — BLAS-Dgemm × scalar ReLU × BLAS-Dgemm.
            // Skips the L1-resident fused kernel (float-only) but still avoids
            // the eager tape recording/dispatch pipeline. The two matmuls
            // dispatch through OpenBLAS Dgemm directly (~75 GFLOPS) instead of
            // going through eager TensorMatMul → tape recording → AutoTracer.
            // Gated on BlasProvider.HasNativeDgemm at compile-build time — if
            // the user has explicitly opted out of BLAS (AIDOTNET_USE_BLAS=0),
            // we don't fuse this triple; the eager path will run and remain
            // correct (just slower for that user). Industry standard: if you
            // disable the BLAS that matmul depends on, you've also disabled
            // the kernels that depend on it.
            if (typeof(T) == typeof(double))
            {
                if (!BlasProvider.HasNativeDgemm)
                {
                    TensorAllocator.Return(activatedBuffer);
                    continue;
                }

                var capturedInputD = inputTensor;
                var capturedW1D = w1Tensor;
                var capturedW2D = w2Tensor;
                var capturedOutputD = outputTensor;
                var capturedActivatedD = activatedBuffer;
                int cmD = m, ckD = k, chD = h, cnD = n;
                var capturedGradMapD = gradMap;

                fusedForward.Add(eng =>
                {
                    var inA = (double[])(object)capturedInputD.GetDataArray();
                    var w1A = (double[])(object)capturedW1D.GetDataArray();
                    var w2A = (double[])(object)capturedW2D.GetDataArray();
                    var outA = (double[])(object)capturedOutputD.GetDataArray();
                    var actA = (double[])(object)capturedActivatedD.GetDataArray();

                    BlasProvider.TryGemm(cmD, chD, ckD, inA, 0, ckD, w1A, 0, chD, actA, 0, chD);

                    int activatedLen = cmD * chD;
                    for (int idx = 0; idx < activatedLen; idx++)
                        if (actA[idx] < 0.0) actA[idx] = 0.0;

                    BlasProvider.TryGemm(cmD, cnD, chD, actA, 0, chD, w2A, 0, cnD, outA, 0, cnD);
                });

                fusedBackward.Add(eng =>
                {
                    if (!capturedGradMapD.TryGetValue(capturedOutputD, out var gradOut)) return;

                    var gOutArr = (double[])(object)gradOut.GetDataArray();
                    var inA = (double[])(object)capturedInputD.GetDataArray();
                    var w1A = (double[])(object)capturedW1D.GetDataArray();
                    var w2A = (double[])(object)capturedW2D.GetDataArray();
                    var actA = (double[])(object)capturedActivatedD.GetDataArray();

                    capturedGradMapD.TryGetValue(capturedW1D, out var gW1Tensor);
                    capturedGradMapD.TryGetValue(capturedW2D, out var gW2Tensor);
                    capturedGradMapD.TryGetValue(capturedInputD, out var gInTensor);

                    // dW2 = activated^T @ gradOut → [h, n]
                    if (gW2Tensor is not null)
                    {
                        var gW2 = (double[])(object)gW2Tensor.GetDataArray();
                        BlasProvider.TryGemmEx(chD, cnD, cmD,
                            actA, 0, chD, transA: true,
                            gOutArr, 0, cnD, transB: false,
                            gW2, 0, cnD);
                    }

                    // gradH = gradOut @ w2^T → [m, h]
                    var gradH = ArrayPool<double>.Shared.Rent(cmD * chD);
                    Array.Clear(gradH, 0, cmD * chD);
                    BlasProvider.TryGemmEx(cmD, chD, cnD,
                        gOutArr, 0, cnD, transA: false,
                        w2A, 0, cnD, transB: true,
                        gradH, 0, chD);

                    // ReLU backward: gradH *= (activated > 0 ? 1 : 0)
                    int gradHLen = cmD * chD;
                    for (int idx = 0; idx < gradHLen; idx++)
                        if (actA[idx] <= 0.0) gradH[idx] = 0.0;

                    // dW1 = input^T @ gradH → [k, h]
                    if (gW1Tensor is not null)
                    {
                        var gW1 = (double[])(object)gW1Tensor.GetDataArray();
                        BlasProvider.TryGemmEx(ckD, chD, cmD,
                            inA, 0, ckD, transA: true,
                            gradH, 0, chD, transB: false,
                            gW1, 0, chD);
                    }

                    // dInput = gradH @ w1^T → [m, k]
                    if (gInTensor is not null)
                    {
                        var gIn = (double[])(object)gInTensor.GetDataArray();
                        BlasProvider.TryGemmEx(cmD, ckD, chD,
                            gradH, 0, chD, transA: false,
                            w1A, 0, chD, transB: true,
                            gIn, 0, ckD);
                    }

                    ArrayPool<double>.Shared.Return(gradH);

                    capturedW1D.Grad = gW1Tensor;
                    capturedW2D.Grad = gW2Tensor;
                    capturedInputD.Grad = gInTensor;
                });

                consumedIndices.Add(i);
                consumedIndices.Add(i + 1);
                consumedIndices.Add(i + 2);
                i += 2;
                continue;
            }

            // Issue #340: the block below is float-only (FusedMultiLayerGemm
            // takes float[]; every cast is to (float[])(object)X). The
            // T=double branch above continues; any other T (e.g. BFloat16)
            // would crash on the cast at first replay. Skip the fused
            // float kernel for non-float T so the plan falls through to
            // the generic step path.
            if (typeof(T) != typeof(float))
            {
                TensorAllocator.Return(activatedBuffer);
                continue;
            }

            // Capture for closures
            var capturedInput = inputTensor;
            var capturedW1 = w1Tensor;
            var capturedW2 = w2Tensor;
            var capturedOutput = outputTensor;
            var capturedActivated = activatedBuffer;
            int cm = m, ck = k, ch = h, cn = n;
            var capturedGradMap = gradMap;

            // Issue #338 Phase G.2: for GELU, we need pre-activation in
            // backward (derivative isn't recoverable from post-activation
            // values). Allocate a second buffer; for ReLU the buffer stays
            // null and the existing code path is unchanged.
            Tensor<T>? preActivationBuffer = isGeluPattern
                ? TensorAllocator.RentUninitialized<T>(new[] { m, h })
                : null;

            Func<float, float> activation = isGeluPattern
                ? new Func<float, float>(GeluTanhApprox)
                : new Func<float, float>(x => x > 0f ? x : 0f);

            // Pre-allocate backward workspace at compile time (zero alloc during replay)
            var backwardWorkspace = new float[cm * ch]; // grad_h buffer
            var emptyBias = new float[0];

            // Capture for closure scope (locals can't be captured by reference into closures)
            bool capturedIsGelu = isGeluPattern;
            var capturedPreAct = preActivationBuffer;

            // Fused forward: single kernel call replaces 3 steps.
            // GELU path uses SIMD GELU (FusedGemmGeluGemm); ReLU path uses
            // the scalar-Func variant — ReLU is cheap enough that the
            // delegate dispatch isn't a bottleneck.
            fusedForward.Add(eng =>
            {
                var inA = (float[])(object)capturedInput.GetDataArray();
                var w1A = (float[])(object)capturedW1.GetDataArray();
                var w2A = (float[])(object)capturedW2.GetDataArray();
                var outA = (float[])(object)capturedOutput.GetDataArray();
                var actA = (float[])(object)capturedActivated.GetDataArray();
                if (capturedIsGelu)
                {
                    var preA = (float[])(object)capturedPreAct!.GetDataArray();
                    FusedMultiLayerGemm.FusedGemmGeluGemm(inA, w1A, w2A, outA, actA, preA, cm, ck, ch, cn);
                }
                else
                {
                    FusedMultiLayerGemm.FusedGemmActivationGemm(inA, w1A, w2A, outA, actA, cm, ck, ch, cn, activation);
                }
            });

            // Fused backward: pre-allocated workspace, zero alloc during replay
            fusedBackward.Add(eng =>
            {
                var gradOut = capturedGradMap.ContainsKey(capturedOutput)
                    ? capturedGradMap[capturedOutput] : null;
                if (gradOut == null) return;

                var gOutArr = (float[])(object)gradOut.GetDataArray();
                var inA = (float[])(object)capturedInput.GetDataArray();
                var w1A = (float[])(object)capturedW1.GetDataArray();
                var w2A = (float[])(object)capturedW2.GetDataArray();
                var actA = (float[])(object)capturedActivated.GetDataArray();

                var gW1 = capturedGradMap.ContainsKey(capturedW1)
                    ? (float[])(object)capturedGradMap[capturedW1].GetDataArray() : new float[ck * ch];
                var gW2 = capturedGradMap.ContainsKey(capturedW2)
                    ? (float[])(object)capturedGradMap[capturedW2].GetDataArray() : new float[ch * cn];
                var gIn = capturedGradMap.ContainsKey(capturedInput)
                    ? (float[])(object)capturedGradMap[capturedInput].GetDataArray() : new float[cm * ck];

                Array.Clear(backwardWorkspace, 0, backwardWorkspace.Length);

                if (capturedIsGelu)
                {
                    var preA = (float[])(object)capturedPreAct!.GetDataArray();
                    FusedMultiLayerBackward.ComputeGradients(
                        gOutArr, inA, w1A, w2A, actA, preA,
                        gW1, gW2, emptyBias, emptyBias, gIn,
                        cm, ck, ch, cn, FusedMultiLayerBackward.GELUDerivative,
                        backwardWorkspace);
                }
                else
                {
                    FusedMultiLayerBackward.ComputeGradients(
                        gOutArr, inA, w1A, w2A, actA,
                        gW1, gW2, emptyBias, emptyBias, gIn,
                        cm, ck, ch, cn, FusedMultiLayerBackward.ReLUDerivative,
                        backwardWorkspace);
                }

                capturedW1.Grad = capturedGradMap.ContainsKey(capturedW1) ? capturedGradMap[capturedW1] : null;
                capturedW2.Grad = capturedGradMap.ContainsKey(capturedW2) ? capturedGradMap[capturedW2] : null;
                capturedInput.Grad = capturedGradMap.ContainsKey(capturedInput) ? capturedGradMap[capturedInput] : null;
            });

            consumedIndices.Add(i);
            consumedIndices.Add(i + 1);
            consumedIndices.Add(i + 2);

            // Skip past the fused group
            i += 2;
        }
    }

    /// <summary>
    /// GELU forward — tanh approximation matching SimdKernels.GELUUnsafe so
    /// the fused-FFN forward stays bit-equivalent to the unfused MatMul → GELU
    /// → MatMul path. <c>GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float GeluTanhApprox(float x)
    {
        const float SQRT_2_OVER_PI = 0.7978845608028654f;
        const float COEFF = 0.044715f;
        float u = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        return 0.5f * x * (1f + MathF.Tanh(u));
    }

    /// <summary>
    /// Issue #338 Phase G.6: cross-layer MatMul→MatMul fusion. Detects
    /// pairs of consecutive MatMul steps where the intermediate is
    /// single-consumer and replaces them with a single fused MatMul
    /// using a pre-packed W_fused = W1 @ W2.
    ///
    /// <para>Forward savings: one MatMul instead of two (saves
    /// (M, K, H) work where H is the intermediate hidden dim) + skip
    /// intermediate materialization.</para>
    /// <para>Backward savings: replace the dominant intermediate-sized
    /// (M, H, N) GEMM with two smaller (K, H, N) and (K, K, H) ones
    /// via chain rule on W_fused. Requires the intermediate to be
    /// single-consumer (the gradient signal flows to dW1 and dW2 only
    /// through dW_fused, never directly).</para>
    /// </summary>
    /// <summary>
    /// Issue #338 Phase G.7: detect MatMul steps whose output is consumed
    /// ONLY by a ReduceSum-all-to-scalar step that produces the loss.
    /// When this pattern matches, the gradient flowing into the MatMul's
    /// output is mathematically <c>α * ones</c> (where α is the loss-grad
    /// scalar, defaults to 1). The backward can be computed analytically
    /// from W and the input directly, skipping the M*N ones materialization
    /// and the two M*N*K GEMMs entirely.
    ///
    /// <para>For Issue #327's d=128 / V=8192 LM head: replaces the 6.3B-MAC
    /// LM-head backward (~30 ms on this CPU) with ~2.5M element ops (~0.2 ms).</para>
    /// </summary>
    private static void DetectAnalyticLossMatMulBackward(
        List<CompiledStep<T>> forwardSteps,
        Dictionary<Tensor<T>, int> consumerCount,
        Dictionary<Tensor<T>, Tensor<T>> gradMap,
        Dictionary<int, Action<IEngine>> analyticBackwardSpecs)
    {
        if (typeof(T) != typeof(float)) return;
        if (forwardSteps.Count < 2) return;

        // Walk pairs (matmul, reducesum) and check the link.
        for (int i = 0; i + 1 < forwardSteps.Count; i++)
        {
            var matmul = forwardSteps[i];
            var reduce = forwardSteps[i + 1];

            if (matmul.OpName != "TensorMatMul") continue;
            if (reduce.OpName != "ReduceSum") continue;
            if (reduce.OutputBuffer.Length != 1) continue;  // sum-to-scalar
            if (matmul.Inputs.Length != 2 || reduce.Inputs.Length != 1) continue;
            if (!ReferenceEquals(matmul.OutputBuffer, reduce.Inputs[0])) continue;

            // MatMul output must be single-consumer (only ReduceSum). If
            // some other op also consumes it, we need to materialize the
            // gradient normally.
            if (consumerCount.TryGetValue(matmul.OutputBuffer, out int mmCount) && mmCount > 1)
                continue;

            // ReduceSum must be the loss output — i.e. nothing consumes it
            // downstream in the forward graph.
            if (consumerCount.TryGetValue(reduce.OutputBuffer, out int rsCount) && rsCount > 0)
                continue;

            var inputA = matmul.Inputs[0];   // [...batch, K]
            var inputW = matmul.Inputs[1];   // [K, V]
            var output = matmul.OutputBuffer;

            if (inputA.Rank < 2 || inputW.Rank != 2) continue;
            if (!inputA.IsContiguous || !inputW.IsContiguous) continue;

            int K = inputA._shape[inputA.Rank - 1];
            int V = inputW._shape[1];
            int M = 1;
            for (int d = 0; d < inputA.Rank - 1; d++) M *= inputA._shape[d];
            if (inputW._shape[0] != K) continue;

            if (!gradMap.ContainsKey(inputA) || !gradMap.ContainsKey(inputW)) continue;
            var gradA = gradMap[inputA];
            var gradW = gradMap[inputW];
            // Loss-grad scalar value lives in gradMap[reduce.OutputBuffer]
            // when backward replay runs. Capture the reference.
            if (!gradMap.ContainsKey(reduce.OutputBuffer)) continue;
            var lossGradTensor = gradMap[reduce.OutputBuffer];

            // Build the analytic backward closure.
            int cM = M, cK = K, cV = V;
            var capA = inputA;
            var capW = inputW;
            var capGradA = gradA;
            var capGradW = gradW;
            var capLossGrad = lossGradTensor;

            analyticBackwardSpecs[i] = eng =>
            {
                var aArr = (float[])(object)capA.GetDataArray();
                var wArr = (float[])(object)capW.GetDataArray();
                var gA = (float[])(object)capGradA.GetDataArray();
                var gW = (float[])(object)capGradW.GetDataArray();
                var lossArr = (float[])(object)capLossGrad.GetDataArray();
                float alpha = lossArr[0];

                // Compute row_sums(W): for each row k, sum across all v.
                //   row_sum_W[k] = sum_v W[k, v]
                // W is row-major [K, V], so row k starts at k*V and has V entries.
                var rowSumW = ArrayPool<float>.Shared.Rent(cK);
                try
                {
                    for (int k = 0; k < cK; k++)
                    {
                        float s = 0f;
                        int baseIdx = k * cV;
                        // Loop unroll-friendly stride-1 read pattern
                        for (int v = 0; v < cV; v++) s += wArr[baseIdx + v];
                        rowSumW[k] = alpha * s;
                    }

                    // dA[m, k] = alpha * row_sum_W[k]  — same value for all m.
                    // Stride-K broadcast write into gA.
                    for (int m = 0; m < cM; m++)
                    {
                        int baseIdx = m * cK;
                        for (int k = 0; k < cK; k++) gA[baseIdx + k] = rowSumW[k];
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(rowSumW);
                }

                // Compute col_sums(A): for each column k of A (= K-th feature),
                // sum across all M rows.
                //   col_sum_A[k] = sum_m A[m, k]
                // A is row-major [M, K], so element (m, k) is at m*K + k.
                var colSumA = ArrayPool<float>.Shared.Rent(cK);
                try
                {
                    for (int k = 0; k < cK; k++) colSumA[k] = 0f;
                    for (int m = 0; m < cM; m++)
                    {
                        int baseIdx = m * cK;
                        for (int k = 0; k < cK; k++) colSumA[k] += aArr[baseIdx + k];
                    }
                    for (int k = 0; k < cK; k++) colSumA[k] *= alpha;

                    // dW[k, v] = alpha * col_sum_A[k]  — same value for all v.
                    for (int k = 0; k < cK; k++)
                    {
                        float c = colSumA[k];
                        int baseIdx = k * cV;
                        for (int v = 0; v < cV; v++) gW[baseIdx + v] = c;
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(colSumA);
                }

                capA.Grad = capGradA;
                capW.Grad = capGradW;
            };
        }
    }

    private static unsafe void TryFuseMatMulMatMul(
        List<CompiledStep<T>> steps,
        Dictionary<Tensor<T>, Tensor<T>> gradMap,
        Dictionary<Tensor<T>, int> consumerCount,
        List<Action<IEngine>> fusedForward,
        List<Action<IEngine>> fusedBackward,
        HashSet<int> consumedIndices)
    {
        if (typeof(T) != typeof(float)) return;

        for (int i = 0; i + 1 < steps.Count; i++)
        {
            if (consumedIndices.Contains(i) || consumedIndices.Contains(i + 1)) continue;

            var step0 = steps[i];
            var step1 = steps[i + 1];

            if (step0.OpName != "TensorMatMul" || step1.OpName != "TensorMatMul") continue;
            if (step0.Inputs.Length != 2 || step1.Inputs.Length != 2) continue;
            if (!ReferenceEquals(step0.OutputBuffer, step1.Inputs[0])) continue;

            // Intermediate must be single-consumer for chain-rule fusion
            // to be valid; multi-consumer intermediates leak gradients
            // through paths the fused chain can't see.
            if (consumerCount.TryGetValue(step0.OutputBuffer, out int interCount) && interCount > 1)
                continue;

            var inputA = step0.Inputs[0];     // [...batch, K]
            var W1 = step0.Inputs[1];         // [K, H]
            var W2 = step1.Inputs[1];         // [H, N]
            var output = step1.OutputBuffer;  // [...batch, N]

            if (inputA.Rank < 2 || W1.Rank != 2 || W2.Rank != 2) continue;
            if (!inputA.IsContiguous || !W1.IsContiguous || !W2.IsContiguous || !output.IsContiguous)
                continue;

            int K = inputA._shape[inputA.Rank - 1];
            int H = W1._shape[1];
            int N = W2._shape[1];
            if (W1._shape[0] != K || W2._shape[0] != H) continue;
            int M = 1;
            for (int d = 0; d < inputA.Rank - 1; d++) M *= inputA._shape[d];

            // Pre-pack W_fused = W1 @ W2 at compile time. Allocated here
            // so the closure captures a stable reference. ONE pre-pack
            // cost (compile-time, never replayed).
            var w1Data = (float[])(object)W1.GetDataArray();
            var w2Data = (float[])(object)W2.GetDataArray();
            var wFused = new float[K * N];
            if (!BlasProvider.TryGemm(K, N, H, w1Data, 0, H, w2Data, 0, N, wFused, 0, N))
            {
                SimdGemm.Sgemm(w1Data.AsSpan(0, K * H), w2Data.AsSpan(0, H * N), wFused.AsSpan(0, K * N), K, H, N);
            }

            // Forward: single MatMul input @ W_fused → output. Skip
            // intermediate materialization.
            var capturedInput = inputA;
            var capturedOutput = output;
            var capturedW1 = W1;
            var capturedW2 = W2;
            int cM = M, cK = K, cH = H, cN = N;
            float[] cWFused = wFused;
            var capturedGradMap = gradMap;

            fusedForward.Add(eng =>
            {
                var inArr = (float[])(object)capturedInput.GetDataArray();
                var outArr = (float[])(object)capturedOutput.GetDataArray();
                if (!BlasProvider.TryGemm(cM, cN, cK, inArr, 0, cK, cWFused, 0, cN, outArr, 0, cN))
                    SimdGemm.Sgemm(inArr.AsSpan(0, cM * cK), cWFused.AsSpan(0, cK * cN),
                        outArr.AsSpan(0, cM * cN), cM, cK, cN);
            });

            // Pre-allocate dW_fused scratch (compile-time, reused per call).
            var dWFused = new float[K * N];

            fusedBackward.Add(eng =>
            {
                if (!capturedGradMap.TryGetValue(capturedOutput, out var gradOut)) return;
                var gOutArr = (float[])(object)gradOut.GetDataArray();
                var inArr = (float[])(object)capturedInput.GetDataArray();
                var w1Arr = (float[])(object)capturedW1.GetDataArray();
                var w2Arr = (float[])(object)capturedW2.GetDataArray();

                // dW_fused = input^T @ gradOut  [K, N]
                if (!BlasProvider.TryGemmEx(cK, cN, cM,
                        inArr, 0, cK, true,
                        gOutArr, 0, cN, false,
                        dWFused, 0, cN))
                {
                    SimdGemm.Sgemm(inArr.AsSpan(0, cM * cK), cK, true,
                        gOutArr.AsSpan(0, cM * cN), cN, false,
                        dWFused.AsSpan(0, cK * cN), cK, cM, cN);
                }

                // dW1 = dW_fused @ W2^T  [K, H]  (chain rule on W_fused = W1 @ W2)
                if (capturedGradMap.TryGetValue(capturedW1, out var gW1Tensor))
                {
                    var gW1 = (float[])(object)gW1Tensor.GetDataArray();
                    if (!BlasProvider.TryGemmEx(cK, cH, cN,
                            dWFused, 0, cN, false,
                            w2Arr, 0, cN, true,
                            gW1, 0, cH))
                    {
                        SimdGemm.Sgemm(dWFused.AsSpan(0, cK * cN), cN, false,
                            w2Arr.AsSpan(0, cH * cN), cN, true,
                            gW1.AsSpan(0, cK * cH), cK, cN, cH);
                    }
                    capturedW1.Grad = gW1Tensor;
                }

                // dW2 = W1^T @ dW_fused  [H, N]
                if (capturedGradMap.TryGetValue(capturedW2, out var gW2Tensor))
                {
                    var gW2 = (float[])(object)gW2Tensor.GetDataArray();
                    if (!BlasProvider.TryGemmEx(cH, cN, cK,
                            w1Arr, 0, cH, true,
                            dWFused, 0, cN, false,
                            gW2, 0, cN))
                    {
                        SimdGemm.Sgemm(w1Arr.AsSpan(0, cK * cH), cH, true,
                            dWFused.AsSpan(0, cK * cN), cN, false,
                            gW2.AsSpan(0, cH * cN), cH, cK, cN);
                    }
                    capturedW2.Grad = gW2Tensor;
                }

                // dInput = gradOut @ W_fused^T  [M, K]
                if (capturedGradMap.TryGetValue(capturedInput, out var gInTensor))
                {
                    var gIn = (float[])(object)gInTensor.GetDataArray();
                    if (!BlasProvider.TryGemmEx(cM, cK, cN,
                            gOutArr, 0, cN, false,
                            cWFused, 0, cN, true,
                            gIn, 0, cK))
                    {
                        SimdGemm.Sgemm(gOutArr.AsSpan(0, cM * cN), cN, false,
                            cWFused.AsSpan(0, cK * cN), cN, true,
                            gIn.AsSpan(0, cM * cK), cM, cN, cK);
                    }
                    capturedInput.Grad = gInTensor;
                }
            });

            consumedIndices.Add(i);
            consumedIndices.Add(i + 1);
            i++; // Skip past i+1 — already consumed
        }
    }
}

/// <summary>
/// A single backward step in a compiled training plan.
/// </summary>
internal sealed class BackwardStep<T>
{
    internal readonly string OpName;
    internal readonly BackwardFunction<T> BackwardFn;
    internal readonly Tensor<T> Output;
    internal readonly Tensor<T>[] Inputs;
    internal readonly object[]? SavedState;

    internal BackwardStep(
        string opName,
        BackwardFunction<T> backwardFn,
        Tensor<T> output,
        Tensor<T>[] inputs,
        object[]? savedState)
    {
        OpName = opName;
        BackwardFn = backwardFn;
        Output = output;
        Inputs = inputs;
        SavedState = savedState;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void Execute(IEngine engine, Tensor<T> gradOutput, Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        BackwardFn(gradOutput, Inputs, Output, SavedState ?? Array.Empty<object>(), engine, gradAccumulator);
    }
}
