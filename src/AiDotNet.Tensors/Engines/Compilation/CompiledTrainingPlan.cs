using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Autodiff;
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
    private readonly Action<IEngine>[] _forwardActions;
    private readonly Action<IEngine>[] _backwardActions;
    private readonly Tensor<T> _lossOutput;
    private readonly IEngine _engine;
    private readonly Tensor<T>[] _parameters;
    private readonly Tensor<T>[] _gradients;
    private readonly Tensor<T>[] _preAllocatedGrads;
    private readonly Tensor<T> _lossGradSeed;
    private readonly List<GCHandle> _pinnedHandles = new();
    private bool _disposed;

    // Indices of gradient buffers that need zeroing (used by generic/accumulating backward only)
    private readonly int[]? _genericGradIndices;

    // Cached raw arrays for zero-overhead Step() — avoid AsSpan()/GetDataArray() per call
    private T[][]? _cachedGradArrays;
    private T[]? _cachedLossGradSeedArray;
    private T[]? _cachedLossGradDestArray;
    private readonly Tensor<T>? _lossGradDest; // Pre-allocated gradient buffer for loss output

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
        List<GCHandle>? pinnedHandles = null)
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

        // Forward: use checkpointing if enabled, otherwise straight-line delegates
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
        var bwd = _backwardActions;
        for (int i = 0; i < bwd.Length; i++)
            bwd[i](engine);

        // Fused optimizer update (if configured via ConfigureOptimizer)
        _optimizerUpdate?.Invoke();

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
        if (typeof(T) != typeof(float))
            throw new NotSupportedException("Fused optimizer updates only support float parameters.");

        // Pre-allocate optimizer state buffers for each parameter
        int paramCount = _parameters.Length;
        var paramArrays = new float[paramCount][];
        var gradArrays = new float[paramCount][];
        var lengths = new int[paramCount];

        // Momentum / first moment buffers (Adam, SGD+momentum, etc.)
        var m = new float[paramCount][];
        // Second moment buffers (Adam, RMSprop, etc.)
        var v = new float[paramCount][];

        for (int p = 0; p < paramCount; p++)
        {
            paramArrays[p] = (float[])(object)_parameters[p].GetDataArray();
            gradArrays[p] = _gradients[p] is not null
                ? (float[])(object)_gradients[p].GetDataArray()
                : Array.Empty<float>();
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
        var lr = learningRate;
        var b1 = beta1;
        var b2 = beta2;
        var epsVal = eps;
        var wd = weightDecay;
        var optType = optimizerType;

        _optimizerUpdate = () =>
        {
            _optimizerStep++;
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

    internal static CompiledTrainingPlan<T> Compile(
        LazyTensorScope scope, IEngine engine, Tensor<T>[] parameters)
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
            var specialized = TryBuildSpecializedForward(step, pinnedHandles);
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
        var lossOutput = forwardSteps.Count > 0
            ? forwardSteps[forwardSteps.Count - 1].OutputBuffer
            : new Tensor<T>(new int[] { 1 });
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
            pinnedHandles);
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

    internal static unsafe Action<IEngine>? TryBuildSpecializedForward(CompiledStep<T> step, List<GCHandle>? handleTracker = null)
    {
        if (typeof(T) != typeof(float)) return null;

        // MatMul forward: direct BLAS into output buffer
        if (step.OpType == OpType.TensorMatMul && step.Inputs.Length == 2
            && step.Inputs[0].Rank == 2 && step.Inputs[1].Rank == 2
            && typeof(T) == typeof(float))
        {
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;
            int M = inputA._shape[0], K = inputA._shape[1], N = inputB._shape[1];

            // Pre-fetch arrays at compile time (bypasses EnsureMaterialized at replay)
            var cA = (float[])(object)inputA.GetDataArray();
            var cB = (float[])(object)inputB.GetDataArray();
            var cOut = (float[])(object)output.GetDataArray();

            return eng =>
            {
                if (!BlasProvider.TryGemm(M, N, K, cA, 0, K, cB, 0, N, cOut, 0, N))
                    SimdGemm.Sgemm(cA, cB, cOut, M, K, N);
            };
        }

        // ReLU forward: direct SIMD into output buffer
        if (step.OpType == OpType.ReLU && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
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
                            Parallel.For(0, numChunks, chunk =>
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
            return eng =>
            {
                cOut ??= (float[])(object)output.GetDataArray();
                T sum = eng.TensorSum(input);
                if (typeof(T) == typeof(float))
                    cOut[0] = Unsafe.As<T, float>(ref sum);
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
                    float maxVal = p[0];
                    for (int j = 1; j < len; j++)
                        if (p[j] > maxVal) maxVal = p[j];
                    cOut[0] = maxVal;
                }
            };
        }

        // TensorAdd forward: pinned SIMD VectorAddUnsafe
        if (step.OpType == OpType.TensorAdd && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && typeof(T) == typeof(float))
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            var aH = PinAndTrack(((Tensor<float>)(object)a).GetDataArray(), handleTracker);
            var bH = PinAndTrack(((Tensor<float>)(object)b).GetDataArray(), handleTracker);
            var oH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = a.Length;
            return eng => { unsafe { SimdKernels.VectorAddUnsafe((float*)aH.AddrOfPinnedObject(), (float*)bH.AddrOfPinnedObject(), (float*)oH.AddrOfPinnedObject(), len); } };
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
            return eng => { unsafe { SimdKernels.VectorSubtractUnsafe((float*)aH.AddrOfPinnedObject(), (float*)bH.AddrOfPinnedObject(), (float*)oH.AddrOfPinnedObject(), len); } };
        }
        if (step.OpType == OpType.TensorSubtract && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous)
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            return eng => eng.TensorSubtractInto(o, a, b);
        }

        // TensorMultiply forward: pinned SIMD VectorMultiplyUnsafe
        if (step.OpType == OpType.TensorMultiply && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous
            && typeof(T) == typeof(float))
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            var aH = PinAndTrack(((Tensor<float>)(object)a).GetDataArray(), handleTracker);
            var bH = PinAndTrack(((Tensor<float>)(object)b).GetDataArray(), handleTracker);
            var oH = PinAndTrack(((Tensor<float>)(object)o).GetDataArray(), handleTracker);
            int len = a.Length;
            return eng => { unsafe { SimdKernels.VectorMultiplyUnsafe((float*)aH.AddrOfPinnedObject(), (float*)bH.AddrOfPinnedObject(), (float*)oH.AddrOfPinnedObject(), len); } };
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
                // Direct BLAS into output + fused bias + activation
                CpuFusedOperations.FusedGemmBiasActivation(
                    inArr, wArr, bArr, oArr, M, N, K, activation);
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
                unsafe
                {
                    // LayerNorm per batch element
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

        // Conv2D forward: use Conv2DInto to write directly into output
        if (step.OpType == OpType.Conv2D && step.Inputs.Length == 2)
        {
            var inp = step.Inputs[0]; var kernel = step.Inputs[1]; var o = step.OutputBuffer;
            var savedState = step.SavedState;
            if (savedState != null && savedState.Length == 3
                && savedState[0] is int[] stride && savedState[1] is int[] padding && savedState[2] is int[] dilation)
            {
                int s = stride[0], p = padding[0], d = dilation[0];
                return eng => eng.Conv2DInto(o, inp, kernel, s, p, d);
            }
            return eng => eng.Conv2DInto(o, inp, kernel, 1, 0, 1);
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

        // Mean forward: pinned SumUnsafe + divide
        // A/B tested: Parallel.For overhead (0.43ms) exceeds single-thread (0.16ms) for 1M.
        // PyTorch likely uses SIMD sum with wider parallelism (internal thread pool).
        if (step.OpType == OpType.Mean && step.Inputs.Length == 1 && typeof(T) == typeof(float))
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
        if (typeof(T) != typeof(float)) return null;

        // MatMul backward: dA = dC @ B^T, dB = A^T @ dC — transposed BLAS, zero alloc
        if (step.OpType == OpType.TensorMatMul && step.Inputs.Length == 2
            && step.Inputs[0].Rank == 2 && step.Inputs[1].Rank == 2
            && BlasProvider.IsAvailable)
        {
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(inputA) || !gradMap.ContainsKey(inputB))
                return null;

            var gradOut = gradMap[output];
            var gradA = gradMap[inputA];
            var gradB = gradMap[inputB];

            int M = inputA._shape[0], K = inputA._shape[1], N = inputB._shape[1];

            // Cache array references on first call to avoid GetDataArray() overhead per step
            float[]? cachedDC = null, cachedA = null, cachedB = null, cachedDestA = null, cachedDestB = null;

            return eng =>
            {
                cachedDC ??= (float[])(object)gradOut.GetDataArray();
                cachedA ??= (float[])(object)inputA.GetDataArray();
                cachedB ??= (float[])(object)inputB.GetDataArray();
                cachedDestA ??= (float[])(object)gradA.GetDataArray();
                cachedDestB ??= (float[])(object)gradB.GetDataArray();

                // dA = dC @ B^T — direct BLAS with engine fallback
                if (!BlasProvider.TryGemmEx(M, K, N, cachedDC, 0, N, false, cachedB, 0, N, true, cachedDestA, 0, K))
                {
                    var dA = eng.TensorMatMul(gradOut, inputB.Transpose());
                    dA.AsSpan().CopyTo(gradA.AsWritableSpan());
                }
                // dB = A^T @ dC — direct BLAS with engine fallback
                if (!BlasProvider.TryGemmEx(K, N, M, cachedA, 0, K, true, cachedDC, 0, N, false, cachedDestB, 0, N))
                {
                    var dB = eng.TensorMatMul(inputA.Transpose(), gradOut);
                    dB.AsSpan().CopyTo(gradB.AsWritableSpan());
                }

                inputA.Grad = gradA;
                inputB.Grad = gradB;
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
                // Create bitmask from forward output (output > 0 ↔ input > 0 for ReLU)
                var outputData = (float[])(object)output.GetDataArray();
                int len = output.Length;
                reluBitmask ??= new byte[(len + 7) / 8];
                ActivationCheckpoint.CreateReluBitmask(outputData, len);

                // Apply backward using bitmask
                var gradOutData = (float[])(object)gradOut.GetDataArray();
                var gradInData = (float[])(object)gradIn.GetDataArray();
                ActivationCheckpoint.ApplyReluBackwardFromBitmask(gradOutData, reluBitmask, gradInData, len);
                input.Grad = gradIn;
            };
        }
        // ReLU backward fallback: full input tensor path (non-float types)
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
        if (typeof(T) != typeof(float)) return;

        for (int i = 0; i + 2 < steps.Count; i++)
        {
            if (consumedIndices.Contains(i)) continue;

            var step0 = steps[i];
            var step1 = steps[i + 1];
            var step2 = steps[i + 2];

            // Pattern: MatMul[i] → ReLU[i+1] → MatMul[i+2]
            // where ReLU's input is MatMul[i]'s output, and MatMul[i+2]'s input is ReLU's output
            if (step0.OpName != "TensorMatMul" || step1.OpName != "ReLU" || step2.OpName != "TensorMatMul")
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

            // Verify all 2D
            if (step0.Inputs[0].Rank != 2 || step0.Inputs[1].Rank != 2 ||
                step2.Inputs[1].Rank != 2)
                continue;

            // Check hidden dim: must fit in L1 (max) AND be large enough for L1 benefit (min 128)
            // Below 128, per-op BLAS is faster. Above 128, L1 residency wins.
            int h = step0.Inputs[1]._shape[1]; // output cols of W1
            if (h > Optimization.TensorCodecOptions.Current.DataflowFusionMaxHidden || h < 128)
                continue;

            // Extract dimensions
            var inputTensor = step0.Inputs[0];   // [M, K]
            var w1Tensor = step0.Inputs[1];      // [K, H]
            var w2Tensor = step2.Inputs[1];      // [H, N]
            var outputTensor = step2.OutputBuffer; // [M, N]

            int m = inputTensor._shape[0];
            int k = inputTensor._shape[1];
            int n = w2Tensor._shape[1];

            // Pre-allocate activated intermediate buffer
            var activatedBuffer = TensorAllocator.RentUninitialized<T>(new[] { m, h });

            // Capture for closures
            var capturedInput = inputTensor;
            var capturedW1 = w1Tensor;
            var capturedW2 = w2Tensor;
            var capturedOutput = outputTensor;
            var capturedActivated = activatedBuffer;
            int cm = m, ck = k, ch = h, cn = n;
            var capturedGradMap = gradMap;

            Func<float, float> relu = x => x > 0f ? x : 0f;

            // Pre-allocate backward workspace at compile time (zero alloc during replay)
            var backwardWorkspace = new float[cm * ch]; // grad_h buffer
            var emptyBias = new float[0];

            // Fused forward: single kernel call replaces 3 steps
            fusedForward.Add(eng =>
            {
                var inA = (float[])(object)capturedInput.GetDataArray();
                var w1A = (float[])(object)capturedW1.GetDataArray();
                var w2A = (float[])(object)capturedW2.GetDataArray();
                var outA = (float[])(object)capturedOutput.GetDataArray();
                var actA = (float[])(object)capturedActivated.GetDataArray();
                FusedMultiLayerGemm.FusedGemmActivationGemm(inA, w1A, w2A, outA, actA, cm, ck, ch, cn, relu);
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

                FusedMultiLayerBackward.ComputeGradients(
                    gOutArr, inA, w1A, w2A, actA,
                    gW1, gW2, emptyBias, emptyBias, gIn,
                    cm, ck, ch, cn, FusedMultiLayerBackward.ReLUDerivative,
                    backwardWorkspace);

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
