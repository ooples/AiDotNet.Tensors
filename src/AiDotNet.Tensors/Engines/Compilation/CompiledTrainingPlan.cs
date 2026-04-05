using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Autodiff;
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
internal sealed class CompiledTrainingPlan<T>
{
    private readonly Action<IEngine>[] _forwardActions;
    private readonly Action<IEngine>[] _backwardActions;
    private readonly Tensor<T> _lossOutput;
    private readonly IEngine _engine;
    private readonly Tensor<T>[] _parameters;
    private readonly Tensor<T>[] _gradients;
    private readonly Tensor<T>[] _preAllocatedGrads;
    private readonly Tensor<T> _lossGradSeed;

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
        Tensor<T>? lossGradDest = null)
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
    }

    internal Tensor<T>[] Gradients => _gradients;
    internal int ForwardStepCount => _forwardActions.Length;
    internal int BackwardStepCount => _backwardActions.Length;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Tensor<T> Step()
    {
        var engine = _engine;

        // Forward: straight-line delegate calls (specialized: direct BLAS, no engine dispatch)
        var fwd = _forwardActions;
        for (int i = 0; i < fwd.Length; i++)
            fwd[i](engine);

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

        return _lossOutput;
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

        // Build forward actions: fused actions first, then remaining unfused steps
        // Build forward actions with specialized direct-BLAS where possible
        var allForwardActions = new List<Action<IEngine>>(fusedForwardActions);
        for (int i = 0; i < forwardSteps.Count; i++)
        {
            if (fusedStepIndices.Contains(i)) continue;
            var step = forwardSteps[i];
            var specialized = TryBuildSpecializedForward(step);
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

            var action = BuildSpecializedBackward(step, gradMap, consumerCount, engine);
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
        // Append fused backward actions (they handle their own gradient routing)
        backwardActions.AddRange(fusedBackwardActions);

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

        // If all backward steps are specialized (overwrite), we can skip gradient zeroing entirely
        int[]? genericGradIndices = genericBackwardCount == 0 ? new int[0] : null;

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
            gradMap.ContainsKey(lossOutput) ? gradMap[lossOutput] : null);
    }

    /// <summary>
    /// Generates a specialized forward delegate that bypasses engine dispatch overhead.
    /// Calls BLAS/SIMD directly into the pre-allocated output buffer.
    /// Eliminates: GraphMode check, tape recording, shape validation, DifferentiableOps.
    /// </summary>
    internal static unsafe Action<IEngine>? TryBuildSpecializedForward(CompiledStep<T> step)
    {
        if (typeof(T) != typeof(float)) return null;

        // MatMul forward: direct BLAS into output buffer
        if (step.OpName == "TensorMatMul" && step.Inputs.Length == 2
            && step.Inputs[0].Rank == 2 && step.Inputs[1].Rank == 2
            && BlasProvider.IsAvailable)
        {
            var inputA = step.Inputs[0];
            var inputB = step.Inputs[1];
            var output = step.OutputBuffer;
            int M = inputA._shape[0], K = inputA._shape[1], N = inputB._shape[1];

            // Cache arrays on first call
            float[]? cA = null, cB = null, cOut = null;

            return eng =>
            {
                cA ??= (float[])(object)inputA.GetDataArray();
                cB ??= (float[])(object)inputB.GetDataArray();
                cOut ??= (float[])(object)output.GetDataArray();
                // No Array.Clear needed — BLAS GEMM with beta=0 overwrites completely
                BlasProvider.TryGemm(M, N, K, cA, 0, K, cB, 0, N, cOut, 0, N);
            };
        }

        // ReLU forward: direct SIMD into output buffer
        if (step.OpName == "ReLU" && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
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
        if (step.OpName == "ReduceSum" && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous
            && step.OutputBuffer.Length == 1)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;
            float[]? cIn = null, cOut = null;

            return eng =>
            {
                cIn ??= (float[])(object)input.GetDataArray();
                cOut ??= (float[])(object)output.GetDataArray();
                float sum = 0;
                int len = input.Length;
                for (int j = 0; j < len; j++) sum += cIn[j];
                cOut[0] = sum;
            };
        }

        // TensorAdd forward: direct SIMD into output buffer
        if (step.OpName == "TensorAdd" && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous)
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            return eng => eng.TensorAddInto(o, a, b);
        }

        // TensorSubtract forward
        if (step.OpName == "TensorSubtract" && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous)
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            return eng => eng.TensorSubtractInto(o, a, b);
        }

        // TensorMultiply forward
        if (step.OpName == "TensorMultiply" && step.Inputs.Length == 2
            && step.Inputs[0].IsContiguous && step.Inputs[1].IsContiguous)
        {
            var a = step.Inputs[0]; var b = step.Inputs[1]; var o = step.OutputBuffer;
            return eng => eng.TensorMultiplyInto(o, a, b);
        }

        // Sigmoid: don't specialize forward — the eager allocating path is faster
        // (SigmoidInto has auto-materialization overhead that exceeds the allocation savings)

        // Tanh forward: direct SIMD
        if (step.OpName == "Tanh" && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => eng.TanhInto(o, inp);
        }

        // Softmax forward: use SoftmaxInto
        if (step.OpName == "Softmax" && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            int axis = step.SavedState != null && step.SavedState.Length > 0 ? (int)step.SavedState[0] : -1;
            return eng => eng.SoftmaxInto(o, inp, axis);
        }

        // TensorNegate forward
        if (step.OpName == "TensorNegate" && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng =>
            {
                MathHelper.GetNumericOperations<T>().Negate(inp.AsSpan(), o.AsWritableSpan());
            };
        }

        // GELU forward: direct SIMD
        if (step.OpName == "GELU" && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => eng.GELUInto(o, inp);
        }

        // LeakyReLU forward: direct SIMD
        if (step.OpName == "LeakyReLU" && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            var alpha = step.SavedState != null && step.SavedState.Length > 0
                ? MathHelper.GetNumericOperations<T>().FromDouble((double)step.SavedState[0])
                : MathHelper.GetNumericOperations<T>().FromDouble(0.01);
            return eng => eng.LeakyReLUInto(o, inp, alpha);
        }

        // Swish forward
        if (step.OpName == "Swish" && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            return eng => { var r = eng.Swish(inp); r.AsSpan().CopyTo(o.AsWritableSpan()); };
        }

        // ELU forward
        if (step.OpName == "ELU" && step.Inputs.Length == 1 && step.Inputs[0].IsContiguous)
        {
            var inp = step.Inputs[0]; var o = step.OutputBuffer;
            double alpha = step.SavedState != null && step.SavedState.Length > 0 ? (double)step.SavedState[0] : 1.0;
            return eng => { var r = eng.ELU(inp, alpha); r.AsSpan().CopyTo(o.AsWritableSpan()); };
        }

        // TensorTranspose: don't specialize (no Into variant, allocate+copy is slower)

        // TensorDivide: no Into specialization (Pin overhead in Into variants
        // exceeds the allocation savings — eager path uses fixed+GetDataArray which is faster)

        // BatchMatMul/BroadcastAdd: don't specialize (allocate+copy is slower than eager)
        // These need proper Into variants to be worth specializing

        // Conv2D forward: use Conv2DInto to write directly into output
        if (step.OpName == "Conv2D" && step.Inputs.Length == 2)
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

        // MaxPool2D: don't specialize (no Into variant, allocate+copy is slower)

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
        IEngine engine)
    {
        if (typeof(T) != typeof(float)) return null;

        // MatMul backward: dA = dC @ B^T, dB = A^T @ dC — transposed BLAS, zero alloc
        if (step.OpName == "TensorMatMul" && step.Inputs.Length == 2
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
            bool accumA = consumerCount.ContainsKey(inputA) && consumerCount[inputA] > 1;
            bool accumB = consumerCount.ContainsKey(inputB) && consumerCount[inputB] > 1;

            // Cache array references on first call to avoid GetDataArray() overhead per step
            float[]? cachedDC = null, cachedA = null, cachedB = null, cachedDestA = null, cachedDestB = null;

            return eng =>
            {
                cachedDC ??= (float[])(object)gradOut.GetDataArray();
                cachedA ??= (float[])(object)inputA.GetDataArray();
                cachedB ??= (float[])(object)inputB.GetDataArray();
                cachedDestA ??= (float[])(object)gradA.GetDataArray();
                cachedDestB ??= (float[])(object)gradB.GetDataArray();

                // dA = dC @ B^T — direct BLAS into pre-allocated buffer
                BlasProvider.TryGemmEx(M, K, N, cachedDC, 0, N, false, cachedB, 0, N, true, cachedDestA, 0, K);
                // dB = A^T @ dC — direct BLAS into pre-allocated buffer
                BlasProvider.TryGemmEx(K, N, M, cachedA, 0, K, true, cachedDC, 0, N, false, cachedDestB, 0, N);

                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // ReLU backward: mask = input > 0, grad = gradOut * mask — SIMD, zero alloc
        if (step.OpName == "ReLU" && step.Inputs.Length == 1
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

        // Add backward: gradA += gradOut, gradB += gradOut — just copy, zero alloc
        if (step.OpName == "TensorAdd" && step.Inputs.Length == 2)
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
                // Add backward: both inputs get the output gradient
                gradOut.AsSpan().CopyTo(gradA.AsWritableSpan());
                gradOut.AsSpan().CopyTo(gradB.AsWritableSpan());
                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // Subtract backward: gradA += gradOut, gradB -= gradOut
        if (step.OpName == "TensorSubtract" && step.Inputs.Length == 2)
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
                gradOut.AsSpan().CopyTo(gradA.AsWritableSpan());
                MathHelper.GetNumericOperations<T>().Negate(gradOut.AsSpan(), gradB.AsWritableSpan());
                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // Multiply backward: gradA = gradOut * B, gradB = gradOut * A — SIMD, zero alloc
        if (step.OpName == "TensorMultiply" && step.Inputs.Length == 2
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
                eng.TensorMultiplyInto(gradA, gradOut, inputB);
                eng.TensorMultiplyInto(gradB, gradOut, inputA);
                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // ReduceSum backward: broadcast scalar gradient to all elements
        if (step.OpName == "ReduceSum" && step.Inputs.Length == 1)
        {
            var input = step.Inputs[0];
            var output = step.OutputBuffer;

            if (!gradMap.ContainsKey(output) || !gradMap.ContainsKey(input))
                return null;

            var gradOut = gradMap[output];
            var gradIn = gradMap[input];
            int length = input.Length;

            return eng =>
            {
                // ReduceSum backward: each input element gets the same scalar gradient
                gradIn.AsWritableSpan().Fill(gradOut.AsSpan()[0]);
                input.Grad = gradIn;
            };
        }

        // Sigmoid backward: grad * sigmoid(out) * (1 - sigmoid(out))
        if (step.OpName == "Sigmoid" && step.Inputs.Length == 1)
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
        if (step.OpName == "Tanh" && step.Inputs.Length == 1)
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
        if (step.OpName == "TensorNegate" && step.Inputs.Length == 1)
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
        if (step.OpName == "TensorTranspose" && step.Inputs.Length == 1)
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
        if (step.OpName == "TensorDivide" && step.Inputs.Length == 2
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
        if (step.OpName == "Softmax" && step.Inputs.Length == 1)
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
        if (step.OpName == "GELU" && step.Inputs.Length == 1)
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
        if (step.OpName == "LeakyReLU" && step.Inputs.Length == 1)
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
        if (step.OpName == "Swish" && step.Inputs.Length == 1)
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
        if (step.OpName == "ELU" && step.Inputs.Length == 1)
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
