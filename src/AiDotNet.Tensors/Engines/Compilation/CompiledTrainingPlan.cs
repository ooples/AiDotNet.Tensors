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

    private CompiledTrainingPlan(
        Action<IEngine>[] forwardActions,
        Action<IEngine>[] backwardActions,
        Tensor<T> lossOutput,
        IEngine engine,
        Tensor<T>[] parameters,
        Tensor<T>[] preAllocatedGrads,
        Tensor<T>[] gradients,
        Tensor<T> lossGradSeed)
    {
        _forwardActions = forwardActions;
        _backwardActions = backwardActions;
        _lossOutput = lossOutput;
        _engine = engine;
        _parameters = parameters;
        _preAllocatedGrads = preAllocatedGrads;
        _gradients = gradients;
        _lossGradSeed = lossGradSeed;
    }

    internal Tensor<T>[] Gradients => _gradients;
    internal int ForwardStepCount => _forwardActions.Length;
    internal int BackwardStepCount => _backwardActions.Length;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Tensor<T> Step()
    {
        var engine = _engine;

        // Forward: straight-line delegate calls
        var fwd = _forwardActions;
        for (int i = 0; i < fwd.Length; i++)
            fwd[i](engine);

        // Zero all gradient buffers
        for (int i = 0; i < _preAllocatedGrads.Length; i++)
            _preAllocatedGrads[i].AsWritableSpan().Clear();

        // Re-seed loss gradient
        _lossGradSeed.AsSpan().CopyTo(_preAllocatedGrads[_preAllocatedGrads.Length - 1].AsWritableSpan());

        // Backward: specialized delegates that write directly into pre-allocated buffers
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
        var allForwardActions = new List<Action<IEngine>>(fusedForwardActions);
        for (int i = 0; i < forwardSteps.Count; i++)
        {
            if (fusedStepIndices.Contains(i)) continue;
            var step = forwardSteps[i];
            var output = step.OutputBuffer;
            var exec = step.Execute;
            allForwardActions.Add(eng => exec(eng, output));
        }
        var forwardActions = allForwardActions.ToArray();

        // Build backward actions: specialized per-step + fused backward for fused groups
        var backwardActions = new List<Action<IEngine>>();
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

        return new CompiledTrainingPlan<T>(
            forwardActions,
            backwardActions.ToArray(),
            lossOutput,
            engine,
            parameters,
            allGrads.ToArray(),
            gradients,
            lossGradSeed);
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

            return eng =>
            {
                var dCArr = (float[])(object)gradOut.GetDataArray();
                var aArr = (float[])(object)inputA.GetDataArray();
                var bArr = (float[])(object)inputB.GetDataArray();
                var destA = (float[])(object)gradA.GetDataArray();
                var destB = (float[])(object)gradB.GetDataArray();

                // dA = dC @ B^T — direct BLAS into pre-allocated buffer
                BlasProvider.TryGemmEx(M, K, N, dCArr, 0, N, false, bArr, 0, N, true, destA, 0, K);
                // dB = A^T @ dC — direct BLAS into pre-allocated buffer
                BlasProvider.TryGemmEx(K, N, M, aArr, 0, K, true, dCArr, 0, N, false, destB, 0, N);

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

            // Check hidden dim fits in L1
            int h = step0.Inputs[1]._shape[1]; // output cols of W1
            if (h > Optimization.TensorCodecOptions.Current.DataflowFusionMaxHidden)
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

            // Fused forward: single kernel call replaces 3 steps
            fusedForward.Add(eng =>
            {
                var inArr = (float[])(object)capturedInput.GetDataArray();
                var w1Arr = (float[])(object)capturedW1.GetDataArray();
                var w2Arr = (float[])(object)capturedW2.GetDataArray();
                var outArr = (float[])(object)capturedOutput.GetDataArray();
                var actArr = (float[])(object)capturedActivated.GetDataArray();
                FusedMultiLayerGemm.FusedGemmActivationGemm(inArr, w1Arr, w2Arr, outArr, actArr, cm, ck, ch, cn, relu);
            });

            // Fused backward: transposed BLAS for all gradients, zero transpose alloc
            fusedBackward.Add(eng =>
            {
                var gradOut = capturedGradMap.ContainsKey(capturedOutput)
                    ? capturedGradMap[capturedOutput] : null;
                if (gradOut == null) return;

                var inArr = (float[])(object)capturedInput.GetDataArray();
                var w1Arr = (float[])(object)capturedW1.GetDataArray();
                var w2Arr = (float[])(object)capturedW2.GetDataArray();
                var actArr = (float[])(object)capturedActivated.GetDataArray();
                var gOutArr = (float[])(object)gradOut.GetDataArray();

                // Pre-allocate or reuse gradient buffers from gradMap
                var gW1 = capturedGradMap.ContainsKey(capturedW1)
                    ? (float[])(object)capturedGradMap[capturedW1].GetDataArray() : new float[ck * ch];
                var gW2 = capturedGradMap.ContainsKey(capturedW2)
                    ? (float[])(object)capturedGradMap[capturedW2].GetDataArray() : new float[ch * cn];
                var gInput = capturedGradMap.ContainsKey(capturedInput)
                    ? (float[])(object)capturedGradMap[capturedInput].GetDataArray() : new float[cm * ck];

                FusedMultiLayerBackward.ComputeGradients(
                    gOutArr, inArr, w1Arr, w2Arr, actArr,
                    gW1, gW2, new float[0], new float[0], gInput,
                    cm, ck, ch, cn, FusedMultiLayerBackward.ReLUDerivative);

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
