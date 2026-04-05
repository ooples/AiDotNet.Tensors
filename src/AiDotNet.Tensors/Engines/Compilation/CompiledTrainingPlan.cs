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
    private readonly Tensor<T> _lossGrad;

    private CompiledTrainingPlan(
        Action<IEngine>[] forwardActions,
        Action<IEngine>[] backwardActions,
        Tensor<T> lossOutput,
        IEngine engine,
        Tensor<T>[] parameters,
        Tensor<T>[] preAllocatedGrads,
        Tensor<T>[] gradients,
        Tensor<T> lossGradSeed,
        Tensor<T> lossGrad)
    {
        _forwardActions = forwardActions;
        _backwardActions = backwardActions;
        _lossOutput = lossOutput;
        _engine = engine;
        _parameters = parameters;
        _preAllocatedGrads = preAllocatedGrads;
        _gradients = gradients;
        _lossGradSeed = lossGradSeed;
        _lossGrad = lossGrad;
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

        // Re-seed loss gradient into the loss output's gradient buffer (not positional)
        _lossGradSeed.AsSpan().CopyTo(_lossGrad.AsWritableSpan());

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

        // Build forward actions (capture step + output buffer)
        var forwardActions = new Action<IEngine>[forwardSteps.Count];
        for (int i = 0; i < forwardSteps.Count; i++)
        {
            var step = forwardSteps[i];
            var output = step.OutputBuffer;
            var exec = step.Execute;
            forwardActions[i] = eng => exec(eng, output);
        }

        // Build specialized backward actions
        var backwardActions = new List<Action<IEngine>>();
        for (int i = forwardSteps.Count - 1; i >= 0; i--)
        {
            var step = forwardSteps[i];
            if (step.BackwardFn == null) continue;

            var action = BuildSpecializedBackward(step, gradMap, consumerCount, engine);
            if (action != null)
                backwardActions.Add(action);
            else
            {
                // Fallback: use generic backward function with pre-allocated grad accumulator
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

        // Loss gradient seed — use the loss output tensor's gradient buffer explicitly
        var numOps = MathHelper.GetNumericOperations<T>();
        var lossOutput = forwardSteps.Count > 0
            ? forwardSteps[forwardSteps.Count - 1].OutputBuffer
            : new Tensor<T>(new int[] { 1 });
        var lossGradSeed = TensorAllocator.RentUninitialized<T>(lossOutput._shape);
        lossGradSeed.AsWritableSpan().Fill(numOps.One);

        // Look up the loss output's gradient buffer by tensor identity, not array position
        var lossGrad = gradMap.ContainsKey(lossOutput) ? gradMap[lossOutput] : allGrads[allGrads.Count - 1];

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
            lossGradSeed,
            lossGrad);
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
                if (!BlasProvider.TryGemmEx(M, K, N, dCArr, 0, N, false, bArr, 0, N, true, destA, 0, K))
                {
                    // BLAS unavailable — fallback to engine matmul
                    var fallbackA = eng.TensorMatMul(gradOut, eng.TensorTranspose(inputB));
                    fallbackA.AsSpan().CopyTo(gradA.AsWritableSpan());
                }
                // dB = A^T @ dC — direct BLAS into pre-allocated buffer
                if (!BlasProvider.TryGemmEx(K, N, M, aArr, 0, K, true, dCArr, 0, N, false, destB, 0, N))
                {
                    var fallbackB = eng.TensorMatMul(eng.TensorTranspose(inputA), gradOut);
                    fallbackB.AsSpan().CopyTo(gradB.AsWritableSpan());
                }

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

        // Add backward: gradA += gradOut, gradB += gradOut — accumulate for multi-consumer safety
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
                // Accumulate (add) instead of overwrite to handle multi-consumer tensors (e.g. x+x)
                eng.TensorAddInPlace(gradA, gradOut);
                eng.TensorAddInPlace(gradB, gradOut);
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
                // Accumulate for multi-consumer safety
                eng.TensorAddInPlace(gradA, gradOut);
                eng.TensorSubtractInPlace(gradB, gradOut);
                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        // Multiply backward: gradA += gradOut * B, gradB += gradOut * A
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
            bool accumA = consumerCount.ContainsKey(inputA) && consumerCount[inputA] > 1;
            bool accumB = consumerCount.ContainsKey(inputB) && consumerCount[inputB] > 1;

            return eng =>
            {
                if (accumA)
                {
                    // Accumulate: compute temp = gradOut * B, then gradA += temp
                    var temp = eng.TensorMultiply(gradOut, inputB);
                    eng.TensorAddInPlace(gradA, temp);
                }
                else
                {
                    eng.TensorMultiplyInto(gradA, gradOut, inputB);
                }
                if (accumB)
                {
                    var temp = eng.TensorMultiply(gradOut, inputA);
                    eng.TensorAddInPlace(gradB, temp);
                }
                else
                {
                    eng.TensorMultiplyInto(gradB, gradOut, inputA);
                }
                inputA.Grad = gradA;
                inputB.Grad = gradB;
            };
        }

        return null; // Not specialized — use generic fallback
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
