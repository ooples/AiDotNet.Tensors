using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 4.3: Element-wise kernel fusion — merge consecutive pointwise operations
/// on the same tensor into a single pass over memory.
///
/// Example: sigmoid(x) * x (Swish) normally requires two passes:
///   Pass 1: compute sigmoid(x) → temp
///   Pass 2: compute temp * x → output
///
/// Fused: single pass computing sigmoid(x[i]) * x[i] per element, halving memory bandwidth.
///
/// Only fuses chains where each op consumes the previous op's output (linear chain),
/// all tensors have the same shape, and ops are pointwise (element-independent).
/// </summary>
internal sealed class PointwiseFusionPass : ICpuOptimizationPass
{
    public string Name => "PointwiseFusion";

    public bool IsEnabled => TensorCodecOptions.Current.EnablePointwiseFusion;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || typeof(T) != typeof(float) || steps.Length < 2) return null;

        var result = new List<CompiledStep<T>>(steps.Length);
        bool anyFused = false;

        // Readers per tensor: a chain may only swallow an intermediate the next link is the sole reader of. Fusing
        // through one that another step also reads left that step reading a tensor nothing writes any more (measured:
        // Sigmoid -> Tanh fused while a reshape of the Sigmoid output fed the next layer, which then read zeros).
        var readers = new Dictionary<Tensor<T>, int>(ReaderKeyComparer<Tensor<T>>.Instance);
        foreach (var step in steps)
            foreach (var input in step.Inputs)
                readers[input] = readers.TryGetValue(input, out var r) ? r + 1 : 1;

        int i = 0;
        while (i < steps.Length)
        {
            // Try to find a chain of consecutive pointwise unary ops
            if (IsPointwiseUnary(steps[i].OpName) && steps[i].Inputs.Length == 1)
            {
                int chainStart = i;
                int chainEnd = i;

                // Extend chain while next step consumes current output and is also pointwise
                while (chainEnd + 1 < steps.Length
                    && IsPointwiseUnary(steps[chainEnd + 1].OpName)
                    && steps[chainEnd + 1].Inputs.Length == 1
                    && ReferenceEquals(steps[chainEnd].OutputBuffer, steps[chainEnd + 1].Inputs[0])
                    && readers.TryGetValue(steps[chainEnd].OutputBuffer, out var intermediateReaders)
                    && intermediateReaders == 1)
                {
                    chainEnd++;
                }

                if (chainEnd > chainStart)
                {
                    // We have a chain of 2+ pointwise ops — fuse them
                    var fusedStep = FuseChain(steps, chainStart, chainEnd);
                    if (fusedStep is not null)
                    {
                        result.Add(fusedStep);
                        i = chainEnd + 1;
                        anyFused = true;
                        continue;
                    }
                }
            }

            result.Add(steps[i]);
            i++;
        }

        return anyFused ? result.ToArray() : null;
    }

    private static CompiledStep<T>? FuseChain<T>(CompiledStep<T>[] steps, int start, int end)
    {
        // Collect the chain of ops
        var chainOps = new List<string>();
        for (int i = start; i <= end; i++)
            chainOps.Add(steps[i].OpName);

        var firstInput = steps[start].Inputs[0];
        var lastOutput = steps[end].OutputBuffer;
        string fusedName = "Fused_" + string.Join("_", chainOps);

        // Try hardcoded SIMD kernel first (zero delegate overhead)
        if (typeof(T) == typeof(float) && firstInput.IsContiguous
            && Simd.FusedKernels.TryGetFusedKernel(chainOps.ToArray(), out var kernelName))
        {
            var capturedInput = firstInput;
            var capturedKernel = kernelName;

            return new CompiledStep<T>(
                "HardcodedFused_" + capturedKernel,
                (eng, output) =>
                {
                    var iMem = ((Tensor<float>)(object)capturedInput).Data;
                    var oMem = ((Tensor<float>)(object)output).Data;
                    using var pinI = iMem.Pin();
                    using var pinO = oMem.Pin();
                    unsafe
                    {
                        float* pIn = (float*)pinI.Pointer;
                        float* pOut = (float*)pinO.Pointer;
                        int len = capturedInput.Length;
                        switch (capturedKernel)
                        {
                            case "Swish": Simd.FusedKernels.SwishUnsafe(pIn, pOut, len); break;
                            case "GELU": Simd.FusedKernels.GeluUnsafe(pIn, pOut, len); break;
                            case "Mish": Simd.FusedKernels.MishUnsafe(pIn, pOut, len); break;
                            default: Simd.FusedPointwise.ApplyFused(pIn, pOut, len, x => x); break;
                        }
                    }
                },
                lastOutput,
                new[] { firstInput },
                null,
                steps[end].SavedState);
        }

        // Fallback: delegate composition (still single-pass but with delegate overhead)
        if (typeof(T) == typeof(float) && firstInput.IsContiguous)
        {
            var fusedDelegate = Simd.FusedPointwise.BuildFusedDelegate(chainOps.ToArray());
            if (fusedDelegate is not null)
            {
                var capturedInput = firstInput;
                var capturedFused = fusedDelegate;

                return new CompiledStep<T>(
                    fusedName,
                    (eng, output) =>
                    {
                        var iMem = ((Tensor<float>)(object)capturedInput).Data;
                        var oMem = ((Tensor<float>)(object)output).Data;
                        using var pinI = iMem.Pin();
                        using var pinO = oMem.Pin();
                        unsafe
                        {
                            Simd.FusedPointwise.ApplyFused(
                                (float*)pinI.Pointer, (float*)pinO.Pointer,
                                capturedInput.Length, capturedFused);
                        }
                    },
                    lastOutput,
                    new[] { firstInput },
                    null, // Backward incompatible with fused inputs — inference-only
                    steps[end].SavedState);
            }
        }

        // Fallback: dispatch fusion (sequential execution, fewer dispatch steps)
        var capturedSteps = new CompiledStep<T>[end - start + 1];
        Array.Copy(steps, start, capturedSteps, 0, capturedSteps.Length);

        return new CompiledStep<T>(
            fusedName,
            (eng, output) =>
            {
                foreach (var step in capturedSteps)
                    step.Execute(eng, step.OutputBuffer);
            },
            lastOutput,
            new[] { firstInput },
            null,
            steps[end].SavedState);
    }

    private static bool IsPointwiseUnary(string opName)
    {
        // Use OpType enum for reliable matching (handles "Exp" vs "TensorExp" differences)
        var opType = OpTypeParser.Parse(opName);
        return IsPointwiseUnaryOpType(opType);
    }

    private static bool IsPointwiseUnaryOpType(OpType opType)
    {
        return opType is OpType.ReLU or OpType.Sigmoid or OpType.Tanh or OpType.GELU
            or OpType.Swish or OpType.Mish or OpType.TensorExp or OpType.TensorLog
            or OpType.TensorAbs or OpType.TensorSqrt or OpType.TensorNegate or OpType.Sign
            or OpType.Softplus or OpType.HardSwish or OpType.HardSigmoid or OpType.ELU
            or OpType.SELU or OpType.LeakyReLU or OpType.Reciprocal or OpType.Floor
            or OpType.Ceiling or OpType.Round or OpType.Clamp;
    }

    private sealed class ReaderKeyComparer<TItem> : IEqualityComparer<TItem> where TItem : class
    {
        public static readonly ReaderKeyComparer<TItem> Instance = new();
        public bool Equals(TItem? x, TItem? y) => ReferenceEquals(x, y);
        public int GetHashCode(TItem obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }
}
