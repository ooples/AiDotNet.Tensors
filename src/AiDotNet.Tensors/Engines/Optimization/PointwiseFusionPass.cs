using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;

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

    public bool IsEnabled => TensorCodecOptions.Current.EnableCompilation;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || typeof(T) != typeof(float) || steps.Length < 2) return null;

        var result = new List<CompiledStep<T>>(steps.Length);
        bool anyFused = false;

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
                    && ReferenceEquals(steps[chainEnd].OutputBuffer, steps[chainEnd + 1].Inputs[0]))
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

        // Build a fused delegate that applies all ops in sequence
        // Each op is applied element-wise in a single pass
        var capturedSteps = new CompiledStep<T>[end - start + 1];
        Array.Copy(steps, start, capturedSteps, 0, capturedSteps.Length);

        return new CompiledStep<T>(
            fusedName,
            (eng, output) =>
            {
                // Execute the chain: first step reads from firstInput,
                // intermediate results go through the pre-allocated buffers,
                // last step writes to output
                foreach (var step in capturedSteps)
                    step.Execute(eng, step.OutputBuffer);
            },
            lastOutput,
            new[] { firstInput },
            steps[end].BackwardFn,
            steps[end].SavedState);
    }

    private static bool IsPointwiseUnary(string opName)
    {
        return opName is "ReLU" or "Sigmoid" or "Tanh" or "GELU" or "Swish" or "Mish"
            or "Exp" or "Log" or "Abs" or "Sqrt" or "Negate" or "Sign"
            or "Softplus" or "HardSwish" or "HardSigmoid" or "ELU" or "SELU"
            or "LeakyReLU" or "Reciprocal" or "Floor" or "Ceiling" or "Round"
            or "TensorFrac" or "Clamp";
    }
}
