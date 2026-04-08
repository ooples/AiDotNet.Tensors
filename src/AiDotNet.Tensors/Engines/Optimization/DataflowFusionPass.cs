using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase B: Dataflow fusion pass that pattern-matches consecutive MatMul → Activation → MatMul
/// chains and replaces them with a single fused multi-layer GEMM kernel.
///
/// For input[M,K] → W1[K,H] → ReLU → W2[H,N]:
/// Traditional: 3 kernel dispatches with 2 store/load round-trips to DRAM.
/// Fused: 1 kernel dispatch, inter-layer tile [Mr,H] stays in L1 cache.
///
/// Constraint: hidden dimension H must fit in L1 (H ≤ DataflowFusionMaxHidden, default 512).
/// Result is bitwise identical to unfused execution.
/// </summary>
internal sealed class DataflowFusionPass : ICpuOptimizationPass
{
    public string Name => "DataflowFusion";

    public bool IsEnabled => TensorCodecOptions.Current.EnableDataflowFusion;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || typeof(T) != typeof(float)) return null;

        var options = TensorCodecOptions.Current;
        var result = new List<CompiledStep<T>>(steps.Length);
        bool anyFused = false;

        for (int i = 0; i < steps.Length; i++)
        {
            // Pattern: MatMul[i] → Activation[i+1] → MatMul[i+2]
            if (i + 2 < steps.Length && TryMatchFusionPattern(steps, i, options, out var fused))
            {
                if (fused != null)
                {
                    result.Add(fused);
                    i += 2; // Skip the 2 consumed steps
                    anyFused = true;
                    continue;
                }
            }
            result.Add(steps[i]);
        }

        return anyFused ? result.ToArray() : null;
    }

    private static bool TryMatchFusionPattern<T>(
        CompiledStep<T>[] steps, int index,
        TensorCodecOptions options,
        out CompiledStep<T>? fused)
    {
        fused = null;
        var matmul1 = steps[index];
        var activation = steps[index + 1];
        var matmul2 = steps[index + 2];

        // Verify pattern: MatMul → pointwise activation → MatMul
        if (matmul1.OpName != "TensorMatMul" || matmul2.OpName != "TensorMatMul")
            return false;

        // Activation must be a pointwise op (ReLU, GELU, Tanh, etc.)
        if (!IsPointwiseActivation(activation.OpName))
            return false;

        // Verify correct input counts
        if (matmul1.Inputs.Length != 2 || matmul2.Inputs.Length != 2)
            return false;
        if (activation.Inputs.Length < 1)
            return false;

        // Verify data dependency chain: matmul1.output feeds activation, activation.output feeds matmul2
        if (!ReferenceEquals(matmul1.OutputBuffer, activation.Inputs[0]))
            return false;
        if (!ReferenceEquals(activation.OutputBuffer, matmul2.Inputs[0]))
            return false;

        // Verify activation type is supported by the fused kernel
        var activationType = MapOpNameToActivationType(activation.OpName);
        if (activationType == FusedActivationType.None)
            return false;

        // Extract dimensions
        var input = matmul1.Inputs[0]; // [M, K]
        var w1 = matmul1.Inputs[1];    // [K, H]
        var w2 = matmul2.Inputs[1];    // [H, N]

        if (input.Rank != 2 || w1.Rank != 2 || w2.Rank != 2)
            return false;

        int m = input._shape[0];
        int k = input._shape[1];
        int h = w1._shape[1];
        int n = w2._shape[1];

        // Check hidden dim fits in L1
        if (!TensorCodecOptimizer.CanFuseDataflow(h))
            return false;

        // Get activation function delegate (activationType already validated above)
        var activationFn = CpuFusedOperations.GetFloatActivation(activationType);

        var capturedInput = input;
        var capturedW1 = w1;
        var capturedW2 = w2;
        var capturedOutput = matmul2.OutputBuffer;
        int cm = m, ck = k, ch = h, cn = n;
        var capturedActivationFn = activationFn;

        fused = new CompiledStep<T>(
            "FusedTwoLayer",
            (eng, output) =>
            {
                var inArr = (float[])(object)capturedInput.GetDataArray();
                var w1Arr = (float[])(object)capturedW1.GetDataArray();
                var w2Arr = (float[])(object)capturedW2.GetDataArray();
                var outArr = (float[])(object)output.GetDataArray();
                var activated = new float[cm * ch];
                FusedMultiLayerGemm.FusedGemmActivationGemm(
                    inArr, w1Arr, w2Arr, outArr, activated, cm, ck, ch, cn, capturedActivationFn);
            },
            capturedOutput,
            new[] { input, w1, w2 },
            matmul2.BackwardFn,
            matmul2.SavedState);

        return true;
    }

    private static bool IsPointwiseActivation(string opName)
    {
        return MapOpNameToActivationType(opName) != FusedActivationType.None;
    }

    private static FusedActivationType MapOpNameToActivationType(string opName) => opName switch
    {
        "ReLU" => FusedActivationType.ReLU,
        "GELU" => FusedActivationType.GELU,
        "Sigmoid" => FusedActivationType.Sigmoid,
        "Tanh" => FusedActivationType.Tanh,
        "Swish" => FusedActivationType.Swish,
        "LeakyReLU" => FusedActivationType.LeakyReLU,
        "ELU" => FusedActivationType.ELU,
        "Mish" => FusedActivationType.Mish,
        "Softplus" => FusedActivationType.Softplus,
        "HardSwish" => FusedActivationType.HardSwish,
        _ => FusedActivationType.None
    };
}
