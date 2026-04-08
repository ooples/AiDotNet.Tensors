using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 4.1: Conv + BatchNorm + optional ReLU fusion for inference.
///
/// At compile time, folds BatchNorm affine parameters (gamma, beta, mean, var)
/// into the Conv2D weights and bias. The fused convolution produces the same
/// output as Conv -> BatchNorm -> ReLU in a single kernel, eliminating two
/// kernel launches and two DRAM round-trips.
///
/// Fusion formula:
///   W_fused[oc] = gamma[oc] / sqrt(var[oc] + eps) * W[oc]
///   b_fused[oc] = gamma[oc] / sqrt(var[oc] + eps) * (b[oc] - mean[oc]) + beta[oc]
///
/// This is inference-only: during training, BN stats are updated and cannot be folded.
/// </summary>
internal sealed class ConvBnFusionPass : ICpuOptimizationPass
{
    public string Name => "ConvBnFusion";

    public bool IsEnabled => TensorCodecOptions.Current.EnableCompilation;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || typeof(T) != typeof(float)) return null;

        var result = new List<CompiledStep<T>>(steps.Length);
        bool anyFused = false;

        for (int i = 0; i < steps.Length; i++)
        {
            // Pattern: Conv2D[i] -> BatchNorm[i+1] -> optional ReLU[i+2]
            if (i + 1 < steps.Length && TryMatchConvBn(steps, i, out var fused, out int consumed))
            {
                if (fused is not null)
                {
                    result.Add(fused);
                    i += consumed - 1; // Skip consumed steps (loop increments i)
                    anyFused = true;
                    continue;
                }
            }
            result.Add(steps[i]);
        }

        return anyFused ? result.ToArray() : null;
    }

    private static bool TryMatchConvBn<T>(
        CompiledStep<T>[] steps, int index,
        out CompiledStep<T>? fused, out int consumed)
    {
        fused = null;
        consumed = 0;

        var conv = steps[index];
        var bn = steps[index + 1];

        // Verify: Conv2D followed by BatchNorm
        if (conv.OpName is not ("Conv2D" or "DepthwiseConv2D"))
            return false;
        if (bn.OpName is not ("BatchNorm" or "InstanceNorm"))
            return false;
        if (conv.Inputs.Length < 2 || bn.Inputs.Length < 5)
            return false;

        // Verify data dependency: conv output feeds BN input
        if (!ReferenceEquals(conv.OutputBuffer, bn.Inputs[0]))
            return false;

        // Check for optional activation after BN
        bool hasActivation = false;
        FusedActivationType activationType = FusedActivationType.None;
        if (index + 2 < steps.Length)
        {
            var activation = steps[index + 2];
            if (ReferenceEquals(bn.OutputBuffer, activation.Inputs[0])
                && IsPointwiseActivation(activation.OpName))
            {
                activationType = MapActivation(activation.OpName);
                if (activationType != FusedActivationType.None)
                    hasActivation = true;
            }
        }

        consumed = hasActivation ? 3 : 2;

        // Extract BN parameters: input, gamma, beta, runningMean, runningVar
        // BN formula: output = gamma * (input - mean) / sqrt(var + eps) + beta
        // After folding into conv:
        //   W_fused = gamma / sqrt(var + eps) * W
        //   b_fused = gamma / sqrt(var + eps) * (b - mean) + beta
        if (typeof(T) != typeof(float)) return false;

        var convInput = conv.Inputs[0];
        var convWeights = conv.Inputs[1];
        var convBias = conv.Inputs.Length > 2 ? conv.Inputs[2] : null;

        // BN inputs: [input, gamma, beta, runningMean, runningVar]
        var gamma = bn.Inputs[1];
        var beta = bn.Inputs[2];
        var runningMean = bn.Inputs[3];
        var runningVar = bn.Inputs[4];

        // Get epsilon from savedState if available
        double epsilon = 1e-5;
        if (bn.SavedState is { Length: > 0 } && bn.SavedState[0] is double eps)
            epsilon = eps;

        // Compute fused weights and bias at compile time
        int outChannels = gamma.Length;
        var gammaData = (float[])(object)gamma.GetDataArray();
        var betaData = (float[])(object)beta.GetDataArray();
        var meanData = (float[])(object)runningMean.GetDataArray();
        var varData = (float[])(object)runningVar.GetDataArray();
        var weightsData = (float[])(object)convWeights.GetDataArray();
        float[]? biasData = convBias is not null ? (float[])(object)convBias.GetDataArray() : null;

        // Compute scale factors: scale[oc] = gamma[oc] / sqrt(var[oc] + eps)
        var scale = new float[outChannels];
        for (int oc = 0; oc < outChannels; oc++)
            scale[oc] = gammaData[oc] / MathF.Sqrt(varData[oc] + (float)epsilon);

        // Fuse into weights: W_fused[oc, ...] = scale[oc] * W[oc, ...]
        int weightsPerChannel = weightsData.Length / outChannels;
        var fusedWeightsData = new float[weightsData.Length];
        for (int oc = 0; oc < outChannels; oc++)
        {
            int offset = oc * weightsPerChannel;
            for (int j = 0; j < weightsPerChannel; j++)
                fusedWeightsData[offset + j] = scale[oc] * weightsData[offset + j];
        }

        // Fuse into bias: b_fused[oc] = scale[oc] * (b[oc] - mean[oc]) + beta[oc]
        var fusedBiasData = new float[outChannels];
        for (int oc = 0; oc < outChannels; oc++)
        {
            float b = biasData is not null ? biasData[oc] : 0f;
            fusedBiasData[oc] = scale[oc] * (b - meanData[oc]) + betaData[oc];
        }

        // Create fused weight and bias tensors
        var fusedWeights = new Tensor<float>(convWeights._shape, new Vector<float>(fusedWeightsData));
        var fusedBias = new Tensor<float>(new[] { outChannels }, new Vector<float>(fusedBiasData));

        // Build fused step: Conv2D with folded BN weights + optional activation
        var finalOutput = hasActivation ? steps[index + 2].OutputBuffer : bn.OutputBuffer;
        var capturedInput = convInput;
        var capturedFusedWeights = (Tensor<T>)(object)fusedWeights;
        var capturedFusedBias = (Tensor<T>)(object)fusedBias;
        var capturedActivation = activationType;

        // Extract original conv parameters from SavedState: [int[] strides, int[] paddings, int[] dilations]
        int[] strides = new[] { 1, 1 };
        int[] paddings = new[] { 0, 0 };
        int[] dilations = new[] { 1, 1 };
        if (conv.SavedState is { Length: >= 3 })
        {
            if (conv.SavedState[0] is int[] s) strides = s;
            if (conv.SavedState[1] is int[] p) paddings = p;
            if (conv.SavedState[2] is int[] d) dilations = d;
        }

        var capturedStrides = strides;
        var capturedPaddings = paddings;
        var capturedDilations = dilations;

        fused = new CompiledStep<T>(
            "FusedConvBnReLU",
            (eng, output) =>
            {
                Tensor<T> result;
                if (capturedActivation != FusedActivationType.None)
                    result = eng.FusedConv2D(capturedInput, capturedFusedWeights, capturedFusedBias,
                        capturedStrides[0], capturedStrides[1],
                        capturedPaddings[0], capturedPaddings[1],
                        capturedDilations[0], capturedDilations[1], capturedActivation);
                else
                    result = eng.Conv2D(capturedInput, capturedFusedWeights,
                        capturedStrides, capturedPaddings, capturedDilations);

                result.AsSpan().CopyTo(output.AsWritableSpan());
            },
            finalOutput,
            new[] { convInput, capturedFusedWeights, capturedFusedBias },
            null, // BackwardFn is incompatible with fused inputs — this is inference-only
            conv.SavedState);

        return true;
    }

    private static bool IsPointwiseActivation(string opName)
    {
        return opName is "ReLU" or "GELU" or "Sigmoid" or "Tanh" or "Swish"
            or "LeakyReLU" or "ELU" or "HardSwish";
    }

    private static FusedActivationType MapActivation(string opName) => opName switch
    {
        "ReLU" => FusedActivationType.ReLU,
        "GELU" => FusedActivationType.GELU,
        "Sigmoid" => FusedActivationType.Sigmoid,
        "Tanh" => FusedActivationType.Tanh,
        "Swish" => FusedActivationType.Swish,
        "LeakyReLU" => FusedActivationType.LeakyReLU,
        "ELU" => FusedActivationType.ELU,
        "HardSwish" => FusedActivationType.HardSwish,
        _ => FusedActivationType.None
    };
}
