#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Real-device execution tests: launch the direct-PTX convolution kernels on the
/// GPU and compare the downloaded result against a CPU fp64 reference. These are
/// the first tests that actually assemble and run the hand-emitted PTX (the
/// static tests only inspect the PTX string), so they are the correctness gate
/// for the v1 golden slice and the shared-memory tiled GEMM. Skips cleanly when
/// no CUDA device is present.
/// </summary>
public sealed class DirectPtxConvolutionGpuExecutionTests
{
    private const float Tolerance = 2e-4f;

    private static float DeterministicInput(int i) => ((i % 7) - 3) * 0.1f;
    private static float DeterministicWeight(int i) => ((i % 5) - 2) * 0.05f;
    private static float DeterministicBias(int k) => ((k % 3) - 1) * 0.1f;

    [Fact]
    public void V1_ExactN1C64H16W16K64_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int c = PtxFusedConv2DNchwK1Kernel.InputChannels; // 64
        const int hw = PtxFusedConv2DNchwK1Kernel.SpatialElements; // 256
        const int k = PtxFusedConv2DNchwK1Kernel.OutputChannels; // 64

        var input = new float[c * hw];
        var weights = new float[k * c];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);

        float[] expected = ReferenceConv1x1(input, weights, bias, batch: 1, k, c, hw);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return; // only the SM86 specialization exists

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            float[] actual = LaunchV1(runtime, input, weights, bias);
            AssertClose(expected, actual);
        }
        finally
        {
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    [Fact]
    public void Tiled_SmallCleanShape_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // Small contract divisible by the tile so no boundary predicate is needed.
        const int n = 2, k = 32, cch = 32, hw = 64, tile = 16;
        var shape = new Conv2DTiledShape(n, k, cch, hw, tile);

        var input = new float[n * cch * hw];
        var weights = new float[k * cch];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);

        float[] expected = ReferenceConv1x1(input, weights, bias, n, k, cch, hw);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        // No release cubin is committed yet, so allow the experiment JIT fallback.
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            float[] actual = LaunchTiled(runtime, shape, input, weights, bias);
            AssertClose(expected, actual);
        }
        finally
        {
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    private static unsafe float[] LaunchV1(
        DirectPtxRuntime runtime, float[] input, float[] weights, float[] bias)
    {
        using var kernel = new PtxFusedConv2DNchwK1Kernel(runtime);
        using var dInput = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.InputBytes);
        using var dWeights = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.WeightBytes);
        using var dBias = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.OutputBytes);
        dInput.Upload<float>(input);
        dWeights.Upload<float>(weights);
        dBias.Upload<float>(bias);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(dWeights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(dOutput, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[PtxFusedConv2DNchwK1Kernel.OutputElements];
        dOutput.Download<float>(actual);
        return actual;
    }

    private static float[] LaunchTiled(
        DirectPtxRuntime runtime, Conv2DTiledShape shape,
        float[] input, float[] weights, float[] bias)
    {
        using var kernel = new PtxConv2DNchwK1TiledKernel(runtime, shape);
        using var dInput = runtime.AllocateBytes((nuint)shape.InputBytes);
        using var dWeights = runtime.AllocateBytes((nuint)shape.WeightBytes);
        using var dBias = runtime.AllocateBytes((nuint)shape.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)shape.OutputBytes);
        dInput.Upload<float>(input);
        dWeights.Upload<float>(weights);
        dBias.Upload<float>(bias);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(dWeights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(dOutput, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[shape.Batch * shape.OutputChannels * shape.Spatial];
        dOutput.Download<float>(actual);
        return actual;
    }

    // O[n,k,hw] = ReLU(bias[k] + sum_c X[n,c,hw] * W[k,c]) with fp64 accumulation.
    private static float[] ReferenceConv1x1(
        float[] input, float[] weights, float[] bias, int batch, int k, int c, int hw)
    {
        var output = new float[batch * k * hw];
        for (int n = 0; n < batch; n++)
            for (int oc = 0; oc < k; oc++)
                for (int p = 0; p < hw; p++)
                {
                    double acc = bias[oc];
                    for (int ic = 0; ic < c; ic++)
                        acc += (double)input[(n * c + ic) * hw + p] * weights[oc * c + ic];
                    output[(n * k + oc) * hw + p] = (float)Math.Max(acc, 0.0);
                }
        return output;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxErr = 0f;
        int worst = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = Math.Abs(expected[i] - actual[i]);
            if (e > maxErr) { maxErr = e; worst = i; }
        }
        Assert.True(maxErr <= Tolerance,
            $"max abs error {maxErr:E3} > {Tolerance:E3} at index {worst} " +
            $"(expected {(worst >= 0 ? expected[worst] : 0)}, actual {(worst >= 0 ? actual[worst] : 0)})");
    }
}
#endif
