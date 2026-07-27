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

    [Fact]
    public void RegBlocked_SmallCleanShape_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // K=64 (BM64), C=64 (BK16), HW=64 (BN64), N=2. 256 threads, 4x4 micro-tile.
        const int n = 2, k = 64, cch = 64, hw = 64;
        var shape = new Conv2DRegBlockShape(n, k, cch, hw, blockM: 64, blockN: 64, blockK: 16, threadM: 4, threadN: 4);

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

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            float[] actual = LaunchRegBlocked(runtime, shape, input, weights, bias);
            AssertClose(expected, actual);
        }
        finally
        {
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    [Fact]
    public void RegBlocked_ResNetC64_ExactShape_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // The exact shape the >=1.10x-vs-cuDNN win is claimed on: N32/C64/56x56/K64.
        const int n = 32, k = 64, cch = 64, hw = 3136;
        var shape = new Conv2DRegBlockShape(n, k, cch, hw, 64, 64, 16, 4, 4);

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

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            float[] actual = LaunchRegBlocked(runtime, shape, input, weights, bias);
            AssertClose(expected, actual);
        }
        finally
        {
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    [Fact]
    public void RegBlocked_ProductionEmbeddedCubin_NoOverride_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // The promoted c64 config, loaded via the EMBEDDED committed cubin with
        // the experiment JIT fallback OFF — i.e. the real production path.
        const int n = 32, k = 64, cch = 64, hw = 3136;
        var shape = new Conv2DRegBlockShape(n, k, cch, hw, 64, 64, 16, 4, 4);

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

        // Override OFF: construction must resolve the embedded cubin, not JIT.
        Assert.False(DirectPtxFeatureGate.ConvolutionExperimentOverride);
        float[] actual = LaunchRegBlocked(runtime, shape, input, weights, bias);
        AssertClose(expected, actual);
    }

    private static float[] LaunchRegBlocked(
        DirectPtxRuntime runtime, Conv2DRegBlockShape shape,
        float[] input, float[] weights, float[] bias)
    {
        using var kernel = new PtxConv2DNchwK1RegBlockedKernel(runtime, shape);
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

    [Fact]
    public void Winograd3x3_SmallShape_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // N2/C4/H8/W8/K4 -> 128 tiles (one block). 3x3 stride-1 same-padded.
        const int n = 2, cch = 4, h = 8, w = 8, k = 4;
        var shape = new Conv2DWinogradShape(n, cch, h, w, k);

        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);

        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            float[] actual = LaunchWinograd(runtime, shape, input, weights, bias);
            AssertClose(expected, actual, 2e-3f); // Winograd rounds differently than direct
        }
        finally
        {
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    [Fact]
    public void Winograd3x3_FilterPretransformed_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // N2/C16/H8/W8/K8: K*C=128 (filter transform block) and 256 tiles.
        const int n = 2, cch = 16, h = 8, w = 8, k = 8;
        var shape = new Conv2DWinogradShape(n, cch, h, w, k, filterPretransformed: true);

        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);

        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            // Stage 1: filter transform weights[K,C,3,3] -> U[K,C,4,4].
            using var filter = new PtxWinogradF23FilterTransformKernel(runtime, k, cch);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(
                DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
            runtime.Synchronize();

            // Stage 2: main Winograd kernel reads U (its "weights" tensor).
            using var kernel = new PtxConv2DNchw3x3WinogradF23Kernel(runtime, shape);
            using var dInput = runtime.AllocateBytes((nuint)shape.InputBytes);
            using var dBias = runtime.AllocateBytes((nuint)shape.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)shape.OutputBytes);
            dInput.Upload<float>(input);
            dBias.Upload<float>(bias);
            kernel.Launch(
                DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(dU, kernel.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(dOutput, kernel.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally
        {
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    [Fact]
    public void Winograd3x3_FusedGemmPipeline_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // N2/C16/H8/W8/K16: K%16=0, C%8=0, P=N*TH*TW=2*4*4=32 %16=0.
        const int n = 2, cch = 16, h = 8, w = 8, k = 16;
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);

        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            // Stage 1: filter transform (position-major U[16,K,C]).
            using var filter = new PtxWinogradF23FilterTransformKernel(runtime, k, cch, positionMajor: true);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(
                DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            // Stage 2: input transform V[16,C,P].
            using var inputT = new PtxWinogradF23InputTransformKernel(runtime, n, cch, h, w);
            using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
            using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
            dInput.Upload<float>(input);
            inputT.Launch(
                DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));
            runtime.Synchronize();

            // Stage 3: fused batched GEMM + output transform.
            using var fused = new PtxWinogradF23FusedGemmKernel(runtime, n, cch, h, w, k, 16, 16, 8);
            using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
            dBias.Upload<float>(bias);
            fused.Launch(
                DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(dV, fused.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally
        {
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    [Fact]
    public void Winograd3x3_BatchedGemmPipeline_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, cch = 16, h = 8, w = 8, k = 16;   // P = 32
        int tiles = n * (h / 2) * (w / 2);
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var filter = new PtxWinogradF23FilterTransformKernel(runtime, k, cch, positionMajor: true);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var inputT = new PtxWinogradF23InputTransformKernel(runtime, n, cch, h, w);
            using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
            using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
            dInput.Upload<float>(input);
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));

            using var gemm = new PtxWinogradBatchedGemmKernel(runtime, k, cch, tiles, 16, 16, 8, 4, 4);
            using var dM = runtime.AllocateBytes((nuint)gemm.MBytes);
            gemm.Launch(DirectPtxTensorView.CreateOwned(dU, gemm.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dV, gemm.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dM, gemm.Blueprint.Tensors[2]));

            using var outT = new PtxWinogradF23OutputTransformKernel(runtime, n, h, w, k);
            using var dBias = runtime.AllocateBytes((nuint)outT.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)outT.OutputBytes);
            dBias.Upload<float>(bias);
            outT.Launch(DirectPtxTensorView.CreateOwned(dM, outT.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dBias, outT.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dOutput, outT.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Winograd3x3_FusedRegBlocked_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, cch = 16, h = 8, w = 8, k = 16;   // P = 32
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var filter = new PtxWinogradF23FilterTransformKernel(runtime, k, cch, positionMajor: true);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var inputT = new PtxWinogradF23InputTransformKernel(runtime, n, cch, h, w);
            using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
            using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
            dInput.Upload<float>(input);
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));

            using var fused = new PtxWinogradF23FusedRegBlockedKernel(runtime, n, cch, h, w, k, 16, 16, 8, 2, 2);
            using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
            dBias.Upload<float>(bias);
            fused.Launch(DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                         DirectPtxTensorView.CreateOwned(dV, fused.Blueprint.Tensors[1]),
                         DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                         DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Winograd3x3_WmmaTensorCorePipeline_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // WMMA constraints: K multiple of 32, P multiple of 32, C multiple of 16.
        const int n = 2, cch = 16, h = 8, w = 8, k = 32;   // P = 32
        int tiles = n * (h / 2) * (w / 2);
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 7 ||
            !DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, k, cch);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
            using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
            using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
            dInput.Upload<float>(input);
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));

            using var gemm = new PtxWinogradWmmaBatchedGemmKernel(runtime, k, cch, tiles);
            using var dM = runtime.AllocateBytes((nuint)gemm.MBytes);
            gemm.Launch(DirectPtxTensorView.CreateOwned(dU, gemm.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dV, gemm.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dM, gemm.Blueprint.Tensors[2]));

            using var outT = new PtxWinogradF23OutputTransformKernel(runtime, n, h, w, k);
            using var dBias = runtime.AllocateBytes((nuint)outT.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)outT.OutputBytes);
            dBias.Upload<float>(bias);
            outT.Launch(DirectPtxTensorView.CreateOwned(dM, outT.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dBias, outT.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dOutput, outT.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            // fp16 U/V operands: accuracy-matched to cuDNN's fp16 Winograd regime.
            AssertClose(expected, actual, 5e-2f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Winograd3x3_WmmaFused_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // Fused WMMA: K multiple of 16, C multiple of 16, P multiple of 8.
        const int n = 2, cch = 16, h = 8, w = 8, k = 32;   // P = 32
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 7 ||
            !DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, k, cch);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
            using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
            using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
            dInput.Upload<float>(input);
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));

            using var fused = new PtxWinogradWmmaFusedKernel(runtime, n, cch, h, w, k);
            using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
            dBias.Upload<float>(bias);
            fused.Launch(DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                         DirectPtxTensorView.CreateOwned(dV, fused.Blueprint.Tensors[1]),
                         DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                         DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 5e-2f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Winograd3x3_WmmaFusedStaged_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        // Staged fused WMMA: K multiple of 16, C multiple of 16, P multiple of 32.
        const int n = 2, cch = 16, h = 8, w = 8, k = 32;   // P = 32
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 8 ||
            !DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, k, cch);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
            using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
            using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
            dInput.Upload<float>(input);
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));

            using var fused = new PtxWinogradWmmaFusedStagedKernel(runtime, n, cch, h, w, k);
            using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
            dBias.Upload<float>(bias);
            fused.Launch(DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                         DirectPtxTensorView.CreateOwned(dV, fused.Blueprint.Tensors[1]),
                         DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                         DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 5e-2f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Winograd3x3_WmmaCoop_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, cch = 16, h = 8, w = 8, k = 32;   // P = 32
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 7 ||
            !DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, k, cch);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
            using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
            using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
            dInput.Upload<float>(input);
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));

            using var coop = new PtxWinogradWmmaCoopKernel(runtime, n, cch, h, w, k);
            using var dBias = runtime.AllocateBytes((nuint)coop.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)coop.OutputBytes);
            dBias.Upload<float>(bias);
            coop.Launch(DirectPtxTensorView.CreateOwned(dU, coop.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dV, coop.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dBias, coop.Blueprint.Tensors[2]),
                        DirectPtxTensorView.CreateOwned(dOutput, coop.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 5e-2f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Winograd3x3_WmmaCoopBlocked_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, cch = 16, h = 8, w = 8, k = 32;   // P = 32
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 7 ||
            !DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, k, cch);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
            using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
            using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
            dInput.Upload<float>(input);
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));

            using var coop = new PtxWinogradWmmaCoopBlockedKernel(runtime, n, cch, h, w, k);
            using var dBias = runtime.AllocateBytes((nuint)coop.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)coop.OutputBytes);
            dBias.Upload<float>(bias);
            coop.Launch(DirectPtxTensorView.CreateOwned(dU, coop.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dV, coop.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dBias, coop.Blueprint.Tensors[2]),
                        DirectPtxTensorView.CreateOwned(dOutput, coop.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 5e-2f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Winograd3x3_WmmaFullyFused_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, cch = 16, h = 8, w = 8, k = 32;   // P = 32
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 7 ||
            !DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            // Only the filter transform (U precomputed, fp16) runs before the fused kernel;
            // the input transform is fused in-kernel.
            using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, k, cch);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var fused = new PtxWinogradWmmaFullyFusedKernel(runtime, n, cch, h, w, k);
            using var dInput = runtime.AllocateBytes((nuint)fused.InputBytes);
            using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
            dInput.Upload<float>(input);
            dBias.Upload<float>(bias);
            fused.Launch(DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                         DirectPtxTensorView.CreateOwned(dInput, fused.Blueprint.Tensors[1]),
                         DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                         DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 5e-2f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Winograd3x3_WmmaFusedAllK_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, cch = 16, h = 8, w = 8, k = 32;   // P = 32, K<=64
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 7 ||
            !DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, k, cch);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var fused = new PtxWinogradWmmaFusedAllKKernel(runtime, n, cch, h, w, k);
            using var dInput = runtime.AllocateBytes((nuint)fused.InputBytes);
            using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
            dInput.Upload<float>(input);
            dBias.Upload<float>(bias);
            fused.Launch(DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                         DirectPtxTensorView.CreateOwned(dInput, fused.Blueprint.Tensors[1]),
                         DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                         DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 5e-2f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Winograd3x3_WmmaPipelined_MatchesDirectConvReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, cch = 32, h = 8, w = 8, k = 32;   // P=32, C=32 -> 2 chunks (exercises pipeline)
        var input = new float[n * cch * h * w];
        var weights = new float[k * cch * 9];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        float[] expected = ReferenceConv3x3Same(input, weights, bias, n, cch, h, w, k);

        using var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 7 ||
            !DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, k, cch);
            using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
            using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
            dWeights.Upload<float>(weights);
            filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));

            using var pipe = new PtxWinogradWmmaPipelinedKernel(runtime, n, cch, h, w, k);
            using var dInput = runtime.AllocateBytes((nuint)pipe.InputBytes);
            using var dBias = runtime.AllocateBytes((nuint)pipe.BiasBytes);
            using var dOutput = runtime.AllocateBytes((nuint)pipe.OutputBytes);
            dInput.Upload<float>(input);
            dBias.Upload<float>(bias);
            pipe.Launch(DirectPtxTensorView.CreateOwned(dU, pipe.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dInput, pipe.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dBias, pipe.Blueprint.Tensors[2]),
                        DirectPtxTensorView.CreateOwned(dOutput, pipe.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * h * w];
            dOutput.Download<float>(actual);
            AssertClose(expected, actual, 5e-2f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Conv2DBackwardBias_MatchesCpuReduction()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 3, k = 8, h = 7, w = 5;   // non-power-of-2 spatial exercises the loop tails
        var grad = new float[n * k * h * w];
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicInput(i) - 0.5f;
        // CPU oracle: sum over batch + spatial per channel.
        var expected = new float[k];
        int hw = h * w;
        for (int b = 0; b < n; b++)
            for (int c = 0; c < k; c++)
            {
                double acc = 0;
                for (int s = 0; s < hw; s++) acc += grad[(b * k + c) * hw + s];
                expected[c] += (float)acc;
            }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConv2DBackwardBiasKernel(runtime, n, k, h, w);
            using var dGrad = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.GradBiasBytes);
            dGrad.Upload<float>(grad);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dGrad, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[1]));
            runtime.Synchronize();
            var actual = new float[k];
            dBias.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Conv2DBackwardWeight3x3_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, k = 6, cch = 4, h = 7, w = 5;
        var input = new float[n * cch * h * w];
        var grad = new float[n * k * h * w];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i) - 0.1f;
        // CPU oracle: dW[k,c,r,s] = sum_{n,oh,ow} input[n,c,oh+r-1,ow+s-1]*gradOut[n,k,oh,ow]
        var expected = new float[k * cch * 9];
        for (int oc = 0; oc < k; oc++)
            for (int ic = 0; ic < cch; ic++)
                for (int r = 0; r < 3; r++)
                    for (int sK = 0; sK < 3; sK++)
                    {
                        double acc = 0;
                        for (int b = 0; b < n; b++)
                            for (int oh = 0; oh < h; oh++)
                                for (int ow = 0; ow < w; ow++)
                                {
                                    int ih = oh + r - 1, iw = ow + sK - 1;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    acc += (double)input[((b * cch + ic) * h + ih) * w + iw] *
                                           grad[((b * k + oc) * h + oh) * w + ow];
                                }
                        expected[((oc * cch + ic) * 3 + r) * 3 + sK] = (float)acc;
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConv2DBackwardWeight3x3Kernel(runtime, n, k, cch, h, w);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dGrad = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.GradWeightBytes);
            dInput.Upload<float>(input);
            dGrad.Upload<float>(grad);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dGrad, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[k * cch * 9];
            dW.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Conv2DBackwardInput3x3_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, k = 6, cch = 4, h = 8, w = 8;   // N*C*H*W = 512 (mult 256)
        var grad = new float[n * k * h * w];
        var weights = new float[k * cch * 9];
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        // CPU oracle: dX[n,c,ih,iw] = sum_{k,r,s} W[k,c,r,s]*gradOut[n,k,ih-r+1,iw-s+1]
        var expected = new float[n * cch * h * w];
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < cch; ic++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < w; iw++)
                    {
                        double acc = 0;
                        for (int oc = 0; oc < k; oc++)
                            for (int r = 0; r < 3; r++)
                                for (int sK = 0; sK < 3; sK++)
                                {
                                    int oh = ih - r + 1, ow = iw - sK + 1;
                                    if (oh < 0 || oh >= h || ow < 0 || ow >= w) continue;
                                    acc += (double)weights[((oc * cch + ic) * 3 + r) * 3 + sK] *
                                           grad[((b * k + oc) * h + oh) * w + ow];
                                }
                        expected[((b * cch + ic) * h + ih) * w + iw] = (float)acc;
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConv2DBackwardInput3x3Kernel(runtime, n, k, cch, h, w);
            using var dGrad = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dX = runtime.AllocateBytes((nuint)kernel.GradInputBytes);
            dGrad.Upload<float>(grad);
            dW.Upload<float>(weights);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dGrad, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dX, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[n * cch * h * w];
            dX.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DepthwiseConv2D3x3_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 8, h = 8, w = 8;   // N*C*H*W = 1024
        var input = new float[n * c * h * w];
        var weights = new float[c * 9];
        var bias = new float[c];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        var expected = new float[n * c * h * w];
        for (int b = 0; b < n; b++)
            for (int ch = 0; ch < c; ch++)
                for (int oh = 0; oh < h; oh++)
                    for (int ow = 0; ow < w; ow++)
                    {
                        double acc = bias[ch];
                        for (int r = 0; r < 3; r++)
                            for (int sK = 0; sK < 3; sK++)
                            {
                                int ih = oh + r - 1, iw = ow + sK - 1;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                acc += (double)input[((b * c + ch) * h + ih) * w + iw] * weights[ch * 9 + r * 3 + sK];
                            }
                        expected[((b * c + ch) * h + oh) * w + ow] = (float)Math.Max(acc, 0.0);
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDepthwiseConv2D3x3Kernel(runtime, n, c, h, w, relu: true);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input);
            dW.Upload<float>(weights);
            dBias.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * c * h * w];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DepthwiseConv2D3x3Backward_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 8, h = 8, w = 8;
        var input = new float[n * c * h * w];
        var grad = new float[n * c * h * w];
        var weights = new float[c * 9];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i) - 0.1f;
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicBias(i) + 0.3f;
        // CPU oracles
        var expDx = new float[n * c * h * w];
        var expDw = new float[c * 9];
        for (int b = 0; b < n; b++)
            for (int ch = 0; ch < c; ch++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < w; iw++)
                    {
                        double acc = 0;
                        for (int r = 0; r < 3; r++)
                            for (int sK = 0; sK < 3; sK++)
                            {
                                int oh = ih - r + 1, ow = iw - sK + 1;
                                if (oh < 0 || oh >= h || ow < 0 || ow >= w) continue;
                                acc += (double)weights[ch * 9 + r * 3 + sK] * grad[((b * c + ch) * h + oh) * w + ow];
                            }
                        expDx[((b * c + ch) * h + ih) * w + iw] = (float)acc;
                    }
        for (int ch = 0; ch < c; ch++)
            for (int r = 0; r < 3; r++)
                for (int sK = 0; sK < 3; sK++)
                {
                    double acc = 0;
                    for (int b = 0; b < n; b++)
                        for (int oh = 0; oh < h; oh++)
                            for (int ow = 0; ow < w; ow++)
                            {
                                int ih = oh + r - 1, iw = ow + sK - 1;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                acc += (double)input[((b * c + ch) * h + ih) * w + iw] * grad[((b * c + ch) * h + oh) * w + ow];
                            }
                    expDw[ch * 9 + r * 3 + sK] = (float)acc;
                }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var bin = new PtxDepthwiseConv2D3x3BackwardInputKernel(runtime, n, c, h, w);
            using var dGrad = runtime.AllocateBytes((nuint)bin.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)bin.WeightBytes);
            using var dX = runtime.AllocateBytes((nuint)bin.GradInputBytes);
            dGrad.Upload<float>(grad); dW.Upload<float>(weights);
            bin.Launch(DirectPtxTensorView.CreateOwned(dGrad, bin.Blueprint.Tensors[0]),
                       DirectPtxTensorView.CreateOwned(dW, bin.Blueprint.Tensors[1]),
                       DirectPtxTensorView.CreateOwned(dX, bin.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDx = new float[n * c * h * w];
            dX.Download<float>(actDx);
            AssertClose(expDx, actDx, 2e-3f);

            using var bw = new PtxDepthwiseConv2D3x3BackwardWeightKernel(runtime, n, c, h, w);
            using var dInput = runtime.AllocateBytes((nuint)bw.InputBytes);
            using var dGrad2 = runtime.AllocateBytes((nuint)bw.GradOutputBytes);
            using var dDw = runtime.AllocateBytes((nuint)bw.GradWeightBytes);
            dInput.Upload<float>(input); dGrad2.Upload<float>(grad);
            bw.Launch(DirectPtxTensorView.CreateOwned(dInput, bw.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dGrad2, bw.Blueprint.Tensors[1]),
                      DirectPtxTensorView.CreateOwned(dDw, bw.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDw = new float[c * 9];
            dDw.Download<float>(actDw);
            AssertClose(expDw, actDw, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Conv1D_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, k = 8, l = 16, kl = 3, stride = 1, pad = 1;
        int ol = (l + 2 * pad - kl) / stride + 1;
        var input = new float[n * c * l];
        var weights = new float[k * c * kl];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        var expected = new float[n * k * ol];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < k; oc++)
                for (int o = 0; o < ol; o++)
                {
                    double acc = bias[oc];
                    for (int ic = 0; ic < c; ic++)
                        for (int t = 0; t < kl; t++)
                        {
                            int il = o * stride + t - pad;
                            if (il < 0 || il >= l) continue;
                            acc += (double)input[(b * c + ic) * l + il] * weights[(oc * c + ic) * kl + t];
                        }
                    expected[(b * k + oc) * ol + o] = (float)Math.Max(acc, 0.0);
                }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConv1DKernel(runtime, n, c, k, l, kl, stride, pad, relu: true);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dBias.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * ol];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Conv1DBackward_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, k = 8, l = 32, kl = 3, stride = 1, pad = 1;   // N*C*L = 256
        int ol = (l + 2 * pad - kl) / stride + 1;
        var input = new float[n * c * l];
        var grad = new float[n * k * ol];
        var weights = new float[k * c * kl];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i) - 0.1f;
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicBias(i) + 0.2f;
        var expDx = new float[n * c * l];
        var expDw = new float[k * c * kl];
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < c; ic++)
                for (int il = 0; il < l; il++)
                {
                    double acc = 0;
                    for (int oc = 0; oc < k; oc++)
                        for (int t = 0; t < kl; t++)
                        {
                            int num = il + pad - t;
                            if (num < 0 || num % stride != 0) continue;
                            int o = num / stride;
                            if (o < 0 || o >= ol) continue;
                            acc += (double)weights[(oc * c + ic) * kl + t] * grad[(b * k + oc) * ol + o];
                        }
                    expDx[(b * c + ic) * l + il] = (float)acc;
                }
        for (int oc = 0; oc < k; oc++)
            for (int ic = 0; ic < c; ic++)
                for (int t = 0; t < kl; t++)
                {
                    double acc = 0;
                    for (int b = 0; b < n; b++)
                        for (int o = 0; o < ol; o++)
                        {
                            int il = o * stride + t - pad;
                            if (il < 0 || il >= l) continue;
                            acc += (double)input[(b * c + ic) * l + il] * grad[(b * k + oc) * ol + o];
                        }
                    expDw[(oc * c + ic) * kl + t] = (float)acc;
                }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var bin = new PtxConv1DBackwardInputKernel(runtime, n, c, k, l, kl, stride, pad);
            using var dGrad = runtime.AllocateBytes((nuint)bin.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)bin.WeightBytes);
            using var dX = runtime.AllocateBytes((nuint)bin.GradInputBytes);
            dGrad.Upload<float>(grad); dW.Upload<float>(weights);
            bin.Launch(DirectPtxTensorView.CreateOwned(dGrad, bin.Blueprint.Tensors[0]),
                       DirectPtxTensorView.CreateOwned(dW, bin.Blueprint.Tensors[1]),
                       DirectPtxTensorView.CreateOwned(dX, bin.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDx = new float[n * c * l];
            dX.Download<float>(actDx);
            AssertClose(expDx, actDx, 2e-3f);

            using var bw = new PtxConv1DBackwardWeightKernel(runtime, n, c, k, l, kl, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)bw.InputBytes);
            using var dGrad2 = runtime.AllocateBytes((nuint)bw.GradOutputBytes);
            using var dDw = runtime.AllocateBytes((nuint)bw.GradWeightBytes);
            dInput.Upload<float>(input); dGrad2.Upload<float>(grad);
            bw.Launch(DirectPtxTensorView.CreateOwned(dInput, bw.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dGrad2, bw.Blueprint.Tensors[1]),
                      DirectPtxTensorView.CreateOwned(dDw, bw.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDw = new float[k * c * kl];
            dDw.Download<float>(actDw);
            AssertClose(expDw, actDw, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    // Stride != 1 takes a DIFFERENT codegen branch in the emitter (rem/div validity checks
    // instead of a plain move). This shape is exactly the one exported as a released cubin,
    // so the committed artifact and the numerically-verified specialization are the same.
    [Fact]
    public void ConvTranspose2DStride2_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, ci = 3, co = 4, h = 4, w = 4, kh = 3, kw = 3, stride = 2, pad = 1, outpad = 1;
        int oh = (h - 1) * stride - 2 * pad + kh + outpad;
        int ow = (w - 1) * stride - 2 * pad + kw + outpad;
        var input = new float[n * ci * h * w];
        var weights = new float[ci * co * kh * kw];
        var bias = new float[co];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        var expected = new float[n * co * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < co; oc++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        double acc = bias[oc];
                        for (int ic = 0; ic < ci; ic++)
                            for (int r = 0; r < kh; r++)
                                for (int t = 0; t < kw; t++)
                                {
                                    int nh = y + pad - r, nw = x + pad - t;
                                    if (nh < 0 || nh % stride != 0 || nw < 0 || nw % stride != 0) continue;
                                    int ih = nh / stride, iw = nw / stride;
                                    if (ih >= h || iw >= w) continue;
                                    acc += (double)input[((b * ci + ic) * h + ih) * w + iw] *
                                           weights[((ic * co + oc) * kh + r) * kw + t];
                                }
                        expected[((b * co + oc) * oh + y) * ow + x] = (float)Math.Max(acc, 0.0);
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConvTranspose2DKernel(runtime, n, ci, co, h, w, kh, kw, stride, pad, outpad, relu: true);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dBias.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * co * oh * ow];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void ConvTranspose2D_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, ci = 4, co = 8, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1, outpad = 0;
        int oh = (h - 1) * stride - 2 * pad + kh + outpad;
        int ow = (w - 1) * stride - 2 * pad + kw + outpad;
        var input = new float[n * ci * h * w];
        var weights = new float[ci * co * kh * kw];
        var bias = new float[co];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        var expected = new float[n * co * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < co; oc++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        double acc = bias[oc];
                        for (int ic = 0; ic < ci; ic++)
                            for (int r = 0; r < kh; r++)
                                for (int t = 0; t < kw; t++)
                                {
                                    int nh = y + pad - r, nw = x + pad - t;
                                    if (nh < 0 || nh % stride != 0 || nw < 0 || nw % stride != 0) continue;
                                    int ih = nh / stride, iw = nw / stride;
                                    if (ih >= h || iw >= w) continue;
                                    acc += (double)input[((b * ci + ic) * h + ih) * w + iw] *
                                           weights[((ic * co + oc) * kh + r) * kw + t];
                                }
                        expected[((b * co + oc) * oh + y) * ow + x] = (float)Math.Max(acc, 0.0);
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConvTranspose2DKernel(runtime, n, ci, co, h, w, kh, kw, stride, pad, outpad, relu: true);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dBias.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * co * oh * ow];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void ConvTranspose2DBackward_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, ci = 4, co = 8, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1, outpad = 0;
        int oh = (h - 1) * stride - 2 * pad + kh + outpad;
        int ow = (w - 1) * stride - 2 * pad + kw + outpad;
        var input = new float[n * ci * h * w];
        var grad = new float[n * co * oh * ow];
        var weights = new float[ci * co * kh * kw];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i) - 0.1f;
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicBias(i) + 0.2f;
        var expDx = new float[n * ci * h * w];
        var expDw = new float[ci * co * kh * kw];
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < ci; ic++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < w; iw++)
                    {
                        double acc = 0;
                        for (int oc = 0; oc < co; oc++)
                            for (int r = 0; r < kh; r++)
                                for (int t = 0; t < kw; t++)
                                {
                                    int y = ih * stride - pad + r, x = iw * stride - pad + t;
                                    if (y < 0 || y >= oh || x < 0 || x >= ow) continue;
                                    acc += (double)grad[((b * co + oc) * oh + y) * ow + x] *
                                           weights[((ic * co + oc) * kh + r) * kw + t];
                                }
                        expDx[((b * ci + ic) * h + ih) * w + iw] = (float)acc;
                    }
        for (int ic = 0; ic < ci; ic++)
            for (int oc = 0; oc < co; oc++)
                for (int r = 0; r < kh; r++)
                    for (int t = 0; t < kw; t++)
                    {
                        double acc = 0;
                        for (int b = 0; b < n; b++)
                            for (int ih = 0; ih < h; ih++)
                                for (int iw = 0; iw < w; iw++)
                                {
                                    int y = ih * stride - pad + r, x = iw * stride - pad + t;
                                    if (y < 0 || y >= oh || x < 0 || x >= ow) continue;
                                    acc += (double)input[((b * ci + ic) * h + ih) * w + iw] *
                                           grad[((b * co + oc) * oh + y) * ow + x];
                                }
                        expDw[((ic * co + oc) * kh + r) * kw + t] = (float)acc;
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var bin = new PtxConvTranspose2DBackwardInputKernel(runtime, n, ci, co, h, w, kh, kw, stride, pad, outpad);
            using var dGrad = runtime.AllocateBytes((nuint)bin.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)bin.WeightBytes);
            using var dX = runtime.AllocateBytes((nuint)bin.GradInputBytes);
            dGrad.Upload<float>(grad); dW.Upload<float>(weights);
            bin.Launch(DirectPtxTensorView.CreateOwned(dGrad, bin.Blueprint.Tensors[0]),
                       DirectPtxTensorView.CreateOwned(dW, bin.Blueprint.Tensors[1]),
                       DirectPtxTensorView.CreateOwned(dX, bin.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDx = new float[n * ci * h * w];
            dX.Download<float>(actDx);
            AssertClose(expDx, actDx, 2e-3f);

            using var bw = new PtxConvTranspose2DBackwardWeightKernel(runtime, n, ci, co, h, w, kh, kw, stride, pad, outpad);
            using var dInput = runtime.AllocateBytes((nuint)bw.InputBytes);
            using var dGrad2 = runtime.AllocateBytes((nuint)bw.GradOutputBytes);
            using var dDw = runtime.AllocateBytes((nuint)bw.GradWeightBytes);
            dInput.Upload<float>(input); dGrad2.Upload<float>(grad);
            bw.Launch(DirectPtxTensorView.CreateOwned(dInput, bw.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dGrad2, bw.Blueprint.Tensors[1]),
                      DirectPtxTensorView.CreateOwned(dDw, bw.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDw = new float[ci * co * kh * kw];
            dDw.Download<float>(actDw);
            AssertClose(expDw, actDw, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Conv3D_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 3, k = 4, d = 8, h = 8, w = 8, kd = 3, kh = 3, kw = 3, stride = 1, pad = 1;   // N*K*OD*OH*OW=4096
        int od = (d + 2 * pad - kd) / stride + 1, oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
        var input = new float[n * c * d * h * w];
        var weights = new float[k * c * kd * kh * kw];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        var expected = new float[n * k * od * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < k; oc++)
                for (int z = 0; z < od; z++)
                    for (int y = 0; y < oh; y++)
                        for (int x = 0; x < ow; x++)
                        {
                            double acc = bias[oc];
                            for (int ic = 0; ic < c; ic++)
                                for (int a = 0; a < kd; a++)
                                    for (int rr = 0; rr < kh; rr++)
                                        for (int t = 0; t < kw; t++)
                                        {
                                            int id = z * stride + a - pad, ih = y * stride + rr - pad, iw = x * stride + t - pad;
                                            if (id < 0 || id >= d || ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                            acc += (double)input[(((b * c + ic) * d + id) * h + ih) * w + iw] *
                                                   weights[(((oc * c + ic) * kd + a) * kh + rr) * kw + t];
                                        }
                            expected[(((b * k + oc) * od + z) * oh + y) * ow + x] = (float)Math.Max(acc, 0.0);
                        }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConv3DKernel(runtime, n, c, k, d, h, w, kd, kh, kw, stride, pad, relu: true);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dBias.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * od * oh * ow];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Conv3DBackward_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 3, k = 4, d = 8, h = 8, w = 8, kd = 3, kh = 3, kw = 3, stride = 1, pad = 1;
        int od = (d + 2 * pad - kd) / stride + 1, oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
        var input = new float[n * c * d * h * w];
        var grad = new float[n * k * od * oh * ow];
        var weights = new float[k * c * kd * kh * kw];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i) - 0.1f;
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicBias(i) + 0.2f;
        var expDx = new float[n * c * d * h * w];
        var expDw = new float[k * c * kd * kh * kw];
        int Iin(int b, int ic, int z, int y, int x) => (((b * c + ic) * d + z) * h + y) * w + x;
        int Ig(int b, int oc, int z, int y, int x) => (((b * k + oc) * od + z) * oh + y) * ow + x;
        int Iw(int oc, int ic, int a, int rr, int t) => (((oc * c + ic) * kd + a) * kh + rr) * kw + t;
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < c; ic++)
                for (int id = 0; id < d; id++)
                    for (int ih = 0; ih < h; ih++)
                        for (int iw = 0; iw < w; iw++)
                        {
                            double acc = 0;
                            for (int oc = 0; oc < k; oc++)
                                for (int a = 0; a < kd; a++)
                                    for (int rr = 0; rr < kh; rr++)
                                        for (int t = 0; t < kw; t++)
                                        {
                                            int z = id - a + pad, y = ih - rr + pad, x = iw - t + pad;
                                            if (z < 0 || z >= od || y < 0 || y >= oh || x < 0 || x >= ow) continue;
                                            acc += (double)weights[Iw(oc, ic, a, rr, t)] * grad[Ig(b, oc, z, y, x)];
                                        }
                            expDx[Iin(b, ic, id, ih, iw)] = (float)acc;
                        }
        for (int oc = 0; oc < k; oc++)
            for (int ic = 0; ic < c; ic++)
                for (int a = 0; a < kd; a++)
                    for (int rr = 0; rr < kh; rr++)
                        for (int t = 0; t < kw; t++)
                        {
                            double acc = 0;
                            for (int b = 0; b < n; b++)
                                for (int z = 0; z < od; z++)
                                    for (int y = 0; y < oh; y++)
                                        for (int x = 0; x < ow; x++)
                                        {
                                            int id = z * stride + a - pad, ih = y * stride + rr - pad, iw = x * stride + t - pad;
                                            if (id < 0 || id >= d || ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                            acc += (double)input[Iin(b, ic, id, ih, iw)] * grad[Ig(b, oc, z, y, x)];
                                        }
                            expDw[Iw(oc, ic, a, rr, t)] = (float)acc;
                        }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var bin = new PtxConv3DBackwardInputKernel(runtime, n, c, k, d, h, w, kd, kh, kw, stride, pad);
            using var dGrad = runtime.AllocateBytes((nuint)bin.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)bin.WeightBytes);
            using var dX = runtime.AllocateBytes((nuint)bin.GradInputBytes);
            dGrad.Upload<float>(grad); dW.Upload<float>(weights);
            bin.Launch(DirectPtxTensorView.CreateOwned(dGrad, bin.Blueprint.Tensors[0]),
                       DirectPtxTensorView.CreateOwned(dW, bin.Blueprint.Tensors[1]),
                       DirectPtxTensorView.CreateOwned(dX, bin.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDx = new float[n * c * d * h * w];
            dX.Download<float>(actDx);
            AssertClose(expDx, actDx, 2e-3f);

            using var bw = new PtxConv3DBackwardWeightKernel(runtime, n, c, k, d, h, w, kd, kh, kw, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)bw.InputBytes);
            using var dGrad2 = runtime.AllocateBytes((nuint)bw.GradOutputBytes);
            using var dDw = runtime.AllocateBytes((nuint)bw.GradWeightBytes);
            dInput.Upload<float>(input); dGrad2.Upload<float>(grad);
            bw.Launch(DirectPtxTensorView.CreateOwned(dInput, bw.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dGrad2, bw.Blueprint.Tensors[1]),
                      DirectPtxTensorView.CreateOwned(dDw, bw.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDw = new float[k * c * kd * kh * kw];
            dDw.Download<float>(actDw);
            AssertClose(expDw, actDw, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Unfold2D_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
        int patchRows = c * kh * kw, cols = oh * ow;
        var input = new float[n * c * h * w];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        var expected = new float[n * patchRows * cols];
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < c; ic++)
                for (int r = 0; r < kh; r++)
                    for (int t = 0; t < kw; t++)
                        for (int y = 0; y < oh; y++)
                            for (int x = 0; x < ow; x++)
                            {
                                int ih = y * stride + r - pad, iw = x * stride + t - pad;
                                float v = 0;
                                if (ih >= 0 && ih < h && iw >= 0 && iw < w) v = input[((b * c + ic) * h + ih) * w + iw];
                                int prow = ic * kh * kw + r * kw + t;
                                expected[(b * patchRows + prow) * cols + (y * ow + x)] = v;
                            }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxUnfold2DKernel(runtime, n, c, h, w, kh, kw, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[1]));
            runtime.Synchronize();
            var actual = new float[n * patchRows * cols];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 1e-5f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Im2colKNFp16_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
        int patchRows = c * kh * kw, cols = oh * ow;
        var input = new float[n * c * h * w];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        var expected = new float[n * patchRows * cols];
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < c; ic++)
                for (int r = 0; r < kh; r++)
                    for (int t = 0; t < kw; t++)
                        for (int y = 0; y < oh; y++)
                            for (int x = 0; x < ow; x++)
                            {
                                int ih = y * stride + r - pad, iw = x * stride + t - pad;
                                float v = 0;
                                if (ih >= 0 && ih < h && iw >= 0 && iw < w) v = input[((b * c + ic) * h + ih) * w + iw];
                                expected[(b * patchRows + (ic * kh * kw + r * kw + t)) * cols + (y * ow + x)] = (float)(Half)v;
                            }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxUnfold2DFp16Kernel(runtime, n, c, h, w, kh, kw, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[1]));
            runtime.Synchronize();
            var raw = new ushort[n * patchRows * cols];
            dOut.Download<ushort>(raw);
            var actual = new float[raw.Length];
            for (int i = 0; i < raw.Length; i++) actual[i] = (float)BitConverter.UInt16BitsToHalf(raw[i]);
            AssertClose(expected, actual, 1e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void Conv2dDirectFp16Hw_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, k = 8, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
        var input = new float[n * c * h * w];
        var wHalf = new ushort[k * c * kh * kw];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < wHalf.Length; i++) wHalf[i] = BitConverter.HalfToUInt16Bits((Half)DeterministicWeight(i));
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        var expected = new float[n * k * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < k; oc++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        double acc = bias[oc];
                        for (int ic = 0; ic < c; ic++)
                            for (int r = 0; r < kh; r++)
                                for (int t = 0; t < kw; t++)
                                {
                                    int ih = y * stride + r - pad, iw = x * stride + t - pad;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    float iv = (float)(Half)input[((b * c + ic) * h + ih) * w + iw];
                                    float wv = (float)BitConverter.UInt16BitsToHalf(wHalf[((oc * c + ic) * kh + r) * kw + t]);
                                    acc += (double)iv * wv;
                                }
                        expected[((b * k + oc) * oh + y) * ow + x] = (float)Math.Max(acc, 0.0);
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConv2DDirectFp16Kernel(runtime, n, c, k, h, w, kh, kw, stride, pad, relu: true);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input); dW.Upload<ushort>(wHalf); dBias.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * k * oh * ow];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void LocallyConnected2DForwardBias_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 3, k = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
        var input = new float[n * c * h * w];
        var weights = new float[oh * ow * k * c * kh * kw];
        var bias = new float[k * oh * ow];
        var grad = new float[n * k * oh * ow];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicInput(i + 7) - 0.3f;
        int Wi(int y, int x, int oc, int ic, int r, int t) => ((((y * ow + x) * k + oc) * c + ic) * kh + r) * kw + t;
        var expOut = new float[n * k * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < k; oc++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        double acc = bias[(oc * oh + y) * ow + x];
                        for (int ic = 0; ic < c; ic++)
                            for (int r = 0; r < kh; r++)
                                for (int t = 0; t < kw; t++)
                                {
                                    int ih = y * stride + r - pad, iw = x * stride + t - pad;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    acc += (double)input[((b * c + ic) * h + ih) * w + iw] * weights[Wi(y, x, oc, ic, r, t)];
                                }
                        expOut[((b * k + oc) * oh + y) * ow + x] = (float)Math.Max(acc, 0.0);
                    }
        var expBias = new float[k * oh * ow];
        for (int oc = 0; oc < k; oc++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                {
                    double acc = 0;
                    for (int b = 0; b < n; b++) acc += grad[((b * k + oc) * oh + y) * ow + x];
                    expBias[(oc * oh + y) * ow + x] = (float)acc;
                }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var fwd = new PtxLocallyConnected2DKernel(runtime, n, c, k, h, w, kh, kw, stride, pad, relu: true);
            using var dInput = runtime.AllocateBytes((nuint)fwd.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)fwd.WeightBytes);
            using var dBias = runtime.AllocateBytes((nuint)fwd.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)fwd.OutputBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dBias.Upload<float>(bias);
            fwd.Launch(DirectPtxTensorView.CreateOwned(dInput, fwd.Blueprint.Tensors[0]),
                       DirectPtxTensorView.CreateOwned(dW, fwd.Blueprint.Tensors[1]),
                       DirectPtxTensorView.CreateOwned(dBias, fwd.Blueprint.Tensors[2]),
                       DirectPtxTensorView.CreateOwned(dOut, fwd.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actOut = new float[n * k * oh * ow];
            dOut.Download<float>(actOut);
            AssertClose(expOut, actOut, 2e-3f);

            using var bb = new PtxLocallyConnected2DBackwardBiasKernel(runtime, n, k, oh, ow);
            using var dGrad = runtime.AllocateBytes((nuint)bb.GradOutputBytes);
            using var dDbias = runtime.AllocateBytes((nuint)bb.GradBiasBytes);
            dGrad.Upload<float>(grad);
            bb.Launch(DirectPtxTensorView.CreateOwned(dGrad, bb.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dDbias, bb.Blueprint.Tensors[1]));
            runtime.Synchronize();
            var actBias = new float[k * oh * ow];
            dDbias.Download<float>(actBias);
            AssertClose(expBias, actBias, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void LocallyConnected2DBackward_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, k = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;   // N*C*H*W=512
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
        var input = new float[n * c * h * w];
        var grad = new float[n * k * oh * ow];
        var weights = new float[oh * ow * k * c * kh * kw];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i) - 0.1f;
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicBias(i) + 0.2f;
        int Wi(int y, int x, int oc, int ic, int r, int t) => ((((y * ow + x) * k + oc) * c + ic) * kh + r) * kw + t;
        var expDx = new float[n * c * h * w];
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < c; ic++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < w; iw++)
                    {
                        double acc = 0;
                        for (int oc = 0; oc < k; oc++)
                            for (int r = 0; r < kh; r++)
                                for (int t = 0; t < kw; t++)
                                {
                                    int nh = ih + pad - r, nw = iw + pad - t;
                                    if (nh < 0 || nh % stride != 0 || nw < 0 || nw % stride != 0) continue;
                                    int y = nh / stride, x = nw / stride;
                                    if (y >= oh || x >= ow) continue;
                                    acc += (double)weights[Wi(y, x, oc, ic, r, t)] * grad[((b * k + oc) * oh + y) * ow + x];
                                }
                        expDx[((b * c + ic) * h + ih) * w + iw] = (float)acc;
                    }
        var expDw = new float[oh * ow * k * c * kh * kw];
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                for (int oc = 0; oc < k; oc++)
                    for (int ic = 0; ic < c; ic++)
                        for (int r = 0; r < kh; r++)
                            for (int t = 0; t < kw; t++)
                            {
                                int ih = y * stride + r - pad, iw = x * stride + t - pad;
                                double acc = 0;
                                if (ih >= 0 && ih < h && iw >= 0 && iw < w)
                                    for (int b = 0; b < n; b++)
                                        acc += (double)input[((b * c + ic) * h + ih) * w + iw] * grad[((b * k + oc) * oh + y) * ow + x];
                                expDw[Wi(y, x, oc, ic, r, t)] = (float)acc;
                            }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var bin = new PtxLocallyConnected2DBackwardInputKernel(runtime, n, c, k, h, w, kh, kw, stride, pad);
            using var dGrad = runtime.AllocateBytes((nuint)bin.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)bin.WeightBytes);
            using var dX = runtime.AllocateBytes((nuint)bin.GradInputBytes);
            dGrad.Upload<float>(grad); dW.Upload<float>(weights);
            bin.Launch(DirectPtxTensorView.CreateOwned(dGrad, bin.Blueprint.Tensors[0]),
                       DirectPtxTensorView.CreateOwned(dW, bin.Blueprint.Tensors[1]),
                       DirectPtxTensorView.CreateOwned(dX, bin.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDx = new float[n * c * h * w];
            dX.Download<float>(actDx);
            AssertClose(expDx, actDx, 2e-3f);

            using var bw = new PtxLocallyConnected2DBackwardWeightKernel(runtime, n, c, k, h, w, kh, kw, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)bw.InputBytes);
            using var dGrad2 = runtime.AllocateBytes((nuint)bw.GradOutputBytes);
            using var dDw = runtime.AllocateBytes((nuint)bw.GradWeightBytes);
            dInput.Upload<float>(input); dGrad2.Upload<float>(grad);
            bw.Launch(DirectPtxTensorView.CreateOwned(dInput, bw.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dGrad2, bw.Blueprint.Tensors[1]),
                      DirectPtxTensorView.CreateOwned(dDw, bw.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDw = new float[oh * ow * k * c * kh * kw];
            dDw.Download<float>(actDw);
            AssertClose(expDw, actDw, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    // Stride != 1 is a separate codegen branch (rem/div validity checks vs a plain move) and is
    // the shape exported as a released cubin; without this the committed artifact had no numerics.
    [Fact]
    public void ConvTranspose3DStride2_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, ci = 2, co = 4, d = 4, h = 4, w = 4, kd = 3, kh = 3, kw = 3, stride = 2, pad = 1, outpad = 1;
        int od = (d - 1) * stride - 2 * pad + kd + outpad, oh = (h - 1) * stride - 2 * pad + kh + outpad, ow = (w - 1) * stride - 2 * pad + kw + outpad;
        var input = new float[n * ci * d * h * w];
        var weights = new float[ci * co * kd * kh * kw];
        var bias = new float[co];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        int Iin(int b, int ic, int z, int y, int x) => (((b * ci + ic) * d + z) * h + y) * w + x;
        int Iw(int ic, int oc, int a, int r, int t) => (((ic * co + oc) * kd + a) * kh + r) * kw + t;
        var expected = new float[n * co * od * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < co; oc++)
                for (int z = 0; z < od; z++)
                    for (int y = 0; y < oh; y++)
                        for (int x = 0; x < ow; x++)
                        {
                            double acc = bias[oc];
                            for (int ic = 0; ic < ci; ic++)
                                for (int a = 0; a < kd; a++)
                                    for (int r = 0; r < kh; r++)
                                        for (int t = 0; t < kw; t++)
                                        {
                                            int nd = z + pad - a, nh = y + pad - r, nw = x + pad - t;
                                            if (nd < 0 || nd % stride != 0 || nh < 0 || nh % stride != 0 || nw < 0 || nw % stride != 0) continue;
                                            int id = nd / stride, ih = nh / stride, iw = nw / stride;
                                            if (id >= d || ih >= h || iw >= w) continue;
                                            acc += (double)input[Iin(b, ic, id, ih, iw)] * weights[Iw(ic, oc, a, r, t)];
                                        }
                            expected[(((b * co + oc) * od + z) * oh + y) * ow + x] = (float)Math.Max(acc, 0.0);
                        }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConvTranspose3DKernel(runtime, n, ci, co, d, h, w, kd, kh, kw, stride, pad, outpad, relu: true);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dBias.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * co * od * oh * ow];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void ConvTranspose3D_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, ci = 3, co = 4, d = 8, h = 8, w = 8, kd = 3, kh = 3, kw = 3, stride = 1, pad = 1, outpad = 0;
        int od = (d - 1) * stride - 2 * pad + kd + outpad, oh = (h - 1) * stride - 2 * pad + kh + outpad, ow = (w - 1) * stride - 2 * pad + kw + outpad;
        var input = new float[n * ci * d * h * w];
        var weights = new float[ci * co * kd * kh * kw];
        var bias = new float[co];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        int Iin(int b, int ic, int z, int y, int x) => (((b * ci + ic) * d + z) * h + y) * w + x;
        int Iw(int ic, int oc, int a, int r, int t) => (((ic * co + oc) * kd + a) * kh + r) * kw + t;
        var expected = new float[n * co * od * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < co; oc++)
                for (int z = 0; z < od; z++)
                    for (int y = 0; y < oh; y++)
                        for (int x = 0; x < ow; x++)
                        {
                            double acc = bias[oc];
                            for (int ic = 0; ic < ci; ic++)
                                for (int a = 0; a < kd; a++)
                                    for (int r = 0; r < kh; r++)
                                        for (int t = 0; t < kw; t++)
                                        {
                                            int nd = z + pad - a, nh = y + pad - r, nw = x + pad - t;
                                            if (nd < 0 || nd % stride != 0 || nh < 0 || nh % stride != 0 || nw < 0 || nw % stride != 0) continue;
                                            int id = nd / stride, ih = nh / stride, iw = nw / stride;
                                            if (id >= d || ih >= h || iw >= w) continue;
                                            acc += (double)input[Iin(b, ic, id, ih, iw)] * weights[Iw(ic, oc, a, r, t)];
                                        }
                            expected[(((b * co + oc) * od + z) * oh + y) * ow + x] = (float)Math.Max(acc, 0.0);
                        }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxConvTranspose3DKernel(runtime, n, ci, co, d, h, w, kd, kh, kw, stride, pad, outpad, relu: true);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dBias.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[3]));
            runtime.Synchronize();
            var actual = new float[n * co * od * oh * ow];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void ConvTranspose3DBackward_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, ci = 3, co = 4, d = 8, h = 8, w = 8, kd = 3, kh = 3, kw = 3, stride = 1, pad = 1, outpad = 0;
        int od = (d - 1) * stride - 2 * pad + kd + outpad, oh = (h - 1) * stride - 2 * pad + kh + outpad, ow = (w - 1) * stride - 2 * pad + kw + outpad;
        var input = new float[n * ci * d * h * w];
        var grad = new float[n * co * od * oh * ow];
        var weights = new float[ci * co * kd * kh * kw];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i) - 0.1f;
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicBias(i) + 0.2f;
        int Iin(int b, int c2, int z, int y, int x) => (((b * ci + c2) * d + z) * h + y) * w + x;
        int Ig(int b, int c2, int z, int y, int x) => (((b * co + c2) * od + z) * oh + y) * ow + x;
        int Iw(int c2, int oc, int a, int r, int t) => (((c2 * co + oc) * kd + a) * kh + r) * kw + t;
        var expDx = new float[n * ci * d * h * w];
        for (int b = 0; b < n; b++)
            for (int c2 = 0; c2 < ci; c2++)
                for (int id = 0; id < d; id++)
                    for (int ih = 0; ih < h; ih++)
                        for (int iw = 0; iw < w; iw++)
                        {
                            double acc = 0;
                            for (int oc = 0; oc < co; oc++)
                                for (int a = 0; a < kd; a++)
                                    for (int r = 0; r < kh; r++)
                                        for (int t = 0; t < kw; t++)
                                        {
                                            int z = id * stride - pad + a, y = ih * stride - pad + r, x = iw * stride - pad + t;
                                            if (z < 0 || z >= od || y < 0 || y >= oh || x < 0 || x >= ow) continue;
                                            acc += (double)grad[Ig(b, oc, z, y, x)] * weights[Iw(c2, oc, a, r, t)];
                                        }
                            expDx[Iin(b, c2, id, ih, iw)] = (float)acc;
                        }
        var expDw = new float[ci * co * kd * kh * kw];
        for (int c2 = 0; c2 < ci; c2++)
            for (int oc = 0; oc < co; oc++)
                for (int a = 0; a < kd; a++)
                    for (int r = 0; r < kh; r++)
                        for (int t = 0; t < kw; t++)
                        {
                            double acc = 0;
                            for (int b = 0; b < n; b++)
                                for (int id = 0; id < d; id++)
                                    for (int ih = 0; ih < h; ih++)
                                        for (int iw = 0; iw < w; iw++)
                                        {
                                            int z = id * stride - pad + a, y = ih * stride - pad + r, x = iw * stride - pad + t;
                                            if (z < 0 || z >= od || y < 0 || y >= oh || x < 0 || x >= ow) continue;
                                            acc += (double)input[Iin(b, c2, id, ih, iw)] * grad[Ig(b, oc, z, y, x)];
                                        }
                            expDw[Iw(c2, oc, a, r, t)] = (float)acc;
                        }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var bin = new PtxConvTranspose3DBackwardInputKernel(runtime, n, ci, co, d, h, w, kd, kh, kw, stride, pad, outpad);
            using var dGrad = runtime.AllocateBytes((nuint)bin.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)bin.WeightBytes);
            using var dX = runtime.AllocateBytes((nuint)bin.GradInputBytes);
            dGrad.Upload<float>(grad); dW.Upload<float>(weights);
            bin.Launch(DirectPtxTensorView.CreateOwned(dGrad, bin.Blueprint.Tensors[0]),
                       DirectPtxTensorView.CreateOwned(dW, bin.Blueprint.Tensors[1]),
                       DirectPtxTensorView.CreateOwned(dX, bin.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDx = new float[n * ci * d * h * w];
            dX.Download<float>(actDx);
            AssertClose(expDx, actDx, 2e-3f);

            using var bw = new PtxConvTranspose3DBackwardWeightKernel(runtime, n, ci, co, d, h, w, kd, kh, kw, stride, pad, outpad);
            using var dInput = runtime.AllocateBytes((nuint)bw.InputBytes);
            using var dGrad2 = runtime.AllocateBytes((nuint)bw.GradOutputBytes);
            using var dDw = runtime.AllocateBytes((nuint)bw.GradWeightBytes);
            dInput.Upload<float>(input); dGrad2.Upload<float>(grad);
            bw.Launch(DirectPtxTensorView.CreateOwned(dInput, bw.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dGrad2, bw.Blueprint.Tensors[1]),
                      DirectPtxTensorView.CreateOwned(dDw, bw.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actDw = new float[ci * co * kd * kh * kw];
            dDw.Download<float>(actDw);
            AssertClose(expDw, actDw, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void UnfoldKNFp16FromFp16_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
        int patchRows = c * kh * kw, cols = oh * ow;
        var inputHalf = new ushort[n * c * h * w];
        for (int i = 0; i < inputHalf.Length; i++) inputHalf[i] = BitConverter.HalfToUInt16Bits((Half)DeterministicInput(i));
        var expected = new ushort[n * patchRows * cols];   // 0-init = fp16 zero
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < c; ic++)
                for (int r = 0; r < kh; r++)
                    for (int t = 0; t < kw; t++)
                        for (int y = 0; y < oh; y++)
                            for (int x = 0; x < ow; x++)
                            {
                                int ih = y * stride + r - pad, iw = x * stride + t - pad;
                                ushort v = 0;
                                if (ih >= 0 && ih < h && iw >= 0 && iw < w) v = inputHalf[((b * c + ic) * h + ih) * w + iw];
                                expected[(b * patchRows + (ic * kh * kw + r * kw + t)) * cols + (y * ow + x)] = v;
                            }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxUnfold2DFp16FromFp16Kernel(runtime, n, c, h, w, kh, kw, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<ushort>(inputHalf);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[1]));
            runtime.Synchronize();
            var actual = new ushort[n * patchRows * cols];
            dOut.Download<ushort>(actual);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i]);   // exact half-to-half copy
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DeformableConv2D_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 3, k = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw;
        var input = new float[n * c * h * w];
        var weights = new float[k * c * taps];
        var offset = new float[n * 2 * taps * oh * ow];
        var mask = new float[n * taps * oh * ow];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 3) * 1.5f;   // fractional offsets
        for (int i = 0; i < mask.Length; i++) mask[i] = 0.5f + 0.4f * DeterministicWeight(i + 5);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        double Sample(int b, int ic, double py, double px)
        {
            int y0 = (int)Math.Floor(py), x0 = (int)Math.Floor(px);
            double wy1 = py - y0, wx1 = px - x0, wy0 = 1 - wy1, wx0 = 1 - wx1;
            double v = 0;
            void Corner(int yy, int xx, double cw) { if (yy >= 0 && yy < h && xx >= 0 && xx < w) v += cw * input[((b * c + ic) * h + yy) * w + xx]; }
            Corner(y0, x0, wy0 * wx0); Corner(y0, x0 + 1, wy0 * wx1); Corner(y0 + 1, x0, wy1 * wx0); Corner(y0 + 1, x0 + 1, wy1 * wx1);
            return v;
        }
        var expected = new float[n * k * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < k; oc++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        double acc = bias[oc];
                        for (int ic = 0; ic < c; ic++)
                            for (int pos = 0; pos < taps; pos++)
                            {
                                int r = pos / kw, t = pos % kw;
                                double offY = offset[((b * 2 * taps + 2 * pos) * oh + y) * ow + x];
                                double offX = offset[((b * 2 * taps + 2 * pos + 1) * oh + y) * ow + x];
                                double m = mask[((b * taps + pos) * oh + y) * ow + x];
                                double py = y * stride + r - pad + offY, px = x * stride + t - pad + offX;
                                acc += weights[(oc * c + ic) * taps + pos] * m * Sample(b, ic, py, px);
                            }
                        expected[((b * k + oc) * oh + y) * ow + x] = (float)acc;
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DKernel(runtime, n, c, k, h, w, kh, kw, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dMask = runtime.AllocateBytes((nuint)kernel.MaskBytes);
            using var dBias = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dOut = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dOff.Upload<float>(offset); dMask.Upload<float>(mask); dBias.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dMask, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dBias, kernel.Blueprint.Tensors[4]),
                          DirectPtxTensorView.CreateOwned(dOut, kernel.Blueprint.Tensors[5]));
            runtime.Synchronize();
            var actual = new float[n * k * oh * ow];
            dOut.Download<float>(actual);
            AssertClose(expected, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DeformableConv2DBackwardMask_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 4, c = 3, k = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;   // N*taps*OH*OW=2304
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw;
        var input = new float[n * c * h * w];
        var weights = new float[k * c * taps];
        var offset = new float[n * 2 * taps * oh * ow];
        var grad = new float[n * k * oh * ow];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 3) * 1.5f;
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 2) - 0.1f;
        double Sample(int b, int ic, double py, double px)
        {
            int y0 = (int)Math.Floor(py), x0 = (int)Math.Floor(px);
            double wy1 = py - y0, wx1 = px - x0, wy0 = 1 - wy1, wx0 = 1 - wx1, v = 0;
            void Cn(int yy, int xx, double cw) { if (yy >= 0 && yy < h && xx >= 0 && xx < w) v += cw * input[((b * c + ic) * h + yy) * w + xx]; }
            Cn(y0, x0, wy0 * wx0); Cn(y0, x0 + 1, wy0 * wx1); Cn(y0 + 1, x0, wy1 * wx0); Cn(y0 + 1, x0 + 1, wy1 * wx1);
            return v;
        }
        var expected = new float[n * taps * oh * ow];
        for (int b = 0; b < n; b++)
            for (int pos = 0; pos < taps; pos++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        int r = pos / kw, t = pos % kw;
                        double offY = offset[((b * 2 * taps + 2 * pos) * oh + y) * ow + x];
                        double offX = offset[((b * 2 * taps + 2 * pos + 1) * oh + y) * ow + x];
                        double py = y * stride + r - pad + offY, px = x * stride + t - pad + offX;
                        double acc = 0;
                        for (int ic = 0; ic < c; ic++)
                        {
                            double gk = 0;
                            for (int oc = 0; oc < k; oc++) gk += grad[((b * k + oc) * oh + y) * ow + x] * weights[(oc * c + ic) * taps + pos];
                            acc += gk * Sample(b, ic, py, px);
                        }
                        expected[((b * taps + pos) * oh + y) * ow + x] = (float)acc;
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DBackwardMaskKernel(runtime, n, c, k, h, w, kh, kw, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dGrad = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dMask = runtime.AllocateBytes((nuint)kernel.GradMaskBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dOff.Upload<float>(offset); dGrad.Upload<float>(grad);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dGrad, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dMask, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            var actual = new float[n * taps * oh * ow];
            dMask.Download<float>(actual);
            AssertClose(expected, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DeformableConv2DBackwardWeight_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 3, k = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw;
        var input = new float[n * c * h * w];
        var offset = new float[n * 2 * taps * oh * ow];
        var mask = new float[n * taps * oh * ow];
        var grad = new float[n * k * oh * ow];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 3) * 1.5f;
        for (int i = 0; i < mask.Length; i++) mask[i] = 0.5f + 0.4f * DeterministicWeight(i + 5);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 2) - 0.1f;
        double Sample(int b, int ic, double py, double px)
        {
            int y0 = (int)Math.Floor(py), x0 = (int)Math.Floor(px);
            double wy1 = py - y0, wx1 = px - x0, wy0 = 1 - wy1, wx0 = 1 - wx1, v = 0;
            void Cn(int yy, int xx, double cw) { if (yy >= 0 && yy < h && xx >= 0 && xx < w) v += cw * input[((b * c + ic) * h + yy) * w + xx]; }
            Cn(y0, x0, wy0 * wx0); Cn(y0, x0 + 1, wy0 * wx1); Cn(y0 + 1, x0, wy1 * wx0); Cn(y0 + 1, x0 + 1, wy1 * wx1);
            return v;
        }
        var expected = new float[k * c * taps];
        for (int oc = 0; oc < k; oc++)
            for (int ic = 0; ic < c; ic++)
                for (int pos = 0; pos < taps; pos++)
                {
                    int r = pos / kw, t = pos % kw;
                    double acc = 0;
                    for (int b = 0; b < n; b++)
                        for (int y = 0; y < oh; y++)
                            for (int x = 0; x < ow; x++)
                            {
                                double offY = offset[((b * 2 * taps + 2 * pos) * oh + y) * ow + x];
                                double offX = offset[((b * 2 * taps + 2 * pos + 1) * oh + y) * ow + x];
                                double m = mask[((b * taps + pos) * oh + y) * ow + x];
                                double g = grad[((b * k + oc) * oh + y) * ow + x];
                                double py = y * stride + r - pad + offY, px = x * stride + t - pad + offX;
                                acc += g * m * Sample(b, ic, py, px);
                            }
                    expected[(oc * c + ic) * taps + pos] = (float)acc;
                }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DBackwardWeightKernel(runtime, n, c, k, h, w, kh, kw, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dMask = runtime.AllocateBytes((nuint)kernel.MaskBytes);
            using var dGrad = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.GradWeightBytes);
            dInput.Upload<float>(input); dOff.Upload<float>(offset); dMask.Upload<float>(mask); dGrad.Upload<float>(grad);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dMask, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dGrad, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            var actual = new float[k * c * taps];
            dW.Download<float>(actual);
            AssertClose(expected, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DeformableConv2DBackwardOffset_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 4, c = 3, k = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;   // N*taps*OH*OW=2304
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw;
        var input = new float[n * c * h * w];
        var weights = new float[k * c * taps];
        var offset = new float[n * 2 * taps * oh * ow];
        var mask = new float[n * taps * oh * ow];
        var grad = new float[n * k * oh * ow];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 3) * 1.5f;
        for (int i = 0; i < mask.Length; i++) mask[i] = 0.5f + 0.4f * DeterministicWeight(i + 5);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 2) - 0.1f;
        double In(int b, int ic, int yy, int xx) => (yy >= 0 && yy < h && xx >= 0 && xx < w) ? input[((b * c + ic) * h + yy) * w + xx] : 0.0;
        var expected = new float[n * 2 * taps * oh * ow];
        for (int b = 0; b < n; b++)
            for (int pos = 0; pos < taps; pos++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        int r = pos / kw, t = pos % kw;
                        double offY = offset[((b * 2 * taps + 2 * pos) * oh + y) * ow + x];
                        double offX = offset[((b * 2 * taps + 2 * pos + 1) * oh + y) * ow + x];
                        double m = mask[((b * taps + pos) * oh + y) * ow + x];
                        double py = y * stride + r - pad + offY, px = x * stride + t - pad + offX;
                        int y0 = (int)Math.Floor(py), x0 = (int)Math.Floor(px);
                        double wy1 = py - y0, wx1 = px - x0, wy0 = 1 - wy1, wx0 = 1 - wx1;
                        double accY = 0, accX = 0;
                        for (int ic = 0; ic < c; ic++)
                        {
                            double v00 = In(b, ic, y0, x0), v01 = In(b, ic, y0, x0 + 1), v10 = In(b, ic, y0 + 1, x0), v11 = In(b, ic, y0 + 1, x0 + 1);
                            double dvaly = wx0 * (v10 - v00) + wx1 * (v11 - v01);
                            double dvalx = wy0 * (v01 - v00) + wy1 * (v11 - v10);
                            double gk = 0;
                            for (int oc = 0; oc < k; oc++) gk += grad[((b * k + oc) * oh + y) * ow + x] * weights[(oc * c + ic) * taps + pos];
                            accY += gk * dvaly; accX += gk * dvalx;
                        }
                        expected[((b * 2 * taps + 2 * pos) * oh + y) * ow + x] = (float)(m * accY);
                        expected[((b * 2 * taps + 2 * pos + 1) * oh + y) * ow + x] = (float)(m * accX);
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DBackwardOffsetKernel(runtime, n, c, k, h, w, kh, kw, stride, pad);
            using var dInput = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dMask = runtime.AllocateBytes((nuint)kernel.MaskBytes);
            using var dGrad = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dDoff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            dInput.Upload<float>(input); dW.Upload<float>(weights); dOff.Upload<float>(offset); dMask.Upload<float>(mask); dGrad.Upload<float>(grad);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dInput, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dMask, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dGrad, kernel.Blueprint.Tensors[4]),
                          DirectPtxTensorView.CreateOwned(dDoff, kernel.Blueprint.Tensors[5]));
            runtime.Synchronize();
            var actual = new float[n * 2 * taps * oh * ow];
            dDoff.Download<float>(actual);
            AssertClose(expected, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void FusedConv3D_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 2, k = 4, d = 4, h = 4, w = 4, kd = 3, kh = 3, kw = 3, stride = 1, pad = 1;
        int od = (d + 2 * pad - kd) / stride + 1, oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1;
        var input = new float[n * c * d * h * w];
        var weights = new float[k * c * kd * kh * kw];
        var bias = new float[k];
        var scale = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        for (int i = 0; i < scale.Length; i++) scale[i] = 0.5f + 0.5f * DeterministicWeight(i + 9);
        var expected = new float[n * k * od * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < k; oc++)
                for (int z = 0; z < od; z++)
                    for (int y = 0; y < oh; y++)
                        for (int x = 0; x < ow; x++)
                        {
                            double acc = bias[oc];
                            for (int ic = 0; ic < c; ic++)
                                for (int a = 0; a < kd; a++)
                                    for (int rr = 0; rr < kh; rr++)
                                        for (int t = 0; t < kw; t++)
                                        {
                                            int iz = z * stride + a - pad, iy = y * stride + rr - pad, ix = x * stride + t - pad;
                                            if (iz >= 0 && iz < d && iy >= 0 && iy < h && ix >= 0 && ix < w)
                                                acc += (double)input[(((b * c + ic) * d + iz) * h + iy) * w + ix]
                                                     * weights[(((oc * c + ic) * kd + a) * kh + rr) * kw + t];
                                        }
                            double v = scale[oc] * acc;
                            expected[(((b * k + oc) * od + z) * oh + y) * ow + x] = (float)Math.Max(v, 0);
                        }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxFusedConv3DKernel(runtime, n, c, k, d, h, w, kd, kh, kw, stride, pad);
            using var dI = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dB = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dS = runtime.AllocateBytes((nuint)kernel.ScaleBytes);
            using var dO = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dI.Upload<float>(input); dW.Upload<float>(weights); dB.Upload<float>(bias); dS.Upload<float>(scale);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dI, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dB, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dS, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dO, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            var actual = new float[n * k * od * oh * ow];
            dO.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void FusedConvTranspose2D_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, ci = 3, co = 4, h = 4, w = 4, kh = 3, kw = 3, stride = 2, pad = 1, outpad = 1;
        int oh = (h - 1) * stride - 2 * pad + kh + outpad, ow = (w - 1) * stride - 2 * pad + kw + outpad;
        var input = new float[n * ci * h * w];
        var weights = new float[ci * co * kh * kw];
        var bias = new float[co];
        var scale = new float[co];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        for (int i = 0; i < scale.Length; i++) scale[i] = 0.5f + 0.5f * DeterministicWeight(i + 9);
        var expected = new float[n * co * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < co; oc++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        double acc = bias[oc];
                        for (int c2 = 0; c2 < ci; c2++)
                            for (int r = 0; r < kh; r++)
                                for (int t = 0; t < kw; t++)
                                {
                                    int numH = y + pad - r, numW = x + pad - t;
                                    if (numH >= 0 && numH % stride == 0 && numW >= 0 && numW % stride == 0)
                                    {
                                        int ih = numH / stride, iw = numW / stride;
                                        if (ih < h && iw < w)
                                            acc += (double)input[((b * ci + c2) * h + ih) * w + iw]
                                                 * weights[((c2 * co + oc) * kh + r) * kw + t];
                                    }
                                }
                        double v = scale[oc] * acc;
                        expected[((b * co + oc) * oh + y) * ow + x] = (float)Math.Max(v, 0);
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxFusedConvTranspose2DKernel(runtime, n, ci, co, h, w, kh, kw, stride, pad, outpad);
            using var dI = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dB = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dS = runtime.AllocateBytes((nuint)kernel.ScaleBytes);
            using var dO = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dI.Upload<float>(input); dW.Upload<float>(weights); dB.Upload<float>(bias); dS.Upload<float>(scale);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dI, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dB, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dS, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dO, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            var actual = new float[n * co * oh * ow];
            dO.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    private static double GroupedBilinear(float[] input, int inBase, double py, double px, int h, int w)
    {
        int y0 = (int)Math.Floor(py), x0 = (int)Math.Floor(px);
        double wy1 = py - y0, wx1 = px - x0, wy0 = 1 - wy1, wx0 = 1 - wx1;
        double v = 0;
        void C(int yy, int xx, double cw) { if (yy >= 0 && yy < h && xx >= 0 && xx < w) v += cw * input[inBase + yy * w + xx]; }
        C(y0, x0, wy0 * wx0); C(y0, x0 + 1, wy0 * wx1); C(y0 + 1, x0, wy1 * wx0); C(y0 + 1, x0 + 1, wy1 * wx1);
        return v;
    }

    [Fact]
    public void DeformableConv2DGroupedBackwardOffset_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, k = 3, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1, dg = 2;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw, cpg = c / dg;
        var input = new float[n * c * h * w];
        var weights = new float[k * c * taps];
        var offset = new float[n * dg * 2 * taps * oh * ow];
        var mask = new float[n * dg * taps * oh * ow];
        var grad = new float[n * k * oh * ow];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 7) * 1.5f;
        for (int i = 0; i < mask.Length; i++) mask[i] = 0.5f + 0.4f * DeterministicWeight(i + 3);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 2) - 0.1f;
        var expected = new float[n * dg * 2 * taps * oh * ow];
        for (int b = 0; b < n; b++)
            for (int g = 0; g < dg; g++)
                for (int pos = 0; pos < taps; pos++)
                    for (int y = 0; y < oh; y++)
                        for (int x = 0; x < ow; x++)
                        {
                            int r = pos / kw, t = pos % kw;
                            int offYc = b * dg * 2 * taps + g * 2 * taps + 2 * pos;
                            double oY = offset[(offYc * oh + y) * ow + x], oX = offset[((offYc + 1) * oh + y) * ow + x];
                            double m = mask[((b * dg * taps + g * taps + pos) * oh + y) * ow + x];
                            double py = y * stride + r - pad + oY, px = x * stride + t - pad + oX;
                            int y0 = (int)Math.Floor(py), x0 = (int)Math.Floor(px);
                            double wy1 = py - y0, wx1 = px - x0, wy0 = 1 - wy1, wx0 = 1 - wx1;
                            double accY = 0, accX = 0;
                            for (int ic = g * cpg; ic < (g + 1) * cpg; ic++)
                            {
                                int ib = (b * c + ic) * h * w;
                                double v00 = (y0 >= 0 && y0 < h && x0 >= 0 && x0 < w) ? input[ib + y0 * w + x0] : 0;
                                double v01 = (y0 >= 0 && y0 < h && x0 + 1 >= 0 && x0 + 1 < w) ? input[ib + y0 * w + x0 + 1] : 0;
                                double v10 = (y0 + 1 >= 0 && y0 + 1 < h && x0 >= 0 && x0 < w) ? input[ib + (y0 + 1) * w + x0] : 0;
                                double v11 = (y0 + 1 >= 0 && y0 + 1 < h && x0 + 1 >= 0 && x0 + 1 < w) ? input[ib + (y0 + 1) * w + x0 + 1] : 0;
                                double dvaly = wx0 * (v10 - v00) + wx1 * (v11 - v01);
                                double dvalx = wy0 * (v01 - v00) + wy1 * (v11 - v10);
                                double gk = 0;
                                for (int oc = 0; oc < k; oc++) gk += grad[((b * k + oc) * oh + y) * ow + x] * weights[(oc * c + ic) * taps + pos];
                                accY += gk * dvaly; accX += gk * dvalx;
                            }
                            expected[(offYc * oh + y) * ow + x] = (float)(accY * m);
                            expected[((offYc + 1) * oh + y) * ow + x] = (float)(accX * m);
                        }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DGroupedBackwardOffsetKernel(runtime, n, c, k, h, w, kh, kw, stride, pad, dg);
            using var dI = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dM = runtime.AllocateBytes((nuint)kernel.MaskBytes);
            using var dG = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dD = runtime.AllocateBytes((nuint)kernel.GradOffsetBytes);
            dI.Upload<float>(input); dW.Upload<float>(weights); dOff.Upload<float>(offset); dM.Upload<float>(mask); dG.Upload<float>(grad);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dI, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dM, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dG, kernel.Blueprint.Tensors[4]),
                          DirectPtxTensorView.CreateOwned(dD, kernel.Blueprint.Tensors[5]));
            runtime.Synchronize();
            var actual = new float[n * dg * 2 * taps * oh * ow];
            dD.Download<float>(actual);
            AssertClose(expected, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DeformableConv2DGroupedBackwardMask_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, k = 3, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1, dg = 2;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw, cpg = c / dg;
        var input = new float[n * c * h * w];
        var weights = new float[k * c * taps];
        var offset = new float[n * dg * 2 * taps * oh * ow];
        var grad = new float[n * k * oh * ow];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 7) * 1.5f;
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 2) - 0.1f;
        var expected = new float[n * dg * taps * oh * ow];
        for (int b = 0; b < n; b++)
            for (int g = 0; g < dg; g++)
                for (int pos = 0; pos < taps; pos++)
                    for (int y = 0; y < oh; y++)
                        for (int x = 0; x < ow; x++)
                        {
                            int r = pos / kw, t = pos % kw;
                            int offYc = b * dg * 2 * taps + g * 2 * taps + 2 * pos;
                            double oY = offset[(offYc * oh + y) * ow + x], oX = offset[((offYc + 1) * oh + y) * ow + x];
                            double py = y * stride + r - pad + oY, px = x * stride + t - pad + oX;
                            double acc = 0;
                            for (int ic = g * cpg; ic < (g + 1) * cpg; ic++)
                            {
                                double val = GroupedBilinear(input, (b * c + ic) * h * w, py, px, h, w);
                                double gk = 0;
                                for (int oc = 0; oc < k; oc++) gk += grad[((b * k + oc) * oh + y) * ow + x] * weights[(oc * c + ic) * taps + pos];
                                acc += gk * val;
                            }
                            expected[((b * dg * taps + g * taps + pos) * oh + y) * ow + x] = (float)acc;
                        }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DGroupedBackwardMaskKernel(runtime, n, c, k, h, w, kh, kw, stride, pad, dg);
            using var dI = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dG = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dM = runtime.AllocateBytes((nuint)kernel.GradMaskBytes);
            dI.Upload<float>(input); dW.Upload<float>(weights); dOff.Upload<float>(offset); dG.Upload<float>(grad);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dI, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dG, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dM, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            var actual = new float[n * dg * taps * oh * ow];
            dM.Download<float>(actual);
            AssertClose(expected, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DeformableConv2DGroupedForward_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, k = 3, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1, dg = 2;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw, cpg = c / dg;
        var input = new float[n * c * h * w];
        var weights = new float[k * c * taps];
        var offset = new float[n * dg * 2 * taps * oh * ow];
        var mask = new float[n * dg * taps * oh * ow];
        var bias = new float[k];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 7) * 1.5f;
        for (int i = 0; i < mask.Length; i++) mask[i] = 0.5f + 0.4f * DeterministicWeight(i + 3);
        for (int i = 0; i < bias.Length; i++) bias[i] = DeterministicBias(i);
        var expected = new float[n * k * oh * ow];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < k; oc++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                    {
                        double acc = bias[oc];
                        for (int ic = 0; ic < c; ic++)
                        {
                            int g = ic / cpg;
                            for (int pos = 0; pos < taps; pos++)
                            {
                                int r = pos / kw, t = pos % kw;
                                int offY = ((b * dg * 2 * taps + g * 2 * taps + 2 * pos) * oh + y) * ow + x;
                                double oY = offset[offY], oX = offset[offY + oh * ow];
                                double m = mask[((b * dg * taps + g * taps + pos) * oh + y) * ow + x];
                                double py = y * stride + r - pad + oY, px = x * stride + t - pad + oX;
                                double val = GroupedBilinear(input, (b * c + ic) * h * w, py, px, h, w);
                                acc += weights[(oc * c + ic) * taps + pos] * m * val;
                            }
                        }
                        expected[((b * k + oc) * oh + y) * ow + x] = (float)acc;
                    }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DGroupedForwardKernel(runtime, n, c, k, h, w, kh, kw, stride, pad, dg);
            using var dI = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dM = runtime.AllocateBytes((nuint)kernel.MaskBytes);
            using var dB = runtime.AllocateBytes((nuint)kernel.BiasBytes);
            using var dO = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dI.Upload<float>(input); dW.Upload<float>(weights); dOff.Upload<float>(offset); dM.Upload<float>(mask); dB.Upload<float>(bias);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dI, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dM, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dB, kernel.Blueprint.Tensors[4]),
                          DirectPtxTensorView.CreateOwned(dO, kernel.Blueprint.Tensors[5]));
            runtime.Synchronize();
            var actual = new float[n * k * oh * ow];
            dO.Download<float>(actual);
            AssertClose(expected, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DeformableConv2DGroupedBackwardInput_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, k = 3, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1, dg = 2;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw, cpg = c / dg;
        var weights = new float[k * c * taps];
        var offset = new float[n * dg * 2 * taps * oh * ow];
        var mask = new float[n * dg * taps * oh * ow];
        var grad = new float[n * k * oh * ow];
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 7) * 1.5f;
        for (int i = 0; i < mask.Length; i++) mask[i] = 0.5f + 0.4f * DeterministicWeight(i + 3);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 2) - 0.1f;
        var expected = new double[n * c * h * w];
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < c; ic++)
            {
                int g = ic / cpg;
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                        for (int pos = 0; pos < taps; pos++)
                        {
                            int r = pos / kw, t = pos % kw;
                            int offY = ((b * dg * 2 * taps + g * 2 * taps + 2 * pos) * oh + y) * ow + x;
                            double oY = offset[offY], oX = offset[offY + oh * ow];
                            double m = mask[((b * dg * taps + g * taps + pos) * oh + y) * ow + x];
                            double top = 0;
                            for (int oc = 0; oc < k; oc++) top += grad[((b * k + oc) * oh + y) * ow + x] * weights[(oc * c + ic) * taps + pos];
                            double contrib = top * m;
                            double py = y * stride + r - pad + oY, px = x * stride + t - pad + oX;
                            int y0 = (int)Math.Floor(py), x0 = (int)Math.Floor(px);
                            double wy1 = py - y0, wx1 = px - x0, wy0 = 1 - wy1, wx0 = 1 - wx1;
                            void Sc(int yy, int xx, double cw) { if (yy >= 0 && yy < h && xx >= 0 && xx < w) expected[((b * c + ic) * h + yy) * w + xx] += contrib * cw; }
                            Sc(y0, x0, wy0 * wx0); Sc(y0, x0 + 1, wy0 * wx1); Sc(y0 + 1, x0, wy1 * wx0); Sc(y0 + 1, x0 + 1, wy1 * wx1);
                        }
            }
        var expF = new float[n * c * h * w];
        for (int i = 0; i < expF.Length; i++) expF[i] = (float)expected[i];

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DGroupedBackwardInputKernel(runtime, n, c, k, h, w, kh, kw, stride, pad, dg);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dM = runtime.AllocateBytes((nuint)kernel.MaskBytes);
            using var dG = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dX = runtime.AllocateBytes((nuint)kernel.GradInputBytes);
            dW.Upload<float>(weights); dOff.Upload<float>(offset); dM.Upload<float>(mask); dG.Upload<float>(grad);
            dX.Upload<float>(new float[n * c * h * w]);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dM, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dG, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dX, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            var actual = new float[n * c * h * w];
            dX.Download<float>(actual);
            AssertClose(expF, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DeformableConv2DGroupedBackwardWeight_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 4, k = 3, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1, dg = 2;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw, cpg = c / dg;
        var input = new float[n * c * h * w];
        var offset = new float[n * dg * 2 * taps * oh * ow];
        var mask = new float[n * dg * taps * oh * ow];
        var grad = new float[n * k * oh * ow];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 7) * 1.5f;
        for (int i = 0; i < mask.Length; i++) mask[i] = 0.5f + 0.4f * DeterministicWeight(i + 3);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 2) - 0.1f;
        var expected = new float[k * c * taps];
        for (int oc = 0; oc < k; oc++)
            for (int ic = 0; ic < c; ic++)
            {
                int g = ic / cpg;
                for (int pos = 0; pos < taps; pos++)
                {
                    int r = pos / kw, t = pos % kw;
                    double acc = 0;
                    for (int b = 0; b < n; b++)
                        for (int y = 0; y < oh; y++)
                            for (int x = 0; x < ow; x++)
                            {
                                int offY = ((b * dg * 2 * taps + g * 2 * taps + 2 * pos) * oh + y) * ow + x;
                                double oY = offset[offY], oX = offset[offY + oh * ow];
                                double m = mask[((b * dg * taps + g * taps + pos) * oh + y) * ow + x];
                                double gv = grad[((b * k + oc) * oh + y) * ow + x];
                                double py = y * stride + r - pad + oY, px = x * stride + t - pad + oX;
                                double val = GroupedBilinear(input, (b * c + ic) * h * w, py, px, h, w);
                                acc += gv * m * val;
                            }
                    expected[(oc * c + ic) * taps + pos] = (float)acc;
                }
            }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DGroupedBackwardWeightKernel(runtime, n, c, k, h, w, kh, kw, stride, pad, dg);
            using var dI = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dM = runtime.AllocateBytes((nuint)kernel.MaskBytes);
            using var dG = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.GradWeightBytes);
            dI.Upload<float>(input); dOff.Upload<float>(offset); dM.Upload<float>(mask); dG.Upload<float>(grad);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dI, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dM, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dG, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            var actual = new float[k * c * taps];
            dW.Download<float>(actual);
            AssertClose(expected, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DepthwiseConv1DForward_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 3, c = 5, l = 32, kl = 3, stride = 1, pad = 1;
        int ol = (l + 2 * pad - kl) / stride + 1;
        var input = new float[n * c * l];
        var weights = new float[c * kl];
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        var expected = new float[n * c * ol];
        for (int b = 0; b < n; b++)
            for (int ch = 0; ch < c; ch++)
                for (int o = 0; o < ol; o++)
                {
                    double acc = 0;
                    for (int t = 0; t < kl; t++)
                    {
                        int il = o * stride + t - pad;
                        if (il >= 0 && il < l) acc += (double)input[(b * c + ch) * l + il] * weights[ch * kl + t];
                    }
                    expected[(b * c + ch) * ol + o] = (float)acc;
                }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDepthwiseConv1DForwardKernel(runtime, n, c, l, kl, stride, pad);
            using var dI = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dO = runtime.AllocateBytes((nuint)kernel.OutputBytes);
            dI.Upload<float>(input); dW.Upload<float>(weights);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dI, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dO, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[n * c * ol];
            dO.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DepthwiseConv1DBackwardInput_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 3, c = 5, l = 32, kl = 3, stride = 1, pad = 1;
        int ol = (l + 2 * pad - kl) / stride + 1;
        var grad = new float[n * c * ol];
        var weights = new float[c * kl];
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 1);
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        var expected = new float[n * c * l];
        for (int b = 0; b < n; b++)
            for (int ch = 0; ch < c; ch++)
                for (int il = 0; il < l; il++)
                {
                    double acc = 0;
                    for (int t = 0; t < kl; t++)
                    {
                        int num = il + pad - t;
                        if (num >= 0 && num % stride == 0)
                        {
                            int o = num / stride;
                            if (o < ol) acc += (double)grad[(b * c + ch) * ol + o] * weights[ch * kl + t];
                        }
                    }
                    expected[(b * c + ch) * l + il] = (float)acc;
                }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDepthwiseConv1DBackwardInputKernel(runtime, n, c, l, kl, stride, pad);
            using var dG = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dX = runtime.AllocateBytes((nuint)kernel.GradInputBytes);
            dG.Upload<float>(grad); dW.Upload<float>(weights);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dG, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dX, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[n * c * l];
            dX.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DepthwiseConv1DBackwardWeight_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 3, c = 5, l = 32, kl = 3, stride = 1, pad = 1;
        int ol = (l + 2 * pad - kl) / stride + 1;
        var grad = new float[n * c * ol];
        var input = new float[n * c * l];
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 1);
        for (int i = 0; i < input.Length; i++) input[i] = DeterministicInput(i);
        var expected = new float[c * kl];
        for (int ch = 0; ch < c; ch++)
            for (int t = 0; t < kl; t++)
            {
                double acc = 0;
                for (int b = 0; b < n; b++)
                    for (int o = 0; o < ol; o++)
                    {
                        int il = o * stride + t - pad;
                        if (il >= 0 && il < l) acc += (double)grad[(b * c + ch) * ol + o] * input[(b * c + ch) * l + il];
                    }
                expected[ch * kl + t] = (float)acc;
            }

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor)) return;
        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDepthwiseConv1DBackwardWeightKernel(runtime, n, c, l, kl, stride, pad);
            using var dG = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dI = runtime.AllocateBytes((nuint)kernel.InputBytes);
            using var dW = runtime.AllocateBytes((nuint)kernel.GradWeightBytes);
            dG.Upload<float>(grad); dI.Upload<float>(input);
            kernel.Launch(DirectPtxTensorView.CreateOwned(dG, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dI, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[2]));
            runtime.Synchronize();
            var actual = new float[c * kl];
            dW.Download<float>(actual);
            AssertClose(expected, actual, 2e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DeformableConv2DBackwardInput_MatchesCpuReference()
    {
        if (!DirectPtxRuntime.IsAvailable) return;

        const int n = 2, c = 3, k = 4, h = 8, w = 8, kh = 3, kw = 3, stride = 1, pad = 1;
        int oh = (h + 2 * pad - kh) / stride + 1, ow = (w + 2 * pad - kw) / stride + 1, taps = kh * kw;
        var weights = new float[k * c * taps];
        var offset = new float[n * 2 * taps * oh * ow];
        var mask = new float[n * taps * oh * ow];
        var grad = new float[n * k * oh * ow];
        for (int i = 0; i < weights.Length; i++) weights[i] = DeterministicWeight(i);
        for (int i = 0; i < offset.Length; i++) offset[i] = DeterministicInput(i + 3) * 1.5f;
        for (int i = 0; i < mask.Length; i++) mask[i] = 0.5f + 0.4f * DeterministicWeight(i + 5);
        for (int i = 0; i < grad.Length; i++) grad[i] = DeterministicWeight(i + 2) - 0.1f;
        var expected = new double[n * c * h * w];
        for (int b = 0; b < n; b++)
            for (int ic = 0; ic < c; ic++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++)
                        for (int pos = 0; pos < taps; pos++)
                        {
                            int r = pos / kw, t = pos % kw;
                            double offY = offset[((b * 2 * taps + 2 * pos) * oh + y) * ow + x];
                            double offX = offset[((b * 2 * taps + 2 * pos + 1) * oh + y) * ow + x];
                            double m = mask[((b * taps + pos) * oh + y) * ow + x];
                            double top = 0;
                            for (int oc = 0; oc < k; oc++) top += grad[((b * k + oc) * oh + y) * ow + x] * weights[(oc * c + ic) * taps + pos];
                            double contrib = top * m;
                            double py = y * stride + r - pad + offY, px = x * stride + t - pad + offX;
                            int y0 = (int)Math.Floor(py), x0 = (int)Math.Floor(px);
                            double wy1 = py - y0, wx1 = px - x0, wy0 = 1 - wy1, wx0 = 1 - wx1;
                            void Sc(int yy, int xx, double cw) { if (yy >= 0 && yy < h && xx >= 0 && xx < w) expected[((b * c + ic) * h + yy) * w + xx] += contrib * cw; }
                            Sc(y0, x0, wy0 * wx0); Sc(y0, x0 + 1, wy0 * wx1); Sc(y0 + 1, x0, wy1 * wx0); Sc(y0 + 1, x0 + 1, wy1 * wx1);
                        }
        var expF = new float[n * c * h * w];
        for (int i = 0; i < expF.Length; i++) expF[i] = (float)expected[i];

        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            return;

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            using var kernel = new PtxDeformableConv2DBackwardInputKernel(runtime, n, c, k, h, w, kh, kw, stride, pad);
            using var dW = runtime.AllocateBytes((nuint)kernel.WeightBytes);
            using var dOff = runtime.AllocateBytes((nuint)kernel.OffsetBytes);
            using var dMask = runtime.AllocateBytes((nuint)kernel.MaskBytes);
            using var dGrad = runtime.AllocateBytes((nuint)kernel.GradOutputBytes);
            using var dX = runtime.AllocateBytes((nuint)kernel.GradInputBytes);
            dW.Upload<float>(weights); dOff.Upload<float>(offset); dMask.Upload<float>(mask); dGrad.Upload<float>(grad);
            dX.Upload<float>(new float[n * c * h * w]);   // zero-init for atomic accumulation
            kernel.Launch(DirectPtxTensorView.CreateOwned(dW, kernel.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dOff, kernel.Blueprint.Tensors[1]),
                          DirectPtxTensorView.CreateOwned(dMask, kernel.Blueprint.Tensors[2]),
                          DirectPtxTensorView.CreateOwned(dGrad, kernel.Blueprint.Tensors[3]),
                          DirectPtxTensorView.CreateOwned(dX, kernel.Blueprint.Tensors[4]));
            runtime.Synchronize();
            var actual = new float[n * c * h * w];
            dX.Download<float>(actual);
            AssertClose(expF, actual, 3e-3f);
        }
        finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = prior; }
    }

    [Fact]
    public void DumpWinogradPtxForSassAnalysis()
    {
        string dir = Environment.GetEnvironmentVariable("PTX_DUMP_DIR");
        if (string.IsNullOrEmpty(dir)) return;   // opt-in only
        System.IO.Directory.CreateDirectory(dir);
        void W(string name, string ptx) => System.IO.File.WriteAllText(System.IO.Path.Combine(dir, name), ptx);
        // ResNet C64 perf shape.
        W("coopblk.ptx", PtxWinogradWmmaCoopBlockedKernel.EmitPtx(8, 6, 32, 64, 56, 56, 64));
        W("coop.ptx", PtxWinogradWmmaCoopKernel.EmitPtx(8, 6, 32, 64, 56, 56, 64));
        if (DirectPtxRuntime.IsAvailable)
        {
            using var rt = new DirectPtxRuntime();
            bool pr = DirectPtxFeatureGate.ConvolutionExperimentOverride;
            DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
            try { W("allk.ptx", new PtxWinogradWmmaFusedAllKKernel(rt, 32, 64, 56, 56, 64).Ptx); }
            finally { DirectPtxFeatureGate.ConvolutionExperimentOverride = pr; }
        }
        W("staged.ptx", PtxWinogradWmmaFusedStagedKernel.EmitPtx(8, 6, 32, 64, 56, 56, 64));
        W("fused.ptx", PtxWinogradWmmaFusedKernel.EmitPtx(8, 6, 32, 64, 56, 56, 64));
        W("wmmagemm.ptx", PtxWinogradWmmaBatchedGemmKernel.EmitPtx(8, 6, 64, 64, 25088));
    }

    private static float[] LaunchWinograd(
        DirectPtxRuntime runtime, Conv2DWinogradShape shape,
        float[] input, float[] weights, float[] bias)
    {
        using var kernel = new PtxConv2DNchw3x3WinogradF23Kernel(runtime, shape);
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
        var actual = new float[shape.Batch * shape.OutputChannels * shape.Height * shape.Width];
        dOutput.Download<float>(actual);
        return actual;
    }

    // Direct 3x3 stride-1 same-padded conv + bias + ReLU, fp64 accumulation.
    private static float[] ReferenceConv3x3Same(
        float[] input, float[] weights, float[] bias, int n, int c, int h, int w, int k)
    {
        var output = new float[n * k * h * w];
        for (int b = 0; b < n; b++)
            for (int oc = 0; oc < k; oc++)
                for (int oh = 0; oh < h; oh++)
                    for (int ow = 0; ow < w; ow++)
                    {
                        double acc = bias[oc];
                        for (int ic = 0; ic < c; ic++)
                            for (int gi = 0; gi < 3; gi++)
                                for (int gj = 0; gj < 3; gj++)
                                {
                                    int ih = oh + gi - 1, iw = ow + gj - 1;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    acc += (double)input[((b * c + ic) * h + ih) * w + iw] *
                                           weights[((oc * c + ic) * 3 + gi) * 3 + gj];
                                }
                        output[((b * k + oc) * h + oh) * w + ow] = (float)Math.Max(acc, 0.0);
                    }
        return output;
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

    private static void AssertClose(float[] expected, float[] actual) => AssertClose(expected, actual, Tolerance);

    private static void AssertClose(float[] expected, float[] actual, float tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxErr = 0f;
        int worst = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = Math.Abs(expected[i] - actual[i]);
            if (e > maxErr) { maxErr = e; worst = i; }
        }
        Assert.True(maxErr <= tol,
            $"max abs error {maxErr:E3} > {tol:E3} at index {worst} " +
            $"(expected {(worst >= 0 ? expected[worst] : 0)}, actual {(worst >= 0 ? actual[worst] : 0)})");
    }
}
#endif
