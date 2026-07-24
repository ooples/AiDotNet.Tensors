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
