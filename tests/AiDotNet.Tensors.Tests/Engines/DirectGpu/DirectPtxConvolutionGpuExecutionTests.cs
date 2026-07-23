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
