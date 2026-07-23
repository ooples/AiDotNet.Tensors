#if NET5_0_OR_GREATER
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// On-device throughput probe for the direct-PTX convolution kernels. Reports
/// median / P95 per-launch latency and GFLOP/s (30 warmups + 101 timed samples,
/// each a launch followed by a device sync). Not a pass/fail gate — it prints
/// honest measured numbers for the #841 evidence table. Skips without a GPU.
/// </summary>
public sealed class DirectPtxConvolutionPerfProbe
{
    private readonly ITestOutputHelper _out;
    public DirectPtxConvolutionPerfProbe(ITestOutputHelper output) => _out = output;

    private const int Warmups = 30;
    private const int Samples = 101;

    [Fact]
    public void Measure_V1_And_Tiled_ResNetC64()
    {
        if (!DirectPtxRuntime.IsAvailable) { _out.WriteLine("no CUDA device"); return; }
        using var runtime = new DirectPtxRuntime();
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
        { _out.WriteLine("no SM86 specialization"); return; }

        _out.WriteLine($"device={runtime.DeviceName} sm{runtime.ComputeCapabilityMajor}{runtime.ComputeCapabilityMinor} drv{runtime.DriverVersion}");

        bool prior = DirectPtxFeatureGate.ConvolutionExperimentOverride;
        DirectPtxFeatureGate.ConvolutionExperimentOverride = true;
        try
        {
            MeasureV1(runtime);
            // ResNet 1x1 bottleneck: N32, C64, H56, W56, K64 (HW=3136), tile 16.
            MeasureTiled(runtime, new Conv2DTiledShape(32, 64, 64, 3136, 16));
            MeasureRegBlocked(runtime, new Conv2DRegBlockShape(32, 64, 64, 3136, 64, 64, 16, 4, 4));
            // ResNet c64 3x3 same-conv via Winograd F(2,3): N32/C64/56x56/K64.
            MeasureWinograd(runtime, new Conv2DWinogradShape(32, 64, 56, 56, 64));
            MeasureWinogradPretransformed(runtime, 32, 64, 56, 56, 64);
            MeasureWinogradFused(runtime, 32, 64, 56, 56, 64);
            MeasureWinogradBatched(runtime, 32, 64, 56, 56, 64);
            MeasureWinogradFusedRB(runtime, 32, 64, 56, 56, 64);
            MeasureWinogradWmma(runtime, 32, 64, 56, 56, 64);
            MeasureWinogradWmmaFused(runtime, 32, 64, 56, 56, 64);
            MeasureWinogradWmmaFusedStaged(runtime, 32, 64, 56, 56, 64);
            MeasureWinogradWmmaCoop(runtime, 32, 64, 56, 56, 64);
            MeasureWinogradWmmaCoopBlocked(runtime, 32, 64, 56, 56, 64);
            MeasureStagesInIsolation(runtime, 32, 64, 56, 56, 64);
            MeasureWinogradWmmaFullyFused(runtime, 32, 64, 56, 56, 64);
        }
        finally
        {
            DirectPtxFeatureGate.ConvolutionExperimentOverride = prior;
        }
    }

    private void MeasureV1(DirectPtxRuntime runtime)
    {
        using var kernel = new PtxFusedConv2DNchwK1Kernel(runtime);
        using var input = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.InputBytes);
        using var weights = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.WeightBytes);
        using var bias = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.BiasBytes);
        using var output = runtime.AllocateBytes((nuint)PtxFusedConv2DNchwK1Kernel.OutputBytes);
        input.Upload<float>(new float[PtxFusedConv2DNchwK1Kernel.InputBytes / sizeof(float)]);
        weights.Upload<float>(new float[PtxFusedConv2DNchwK1Kernel.WeightBytes / sizeof(float)]);
        bias.Upload<float>(new float[PtxFusedConv2DNchwK1Kernel.BiasBytes / sizeof(float)]);

        void Launch() => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));

        long flops = 2L * PtxFusedConv2DNchwK1Kernel.OutputElements * PtxFusedConv2DNchwK1Kernel.InputChannels;
        Report("v1  N1/C64/H16/W16/K64 1x1", runtime, Launch, flops);
    }

    private void MeasureTiled(DirectPtxRuntime runtime, Conv2DTiledShape shape)
    {
        using var kernel = new PtxConv2DNchwK1TiledKernel(runtime, shape);
        using var input = runtime.AllocateBytes((nuint)shape.InputBytes);
        using var weights = runtime.AllocateBytes((nuint)shape.WeightBytes);
        using var bias = runtime.AllocateBytes((nuint)shape.BiasBytes);
        using var output = runtime.AllocateBytes((nuint)shape.OutputBytes);
        input.Upload<float>(new float[shape.InputBytes / sizeof(float)]);
        weights.Upload<float>(new float[shape.WeightBytes / sizeof(float)]);
        bias.Upload<float>(new float[shape.BiasBytes / sizeof(float)]);

        void Launch() => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));

        long flops = 2L * shape.Batch * shape.OutputChannels * shape.Spatial * shape.InputChannels;
        Report($"tiled N{shape.Batch}/C{shape.InputChannels}/HW{shape.Spatial}/K{shape.OutputChannels} 1x1 tile{shape.Tile}",
            runtime, Launch, flops);
    }

    private void MeasureRegBlocked(DirectPtxRuntime runtime, Conv2DRegBlockShape shape)
    {
        using var kernel = new PtxConv2DNchwK1RegBlockedKernel(runtime, shape);
        using var input = runtime.AllocateBytes((nuint)shape.InputBytes);
        using var weights = runtime.AllocateBytes((nuint)shape.WeightBytes);
        using var bias = runtime.AllocateBytes((nuint)shape.BiasBytes);
        using var output = runtime.AllocateBytes((nuint)shape.OutputBytes);
        input.Upload<float>(new float[shape.InputBytes / sizeof(float)]);
        weights.Upload<float>(new float[shape.WeightBytes / sizeof(float)]);
        bias.Upload<float>(new float[shape.BiasBytes / sizeof(float)]);

        void Launch() => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));

        long flops = 2L * shape.Batch * shape.OutputChannels * shape.Spatial * shape.InputChannels;
        Report($"regblk N{shape.Batch}/C{shape.InputChannels}/HW{shape.Spatial}/K{shape.OutputChannels} 1x1 " +
            $"{shape.BlockM}x{shape.BlockN}x{shape.BlockK}/{shape.ThreadM}x{shape.ThreadN} regs={kernel.FunctionInfo.RegistersPerThread}",
            runtime, Launch, flops);
    }

    private void MeasureWinograd(DirectPtxRuntime runtime, Conv2DWinogradShape shape)
    {
        using var kernel = new PtxConv2DNchw3x3WinogradF23Kernel(runtime, shape);
        using var input = runtime.AllocateBytes((nuint)shape.InputBytes);
        using var weights = runtime.AllocateBytes((nuint)shape.WeightBytes);
        using var bias = runtime.AllocateBytes((nuint)shape.BiasBytes);
        using var output = runtime.AllocateBytes((nuint)shape.OutputBytes);
        input.Upload<float>(new float[shape.InputBytes / sizeof(float)]);
        weights.Upload<float>(new float[shape.WeightBytes / sizeof(float)]);
        bias.Upload<float>(new float[shape.BiasBytes / sizeof(float)]);

        void Launch() => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weights, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));

        // Report against nominal direct-conv FLOPs (2*N*K*H*W*C*9) — the standard.
        long flops = 2L * shape.Batch * shape.OutputChannels * shape.Height * shape.Width * shape.InputChannels * 9;
        Report($"winograd N{shape.Batch}/C{shape.InputChannels}/{shape.Height}x{shape.Width}/K{shape.OutputChannels} 3x3 " +
            $"regs={kernel.FunctionInfo.RegistersPerThread}", runtime, Launch, flops);
    }

    private void MeasureWinogradPretransformed(
        DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        // Stage 1 (one-time): filter transform weights -> U.
        using var filter = new PtxWinogradF23FilterTransformKernel(runtime, kk, cch);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(
            DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        // Stage 2 (per input): the main kernel reads the precomputed U.
        var shape = new Conv2DWinogradShape(n, cch, h, w, kk, filterPretransformed: true);
        using var kernel = new PtxConv2DNchw3x3WinogradF23Kernel(runtime, shape);
        using var input = runtime.AllocateBytes((nuint)shape.InputBytes);
        using var bias = runtime.AllocateBytes((nuint)shape.BiasBytes);
        using var output = runtime.AllocateBytes((nuint)shape.OutputBytes);
        input.Upload<float>(new float[shape.InputBytes / sizeof(float)]);
        bias.Upload<float>(new float[shape.BiasBytes / sizeof(float)]);

        void Launch() => kernel.Launch(
            DirectPtxTensorView.CreateOwned(input, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(dU, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(bias, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));

        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-pretf N{n}/C{cch}/{h}x{w}/K{kk} 3x3 (filter U precomputed) regs={kernel.FunctionInfo.RegistersPerThread}",
            runtime, Launch, flops);
    }

    private void MeasureWinogradFused(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        // Filter transform (one-time, position-major U).
        using var filter = new PtxWinogradF23FilterTransformKernel(runtime, kk, cch, positionMajor: true);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(
            DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        using var inputT = new PtxWinogradF23InputTransformKernel(runtime, n, cch, h, w);
        using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
        using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
        dInput.Upload<float>(new float[inputT.InputBytes / sizeof(float)]);

        using var fused = new PtxWinogradF23FusedGemmKernel(runtime, n, cch, h, w, kk, 16, 16, 8);
        using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
        dBias.Upload<float>(new float[fused.BiasBytes / sizeof(float)]);

        // Per call = input transform + fused GEMM (filter U precomputed once).
        void Launch()
        {
            inputT.Launch(
                DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));
            fused.Launch(
                DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                DirectPtxTensorView.CreateOwned(dV, fused.Blueprint.Tensors[1]),
                DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
        }

        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-FUSED N{n}/C{cch}/{h}x{w}/K{kk} 3x3 (inputT+gemm, regs={fused.FunctionInfo.RegistersPerThread})",
            runtime, Launch, flops);
    }

    private void MeasureWinogradBatched(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        int tiles = n * (h / 2) * (w / 2);
        using var filter = new PtxWinogradF23FilterTransformKernel(runtime, kk, cch, positionMajor: true);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        using var inputT = new PtxWinogradF23InputTransformKernel(runtime, n, cch, h, w);
        using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
        using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
        dInput.Upload<float>(new float[inputT.InputBytes / sizeof(float)]);
        using var gemm = new PtxWinogradBatchedGemmKernel(runtime, kk, cch, tiles, 64, 64, 8, 4, 4);
        using var dM = runtime.AllocateBytes((nuint)gemm.MBytes);
        using var outT = new PtxWinogradF23OutputTransformKernel(runtime, n, h, w, kk);
        using var dBias = runtime.AllocateBytes((nuint)outT.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)outT.OutputBytes);
        dBias.Upload<float>(new float[outT.BiasBytes / sizeof(float)]);

        void Launch()
        {
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));
            gemm.Launch(DirectPtxTensorView.CreateOwned(dU, gemm.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dV, gemm.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dM, gemm.Blueprint.Tensors[2]));
            outT.Launch(DirectPtxTensorView.CreateOwned(dM, outT.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dBias, outT.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dOutput, outT.Blueprint.Tensors[2]));
        }
        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-BATCHED N{n}/C{cch}/{h}x{w}/K{kk} 3x3 (inT+regGEMM+outT, gemmRegs={gemm.FunctionInfo.RegistersPerThread})",
            runtime, Launch, flops);
    }

    private void MeasureWinogradFusedRB(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        using var filter = new PtxWinogradF23FilterTransformKernel(runtime, kk, cch, positionMajor: true);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        using var inputT = new PtxWinogradF23InputTransformKernel(runtime, n, cch, h, w);
        using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
        using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
        dInput.Upload<float>(new float[inputT.InputBytes / sizeof(float)]);
        using var fused = new PtxWinogradF23FusedRegBlockedKernel(runtime, n, cch, h, w, kk, 32, 32, 8, 2, 2);
        using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
        dBias.Upload<float>(new float[fused.BiasBytes / sizeof(float)]);

        void Launch()
        {
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));
            fused.Launch(DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                         DirectPtxTensorView.CreateOwned(dV, fused.Blueprint.Tensors[1]),
                         DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                         DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
        }
        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-FUSED-RB N{n}/C{cch}/{h}x{w}/K{kk} 3x3 tm2tn2 (inT+fusedGEMM+outT, regs={fused.FunctionInfo.RegistersPerThread})",
            runtime, Launch, flops);
    }

    private void MeasureWinogradWmma(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        if (runtime.ComputeCapabilityMajor < 7) { _out.WriteLine("winograd-WMMA: no Tensor Cores"); return; }
        int tiles = n * (h / 2) * (w / 2);
        // fp16 filter transform (one-time), position-major U[16,K,C].
        using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, kk, cch);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
        using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
        using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
        dInput.Upload<float>(new float[inputT.InputBytes / sizeof(float)]);
        using var gemm = new PtxWinogradWmmaBatchedGemmKernel(runtime, kk, cch, tiles);
        using var dM = runtime.AllocateBytes((nuint)gemm.MBytes);
        using var outT = new PtxWinogradF23OutputTransformKernel(runtime, n, h, w, kk);
        using var dBias = runtime.AllocateBytes((nuint)outT.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)outT.OutputBytes);
        dBias.Upload<float>(new float[outT.BiasBytes / sizeof(float)]);

        void Launch()
        {
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));
            gemm.Launch(DirectPtxTensorView.CreateOwned(dU, gemm.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dV, gemm.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dM, gemm.Blueprint.Tensors[2]));
            outT.Launch(DirectPtxTensorView.CreateOwned(dM, outT.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dBias, outT.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dOutput, outT.Blueprint.Tensors[2]));
        }
        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-WMMA-TC N{n}/C{cch}/{h}x{w}/K{kk} 3x3 fp16-TC (inT+wmmaGEMM+outT, gemmRegs={gemm.FunctionInfo.RegistersPerThread})",
            runtime, Launch, flops);
    }

    private void MeasureWinogradWmmaFused(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        if (runtime.ComputeCapabilityMajor < 7) { _out.WriteLine("winograd-WMMA-FUSED: no Tensor Cores"); return; }
        // fp16 filter transform (one-time), position-major U[16,K,C].
        using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, kk, cch);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
        using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
        using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
        dInput.Upload<float>(new float[inputT.InputBytes / sizeof(float)]);
        using var fused = new PtxWinogradWmmaFusedKernel(runtime, n, cch, h, w, kk);
        using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
        dBias.Upload<float>(new float[fused.BiasBytes / sizeof(float)]);

        // Per call = input transform + fused (GEMM + output transform), filter U precomputed once.
        void Launch()
        {
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));
            fused.Launch(DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                         DirectPtxTensorView.CreateOwned(dV, fused.Blueprint.Tensors[1]),
                         DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                         DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
        }
        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-WMMA-FUSED N{n}/C{cch}/{h}x{w}/K{kk} 3x3 fp16-TC-fused (inT+fusedMMA+outT, regs={fused.FunctionInfo.RegistersPerThread})",
            runtime, Launch, flops);
    }

    private void MeasureWinogradWmmaFusedStaged(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        if (runtime.ComputeCapabilityMajor < 8) { _out.WriteLine("winograd-WMMA-STAGED: no cp.async"); return; }
        using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, kk, cch);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
        using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
        using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
        dInput.Upload<float>(new float[inputT.InputBytes / sizeof(float)]);
        using var fused = new PtxWinogradWmmaFusedStagedKernel(runtime, n, cch, h, w, kk);
        using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
        dBias.Upload<float>(new float[fused.BiasBytes / sizeof(float)]);

        void Launch()
        {
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));
            fused.Launch(DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
                         DirectPtxTensorView.CreateOwned(dV, fused.Blueprint.Tensors[1]),
                         DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
                         DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
        }
        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-WMMA-STAGED N{n}/C{cch}/{h}x{w}/K{kk} 3x3 fp16-TC-cpasync-4warp (inT+stagedMMA+outT, regs={fused.FunctionInfo.RegistersPerThread})",
            runtime, Launch, flops);
    }

    private void MeasureWinogradWmmaCoop(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        if (runtime.ComputeCapabilityMajor < 7) { _out.WriteLine("winograd-WMMA-COOP: no Tensor Cores"); return; }
        using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, kk, cch);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
        using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
        using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
        dInput.Upload<float>(new float[inputT.InputBytes / sizeof(float)]);
        using var coop = new PtxWinogradWmmaCoopKernel(runtime, n, cch, h, w, kk);
        using var dBias = runtime.AllocateBytes((nuint)coop.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)coop.OutputBytes);
        dBias.Upload<float>(new float[coop.BiasBytes / sizeof(float)]);

        void Launch()
        {
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));
            coop.Launch(DirectPtxTensorView.CreateOwned(dU, coop.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dV, coop.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dBias, coop.Blueprint.Tensors[2]),
                        DirectPtxTensorView.CreateOwned(dOutput, coop.Blueprint.Tensors[3]));
        }
        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-WMMA-COOP N{n}/C{cch}/{h}x{w}/K{kk} 3x3 fp16-TC-coop16warp (inT+coopMMA+outT, regs={coop.FunctionInfo.RegistersPerThread})",
            runtime, Launch, flops);
    }

    private void MeasureStagesInIsolation(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        int tiles = n * (h / 2) * (w / 2);
        long flops = 2L * n * kk * h * w * cch * 9;

        // fp16 input transform alone.
        using (var it16 = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w))
        using (var di = runtime.AllocateBytes((nuint)it16.InputBytes))
        using (var dv = runtime.AllocateBytes((nuint)it16.TransformedBytes))
        {
            di.Upload<float>(new float[it16.InputBytes / sizeof(float)]);
            void L() => it16.Launch(DirectPtxTensorView.CreateOwned(di, it16.Blueprint.Tensors[0]),
                                    DirectPtxTensorView.CreateOwned(dv, it16.Blueprint.Tensors[1]));
            Report($"STAGE inputT-fp16 alone (V[16,P,C], regs={it16.FunctionInfo.RegistersPerThread})", runtime, L, flops);
        }
        // fp32 input transform alone.
        using (var it32 = new PtxWinogradF23InputTransformKernel(runtime, n, cch, h, w))
        using (var di = runtime.AllocateBytes((nuint)it32.InputBytes))
        using (var dv = runtime.AllocateBytes((nuint)it32.TransformedBytes))
        {
            di.Upload<float>(new float[it32.InputBytes / sizeof(float)]);
            void L() => it32.Launch(DirectPtxTensorView.CreateOwned(di, it32.Blueprint.Tensors[0]),
                                    DirectPtxTensorView.CreateOwned(dv, it32.Blueprint.Tensors[1]));
            Report($"STAGE inputT-fp32 alone (V[16,C,P], regs={it32.FunctionInfo.RegistersPerThread})", runtime, L, flops);
        }
        // coop GEMM alone (U,V pre-staged; not re-run each call).
        using (var f16 = new PtxWinogradF23FilterTransformFp16Kernel(runtime, kk, cch))
        using (var it16 = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w))
        using (var dw = runtime.AllocateBytes((nuint)f16.WeightBytes))
        using (var du = runtime.AllocateBytes((nuint)f16.TransformedBytes))
        using (var di = runtime.AllocateBytes((nuint)it16.InputBytes))
        using (var dv = runtime.AllocateBytes((nuint)it16.TransformedBytes))
        using (var coop = new PtxWinogradWmmaCoopKernel(runtime, n, cch, h, w, kk))
        using (var db = runtime.AllocateBytes((nuint)coop.BiasBytes))
        using (var doU = runtime.AllocateBytes((nuint)coop.OutputBytes))
        {
            dw.Upload<float>(new float[f16.WeightBytes / sizeof(float)]);
            di.Upload<float>(new float[it16.InputBytes / sizeof(float)]);
            db.Upload<float>(new float[coop.BiasBytes / sizeof(float)]);
            f16.Launch(DirectPtxTensorView.CreateOwned(dw, f16.Blueprint.Tensors[0]),
                       DirectPtxTensorView.CreateOwned(du, f16.Blueprint.Tensors[1]));
            it16.Launch(DirectPtxTensorView.CreateOwned(di, it16.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dv, it16.Blueprint.Tensors[1]));
            runtime.Synchronize();
            void L() => coop.Launch(DirectPtxTensorView.CreateOwned(du, coop.Blueprint.Tensors[0]),
                                    DirectPtxTensorView.CreateOwned(dv, coop.Blueprint.Tensors[1]),
                                    DirectPtxTensorView.CreateOwned(db, coop.Blueprint.Tensors[2]),
                                    DirectPtxTensorView.CreateOwned(doU, coop.Blueprint.Tensors[3]));
            Report($"STAGE coopGEMM alone (regs={coop.FunctionInfo.RegistersPerThread})", runtime, L, flops);
        }
    }

    private void MeasureWinogradWmmaCoopBlocked(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        if (runtime.ComputeCapabilityMajor < 7) { _out.WriteLine("winograd-WMMA-COOPBLK: no Tensor Cores"); return; }
        using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, kk, cch);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        using var inputT = new PtxWinogradF23InputTransformFp16Kernel(runtime, n, cch, h, w);
        using var dInput = runtime.AllocateBytes((nuint)inputT.InputBytes);
        using var dV = runtime.AllocateBytes((nuint)inputT.TransformedBytes);
        dInput.Upload<float>(new float[inputT.InputBytes / sizeof(float)]);
        using var coop = new PtxWinogradWmmaCoopBlockedKernel(runtime, n, cch, h, w, kk);
        using var dBias = runtime.AllocateBytes((nuint)coop.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)coop.OutputBytes);
        dBias.Upload<float>(new float[coop.BiasBytes / sizeof(float)]);

        void Launch()
        {
            inputT.Launch(DirectPtxTensorView.CreateOwned(dInput, inputT.Blueprint.Tensors[0]),
                          DirectPtxTensorView.CreateOwned(dV, inputT.Blueprint.Tensors[1]));
            coop.Launch(DirectPtxTensorView.CreateOwned(dU, coop.Blueprint.Tensors[0]),
                        DirectPtxTensorView.CreateOwned(dV, coop.Blueprint.Tensors[1]),
                        DirectPtxTensorView.CreateOwned(dBias, coop.Blueprint.Tensors[2]),
                        DirectPtxTensorView.CreateOwned(dOutput, coop.Blueprint.Tensors[3]));
        }
        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-WMMA-COOPBLK N{n}/C{cch}/{h}x{w}/K{kk} 3x3 fp16-TC-coop-wn4 (inT+blkMMA+outT, regs={coop.FunctionInfo.RegistersPerThread})",
            runtime, Launch, flops);
    }

    private void MeasureWinogradWmmaFullyFused(DirectPtxRuntime runtime, int n, int cch, int h, int w, int kk)
    {
        if (runtime.ComputeCapabilityMajor < 7) { _out.WriteLine("winograd-WMMA-FULLFUSED: no Tensor Cores"); return; }
        using var filter = new PtxWinogradF23FilterTransformFp16Kernel(runtime, kk, cch);
        using var dWeights = runtime.AllocateBytes((nuint)filter.WeightBytes);
        using var dU = runtime.AllocateBytes((nuint)filter.TransformedBytes);
        dWeights.Upload<float>(new float[filter.WeightBytes / sizeof(float)]);
        filter.Launch(DirectPtxTensorView.CreateOwned(dWeights, filter.Blueprint.Tensors[0]),
                      DirectPtxTensorView.CreateOwned(dU, filter.Blueprint.Tensors[1]));
        runtime.Synchronize();

        using var fused = new PtxWinogradWmmaFullyFusedKernel(runtime, n, cch, h, w, kk);
        using var dInput = runtime.AllocateBytes((nuint)fused.InputBytes);
        using var dBias = runtime.AllocateBytes((nuint)fused.BiasBytes);
        using var dOutput = runtime.AllocateBytes((nuint)fused.OutputBytes);
        dInput.Upload<float>(new float[fused.InputBytes / sizeof(float)]);
        dBias.Upload<float>(new float[fused.BiasBytes / sizeof(float)]);

        // Per call = the single fused kernel (U precomputed once).
        void Launch() => fused.Launch(
            DirectPtxTensorView.CreateOwned(dU, fused.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(dInput, fused.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(dBias, fused.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(dOutput, fused.Blueprint.Tensors[3]));
        long flops = 2L * n * kk * h * w * cch * 9;
        Report($"winograd-WMMA-FULLFUSED N{n}/C{cch}/{h}x{w}/K{kk} 3x3 fp16-TC-1kernel (inT+GEMM+outT fused, regs={fused.FunctionInfo.RegistersPerThread})",
            runtime, Launch, flops);
    }

    private void Report(string label, DirectPtxRuntime runtime, Action launch, long flops)
    {
        for (int i = 0; i < Warmups; i++) launch();
        runtime.Synchronize();

        var us = new double[Samples];
        var sw = new Stopwatch();
        for (int i = 0; i < Samples; i++)
        {
            sw.Restart();
            launch();
            runtime.Synchronize();
            sw.Stop();
            us[i] = sw.Elapsed.TotalMilliseconds * 1000.0;
        }
        Array.Sort(us);
        double median = us[Samples / 2];
        double p95 = us[(int)(Samples * 0.95)];
        double gflops = flops / (median / 1e6) / 1e9;

        // Amortized (kernel-only): 100 launches, one sync — removes per-launch
        // overhead so it is apples-to-apples with cuDNN CUDA-graph replay.
        const int batch = 100;
        var swa = Stopwatch.StartNew();
        for (int i = 0; i < batch; i++) launch();
        runtime.Synchronize();
        swa.Stop();
        double amortUs = swa.Elapsed.TotalMilliseconds * 1000.0 / batch;
        double amortGflops = flops / (amortUs / 1e6) / 1e9;

        _out.WriteLine($"{label}: median={median:F1}us p95={p95:F1}us {gflops:F0} GFLOP/s | " +
            $"amortized={amortUs:F1}us {amortGflops:F0} GFLOP/s ({flops / 1e6:F0} MFLOP)");
    }
}
#endif
