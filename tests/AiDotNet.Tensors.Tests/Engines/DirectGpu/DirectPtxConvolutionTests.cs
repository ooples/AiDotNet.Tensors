#if NET5_0_OR_GREATER
using System;
using System.IO;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxConvolutionTests
{
    [Fact]
    public void Emitter_IsPointerOnlyShapeSpecializedSm86Ptx()
    {
        string ptx = PtxFusedConv2DNchwK1Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedConv2DNchwK1Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.Equal(PtxFusedConv2DNchwK1Kernel.InputChannels, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("setp.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(" bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxFusedConv2DNchwK1Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Blueprint_DeclaresExactContiguousAbiAndNoIntermediate()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxFusedConv2DNchwK1Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("conv2d-bias-relu-v1-Ampere-n1-c64-h16-w16-k64-r1-s1-p0-fp32", blueprint.Id);
        Assert.Equal(
            new[] { DirectPtxPhysicalLayout.Nchw, DirectPtxPhysicalLayout.Oihw,
                DirectPtxPhysicalLayout.Vector, DirectPtxPhysicalLayout.Nchw },
            blueprint.Tensors.Select(t => t.Layout));
        Assert.All(blueprint.Tensors, tensor =>
        {
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode);
            Assert.Equal(16, tensor.AlignmentBytes);
            Assert.Equal(DirectPtxPhysicalType.Float32, tensor.PhysicalType);
        });
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.Equal("experimental-pending-gpu-evidence", blueprint.Semantics["promotion"]);
        Assert.Equal(0, blueprint.ResourceBudget.MaxLocalBytesPerThread);
    }

    [Fact]
    public void Eligibility_AcceptsOnlyExactSm86NonAliasingContract()
    {
        (SyntheticGpuBuffer input, SyntheticGpuBuffer weights,
            SyntheticGpuBuffer bias, SyntheticGpuBuffer output) = ValidBuffers();

        Assert.Null(Validate(true, true, 8, 6, PtxFusedConv2DNchwK1Kernel.Shape,
            input, weights, bias, output));
        Assert.Equal(DirectPtxConvolutionEligibility.FeatureDisabled,
            Validate(false, true, 8, 6, PtxFusedConv2DNchwK1Kernel.Shape,
                input, weights, bias, output));
        Assert.Equal(DirectPtxConvolutionEligibility.BackendUnavailable,
            Validate(true, false, 8, 6, PtxFusedConv2DNchwK1Kernel.Shape,
                input, weights, bias, output));
        Assert.Equal(DirectPtxConvolutionEligibility.ArchitectureNotImplemented,
            Validate(true, true, 8, 9, PtxFusedConv2DNchwK1Kernel.Shape,
                input, weights, bias, output));
        Assert.Equal(DirectPtxConvolutionEligibility.ShapeNotImplemented,
            Validate(true, true, 8, 6,
                PtxFusedConv2DNchwK1Kernel.Shape with { Batch = 2 },
                input, weights, bias, output));
        Assert.Equal(DirectPtxConvolutionEligibility.NullBuffer,
            Validate(true, true, 8, 6, PtxFusedConv2DNchwK1Kernel.Shape,
                null, weights, bias, output));
        Assert.Equal(DirectPtxConvolutionEligibility.InvalidDevicePointer,
            Validate(true, true, 8, 6, PtxFusedConv2DNchwK1Kernel.Shape,
                new SyntheticGpuBuffer(IntPtr.Zero, PtxFusedConv2DNchwK1Kernel.InputBytes),
                weights, bias, output));
        Assert.Equal(DirectPtxConvolutionEligibility.PhysicalExtentMismatch,
            Validate(true, true, 8, 6, PtxFusedConv2DNchwK1Kernel.Shape,
                new SyntheticGpuBuffer(input.Handle, input.SizeInBytes + sizeof(float)),
                weights, bias, output));
        Assert.Equal(DirectPtxConvolutionEligibility.AlignmentMismatch,
            Validate(true, true, 8, 6, PtxFusedConv2DNchwK1Kernel.Shape,
                input, weights, bias,
                new SyntheticGpuBuffer(new IntPtr(output.Handle.ToInt64() + 4), output.SizeInBytes)));
        Assert.Equal(DirectPtxConvolutionEligibility.AliasNotSupported,
            Validate(true, true, 8, 6, PtxFusedConv2DNchwK1Kernel.Shape,
                input, weights, bias,
                new SyntheticGpuBuffer(input.Handle, PtxFusedConv2DNchwK1Kernel.OutputBytes)));
    }

    [Fact]
    public void FeatureGate_IsDedicatedAndFailClosed()
    {
        bool? previous = DirectPtxFeatureGate.TestOverride;
        try
        {
            Assert.Equal("AIDOTNET_DIRECT_PTX_CONVOLUTION",
                DirectPtxFeatureGate.ConvolutionEnvironmentVariable);
            DirectPtxFeatureGate.TestOverride = false;
            Assert.False(DirectPtxFeatureGate.IsConvolutionEnabled);
            DirectPtxFeatureGate.TestOverride = true;
            Assert.True(DirectPtxFeatureGate.IsConvolutionEnabled);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
        }
    }

    [Fact]
    public void CoverageManifest_AssignsEveryEntryExactlyOnce()
    {
        Assert.Equal(DirectPtxConvolutionCoverageManifest.All.Count,
            DirectPtxConvolutionCoverageManifest.All.Select(cell => cell.Api)
                .Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxConvolutionCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
        });
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("DirectGpuTensorEngine.FusedConv2D").Status);
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxConvolutionCoverageManifest.Get("UnassignedConvolutionApi"));
    }

    [Fact]
    public void PublicRouteAndCaptureLifetime_AreWiredBeforeEstablishedFallback()
    {
        string engine = File.ReadAllText(SourcePath(
            "src", "AiDotNet.Tensors", "Engines", "DirectGpuTensorEngine.cs"));
        int direct = engine.IndexOf("TryDirectPtxFusedConv2DBiasRelu", StringComparison.Ordinal);
        int established = engine.IndexOf("backend.Conv2D(inputBuffer.Buffer", direct, StringComparison.Ordinal);
        Assert.True(direct >= 0 && established > direct);

        string backend = File.ReadAllText(SourcePath(
            "src", "AiDotNet.Tensors", "Engines", "DirectGpu", "CUDA",
            "CudaBackend.DirectPtx.Convolution.cs"));
        Assert.Contains("must be prewarmed before CUDA graph capture", backend, StringComparison.Ordinal);
        Assert.Contains("_directPtxConvolutionKernels.Pin(key)", backend, StringComparison.Ordinal);
        string owner = File.ReadAllText(SourcePath(
            "src", "AiDotNet.Tensors", "Engines", "DirectGpu", "CUDA",
            "CudaBackend.DirectPtx.cs"));
        Assert.Contains("_directPtxConvolutionKernels.Dispose()", owner, StringComparison.Ordinal);
    }

    [Fact]
    public void EvidenceScaffolding_RequiresResidentCompetitorsAndCompleteSamples()
    {
        string benchmark = File.ReadAllText(SourcePath(
            "tests", "AiDotNet.Tensors.Benchmarks", "DirectPtxConvolutionExperiment.cs"));
        Assert.Contains("for (int warmup = 0; warmup < 30; warmup++)", benchmark,
            StringComparison.Ordinal);
        Assert.Contains("var samples = new double[101]", benchmark, StringComparison.Ordinal);
        Assert.Contains("AiDotNet established CUDA/cuDNN", benchmark, StringComparison.Ordinal);
        Assert.Contains("Direct PTX fused (experimental)", benchmark, StringComparison.Ordinal);
        Assert.Contains("TemporaryDeviceBytes", benchmark, StringComparison.Ordinal);

        string python = File.ReadAllText(SourcePath(
            "tests", "AiDotNet.Tensors.Benchmarks", "BaselineRunners", "py",
            "run_direct_ptx_convolution_competitors.py"));
        Assert.Contains("WARMUPS = 30", python, StringComparison.Ordinal);
        Assert.Contains("SAMPLES = 101", python, StringComparison.Ordinal);
        Assert.Contains("PyTorch cuDNN eager", python, StringComparison.Ordinal);
        Assert.Contains("PyTorch cuDNN CUDA Graph", python, StringComparison.Ordinal);

        string ncu = File.ReadAllText(SourcePath(
            "tests", "AiDotNet.Tensors.Benchmarks", "Profiling", "run-direct-ptx-ncu.ps1"));
        Assert.Contains("'convolution' { 1 }", ncu, StringComparison.Ordinal);
        Assert.Contains(PtxFusedConv2DNchwK1Kernel.EntryPoint, ncu, StringComparison.Ordinal);
        Assert.Contains("smsp__sass_inst_executed_op_local_ld.sum", ncu,
            StringComparison.Ordinal);
    }

    [Fact]
    public void DepthwiseEmitter_IsPointerOnlyHaloPredicatedSm86Ptx()
    {
        string ptx = PtxFusedDepthwiseConv2D3x3F32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedDepthwiseConv2D3x3F32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        // Nine weights + one center tap + eight halo-predicated boundary taps.
        Assert.Equal(18, Count(ptx, "ld.global.nc.f32"));
        // Only the eight boundary taps are predicated; the center tap never is.
        Assert.Contains("setp.", ptx, StringComparison.Ordinal);
        Assert.Equal(4, Count(ptx, "and.pred"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxFusedDepthwiseConv2D3x3F32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DepthwiseBlueprint_DeclaresExactContiguousAbiAndNoIntermediate()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxFusedDepthwiseConv2D3x3F32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("depthwise-conv2d-v1-Ampere-n1-c64-h16-w16-r3-s1-p1-fp32", blueprint.Id);
        Assert.Equal(
            new[] { DirectPtxPhysicalLayout.Nchw, DirectPtxPhysicalLayout.Oihw,
                DirectPtxPhysicalLayout.Nchw },
            blueprint.Tensors.Select(t => t.Layout));
        Assert.All(blueprint.Tensors, tensor =>
        {
            Assert.Equal(DirectPtxExtentMode.Exact, tensor.ExtentMode);
            Assert.Equal(16, tensor.AlignmentBytes);
            Assert.Equal(DirectPtxPhysicalType.Float32, tensor.PhysicalType);
        });
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.Equal("experimental-pending-gpu-evidence", blueprint.Semantics["promotion"]);
        Assert.Equal(0, blueprint.ResourceBudget.MaxLocalBytesPerThread);
    }

    [Fact]
    public void DepthwiseManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DepthwiseConv2D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDepthwiseConv2D3x3").Status);
    }

    [Fact]
    public void RegisterCeiling_ScalesWithDeviceRegisterFileNotHardcodedLiteral()
    {
        // 65536 regs/SM, 4 target blocks x 128 threads => 65536/512 = 128/thread,
        // and 65536 regs/block / 128 = 512, so the SM bound (128) wins.
        DirectPtxFunctionInfo ampere = Info(regsPerSm: 65536, regsPerBlock: 65536);
        Assert.Equal(128, DirectPtxResourceBudget.DeriveRegisterCeiling(ampere, 128, 4));

        // Half the register file => half the ceiling. Nothing is hardcoded.
        DirectPtxFunctionInfo smaller = Info(regsPerSm: 32768, regsPerBlock: 65536);
        Assert.Equal(64, DirectPtxResourceBudget.DeriveRegisterCeiling(smaller, 128, 4));

        // The per-block cap can be the binding constraint at large block sizes.
        DirectPtxFunctionInfo blockBound = Info(regsPerSm: 131072, regsPerBlock: 32768);
        Assert.Equal(128, DirectPtxResourceBudget.DeriveRegisterCeiling(blockBound, 256, 2));

        // Older drivers that cannot report capacity => no standalone bound; the
        // driver occupancy calculator and zero-local-bytes invariant still apply.
        DirectPtxFunctionInfo unknown = Info(regsPerSm: 0, regsPerBlock: 0);
        Assert.Equal(int.MaxValue, DirectPtxResourceBudget.DeriveRegisterCeiling(unknown, 128, 4));
    }

    private static DirectPtxFunctionInfo Info(int regsPerSm, int regsPerBlock) =>
        new(MaxThreadsPerBlock: 1024, StaticSharedBytes: 0, ConstBytes: 0,
            LocalBytesPerThread: 0, RegistersPerThread: 40, PtxVersion: 0, BinaryVersion: 0,
            MaxRegistersPerMultiprocessor: regsPerSm, MaxRegistersPerBlock: regsPerBlock);

    [SkippableFact]
    public void DriverOnlyDepthwiseConv2D3x3_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 depthwise specialization.");

        const int channels = PtxFusedDepthwiseConv2D3x3F32Kernel.Channels;
        const int height = PtxFusedDepthwiseConv2D3x3F32Kernel.Height;
        const int width = PtxFusedDepthwiseConv2D3x3F32Kernel.Width;
        const int kernelSize = PtxFusedDepthwiseConv2D3x3F32Kernel.KernelSize;

        using var kernel = new PtxFusedDepthwiseConv2D3x3F32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxFusedDepthwiseConv2D3x3F32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxFusedDepthwiseConv2D3x3F32Kernel.WeightBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxFusedDepthwiseConv2D3x3F32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[channels * height * width];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[channels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[channels * height * width];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int c = 0; c < channels; c++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float expected = 0;
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int iy = y + ky - 1;
                int ix = x + kx - 1;
                if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                expected += weights[(c * kernelSize + ky) * kernelSize + kx] *
                    input[(c * height + iy) * width + ix];
            }
            float got = actual[(c * height + y) * width + x];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 5e-5f,
            $"Depthwise 3x3 max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void DepthwiseBackwardInputEmitter_IsPointerOnlyHaloPredicatedSm86Ptx()
    {
        string ptx = PtxDepthwiseConv2D3x3BackwardInputF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDepthwiseConv2D3x3BackwardInputF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(18, Count(ptx, "ld.global.nc.f32"));
        Assert.Equal(4, Count(ptx, "and.pred"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDepthwiseConv2D3x3BackwardInputF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DepthwiseBackwardInputBlueprint_IsTransposeContractWithNoIntermediate()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxDepthwiseConv2D3x3BackwardInputF32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("depthwise-conv2d-bwd-input-v1-Ampere-n1-c64-h16-w16-r3-s1-p1-fp32", blueprint.Id);
        Assert.Equal(
            new[] { DirectPtxPhysicalLayout.Nchw, DirectPtxPhysicalLayout.Oihw,
                DirectPtxPhysicalLayout.Nchw },
            blueprint.Tensors.Select(t => t.Layout));
        Assert.Equal("grad_output", blueprint.Tensors[0].Name);
        Assert.Equal("grad_input", blueprint.Tensors[2].Name);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.Equal(0, blueprint.ResourceBudget.MaxLocalBytesPerThread);
    }

    [Fact]
    public void DepthwiseBackwardInputManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DepthwiseConv2DBackwardInput").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDepthwiseConv2D3x3BackwardInput").Status);
    }

    [SkippableFact]
    public void DriverOnlyDepthwiseConv2D3x3BackwardInput_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 depthwise specialization.");

        const int channels = PtxDepthwiseConv2D3x3BackwardInputF32Kernel.Channels;
        const int height = PtxDepthwiseConv2D3x3BackwardInputF32Kernel.Height;
        const int width = PtxDepthwiseConv2D3x3BackwardInputF32Kernel.Width;
        const int kernelSize = PtxDepthwiseConv2D3x3BackwardInputF32Kernel.KernelSize;

        using var kernel = new PtxDepthwiseConv2D3x3BackwardInputF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv2D3x3BackwardInputF32Kernel.GradOutputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv2D3x3BackwardInputF32Kernel.WeightBytes);
        using var gradInDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv2D3x3BackwardInputF32Kernel.GradInputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[channels * height * width];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[channels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradInDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[channels * height * width];
        gradInDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int c = 0; c < channels; c++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float expected = 0;
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int oy = y - (ky - 1);
                int ox = x - (kx - 1);
                if (oy < 0 || oy >= height || ox < 0 || ox >= width) continue;
                expected += weights[(c * kernelSize + ky) * kernelSize + kx] *
                    gradOut[(c * height + oy) * width + ox];
            }
            float got = actual[(c * height + y) * width + x];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 5e-5f,
            $"Depthwise 3x3 backward-input max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void DepthwiseBackwardWeightEmitter_IsWarpReductionStrideFreeSm86Ptx()
    {
        string ptx = PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        // Nine tap accumulators, each reduced with a 5-step butterfly (9 x 5 = 45).
        Assert.Equal(45, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        // Lane 0 writes the nine reduced weights (all predicated stores).
        Assert.Equal(9, Count(ptx, "st.global.f32"));
        Assert.Contains("REDUCE_SPATIAL:", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal); // deterministic: no atomics
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DepthwiseBackwardWeightBlueprint_IsDeterministicReductionContract()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("depthwise-conv2d-bwd-weight-v1-Ampere-n1-c64-h16-w16-r3-s1-p1-fp32", blueprint.Id);
        Assert.Equal(
            new[] { DirectPtxPhysicalLayout.Nchw, DirectPtxPhysicalLayout.Nchw,
                DirectPtxPhysicalLayout.Oihw },
            blueprint.Tensors.Select(t => t.Layout));
        Assert.Equal("grad_output", blueprint.Tensors[0].Name);
        Assert.Equal("grad_weight", blueprint.Tensors[2].Name);
        Assert.Equal("warp-shuffle-butterfly-deterministic", blueprint.Semantics["reduction"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.Equal(0, blueprint.ResourceBudget.MaxStaticSharedBytes);
        Assert.Equal(0, blueprint.ResourceBudget.MaxLocalBytesPerThread);
    }

    [Fact]
    public void DepthwiseBackwardWeightManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DepthwiseConv2DBackwardKernel").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDepthwiseConv2D3x3BackwardWeight").Status);
    }

    [SkippableFact]
    public void DriverOnlyDepthwiseConv2D3x3BackwardWeight_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 depthwise specialization.");

        const int channels = PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.Channels;
        const int height = PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.Height;
        const int width = PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.Width;
        const int kernelSize = PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.KernelSize;

        using var kernel = new PtxDepthwiseConv2D3x3BackwardWeightF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.InputBytes);
        using var gradWeightDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv2D3x3BackwardWeightF32Kernel.GradWeightBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[channels * height * width];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[channels * height * width];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradWeightDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[channels * kernelSize * kernelSize];
        gradWeightDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int c = 0; c < channels; c++)
        for (int ky = 0; ky < kernelSize; ky++)
        for (int kx = 0; kx < kernelSize; kx++)
        {
            float expected = 0;
            for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int iy = y + (ky - 1);
                int ix = x + (kx - 1);
                if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                expected += gradOut[(c * height + y) * width + x] *
                    input[(c * height + iy) * width + ix];
            }
            float got = actual[(c * kernelSize + ky) * kernelSize + kx];
            // Reduction over 256 terms accumulates more rounding than a 9-tap dot;
            // scale the tolerance to the reduction width.
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Depthwise 3x3 backward-weight max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void ConvBackwardBiasEmitter_IsWarpReductionStrideFreeSm86Ptx()
    {
        string ptx = PtxConv2DBackwardBiasF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv2DBackwardBiasF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(2, Count(ptx, ".param .u64"));
        Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32"));
        // One partial-sum add per spatial step plus the five reduction adds.
        Assert.Equal(PtxConv2DBackwardBiasF32Kernel.SpatialStepsPerLane + 5, Count(ptx, "add.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv2DBackwardBiasF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void ConvBackwardBiasBlueprint_IsNchwToVectorReduction()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxConv2DBackwardBiasF32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("conv2d-bwd-bias-v1-Ampere-n1-k64-h16-w16-fp32", blueprint.Id);
        Assert.Equal(
            new[] { DirectPtxPhysicalLayout.Nchw, DirectPtxPhysicalLayout.Vector },
            blueprint.Tensors.Select(t => t.Layout));
        Assert.Equal("grad_bias", blueprint.Tensors[1].Name);
        Assert.Equal("warp-shuffle-butterfly-deterministic", blueprint.Semantics["reduction"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.Equal(0, blueprint.ResourceBudget.MaxStaticSharedBytes);
    }

    [Fact]
    public void ConvBackwardBiasManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("DirectGpuTensorEngine.Conv2DBackwardBiasGpu").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv2DBackwardBias").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv2DBackwardBias_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int channels = PtxConv2DBackwardBiasF32Kernel.OutputChannels;
        const int height = PtxConv2DBackwardBiasF32Kernel.Height;
        const int width = PtxConv2DBackwardBiasF32Kernel.Width;

        using var kernel = new PtxConv2DBackwardBiasF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConv2DBackwardBiasF32Kernel.GradOutputBytes);
        using var gradBiasDevice = runtime.AllocateBytes((nuint)PtxConv2DBackwardBiasF32Kernel.GradBiasBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[channels * height * width];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(gradBiasDevice, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[channels];
        gradBiasDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int k = 0; k < channels; k++)
        {
            float expected = 0;
            for (int s = 0; s < height * width; s++)
                expected += gradOut[k * height * width + s];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(actual[k] - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Conv2D backward-bias max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void ConvK1BackwardInputEmitter_IsPointerOnlyTransposedSm86Ptx()
    {
        string ptx = PtxConv2DNchwK1BackwardInputF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv2DNchwK1BackwardInputF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        // One FMA per output channel; no bias add, no ReLU (transposed mat-vec only).
        Assert.Equal(PtxConv2DNchwK1BackwardInputF32Kernel.OutputChannels, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("max.f32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("setp.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv2DNchwK1BackwardInputF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void ConvK1BackwardInputBlueprint_IsTransposeContract()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxConv2DNchwK1BackwardInputF32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("conv2d-bwd-input-v1-Ampere-n1-c64-h16-w16-k64-r1-s1-p0-fp32", blueprint.Id);
        Assert.Equal(
            new[] { DirectPtxPhysicalLayout.Nchw, DirectPtxPhysicalLayout.Oihw,
                DirectPtxPhysicalLayout.Nchw },
            blueprint.Tensors.Select(t => t.Layout));
        Assert.Equal("grad_output", blueprint.Tensors[0].Name);
        Assert.Equal("grad_input", blueprint.Tensors[2].Name);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.Equal(0, blueprint.ResourceBudget.MaxLocalBytesPerThread);
    }

    [Fact]
    public void ConvK1BackwardInputManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Conv2DBackwardInput").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv2DBackwardInput").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv2DK1BackwardInput_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv2DNchwK1BackwardInputF32Kernel.InputChannels;
        const int outChannels = PtxConv2DNchwK1BackwardInputF32Kernel.OutputChannels;
        const int spatial = PtxConv2DNchwK1BackwardInputF32Kernel.SpatialElements;

        using var kernel = new PtxConv2DNchwK1BackwardInputF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConv2DNchwK1BackwardInputF32Kernel.GradOutputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConv2DNchwK1BackwardInputF32Kernel.WeightBytes);
        using var gradInDevice = runtime.AllocateBytes((nuint)PtxConv2DNchwK1BackwardInputF32Kernel.GradInputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradInDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[inChannels * spatial];
        gradInDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int c = 0; c < inChannels; c++)
        for (int s = 0; s < spatial; s++)
        {
            float expected = 0;
            for (int k = 0; k < outChannels; k++)
                expected += weights[k * inChannels + c] * gradOut[k * spatial + s];
            float got = actual[c * spatial + s];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 5e-5f,
            $"Conv2D 1x1 backward-input max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void ConvK1BackwardWeightEmitter_IsThreadPrivateDotSm86Ptx()
    {
        string ptx = PtxConv2DNchwK1BackwardWeightF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv2DNchwK1BackwardWeightF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("BWD_WEIGHT_DOT:", ptx, StringComparison.Ordinal);
        // Full spatial dot rolled into one loop body -> one FMA, one store.
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal); // deterministic, no atomics
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv2DNchwK1BackwardWeightF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void ConvK1BackwardWeightBlueprint_IsNchwPairToOihw()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxConv2DNchwK1BackwardWeightF32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("conv2d-bwd-weight-v1-Ampere-n1-c64-h16-w16-k64-r1-s1-p0-fp32", blueprint.Id);
        Assert.Equal(
            new[] { DirectPtxPhysicalLayout.Nchw, DirectPtxPhysicalLayout.Nchw,
                DirectPtxPhysicalLayout.Oihw },
            blueprint.Tensors.Select(t => t.Layout));
        Assert.Equal("grad_weight", blueprint.Tensors[2].Name);
        Assert.Equal("thread-private-full-dot-deterministic", blueprint.Semantics["reduction"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
    }

    [Fact]
    public void ConvK1BackwardWeightManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Conv2DBackwardKernel").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv2DBackwardWeight").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv2DK1BackwardWeight_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv2DNchwK1BackwardWeightF32Kernel.InputChannels;
        const int outChannels = PtxConv2DNchwK1BackwardWeightF32Kernel.OutputChannels;
        const int spatial = PtxConv2DNchwK1BackwardWeightF32Kernel.SpatialElements;

        using var kernel = new PtxConv2DNchwK1BackwardWeightF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConv2DNchwK1BackwardWeightF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConv2DNchwK1BackwardWeightF32Kernel.InputBytes);
        using var gradWeightDevice = runtime.AllocateBytes((nuint)PtxConv2DNchwK1BackwardWeightF32Kernel.GradWeightBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradWeightDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * inChannels];
        gradWeightDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int k = 0; k < outChannels; k++)
        for (int c = 0; c < inChannels; c++)
        {
            float expected = 0;
            for (int s = 0; s < spatial; s++)
                expected += gradOut[k * spatial + s] * input[c * spatial + s];
            float got = actual[k * inChannels + c];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 1e-4f,
            $"Conv2D 1x1 backward-weight max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv1DForwardEmitter_IsHaloPredicatedChannelLoopSm86Ptx()
    {
        string ptx = PtxConv1DNclForwardF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv1DNclForwardF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV1D_CHANNELS:", ptx, StringComparison.Ordinal);
        // Three kernel taps unrolled in the channel-loop body.
        Assert.Equal(3, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        // Two boundary taps predicated (left/right), center tap unpredicated,
        // plus the single loop-continue branch => three total predicate uses.
        Assert.Equal(1, Count(ptx, "@%p0 ld.global.nc.f32"));
        Assert.Equal(1, Count(ptx, "@%p1 ld.global.nc.f32"));
        Assert.Contains("@%p2 bra CONV1D_CHANNELS", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv1DNclForwardF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv1DForwardBlueprint_IsH1NchwContract()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxConv1DNclForwardF32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("conv1d-v1-Ampere-n1-cin64-l256-cout64-r3-s1-p1-fp32", blueprint.Id);
        Assert.Equal(
            new[] { DirectPtxPhysicalLayout.Nchw, DirectPtxPhysicalLayout.Oihw,
                DirectPtxPhysicalLayout.Nchw },
            blueprint.Tensors.Select(t => t.Layout));
        Assert.Equal("zero-pad-1-halo-predicated", blueprint.Semantics["padding"]);
        Assert.Equal("0", blueprint.Semantics["intermediate-global-bytes"]);
        Assert.Equal(0, blueprint.ResourceBudget.MaxLocalBytesPerThread);
    }

    [Fact]
    public void Conv1DForwardManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Conv1D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv1D").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv1DForward_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv1DNclForwardF32Kernel.InputChannels;
        const int outChannels = PtxConv1DNclForwardF32Kernel.OutputChannels;
        const int length = PtxConv1DNclForwardF32Kernel.Length;
        const int kernelSize = PtxConv1DNclForwardF32Kernel.KernelSize;

        using var kernel = new PtxConv1DNclForwardF32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConv1DNclForwardF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConv1DNclForwardF32Kernel.WeightBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxConv1DNclForwardF32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[inChannels * length];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * length];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int l = 0; l < length; l++)
        {
            float expected = 0;
            for (int ci = 0; ci < inChannels; ci++)
            for (int k = 0; k < kernelSize; k++)
            {
                int il = l + k - 1;
                if (il < 0 || il >= length) continue;
                expected += weights[(co * inChannels + ci) * kernelSize + k] *
                    input[ci * length + il];
            }
            float got = actual[co * length + l];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Conv1D forward max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv1DBackwardInputEmitter_IsHaloPredicatedChannelLoopSm86Ptx()
    {
        string ptx = PtxConv1DNclBackwardInputF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv1DNclBackwardInputF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV1D_BWD_INPUT:", ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(1, Count(ptx, "@%p0 ld.global.nc.f32"));
        Assert.Equal(1, Count(ptx, "@%p1 ld.global.nc.f32"));
        Assert.Contains("@%p2 bra CONV1D_BWD_INPUT", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv1DNclBackwardInputF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv1DBackwardInputManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Conv1DBackwardInput").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv1DBackwardInput").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv1DBackwardInput_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv1DNclBackwardInputF32Kernel.InputChannels;
        const int outChannels = PtxConv1DNclBackwardInputF32Kernel.OutputChannels;
        const int length = PtxConv1DNclBackwardInputF32Kernel.Length;
        const int kernelSize = PtxConv1DNclBackwardInputF32Kernel.KernelSize;

        using var kernel = new PtxConv1DNclBackwardInputF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConv1DNclBackwardInputF32Kernel.GradOutputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConv1DNclBackwardInputF32Kernel.WeightBytes);
        using var gradInDevice = runtime.AllocateBytes((nuint)PtxConv1DNclBackwardInputF32Kernel.GradInputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * length];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradInDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[inChannels * length];
        gradInDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int ci = 0; ci < inChannels; ci++)
        for (int l = 0; l < length; l++)
        {
            float expected = 0;
            for (int co = 0; co < outChannels; co++)
            for (int k = 0; k < kernelSize; k++)
            {
                int ol = l - k + 1;
                if (ol < 0 || ol >= length) continue;
                expected += weights[(co * inChannels + ci) * kernelSize + k] *
                    gradOut[co * length + ol];
            }
            float got = actual[ci * length + l];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Conv1D backward-input max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv1DBackwardWeightEmitter_IsThreadPrivateDotSm86Ptx()
    {
        string ptx = PtxConv1DNclBackwardWeightF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv1DNclBackwardWeightF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV1D_BWD_WEIGHT:", ptx, StringComparison.Ordinal);
        // id decomposition uses one integer divide by the kernel size.
        Assert.Contains("div.u32 %r3, %r2, 3", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv1DNclBackwardWeightF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv1DBackwardWeightManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Conv1DBackwardKernel").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv1DBackwardWeight").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv1DBackwardWeight_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv1DNclBackwardWeightF32Kernel.InputChannels;
        const int outChannels = PtxConv1DNclBackwardWeightF32Kernel.OutputChannels;
        const int length = PtxConv1DNclBackwardWeightF32Kernel.Length;
        const int kernelSize = PtxConv1DNclBackwardWeightF32Kernel.KernelSize;

        using var kernel = new PtxConv1DNclBackwardWeightF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConv1DNclBackwardWeightF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConv1DNclBackwardWeightF32Kernel.InputBytes);
        using var gradWeightDevice = runtime.AllocateBytes((nuint)PtxConv1DNclBackwardWeightF32Kernel.GradWeightBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * length];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[inChannels * length];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradWeightDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * inChannels * kernelSize];
        gradWeightDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int ci = 0; ci < inChannels; ci++)
        for (int k = 0; k < kernelSize; k++)
        {
            float expected = 0;
            for (int l = 0; l < length; l++)
            {
                int il = l + k - 1;
                if (il < 0 || il >= length) continue;
                expected += gradOut[co * length + l] * input[ci * length + il];
            }
            float got = actual[(co * inChannels + ci) * kernelSize + k];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Conv1D backward-weight max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv2D3x3ForwardEmitter_IsHaloPredicatedChannelLoopSm86Ptx()
    {
        string ptx = PtxConv2DNchw3x3ForwardF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv2DNchw3x3ForwardF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV2D_CHANNELS:", ptx, StringComparison.Ordinal);
        // Nine taps unrolled in the channel-loop body.
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        // Four corner taps combine two boundary predicates.
        Assert.Equal(4, Count(ptx, "and.pred %p4"));
        // Loop uses %p5, keeping the boundary predicates (%p0-%p3) intact.
        Assert.Contains("@%p5 bra CONV2D_CHANNELS", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv2DNchw3x3ForwardF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv2D3x3ForwardManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Conv2D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv2D3x3").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv2D3x3Forward_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv2DNchw3x3ForwardF32Kernel.InputChannels;
        const int outChannels = PtxConv2DNchw3x3ForwardF32Kernel.OutputChannels;
        const int height = PtxConv2DNchw3x3ForwardF32Kernel.Height;
        const int width = PtxConv2DNchw3x3ForwardF32Kernel.Width;
        const int kernelSize = PtxConv2DNchw3x3ForwardF32Kernel.KernelSize;

        using var kernel = new PtxConv2DNchw3x3ForwardF32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConv2DNchw3x3ForwardF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConv2DNchw3x3ForwardF32Kernel.WeightBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxConv2DNchw3x3ForwardF32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[inChannels * height * width];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * height * width];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float expected = 0;
            for (int ci = 0; ci < inChannels; ci++)
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int iy = y + ky - 1;
                int ix = x + kx - 1;
                if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                expected += weights[((co * inChannels + ci) * kernelSize + ky) * kernelSize + kx] *
                    input[(ci * height + iy) * width + ix];
            }
            float got = actual[(co * height + y) * width + x];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"Conv2D 3x3 forward max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv2D3x3BackwardInputEmitter_IsHaloPredicatedChannelLoopSm86Ptx()
    {
        string ptx = PtxConv2DNchw3x3BackwardInputF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv2DNchw3x3BackwardInputF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV2D_BWD_INPUT:", ptx, StringComparison.Ordinal);
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(4, Count(ptx, "and.pred %p4"));
        Assert.Contains("@%p5 bra CONV2D_BWD_INPUT", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv2DNchw3x3BackwardInputF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv2D3x3BackwardInputManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv2D3x3BackwardInput").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv2D3x3BackwardInput_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv2DNchw3x3BackwardInputF32Kernel.InputChannels;
        const int outChannels = PtxConv2DNchw3x3BackwardInputF32Kernel.OutputChannels;
        const int height = PtxConv2DNchw3x3BackwardInputF32Kernel.Height;
        const int width = PtxConv2DNchw3x3BackwardInputF32Kernel.Width;
        const int kernelSize = PtxConv2DNchw3x3BackwardInputF32Kernel.KernelSize;

        using var kernel = new PtxConv2DNchw3x3BackwardInputF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConv2DNchw3x3BackwardInputF32Kernel.GradOutputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConv2DNchw3x3BackwardInputF32Kernel.WeightBytes);
        using var gradInDevice = runtime.AllocateBytes((nuint)PtxConv2DNchw3x3BackwardInputF32Kernel.GradInputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * height * width];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradInDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[inChannels * height * width];
        gradInDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int ci = 0; ci < inChannels; ci++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float expected = 0;
            for (int co = 0; co < outChannels; co++)
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int oy = y - (ky - 1);
                int ox = x - (kx - 1);
                if (oy < 0 || oy >= height || ox < 0 || ox >= width) continue;
                expected += weights[((co * inChannels + ci) * kernelSize + ky) * kernelSize + kx] *
                    gradOut[(co * height + oy) * width + ox];
            }
            float got = actual[(ci * height + y) * width + x];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"Conv2D 3x3 backward-input max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv2D3x3BackwardWeightEmitter_IsThreadPrivateDotSm86Ptx()
    {
        string ptx = PtxConv2DNchw3x3BackwardWeightF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv2DNchw3x3BackwardWeightF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV2D_BWD_WEIGHT:", ptx, StringComparison.Ordinal);
        // Two integer divides decompose the weight id (by 9 then by the kernel size).
        Assert.Equal(2, Count(ptx, "div.u32"));
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv2DNchw3x3BackwardWeightF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv2D3x3BackwardWeightManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv2D3x3BackwardWeight").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv2D3x3BackwardWeight_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv2DNchw3x3BackwardWeightF32Kernel.InputChannels;
        const int outChannels = PtxConv2DNchw3x3BackwardWeightF32Kernel.OutputChannels;
        const int height = PtxConv2DNchw3x3BackwardWeightF32Kernel.Height;
        const int width = PtxConv2DNchw3x3BackwardWeightF32Kernel.Width;
        const int kernelSize = PtxConv2DNchw3x3BackwardWeightF32Kernel.KernelSize;

        using var kernel = new PtxConv2DNchw3x3BackwardWeightF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConv2DNchw3x3BackwardWeightF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConv2DNchw3x3BackwardWeightF32Kernel.InputBytes);
        using var gradWeightDevice = runtime.AllocateBytes((nuint)PtxConv2DNchw3x3BackwardWeightF32Kernel.GradWeightBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * height * width];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[inChannels * height * width];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradWeightDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * inChannels * kernelSize * kernelSize];
        gradWeightDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int ci = 0; ci < inChannels; ci++)
        for (int ky = 0; ky < kernelSize; ky++)
        for (int kx = 0; kx < kernelSize; kx++)
        {
            float expected = 0;
            for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int iy = y + ky - 1;
                int ix = x + kx - 1;
                if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                expected += gradOut[(co * height + y) * width + x] *
                    input[(ci * height + iy) * width + ix];
            }
            float got = actual[((co * inChannels + ci) * kernelSize + ky) * kernelSize + kx];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Conv2D 3x3 backward-weight max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void FusedConv2D3x3Emitter_HasBiasAddAndReluEpilogue()
    {
        string ptx = PtxFusedConv2DNchw3x3BiasReluF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedConv2DNchw3x3BiasReluF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.Contains("CONV2D_FUSED_CHANNELS:", ptx, StringComparison.Ordinal);
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        // Fused epilogue: one bias add + one ReLU max, single store, no intermediate.
        Assert.Equal(1, Count(ptx, "add.rn.f32"));
        Assert.Equal(1, Count(ptx, "max.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(4, Count(ptx, "and.pred %p4"));
        Assert.Contains("@%p5 bra CONV2D_FUSED_CHANNELS", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxFusedConv2DNchw3x3BiasReluF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void FusedConv2D3x3ManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxFusedConv2D3x3BiasRelu").Status);
    }

    [SkippableFact]
    public void DriverOnlyFusedConv2D3x3BiasRelu_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxFusedConv2DNchw3x3BiasReluF32Kernel.InputChannels;
        const int outChannels = PtxFusedConv2DNchw3x3BiasReluF32Kernel.OutputChannels;
        const int height = PtxFusedConv2DNchw3x3BiasReluF32Kernel.Height;
        const int width = PtxFusedConv2DNchw3x3BiasReluF32Kernel.Width;
        const int kernelSize = PtxFusedConv2DNchw3x3BiasReluF32Kernel.KernelSize;

        using var kernel = new PtxFusedConv2DNchw3x3BiasReluF32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxFusedConv2DNchw3x3BiasReluF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxFusedConv2DNchw3x3BiasReluF32Kernel.WeightBytes);
        using var biasDevice = runtime.AllocateBytes((nuint)PtxFusedConv2DNchw3x3BiasReluF32Kernel.BiasBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxFusedConv2DNchw3x3BiasReluF32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[inChannels * height * width];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        var bias = new float[outChannels];
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);
        biasDevice.Upload<float>(bias);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(biasDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[outChannels * height * width];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float acc = bias[co];
            for (int ci = 0; ci < inChannels; ci++)
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int iy = y + ky - 1;
                int ix = x + kx - 1;
                if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                acc += weights[((co * inChannels + ci) * kernelSize + ky) * kernelSize + kx] *
                    input[(ci * height + iy) * width + ix];
            }
            float expected = MathF.Max(acc, 0f);
            float got = actual[(co * height + y) * width + x];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"Fused Conv2D 3x3 bias+ReLU max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void ConvTranspose2D3x3Emitter_IsHaloPredicatedChannelLoopSm86Ptx()
    {
        string ptx = PtxConvTranspose2DNchw3x3ForwardF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConvTranspose2DNchw3x3ForwardF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV_TRANSPOSE2D_CHANNELS:", ptx, StringComparison.Ordinal);
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(4, Count(ptx, "and.pred %p4"));
        Assert.Contains("@%p5 bra CONV_TRANSPOSE2D_CHANNELS", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConvTranspose2DNchw3x3ForwardF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void ConvTranspose2D3x3Blueprint_UsesIohwWeights()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxConvTranspose2DNchw3x3ForwardF32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal("conv-transpose2d-v1-Ampere-n1-cin64-h16-w16-cout64-r3-s1-p1-fp32", blueprint.Id);
        Assert.Equal(
            new[] { DirectPtxPhysicalLayout.Nchw, DirectPtxPhysicalLayout.Iohw,
                DirectPtxPhysicalLayout.Nchw },
            blueprint.Tensors.Select(t => t.Layout));
    }

    [Fact]
    public void ConvTranspose2D3x3ManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.ConvTranspose2D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConvTranspose2D3x3").Status);
    }

    [SkippableFact]
    public void DriverOnlyConvTranspose2D3x3_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConvTranspose2DNchw3x3ForwardF32Kernel.InputChannels;
        const int outChannels = PtxConvTranspose2DNchw3x3ForwardF32Kernel.OutputChannels;
        const int height = PtxConvTranspose2DNchw3x3ForwardF32Kernel.Height;
        const int width = PtxConvTranspose2DNchw3x3ForwardF32Kernel.Width;
        const int kernelSize = PtxConvTranspose2DNchw3x3ForwardF32Kernel.KernelSize;

        using var kernel = new PtxConvTranspose2DNchw3x3ForwardF32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConvTranspose2DNchw3x3ForwardF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConvTranspose2DNchw3x3ForwardF32Kernel.WeightBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxConvTranspose2DNchw3x3ForwardF32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[inChannels * height * width];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        // IOHW weights: [Cin, Cout, kH, kW].
        var weights = new float[inChannels * outChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * height * width];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float expected = 0;
            for (int ci = 0; ci < inChannels; ci++)
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int iy = y + 1 - ky;
                int ix = x + 1 - kx;
                if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                expected += weights[((ci * outChannels + co) * kernelSize + ky) * kernelSize + kx] *
                    input[(ci * height + iy) * width + ix];
            }
            float got = actual[(co * height + y) * width + x];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"ConvTranspose2D 3x3 max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv3D3x3x3Emitter_Is27TapHaloPredicatedChannelLoopSm86Ptx()
    {
        string ptx = PtxConv3DNcdhw3x3x3ForwardF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv3DNcdhw3x3x3ForwardF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV3D_CHANNELS:", ptx, StringComparison.Ordinal);
        Assert.Equal(27, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        // 12 two-nonzero taps (one AND each) + 8 corner taps (two ANDs each) = 28.
        Assert.Equal(28, Count(ptx, "and.pred %p6"));
        Assert.Contains("@%p7 bra CONV3D_CHANNELS", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv3DNcdhw3x3x3ForwardF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv3D3x3x3ManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Conv3D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv3D3x3x3").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv3D3x3x3_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv3DNcdhw3x3x3ForwardF32Kernel.InputChannels;
        const int outChannels = PtxConv3DNcdhw3x3x3ForwardF32Kernel.OutputChannels;
        const int depth = PtxConv3DNcdhw3x3x3ForwardF32Kernel.Depth;
        const int hgt = PtxConv3DNcdhw3x3x3ForwardF32Kernel.Height;
        const int wid = PtxConv3DNcdhw3x3x3ForwardF32Kernel.Width;
        const int kernelSize = PtxConv3DNcdhw3x3x3ForwardF32Kernel.KernelSize;
        int spatial = depth * hgt * wid;

        using var kernel = new PtxConv3DNcdhw3x3x3ForwardF32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConv3DNcdhw3x3x3ForwardF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConv3DNcdhw3x3x3ForwardF32Kernel.WeightBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxConv3DNcdhw3x3x3ForwardF32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * spatial];
        outputDevice.Download<float>(actual);

        int hw = hgt * wid;
        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int d = 0; d < depth; d++)
        for (int h = 0; h < hgt; h++)
        for (int w = 0; w < wid; w++)
        {
            float expected = 0;
            for (int ci = 0; ci < inChannels; ci++)
            for (int kd = 0; kd < kernelSize; kd++)
            for (int kh = 0; kh < kernelSize; kh++)
            for (int kw = 0; kw < kernelSize; kw++)
            {
                int id = d + kd - 1, ih = h + kh - 1, iw = w + kw - 1;
                if (id < 0 || id >= depth || ih < 0 || ih >= hgt || iw < 0 || iw >= wid) continue;
                int wIdx = (((co * inChannels + ci) * kernelSize + kd) * kernelSize + kh) * kernelSize + kw;
                expected += weights[wIdx] * input[ci * spatial + (id * hw + ih * wid + iw)];
            }
            float got = actual[co * spatial + (d * hw + h * wid + w)];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"Conv3D 3x3x3 max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void DepthwiseConv1DEmitter_IsPerChannelHaloPredicatedSm86Ptx()
    {
        string ptx = PtxDepthwiseConv1DNcl3ForwardF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDepthwiseConv1DNcl3ForwardF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        // Three taps, no channel loop (per-channel depthwise).
        Assert.Equal(3, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal); // no loop
        Assert.Equal(1, Count(ptx, "@%p0 ld.global.nc.f32"));
        Assert.Equal(1, Count(ptx, "@%p1 ld.global.nc.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDepthwiseConv1DNcl3ForwardF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DepthwiseConv1DManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DepthwiseConv1D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDepthwiseConv1D").Status);
    }

    [SkippableFact]
    public void DriverOnlyDepthwiseConv1D_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int channels = PtxDepthwiseConv1DNcl3ForwardF32Kernel.Channels;
        const int length = PtxDepthwiseConv1DNcl3ForwardF32Kernel.Length;
        const int kernelSize = PtxDepthwiseConv1DNcl3ForwardF32Kernel.KernelSize;

        using var kernel = new PtxDepthwiseConv1DNcl3ForwardF32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv1DNcl3ForwardF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv1DNcl3ForwardF32Kernel.WeightBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv1DNcl3ForwardF32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[channels * length];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[channels * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[channels * length];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int c = 0; c < channels; c++)
        for (int l = 0; l < length; l++)
        {
            float expected = 0;
            for (int k = 0; k < kernelSize; k++)
            {
                int il = l + k - 1;
                if (il < 0 || il >= length) continue;
                expected += weights[c * kernelSize + k] * input[c * length + il];
            }
            float got = actual[c * length + l];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 5e-5f,
            $"DepthwiseConv1D max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void UnfoldEmitter_IsHaloMaskedGatherSm86Ptx()
    {
        string ptx = PtxUnfoldIm2ColNchw3x3F32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxUnfoldIm2ColNchw3x3F32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(2, Count(ptx, ".param .u64"));
        // Pure gather: no FMA, exactly one guarded load and one store.
        Assert.Equal(0, Count(ptx, "fma"));
        Assert.Equal(1, Count(ptx, "@%p4 ld.global.nc.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        // Two integer divides decompose the row into channel and kernel indices.
        Assert.Equal(2, Count(ptx, "div.u32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxUnfoldIm2ColNchw3x3F32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void UnfoldManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Unfold").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxUnfold").Status);
    }

    [SkippableFact]
    public void DriverOnlyUnfold_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int channels = PtxUnfoldIm2ColNchw3x3F32Kernel.Channels;
        const int height = PtxUnfoldIm2ColNchw3x3F32Kernel.Height;
        const int width = PtxUnfoldIm2ColNchw3x3F32Kernel.Width;
        const int kernelSize = PtxUnfoldIm2ColNchw3x3F32Kernel.KernelSize;
        const int rows = PtxUnfoldIm2ColNchw3x3F32Kernel.UnfoldRows;
        const int cols = PtxUnfoldIm2ColNchw3x3F32Kernel.SpatialElements;

        using var kernel = new PtxUnfoldIm2ColNchw3x3F32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxUnfoldIm2ColNchw3x3F32Kernel.InputBytes);
        using var unfoldDevice = runtime.AllocateBytes((nuint)PtxUnfoldIm2ColNchw3x3F32Kernel.UnfoldBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[channels * height * width];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(unfoldDevice, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[rows * cols];
        unfoldDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int c = 0; c < channels; c++)
        for (int ky = 0; ky < kernelSize; ky++)
        for (int kx = 0; kx < kernelSize; kx++)
        for (int oy = 0; oy < height; oy++)
        for (int ox = 0; ox < width; ox++)
        {
            int iy = oy + ky - 1, ix = ox + kx - 1;
            float expected = 0;
            if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                expected = input[(c * height + iy) * width + ix];
            int row = (c * kernelSize + ky) * kernelSize + kx;
            int col = oy * width + ox;
            float got = actual[row * cols + col];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        // Pure gather is bit-exact.
        Assert.Equal(0f, maxAbsoluteError);
    }

    [Fact]
    public void DepthwiseConv1DBackwardInputEmitter_IsPerChannelHaloPredicatedSm86Ptx()
    {
        string ptx = PtxDepthwiseConv1DNcl3BackwardInputF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDepthwiseConv1DNcl3BackwardInputF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Equal(3, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        // Negated read offset: k=0 reads dOut[l+1] guarded by p1, k=2 reads dOut[l-1] by p0.
        Assert.Equal(1, Count(ptx, "@%p1 ld.global.nc.f32"));
        Assert.Equal(1, Count(ptx, "@%p0 ld.global.nc.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDepthwiseConv1DNcl3BackwardInputF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DepthwiseConv1DBackwardInputManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DepthwiseConv1DBackwardInput").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDepthwiseConv1DBackwardInput").Status);
    }

    [SkippableFact]
    public void DriverOnlyDepthwiseConv1DBackwardInput_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int channels = PtxDepthwiseConv1DNcl3BackwardInputF32Kernel.Channels;
        const int length = PtxDepthwiseConv1DNcl3BackwardInputF32Kernel.Length;
        const int kernelSize = PtxDepthwiseConv1DNcl3BackwardInputF32Kernel.KernelSize;

        using var kernel = new PtxDepthwiseConv1DNcl3BackwardInputF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv1DNcl3BackwardInputF32Kernel.GradOutputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv1DNcl3BackwardInputF32Kernel.WeightBytes);
        using var gradInDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv1DNcl3BackwardInputF32Kernel.GradInputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[channels * length];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[channels * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradInDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[channels * length];
        gradInDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int c = 0; c < channels; c++)
        for (int l = 0; l < length; l++)
        {
            float expected = 0;
            for (int k = 0; k < kernelSize; k++)
            {
                int ol = l + 1 - k;
                if (ol < 0 || ol >= length) continue;
                expected += weights[c * kernelSize + k] * gradOut[c * length + ol];
            }
            float got = actual[c * length + l];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 5e-5f,
            $"DepthwiseConv1D backward-input max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void DepthwiseConv1DBackwardWeightEmitter_IsThreadPrivateDotSm86Ptx()
    {
        string ptx = PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("DWC1D_BWD_WEIGHT:", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "div.u32"));
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DepthwiseConv1DBackwardWeightManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DepthwiseConv1DBackwardKernel").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDepthwiseConv1DBackwardWeight").Status);
    }

    [SkippableFact]
    public void DriverOnlyDepthwiseConv1DBackwardWeight_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int channels = PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel.Channels;
        const int length = PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel.Length;
        const int kernelSize = PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel.KernelSize;

        using var kernel = new PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel.InputBytes);
        using var gradWeightDevice = runtime.AllocateBytes((nuint)PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel.GradWeightBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[channels * length];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[channels * length];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradWeightDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[channels * kernelSize];
        gradWeightDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int c = 0; c < channels; c++)
        for (int k = 0; k < kernelSize; k++)
        {
            float expected = 0;
            for (int l = 0; l < length; l++)
            {
                int il = l + k - 1;
                if (il < 0 || il >= length) continue;
                expected += gradOut[c * length + l] * input[c * length + il];
            }
            float got = actual[c * kernelSize + k];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"DepthwiseConv1D backward-weight max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv2DFp16K1Emitter_WidensHalvesAndAccumulatesInFp32()
    {
        string ptx = PtxConv2DFp16K1NchwF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv2DFp16K1NchwF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        // FP16 storage: half loads + widening converts; FP32 accumulate.
        Assert.Equal(2 * PtxConv2DFp16K1NchwF32Kernel.InputChannels, Count(ptx, "ld.global.nc.b16"));
        Assert.Equal(2 * PtxConv2DFp16K1NchwF32Kernel.InputChannels, Count(ptx, "cvt.f32.f16"));
        Assert.Equal(PtxConv2DFp16K1NchwF32Kernel.InputChannels, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv2DFp16K1NchwF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv2DFp16K1Blueprint_UsesFp16InputsAndFp32Output()
    {
        DirectPtxKernelBlueprint blueprint =
            PtxConv2DFp16K1NchwF32Kernel.CreateBlueprint(DirectPtxArchitectureFamily.Ampere);

        Assert.Equal(new[]
            {
                DirectPtxPhysicalType.Float16, DirectPtxPhysicalType.Float16,
                DirectPtxPhysicalType.Float32
            },
            blueprint.Tensors.Select(t => t.PhysicalType));
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("CudaBackend.Conv2dDirectFp16Hw").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("CudaBackend.TryDirectPtxConv2DFp16K1").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv2DFp16K1_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv2DFp16K1NchwF32Kernel.InputChannels;
        const int outChannels = PtxConv2DFp16K1NchwF32Kernel.OutputChannels;
        const int spatial = PtxConv2DFp16K1NchwF32Kernel.SpatialElements;

        using var kernel = new PtxConv2DFp16K1NchwF32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConv2DFp16K1NchwF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConv2DFp16K1NchwF32Kernel.WeightBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxConv2DFp16K1NchwF32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        ushort[] inputHalf = RandomHalfBits(random, inChannels * spatial);
        ushort[] weightHalf = RandomHalfBits(random, outChannels * inChannels);
        inputDevice.Upload<ushort>(inputHalf);
        weightDevice.Upload<ushort>(weightHalf);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * spatial];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int k = 0; k < outChannels; k++)
        for (int s = 0; s < spatial; s++)
        {
            float expected = 0;
            for (int c = 0; c < inChannels; c++)
            {
                float w = (float)BitConverter.UInt16BitsToHalf(weightHalf[k * inChannels + c]);
                float x = (float)BitConverter.UInt16BitsToHalf(inputHalf[c * spatial + s]);
                expected += w * x;
            }
            float got = actual[k * spatial + s];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        // Inputs are exact FP16 values on both sides; only fma-vs-mul+add differs.
        Assert.True(maxAbsoluteError <= 5e-3f,
            $"FP16 1x1 conv max absolute error {maxAbsoluteError:G9}.");
    }

    private static ushort[] RandomHalfBits(Random random, int count)
    {
        var bits = new ushort[count];
        for (int i = 0; i < count; i++)
            bits[i] = BitConverter.HalfToUInt16Bits((Half)(float)(random.NextDouble() * 2 - 1));
        return bits;
    }

    [Fact]
    public void ConvTranspose2DBackwardInputEmitter_IsForwardStyleChannelLoopSm86Ptx()
    {
        string ptx = PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONVT2D_BWD_INPUT:", ptx, StringComparison.Ordinal);
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(4, Count(ptx, "and.pred %p4"));
        Assert.Contains("@%p5 bra CONVT2D_BWD_INPUT", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void ConvTranspose2DBackwardInputManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.ConvTranspose2DBackwardInput").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConvTranspose2D3x3BackwardInput").Status);
    }

    [SkippableFact]
    public void DriverOnlyConvTranspose2DBackwardInput_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.InputChannels;
        const int outChannels = PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.OutputChannels;
        const int height = PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.Height;
        const int width = PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.Width;
        const int kernelSize = PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.KernelSize;

        using var kernel = new PtxConvTranspose2DNchw3x3BackwardInputF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.GradOutputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.WeightBytes);
        using var gradInDevice = runtime.AllocateBytes((nuint)PtxConvTranspose2DNchw3x3BackwardInputF32Kernel.GradInputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * height * width];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        // IOHW weights [Cin, Cout, kH, kW].
        var weights = new float[inChannels * outChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradInDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[inChannels * height * width];
        gradInDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int ci = 0; ci < inChannels; ci++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float expected = 0;
            for (int co = 0; co < outChannels; co++)
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int oy = y + ky - 1, ox = x + kx - 1;
                if (oy < 0 || oy >= height || ox < 0 || ox >= width) continue;
                expected += weights[((ci * outChannels + co) * kernelSize + ky) * kernelSize + kx] *
                    gradOut[(co * height + oy) * width + ox];
            }
            float got = actual[(ci * height + y) * width + x];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"ConvTranspose2D backward-input max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void ConvTranspose2DBackwardWeightEmitter_IsThreadPrivateDotSm86Ptx()
    {
        string ptx = PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONVT2D_BWD_WEIGHT:", ptx, StringComparison.Ordinal);
        Assert.Equal(2, Count(ptx, "div.u32"));
        // Negated transpose offset uses neg.s32 on ky-1 and kx-1.
        Assert.Equal(2, Count(ptx, "neg.s32"));
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void ConvTranspose2DBackwardWeightManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.ConvTranspose2DBackwardKernel").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConvTranspose2D3x3BackwardWeight").Status);
    }

    [SkippableFact]
    public void DriverOnlyConvTranspose2DBackwardWeight_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.InputChannels;
        const int outChannels = PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.OutputChannels;
        const int height = PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.Height;
        const int width = PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.Width;
        const int kernelSize = PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.KernelSize;

        using var kernel = new PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.InputBytes);
        using var gradWeightDevice = runtime.AllocateBytes((nuint)PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel.GradWeightBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * height * width];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[inChannels * height * width];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradWeightDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[inChannels * outChannels * kernelSize * kernelSize];
        gradWeightDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int ci = 0; ci < inChannels; ci++)
        for (int co = 0; co < outChannels; co++)
        for (int ky = 0; ky < kernelSize; ky++)
        for (int kx = 0; kx < kernelSize; kx++)
        {
            float expected = 0;
            for (int oy = 0; oy < height; oy++)
            for (int ox = 0; ox < width; ox++)
            {
                int iy = oy + 1 - ky, ix = ox + 1 - kx;
                if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                expected += gradOut[(co * height + oy) * width + ox] *
                    input[(ci * height + iy) * width + ix];
            }
            float got = actual[((ci * outChannels + co) * kernelSize + ky) * kernelSize + kx];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"ConvTranspose2D backward-weight max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void FusedConvTranspose2DEmitter_HasBiasAddAndReluEpilogue()
    {
        string ptx = PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.Contains("CONVT2D_FUSED_CHANNELS:", ptx, StringComparison.Ordinal);
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "add.rn.f32"));
        Assert.Equal(1, Count(ptx, "max.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(4, Count(ptx, "and.pred %p4"));
        Assert.Contains("@%p5 bra CONVT2D_FUSED_CHANNELS", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void FusedConvTranspose2DManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.FusedConvTranspose2D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxFusedConvTranspose2D3x3BiasRelu").Status);
    }

    [SkippableFact]
    public void DriverOnlyFusedConvTranspose2DBiasRelu_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.InputChannels;
        const int outChannels = PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.OutputChannels;
        const int height = PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.Height;
        const int width = PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.Width;
        const int kernelSize = PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.KernelSize;

        using var kernel = new PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.WeightBytes);
        using var biasDevice = runtime.AllocateBytes((nuint)PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.BiasBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[inChannels * height * width];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[inChannels * outChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        var bias = new float[outChannels];
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);
        biasDevice.Upload<float>(bias);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(biasDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[outChannels * height * width];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float acc = bias[co];
            for (int ci = 0; ci < inChannels; ci++)
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int iy = y + 1 - ky, ix = x + 1 - kx;
                if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                acc += weights[((ci * outChannels + co) * kernelSize + ky) * kernelSize + kx] *
                    input[(ci * height + iy) * width + ix];
            }
            float expected = MathF.Max(acc, 0f);
            float got = actual[(co * height + y) * width + x];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"Fused ConvTranspose2D bias+ReLU max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv3D3x3x3BackwardInputEmitter_Is27TapTransposeSm86Ptx()
    {
        string ptx = PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV3D_BWD_INPUT:", ptx, StringComparison.Ordinal);
        Assert.Equal(27, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(28, Count(ptx, "and.pred %p6"));
        Assert.Contains("@%p7 bra CONV3D_BWD_INPUT", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv3D3x3x3BackwardInputManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Conv3DBackwardInput").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv3D3x3x3BackwardInput").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv3D3x3x3BackwardInput_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.InputChannels;
        const int outChannels = PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.OutputChannels;
        const int depth = PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.Depth;
        const int hgt = PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.Height;
        const int wid = PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.Width;
        const int kernelSize = PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.KernelSize;
        int spatial = depth * hgt * wid;
        int hw = hgt * wid;

        using var kernel = new PtxConv3DNcdhw3x3x3BackwardInputF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.GradOutputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.WeightBytes);
        using var gradInDevice = runtime.AllocateBytes((nuint)PtxConv3DNcdhw3x3x3BackwardInputF32Kernel.GradInputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradInDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[inChannels * spatial];
        gradInDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int ci = 0; ci < inChannels; ci++)
        for (int d = 0; d < depth; d++)
        for (int h = 0; h < hgt; h++)
        for (int w = 0; w < wid; w++)
        {
            float expected = 0;
            for (int co = 0; co < outChannels; co++)
            for (int kd = 0; kd < kernelSize; kd++)
            for (int kh = 0; kh < kernelSize; kh++)
            for (int kw = 0; kw < kernelSize; kw++)
            {
                int od = d - (kd - 1), oh = h - (kh - 1), ow = w - (kw - 1);
                if (od < 0 || od >= depth || oh < 0 || oh >= hgt || ow < 0 || ow >= wid) continue;
                int wIdx = (((co * inChannels + ci) * kernelSize + kd) * kernelSize + kh) * kernelSize + kw;
                expected += weights[wIdx] * gradOut[co * spatial + (od * hw + oh * wid + ow)];
            }
            float got = actual[ci * spatial + (d * hw + h * wid + w)];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"Conv3D 3x3x3 backward-input max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void Conv3D3x3x3BackwardWeightEmitter_IsThreadPrivateVolumeDotSm86Ptx()
    {
        string ptx = PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("CONV3D_BWD_WEIGHT:", ptx, StringComparison.Ordinal);
        // Three integer divides decompose the weight id (id/27, m/9, rem9/3).
        Assert.Equal(3, Count(ptx, "div.u32"));
        Assert.Equal(1, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void Conv3D3x3x3BackwardWeightManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.Conv3DBackwardKernel").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxConv3D3x3x3BackwardWeight").Status);
    }

    [SkippableFact]
    public void DriverOnlyConv3D3x3x3BackwardWeight_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.InputChannels;
        const int outChannels = PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.OutputChannels;
        const int depth = PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.Depth;
        const int hgt = PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.Height;
        const int wid = PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.Width;
        const int kernelSize = PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.KernelSize;
        int spatial = depth * hgt * wid;
        int hw = hgt * wid;

        using var kernel = new PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.InputBytes);
        using var gradWeightDevice = runtime.AllocateBytes((nuint)PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel.GradWeightBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradWeightDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * inChannels * kernelSize * kernelSize * kernelSize];
        gradWeightDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int ci = 0; ci < inChannels; ci++)
        for (int kd = 0; kd < kernelSize; kd++)
        for (int kh = 0; kh < kernelSize; kh++)
        for (int kw = 0; kw < kernelSize; kw++)
        {
            float expected = 0;
            for (int d = 0; d < depth; d++)
            for (int h = 0; h < hgt; h++)
            for (int w = 0; w < wid; w++)
            {
                int id = d + kd - 1, ih = h + kh - 1, iw = w + kw - 1;
                if (id < 0 || id >= depth || ih < 0 || ih >= hgt || iw < 0 || iw >= wid) continue;
                expected += gradOut[co * spatial + (d * hw + h * wid + w)] *
                    input[ci * spatial + (id * hw + ih * wid + iw)];
            }
            int wIdx = (((co * inChannels + ci) * kernelSize + kd) * kernelSize + kh) * kernelSize + kw;
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(actual[wIdx] - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Conv3D 3x3x3 backward-weight max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void FusedConv3D3x3x3Emitter_HasBiasAddAndReluEpilogue()
    {
        string ptx = PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(4, Count(ptx, ".param .u64"));
        Assert.Contains("CONV3D_FUSED_CHANNELS:", ptx, StringComparison.Ordinal);
        // 27 taps per input channel, unrolled inside the channel loop.
        Assert.Equal(27, Count(ptx, "fma.rn.f32"));
        // Fused epilogue: one bias add + one ReLU max, single store, no intermediate.
        Assert.Equal(1, Count(ptx, "add.rn.f32"));
        Assert.Equal(1, Count(ptx, "max.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Contains("@%p7 bra CONV3D_FUSED_CHANNELS", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void FusedConv3D3x3x3ManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.FusedConv3D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxFusedConv3D3x3x3BiasRelu").Status);
    }

    [SkippableFact]
    public void DriverOnlyFusedConv3D3x3x3BiasRelu_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.InputChannels;
        const int outChannels = PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.OutputChannels;
        const int depth = PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.Depth;
        const int hgt = PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.Height;
        const int wid = PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.Width;
        const int kernelSize = PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.KernelSize;
        int spatial = depth * hgt * wid;
        int hw = hgt * wid;

        using var kernel = new PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.WeightBytes);
        using var biasDevice = runtime.AllocateBytes((nuint)PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.BiasBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxFusedConv3DNcdhw3x3x3BiasReluF32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        var bias = new float[outChannels];
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);
        biasDevice.Upload<float>(bias);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(biasDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        var actual = new float[outChannels * spatial];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int d = 0; d < depth; d++)
        for (int h = 0; h < hgt; h++)
        for (int w = 0; w < wid; w++)
        {
            float acc = bias[co];
            for (int ci = 0; ci < inChannels; ci++)
            for (int kd = 0; kd < kernelSize; kd++)
            for (int kh = 0; kh < kernelSize; kh++)
            for (int kw = 0; kw < kernelSize; kw++)
            {
                int id = d + kd - 1, ih = h + kh - 1, iw = w + kw - 1;
                if (id < 0 || id >= depth || ih < 0 || ih >= hgt || iw < 0 || iw >= wid) continue;
                acc += weights[(((co * inChannels + ci) * kernelSize + kd) * kernelSize + kh) * kernelSize + kw] *
                    input[ci * spatial + (id * hw + ih * wid + iw)];
            }
            float expected = MathF.Max(acc, 0f);
            float got = actual[co * spatial + (d * hw + h * wid + w)];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Fused Conv3D 3x3x3 bias+ReLU max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void LocallyConnectedConv2DEmitter_IsPerPositionHaloChannelLoopSm86Ptx()
    {
        string ptx = PtxLocallyConnectedConv2DNchw3x3F32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxLocallyConnectedConv2DNchw3x3F32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.Contains("LOCALCONN_CHANNELS:", ptx, StringComparison.Ordinal);
        // Per-position weight base derives from (spatial*Cout + co) — no shared co-only base.
        Assert.Contains("mad.lo.u32 %r7, %r3, 8, %r6", ptx, StringComparison.Ordinal);
        // 9 taps per input channel, unrolled inside the channel loop; single store, no epilogue.
        Assert.Equal(9, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(0, Count(ptx, "add.rn.f32"));   // no bias fusion
        Assert.Equal(0, Count(ptx, "max.f32"));      // no activation fusion
        Assert.Equal(4, Count(ptx, "and.pred %p4"));
        Assert.Contains("@%p5 bra LOCALCONN_CHANNELS", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxLocallyConnectedConv2DNchw3x3F32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void LocallyConnectedConv2DManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.LocallyConnectedConv2D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxLocallyConnectedConv2D").Status);
    }

    [SkippableFact]
    public void DriverOnlyLocallyConnectedConv2D_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxLocallyConnectedConv2DNchw3x3F32Kernel.InputChannels;
        const int outChannels = PtxLocallyConnectedConv2DNchw3x3F32Kernel.OutputChannels;
        const int height = PtxLocallyConnectedConv2DNchw3x3F32Kernel.Height;
        const int width = PtxLocallyConnectedConv2DNchw3x3F32Kernel.Width;
        const int kernelSize = PtxLocallyConnectedConv2DNchw3x3F32Kernel.KernelSize;
        int spatial = height * width;

        using var kernel = new PtxLocallyConnectedConv2DNchw3x3F32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchw3x3F32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchw3x3F32Kernel.WeightBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchw3x3F32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[PtxLocallyConnectedConv2DNchw3x3F32Kernel.WeightElements];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[outChannels * spatial];
        outputDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int oy = 0; oy < height; oy++)
        for (int ox = 0; ox < width; ox++)
        {
            int pos = oy * width + ox;
            float acc = 0;
            for (int ci = 0; ci < inChannels; ci++)
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int iy = oy + ky - 1;
                int ix = ox + kx - 1;
                if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                // Per-position weight index: (((pos*Cout + co)*Cin + ci)*KK) + ky*3 + kx.
                int wIdx = (((pos * outChannels + co) * inChannels + ci) * kernelSize * kernelSize)
                    + ky * kernelSize + kx;
                acc += weights[wIdx] * input[(ci * height + iy) * width + ix];
            }
            float got = actual[(co * height + oy) * width + ox];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - acc));
        }

        Assert.True(maxAbsoluteError <= 1e-4f,
            $"Locally-connected Conv2D 3x3 max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void LocallyConnectedConv2DBackwardInputEmitter_IsTransposeGatherSm86Ptx()
    {
        string ptx = PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        // pos_out = spatial - tapShift folds each tap's output position to a spatial offset.
        Assert.Contains("mad.lo.u32 %r9, %r8, 576, %r7", ptx, StringComparison.Ordinal);
        // 9 taps x 8 output channels, thread-private accumulation; single store, no atomics.
        Assert.Equal(72, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        // Eight non-center taps each guard with a validity skip branch.
        Assert.Equal(8, Count(ptx, "bra LC_BWDIN_SKIP_"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void LocallyConnectedConv2DBackwardInputManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.LocallyConnectedConv2DBackwardInput").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxLocallyConnectedConv2DBackwardInput").Status);
    }

    [SkippableFact]
    public void DriverOnlyLocallyConnectedConv2DBackwardInput_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.InputChannels;
        const int outChannels = PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.OutputChannels;
        const int height = PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.Height;
        const int width = PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.Width;
        const int kernelSize = PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.KernelSize;
        int spatial = height * width;

        using var kernel = new PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.GradOutputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.WeightBytes);
        using var gradInDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.GradInputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[PtxLocallyConnectedConv2DNchw3x3BackwardInputF32Kernel.WeightElements];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        weightDevice.Upload<float>(weights);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradInDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[inChannels * spatial];
        gradInDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int ci = 0; ci < inChannels; ci++)
        for (int iy = 0; iy < height; iy++)
        for (int ix = 0; ix < width; ix++)
        {
            float acc = 0;
            for (int co = 0; co < outChannels; co++)
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int oy = iy - ky + 1;
                int ox = ix - kx + 1;
                if (oy < 0 || oy >= height || ox < 0 || ox >= width) continue;
                int pos = oy * width + ox;
                int wIdx = (((pos * outChannels + co) * inChannels + ci) * kernelSize * kernelSize)
                    + ky * kernelSize + kx;
                acc += weights[wIdx] * gradOut[co * spatial + pos];
            }
            float got = actual[(ci * height + iy) * width + ix];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - acc));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Locally-connected Conv2D 3x3 backward-input max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void LocallyConnectedConv2DBackwardWeightEmitter_IsSingleProductSm86Ptx()
    {
        string ptx = PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        // Two integer divides decompose the flat weight id (id/9 and tap_kk/3).
        Assert.Equal(2, Count(ptx, "div.u32"));
        // Unshared weight = single output usage: exactly one multiply, no fma reduction, single store.
        Assert.Equal(1, Count(ptx, "mul.f32"));
        Assert.Equal(0, Count(ptx, "fma.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        // No loop: straight-line thread-per-weight (no backward branch).
        Assert.DoesNotContain("bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void LocallyConnectedConv2DBackwardWeightManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.LocallyConnectedConv2DBackwardWeights").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxLocallyConnectedConv2DBackwardWeights").Status);
    }

    [SkippableFact]
    public void DriverOnlyLocallyConnectedConv2DBackwardWeight_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.InputChannels;
        const int outChannels = PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.OutputChannels;
        const int height = PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.Height;
        const int width = PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.Width;
        const int kernelSize = PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.KernelSize;
        int spatial = height * width;

        using var kernel = new PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.InputBytes);
        using var gradWeightDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.GradWeightBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(gradWeightDevice, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[PtxLocallyConnectedConv2DNchw3x3BackwardWeightF32Kernel.WeightElements];
        gradWeightDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int oy = 0; oy < height; oy++)
        for (int ox = 0; ox < width; ox++)
        for (int co = 0; co < outChannels; co++)
        for (int ci = 0; ci < inChannels; ci++)
        for (int ky = 0; ky < kernelSize; ky++)
        for (int kx = 0; kx < kernelSize; kx++)
        {
            int pos = oy * width + ox;
            int iy = oy + ky - 1;
            int ix = ox + kx - 1;
            float inVal = (iy < 0 || iy >= height || ix < 0 || ix >= width)
                ? 0f
                : input[ci * spatial + (iy * width + ix)];
            float expected = gradOut[co * spatial + pos] * inVal;
            int wIdx = (((pos * outChannels + co) * inChannels + ci) * kernelSize * kernelSize)
                + ky * kernelSize + kx;
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(actual[wIdx] - expected));
        }

        Assert.True(maxAbsoluteError <= 5e-5f,
            $"Locally-connected Conv2D 3x3 backward-weight max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void LocallyConnectedConv2DBackwardBiasEmitter_IsBatchReductionSm86Ptx()
    {
        string ptx = PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(2, Count(ptx, ".param .u64"));
        // N=4 batch: 4 lane loads, 3 accumulation adds, single store, no atomics.
        Assert.Equal(4, Count(ptx, "ld.global.nc.f32"));
        Assert.Equal(3, Count(ptx, "add.rn.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("bra ", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("red.global", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void LocallyConnectedConv2DBackwardBiasManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.LocallyConnectedConv2DBackwardBias").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxLocallyConnectedConv2DBackwardBias").Status);
    }

    [SkippableFact]
    public void DriverOnlyLocallyConnectedConv2DBackwardBias_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int batch = PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel.Batch;
        const int biasElements = PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel.BiasElements;

        using var kernel = new PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel.GradOutputBytes);
        using var gradBiasDevice = runtime.AllocateBytes((nuint)PtxLocallyConnectedConv2DNchwBackwardBiasF32Kernel.GradBiasBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[batch * biasElements];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(gradBiasDevice, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[biasElements];
        gradBiasDevice.Download<float>(actual);

        float maxAbsoluteError = 0;
        for (int id = 0; id < biasElements; id++)
        {
            float expected = 0;
            for (int n = 0; n < batch; n++) expected += gradOut[n * biasElements + id];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(actual[id] - expected));
        }

        Assert.True(maxAbsoluteError <= 5e-5f,
            $"Locally-connected Conv2D backward-bias max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void DeformableConv2DEmitter_IsBilinearSamplingSm86Ptx()
    {
        string ptx = PtxDeformableConv2DNchw3x3F32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDeformableConv2DNchw3x3F32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(5, Count(ptx, ".param .u64"));
        // Bilinear: floor via round-to-minus-infinity, two per tap (y, x) over 9 taps.
        Assert.Equal(18, Count(ptx, "cvt.rmi.s32.f32"));
        // Nine per-tap channel loops, each guarded by a validity branch.
        Assert.Equal(9, Count(ptx, "bra DEFORM_TAP"));
        Assert.Contains("DEFORM_TAP0_CI:", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDeformableConv2DNchw3x3F32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DeformableConv2DManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DeformableConv2D").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDeformableConv2D").Status);
    }

    [SkippableFact]
    public void DriverOnlyDeformableConv2D_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxDeformableConv2DNchw3x3F32Kernel.InputChannels;
        const int outChannels = PtxDeformableConv2DNchw3x3F32Kernel.OutputChannels;
        const int height = PtxDeformableConv2DNchw3x3F32Kernel.Height;
        const int width = PtxDeformableConv2DNchw3x3F32Kernel.Width;
        const int kernelSize = PtxDeformableConv2DNchw3x3F32Kernel.KernelSize;
        const int taps = PtxDeformableConv2DNchw3x3F32Kernel.TapsPerChannel;
        int spatial = height * width;

        using var kernel = new PtxDeformableConv2DNchw3x3F32Kernel(runtime);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3F32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3F32Kernel.WeightBytes);
        using var offsetDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3F32Kernel.OffsetBytes);
        using var maskDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3F32Kernel.MaskBytes);
        using var outputDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3F32Kernel.OutputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        var offsets = new float[2 * taps * spatial];
        for (int i = 0; i < offsets.Length; i++) offsets[i] = (float)(random.NextDouble() * 2 - 1);
        var mask = new float[taps * spatial];
        for (int i = 0; i < mask.Length; i++) mask[i] = (float)random.NextDouble();
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);
        offsetDevice.Upload<float>(offsets);
        maskDevice.Upload<float>(mask);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(offsetDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(maskDevice, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(outputDevice, kernel.Blueprint.Tensors[4]));
        runtime.Synchronize();
        var actual = new float[outChannels * spatial];
        outputDevice.Download<float>(actual);

        // Same bilinear as the kernel: floor + 4 corner, zero-pad outside the frame.
        static float Bilinear(float[] x, int ci, int spatialSize, int h, int w, float py, float px)
        {
            int y0 = (int)MathF.Floor(py);
            int x0 = (int)MathF.Floor(px);
            int y1 = y0 + 1;
            int x1 = x0 + 1;
            float wy1 = py - y0;
            float wy0 = 1f - wy1;
            float wx1 = px - x0;
            float wx0 = 1f - wx1;
            float Sample(int yy, int xx) =>
                (yy < 0 || yy >= h || xx < 0 || xx >= w) ? 0f : x[ci * spatialSize + (yy * w + xx)];
            return wy0 * wx0 * Sample(y0, x0) + wy0 * wx1 * Sample(y0, x1)
                 + wy1 * wx0 * Sample(y1, x0) + wy1 * wx1 * Sample(y1, x1);
        }

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int oy = 0; oy < height; oy++)
        for (int ox = 0; ox < width; ox++)
        {
            int s = oy * width + ox;
            float acc = 0;
            for (int ky = 0; ky < kernelSize; ky++)
            for (int kx = 0; kx < kernelSize; kx++)
            {
                int t = ky * kernelSize + kx;
                float offY = offsets[(2 * t) * spatial + s];
                float offX = offsets[(2 * t + 1) * spatial + s];
                float py = oy + ky - 1 + offY;
                float px = ox + kx - 1 + offX;
                float m = mask[t * spatial + s];
                for (int ci = 0; ci < inChannels; ci++)
                {
                    float sample = Bilinear(input, ci, spatial, height, width, py, px);
                    acc += weights[((co * inChannels + ci) * kernelSize + ky) * kernelSize + kx] * m * sample;
                }
            }
            float got = actual[co * spatial + s];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - acc));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Deformable Conv2D forward max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void DeformableConv2DBackwardWeightEmitter_IsBilinearRecomputeSm86Ptx()
    {
        string ptx = PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(5, Count(ptx, ".param .u64"));
        Assert.Contains("DEFORM_BWD_WEIGHT:", ptx, StringComparison.Ordinal);
        // Two id-decompose divides (id/9, t/3); single spatial loop recomputing bilinear.
        Assert.Equal(2, Count(ptx, "div.u32"));
        Assert.Equal(2, Count(ptx, "cvt.rmi.s32.f32"));
        Assert.Contains("@%p12 bra DEFORM_BWD_WEIGHT", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DeformableConv2DBackwardWeightManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DeformableConv2DBackwardKernel").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDeformableConv2DBackwardWeights").Status);
    }

    [SkippableFact]
    public void DriverOnlyDeformableConv2DBackwardWeight_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.InputChannels;
        const int outChannels = PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.OutputChannels;
        const int height = PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.Height;
        const int width = PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.Width;
        const int kernelSize = PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.KernelSize;
        const int taps = PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.TapsPerChannel;
        int spatial = height * width;

        using var kernel = new PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.InputBytes);
        using var offsetDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.OffsetBytes);
        using var maskDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.MaskBytes);
        using var gradWeightDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel.GradWeightBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var offsets = new float[2 * taps * spatial];
        for (int i = 0; i < offsets.Length; i++) offsets[i] = (float)(random.NextDouble() * 2 - 1);
        var mask = new float[taps * spatial];
        for (int i = 0; i < mask.Length; i++) mask[i] = (float)random.NextDouble();
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);
        offsetDevice.Upload<float>(offsets);
        maskDevice.Upload<float>(mask);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(offsetDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(maskDevice, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(gradWeightDevice, kernel.Blueprint.Tensors[4]));
        runtime.Synchronize();
        var actual = new float[outChannels * inChannels * kernelSize * kernelSize];
        gradWeightDevice.Download<float>(actual);

        static float Bilinear(float[] x, int ci, int spatialSize, int h, int w, float py, float px)
        {
            int y0 = (int)MathF.Floor(py);
            int x0 = (int)MathF.Floor(px);
            int y1 = y0 + 1;
            int x1 = x0 + 1;
            float wy1 = py - y0;
            float wy0 = 1f - wy1;
            float wx1 = px - x0;
            float wx0 = 1f - wx1;
            float Sample(int yy, int xx) =>
                (yy < 0 || yy >= h || xx < 0 || xx >= w) ? 0f : x[ci * spatialSize + (yy * w + xx)];
            return wy0 * wx0 * Sample(y0, x0) + wy0 * wx1 * Sample(y0, x1)
                 + wy1 * wx0 * Sample(y1, x0) + wy1 * wx1 * Sample(y1, x1);
        }

        float maxAbsoluteError = 0;
        for (int co = 0; co < outChannels; co++)
        for (int ci = 0; ci < inChannels; ci++)
        for (int ky = 0; ky < kernelSize; ky++)
        for (int kx = 0; kx < kernelSize; kx++)
        {
            int t = ky * kernelSize + kx;
            float expected = 0;
            for (int oy = 0; oy < height; oy++)
            for (int ox = 0; ox < width; ox++)
            {
                int s = oy * width + ox;
                float offY = offsets[(2 * t) * spatial + s];
                float offX = offsets[(2 * t + 1) * spatial + s];
                float py = oy + ky - 1 + offY;
                float px = ox + kx - 1 + offX;
                float sample = Bilinear(input, ci, spatial, height, width, py, px);
                expected += gradOut[co * spatial + s] * mask[t * spatial + s] * sample;
            }
            int wIdx = ((co * inChannels + ci) * kernelSize + ky) * kernelSize + kx;
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(actual[wIdx] - expected));
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Deformable Conv2D backward-weight max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void DeformableConv2DBackwardMaskEmitter_IsBilinearReductionSm86Ptx()
    {
        string ptx = PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(5, Count(ptx, ".param .u64"));
        // Bilinear computed once per thread: single floor pair, one div for ky.
        Assert.Equal(2, Count(ptx, "cvt.rmi.s32.f32"));
        Assert.Equal(1, Count(ptx, "div.u32"));
        // Nested ci (outer) and co (inner) loops.
        Assert.Contains("DEFORM_BWD_MASK_CI:", ptx, StringComparison.Ordinal);
        Assert.Contains("DEFORM_BWD_MASK_CO:", ptx, StringComparison.Ordinal);
        Assert.Contains("@%p13 bra DEFORM_BWD_MASK_CI", ptx, StringComparison.Ordinal);
        Assert.Contains("@%p12 bra DEFORM_BWD_MASK_CO", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DeformableConv2DBackwardMaskManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DeformableConv2DBackwardMask").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDeformableConv2DBackwardMask").Status);
    }

    [SkippableFact]
    public void DriverOnlyDeformableConv2DBackwardMask_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.InputChannels;
        const int outChannels = PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.OutputChannels;
        const int height = PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.Height;
        const int width = PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.Width;
        const int kernelSize = PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.KernelSize;
        const int taps = PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.TapsPerChannel;
        int spatial = height * width;

        using var kernel = new PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.WeightBytes);
        using var offsetDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.OffsetBytes);
        using var gradMaskDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel.GradMaskBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        var offsets = new float[2 * taps * spatial];
        for (int i = 0; i < offsets.Length; i++) offsets[i] = (float)(random.NextDouble() * 2 - 1);
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);
        offsetDevice.Upload<float>(offsets);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(offsetDevice, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(gradMaskDevice, kernel.Blueprint.Tensors[4]));
        runtime.Synchronize();
        var actual = new float[taps * spatial];
        gradMaskDevice.Download<float>(actual);

        static float Bilinear(float[] x, int ci, int spatialSize, int h, int w, float py, float px)
        {
            int y0 = (int)MathF.Floor(py);
            int x0 = (int)MathF.Floor(px);
            int y1 = y0 + 1;
            int x1 = x0 + 1;
            float wy1 = py - y0;
            float wy0 = 1f - wy1;
            float wx1 = px - x0;
            float wx0 = 1f - wx1;
            float Sample(int yy, int xx) =>
                (yy < 0 || yy >= h || xx < 0 || xx >= w) ? 0f : x[ci * spatialSize + (yy * w + xx)];
            return wy0 * wx0 * Sample(y0, x0) + wy0 * wx1 * Sample(y0, x1)
                 + wy1 * wx0 * Sample(y1, x0) + wy1 * wx1 * Sample(y1, x1);
        }

        float maxAbsoluteError = 0;
        for (int t = 0; t < taps; t++)
        {
            int ky = t / kernelSize;
            int kx = t % kernelSize;
            for (int oy = 0; oy < height; oy++)
            for (int ox = 0; ox < width; ox++)
            {
                int s = oy * width + ox;
                float offY = offsets[(2 * t) * spatial + s];
                float offX = offsets[(2 * t + 1) * spatial + s];
                float py = oy + ky - 1 + offY;
                float px = ox + kx - 1 + offX;
                float expected = 0;
                for (int ci = 0; ci < inChannels; ci++)
                {
                    float sample = Bilinear(input, ci, spatial, height, width, py, px);
                    for (int co = 0; co < outChannels; co++)
                        expected += gradOut[co * spatial + s]
                            * weights[((co * inChannels + ci) * kernelSize + ky) * kernelSize + kx] * sample;
                }
                maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(actual[t * spatial + s] - expected));
            }
        }

        Assert.True(maxAbsoluteError <= 2e-4f,
            $"Deformable Conv2D backward-mask max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void DeformableConv2DBackwardOffsetEmitter_IsSpatialDerivativeSm86Ptx()
    {
        string ptx = PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(6, Count(ptx, ".param .u64"));
        Assert.Equal(2, Count(ptx, "cvt.rmi.s32.f32"));
        Assert.Equal(1, Count(ptx, "div.u32"));
        Assert.Contains("DEFORM_BWD_OFFSET_CI:", ptx, StringComparison.Ordinal);
        Assert.Contains("DEFORM_BWD_OFFSET_CO:", ptx, StringComparison.Ordinal);
        // Dual outputs: dOffY and dOffX both stored.
        Assert.Equal(2, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DeformableConv2DBackwardOffsetManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DeformableConv2DBackwardOffset").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDeformableConv2DBackwardOffset").Status);
    }

    [SkippableFact]
    public void DriverOnlyDeformableConv2DBackwardOffset_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.InputChannels;
        const int outChannels = PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.OutputChannels;
        const int height = PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.Height;
        const int width = PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.Width;
        const int kernelSize = PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.KernelSize;
        const int taps = PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.TapsPerChannel;
        int spatial = height * width;

        using var kernel = new PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.GradOutputBytes);
        using var inputDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.InputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.WeightBytes);
        using var offsetDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.OffsetBytes);
        using var maskDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.MaskBytes);
        using var gradOffsetDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel.GradOffsetBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var input = new float[inChannels * spatial];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        var offsets = new float[2 * taps * spatial];
        for (int i = 0; i < offsets.Length; i++) offsets[i] = (float)(random.NextDouble() * 2 - 1);
        var mask = new float[taps * spatial];
        for (int i = 0; i < mask.Length; i++) mask[i] = (float)random.NextDouble();
        gradOutDevice.Upload<float>(gradOut);
        inputDevice.Upload<float>(input);
        weightDevice.Upload<float>(weights);
        offsetDevice.Upload<float>(offsets);
        maskDevice.Upload<float>(mask);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(inputDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(offsetDevice, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(maskDevice, kernel.Blueprint.Tensors[4]),
            DirectPtxTensorView.CreateOwned(gradOffsetDevice, kernel.Blueprint.Tensors[5]));
        runtime.Synchronize();
        var actual = new float[2 * taps * spatial];
        gradOffsetDevice.Download<float>(actual);

        // Same bilinear spatial derivative as the kernel (corner values 0 outside frame).
        float maxAbsoluteError = 0;
        for (int t = 0; t < taps; t++)
        {
            int ky = t / kernelSize;
            int kx = t % kernelSize;
            for (int oy = 0; oy < height; oy++)
            for (int ox = 0; ox < width; ox++)
            {
                int s = oy * width + ox;
                float offY = offsets[(2 * t) * spatial + s];
                float offX = offsets[(2 * t + 1) * spatial + s];
                float py = oy + ky - 1 + offY;
                float px = ox + kx - 1 + offX;
                int y0 = (int)MathF.Floor(py);
                int x0 = (int)MathF.Floor(px);
                int y1 = y0 + 1, x1 = x0 + 1;
                float fy = py - y0, fx = px - x0;
                float wy0 = 1f - fy, wx0 = 1f - fx;
                float accY = 0, accX = 0;
                for (int ci = 0; ci < inChannels; ci++)
                {
                    float V00 = (y0 < 0 || y0 >= height || x0 < 0 || x0 >= width) ? 0f : input[ci * spatial + (y0 * width + x0)];
                    float V01 = (y0 < 0 || y0 >= height || x1 < 0 || x1 >= width) ? 0f : input[ci * spatial + (y0 * width + x1)];
                    float V10 = (y1 < 0 || y1 >= height || x0 < 0 || x0 >= width) ? 0f : input[ci * spatial + (y1 * width + x0)];
                    float V11 = (y1 < 0 || y1 >= height || x1 < 0 || x1 >= width) ? 0f : input[ci * spatial + (y1 * width + x1)];
                    float dpy = wx0 * (V10 - V00) + fx * (V11 - V01);
                    float dpx = wy0 * (V01 - V00) + fy * (V11 - V10);
                    float g = 0;
                    for (int co = 0; co < outChannels; co++)
                        g += gradOut[co * spatial + s] * weights[((co * inChannels + ci) * kernelSize + ky) * kernelSize + kx];
                    accY += g * dpy;
                    accX += g * dpx;
                }
                float m = mask[t * spatial + s];
                float expectedY = m * accY;
                float expectedX = m * accX;
                maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(actual[(2 * t) * spatial + s] - expectedY));
                maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(actual[(2 * t + 1) * spatial + s] - expectedX));
            }
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"Deformable Conv2D backward-offset max absolute error {maxAbsoluteError:G9}.");
    }

    [Fact]
    public void DeformableConv2DBackwardInputEmitter_IsTransposeGatherSm86Ptx()
    {
        string ptx = PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.EmitPtx(8, 6);

        Assert.Contains(".target sm_86", ptx, StringComparison.Ordinal);
        Assert.Contains(PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.EntryPoint, ptx, StringComparison.Ordinal);
        Assert.Equal(5, Count(ptx, ".param .u64"));
        // g(d)=max(0,1-|d|): two abs + two max for the separable bilinear transpose weight.
        Assert.Equal(2, Count(ptx, "abs.f32"));
        Assert.Equal(2, Count(ptx, "max.f32"));
        // Triple loop t -> s -> co.
        Assert.Contains("DEFORM_BWD_INPUT_T:", ptx, StringComparison.Ordinal);
        Assert.Contains("DEFORM_BWD_INPUT_S:", ptx, StringComparison.Ordinal);
        Assert.Contains("DEFORM_BWD_INPUT_CO:", ptx, StringComparison.Ordinal);
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("atom", ptx, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() =>
            PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.EmitPtx(8, 9));
    }

    [Fact]
    public void DeformableConv2DBackwardInputManifestCell_IsExperimentalWithDedicatedRoute()
    {
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest.Get("IEngine.DeformableConv2DBackwardInput").Status);
        Assert.Equal(DirectPtxConvolutionCoverageStatus.ExperimentalDirectPtx,
            DirectPtxConvolutionCoverageManifest
                .Get("CudaBackend.TryDirectPtxDeformableConv2DBackwardInput").Status);
    }

    [SkippableFact]
    public void DriverOnlyDeformableConv2DBackwardInput_MatchesCpuReference()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "Requires the experimental SM86 convolution specialization.");

        const int inChannels = PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.InputChannels;
        const int outChannels = PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.OutputChannels;
        const int height = PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.Height;
        const int width = PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.Width;
        const int kernelSize = PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.KernelSize;
        const int taps = PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.TapsPerChannel;
        int spatial = height * width;

        using var kernel = new PtxDeformableConv2DNchw3x3BackwardInputF32Kernel(runtime);
        using var gradOutDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.GradOutputBytes);
        using var weightDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.WeightBytes);
        using var offsetDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.OffsetBytes);
        using var maskDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.MaskBytes);
        using var gradInDevice = runtime.AllocateBytes((nuint)PtxDeformableConv2DNchw3x3BackwardInputF32Kernel.GradInputBytes);

        Random random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var gradOut = new float[outChannels * spatial];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(random.NextDouble() * 2 - 1);
        var weights = new float[outChannels * inChannels * kernelSize * kernelSize];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1);
        var offsets = new float[2 * taps * spatial];
        for (int i = 0; i < offsets.Length; i++) offsets[i] = (float)(random.NextDouble() * 2 - 1);
        var mask = new float[taps * spatial];
        for (int i = 0; i < mask.Length; i++) mask[i] = (float)random.NextDouble();
        gradOutDevice.Upload<float>(gradOut);
        weightDevice.Upload<float>(weights);
        offsetDevice.Upload<float>(offsets);
        maskDevice.Upload<float>(mask);

        kernel.Launch(
            DirectPtxTensorView.CreateOwned(gradOutDevice, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(weightDevice, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(offsetDevice, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(maskDevice, kernel.Blueprint.Tensors[3]),
            DirectPtxTensorView.CreateOwned(gradInDevice, kernel.Blueprint.Tensors[4]));
        runtime.Synchronize();
        var actual = new float[inChannels * spatial];
        gradInDevice.Download<float>(actual);

        static float G(float d) => MathF.Max(0f, 1f - MathF.Abs(d));

        float maxAbsoluteError = 0;
        for (int ci = 0; ci < inChannels; ci++)
        for (int iy = 0; iy < height; iy++)
        for (int ix = 0; ix < width; ix++)
        {
            float acc = 0;
            for (int t = 0; t < taps; t++)
            {
                int ky = t / kernelSize;
                int kx = t % kernelSize;
                for (int oy = 0; oy < height; oy++)
                for (int ox = 0; ox < width; ox++)
                {
                    int s = oy * width + ox;
                    float py = oy + ky - 1 + offsets[(2 * t) * spatial + s];
                    float px = ox + kx - 1 + offsets[(2 * t + 1) * spatial + s];
                    float coeff = G(iy - py) * G(ix - px);
                    if (coeff == 0f) continue;
                    float g = 0;
                    for (int co = 0; co < outChannels; co++)
                        g += gradOut[co * spatial + s] * weights[((co * inChannels + ci) * kernelSize + ky) * kernelSize + kx];
                    acc += coeff * mask[t * spatial + s] * g;
                }
            }
            float got = actual[ci * spatial + (iy * width + ix)];
            maxAbsoluteError = MathF.Max(maxAbsoluteError, MathF.Abs(got - acc));
        }

        Assert.True(maxAbsoluteError <= 3e-4f,
            $"Deformable Conv2D backward-input max absolute error {maxAbsoluteError:G9}.");
    }

    private static string? Validate(
        bool enabled, bool available, int major, int minor,
        DirectPtxConvolutionShape shape, IGpuBuffer? input, IGpuBuffer? weights,
        IGpuBuffer? bias, IGpuBuffer? output) =>
        DirectPtxConvolutionEligibility.Validate(
            enabled, available, major, minor, shape, input, weights, bias, output);

    private static (SyntheticGpuBuffer, SyntheticGpuBuffer, SyntheticGpuBuffer, SyntheticGpuBuffer)
        ValidBuffers() =>
        (
            new SyntheticGpuBuffer(new IntPtr(0x100000), PtxFusedConv2DNchwK1Kernel.InputBytes),
            new SyntheticGpuBuffer(new IntPtr(0x200000), PtxFusedConv2DNchwK1Kernel.WeightBytes),
            new SyntheticGpuBuffer(new IntPtr(0x300000), PtxFusedConv2DNchwK1Kernel.BiasBytes),
            new SyntheticGpuBuffer(new IntPtr(0x400000), PtxFusedConv2DNchwK1Kernel.OutputBytes)
        );

    private static int Count(string text, string value)
    {
        int count = 0;
        for (int index = 0; (index = text.IndexOf(value, index, StringComparison.Ordinal)) >= 0;
             index += value.Length)
            count++;
        return count;
    }

    private static string SourcePath(params string[] relativeParts) =>
        Path.Combine(FindRepoRoot(), Path.Combine(relativeParts));

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName,
                    "src", "AiDotNet.Tensors", "AiDotNet.Tensors.csproj")))
                return directory.FullName;
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException("Could not locate repository root.");
    }

    private sealed class SyntheticGpuBuffer : IGpuBuffer
    {
        internal SyntheticGpuBuffer(IntPtr handle, long sizeInBytes) =>
            (Handle, SizeInBytes) = (handle, sizeInBytes);

        public int Size => checked((int)(SizeInBytes / sizeof(float)));
        public long SizeInBytes { get; }
        public IntPtr Handle { get; }
        public void Dispose() { }
    }
}
#endif
