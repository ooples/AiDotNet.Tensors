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
