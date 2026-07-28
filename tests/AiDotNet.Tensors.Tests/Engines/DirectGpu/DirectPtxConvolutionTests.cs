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
