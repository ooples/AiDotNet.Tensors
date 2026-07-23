using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP16-to-FP32 direct-PTX widening cast
/// family (issue #845), the mirror of <see cref="DirectPtxCastFp16Tests"/>. The
/// emitter and shape-domain assertions run without a GPU; the driver
/// correctness assertion is skipped unless a validated SM86 device is present.
/// The specialization stays disabled by default and fails closed until three
/// clean promotion runs clear the release gate.
/// </summary>
public class DirectPtxCastFp32Tests
{
    [Fact]
    public void WideningCastEmitter_IsRegisterResidentAndPointerOnly()
    {
        string ptx = PtxFusedCastF16ToF32Kernel.EmitPtx(8, 6, 1_048_576);
        Assert.Contains(".maxntid 256, 1, 1", ptx);
        Assert.Contains("exact-shape size=1048576 block=256", ptx);
        Assert.Contains("op=cast-f16-f32", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        // The load/store types are the exact inverse of the narrowing kernel's.
        // Two vectors per thread: both loads issue before either is consumed.
        Assert.Equal(2, Count(ptx, "ld.global.nc.v4.u16"));
        Assert.Equal(8, Count(ptx, "cvt.f32.f16"));
        Assert.Equal(2, Count(ptx, "st.global.v4.f32"));
        // Widening is exact, so no rounding-mode modifier may appear.
        Assert.DoesNotContain("cvt.rn.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bar.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("shfl.sync", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("stride", ptx, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void WideningCastByteStrides_MirrorTheNarrowingKernel()
    {
        string ptx = PtxFusedCastF16ToF32Kernel.EmitPtx(8, 6, 65_536);
        // 8 elements/thread (2 vectors): 16 input bytes (fp16), 32 output (fp32).
        Assert.Contains("mul.wide.u32 %rd2, %r2, 16;", ptx);
        Assert.Contains("mul.wide.u32 %rd3, %r2, 32;", ptx);
        string narrowing = PtxFusedCastF32ToF16Kernel.EmitPtx(8, 6, 65_536);
        Assert.Contains("mul.wide.u32 %rd2, %r2, 32;", narrowing);
        Assert.Contains("mul.wide.u32 %rd3, %r2, 16;", narrowing);
        // The second vector is reached by an immediate offset, not a second
        // address computation.
        Assert.Contains("[%rd4+8];", ptx);
        Assert.Contains("[%rd5+16], {%f4,", ptx);
    }

    [Fact]
    public void WideningCastShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedCastF16ToF32Kernel.IsSupportedShape(65_536));
        Assert.True(PtxFusedCastF16ToF32Kernel.IsSupportedShape(262_144));
        Assert.True(PtxFusedCastF16ToF32Kernel.IsSupportedShape(1_048_576));
        Assert.True(PtxFusedCastF16ToF32Kernel.IsSupportedShape(4_194_304));
        Assert.False(PtxFusedCastF16ToF32Kernel.IsSupportedShape(65_535));
        Assert.False(PtxFusedCastF16ToF32Kernel.IsSupportedShape(1_000_000));
        Assert.False(PtxFusedCastF16ToF32Kernel.IsPromotedShape(1_048_576));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedCastF16ToF32Kernel.EmitPtx(8, 6, 4_194_304, 192));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedCastF16ToF32Kernel.EmitPtx(8, 6, 1_000_000));
    }

    [Fact]
    public void WideningCastArchitectureGate_FailsClosedOutsideSm86()
    {
        Assert.True(DirectPtxArchitecture.HasValidatedCastFp32(8, 6));
        // Other Ampere parts (SM80/SM87) and every other family must fail closed.
        Assert.False(DirectPtxArchitecture.HasValidatedCastFp32(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedCastFp32(8, 7));
        Assert.False(DirectPtxArchitecture.HasValidatedCastFp32(8, 9));
        Assert.False(DirectPtxArchitecture.HasValidatedCastFp32(9, 0));
    }

    [SkippableFact]
    public void BackendWideningCast_PrewarmCaptureAndModuleLifetimeContractsHold()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.CastFp32ExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.CastFp32ExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxCastFp32Enabled, "Requires a GA10x/SM86 CUDA backend.");
            const int size = 65_536;
            using var input = backend.AllocateBuffer(size / 2);
            using var output = backend.AllocateBuffer(new float[size]);

            Assert.True(backend.PrewarmDirectPtxCastFp32(size), backend.DirectPtxLastError);
            bool captured = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captured &= backend.TryDirectPtxCastFp32(input, output, size));
            Assert.True(captured, backend.DirectPtxLastError);
            Assert.NotEqual(IntPtr.Zero, graph);
            Assert.Equal(1, backend.DirectPtxCastFp32PinnedKernelCount);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }
            backend.Synchronize();
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.CastFp32ExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void BackendWideningCast_UnsupportedShapeAndDisabledGateFailClosed()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.CastFp32ExperimentOverride;
        try
        {
            using var backend = new CudaBackend();

            // Gate off: the route must decline with the exact documented reason.
            DirectPtxFeatureGate.TestOverride = false;
            using var input = backend.AllocateBuffer(65_536 / 2);
            using var output = backend.AllocateBuffer(new float[65_536]);
            Assert.False(backend.TryDirectPtxCastFp32(input, output, 65_536));
            Assert.Equal("cast-fp32-feature-disabled", backend.DirectPtxLastError);

            // Gate on but an off-family shape: still fails closed, not silently.
            DirectPtxFeatureGate.TestOverride = true;
            DirectPtxFeatureGate.CastFp32ExperimentOverride = true;
            Skip.IfNot(backend.IsDirectPtxCastFp32Enabled, "Requires a GA10x/SM86 CUDA backend.");
            Assert.False(backend.TryDirectPtxCastFp32(input, output, 1_000_000));
            Assert.Equal("cast-fp32-shape-not-implemented", backend.DirectPtxLastError);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.CastFp32ExperimentOverride = previousExperiment;
        }
    }

    // System.Half is net5+, so the fp16 input synthesis cannot be expressed on
    // net471. The test is driver-gated anyway; every non-driver contract above
    // still runs on net471.
#if NET5_0_OR_GREATER
    [SkippableTheory]
    [InlineData(65_536)]
    [InlineData(262_144)]
    [InlineData(1_048_576)]
    public void DriverOnlyWideningCast_IsBitExactAndHasZeroLocalBytes(int size)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedCastFp32(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in widening cast specialization is admitted only on SM86.");
        using var kernel = new PtxFusedCastF16ToF32Kernel(runtime, size);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        Assert.True(kernel.Audit.ActiveBlocksPerMultiprocessor >= 3);
        Assert.Equal("cast-f16-to-f32", kernel.Blueprint.Operation);
        Assert.Equal(2, kernel.Blueprint.Tensors.Count);

        var random = RandomHelper.CreateSeededRandom(20260722 + size);
        var input = new ushort[size];
        for (int i = 0; i < size; i++)
            input[i] = BitConverter.HalfToUInt16Bits(
                (Half)((random.NextDouble() * 2.0 - 1.0) * 65_504.0));
        // Pin the edge cases the widening path must carry through exactly.
        input[0] = BitConverter.HalfToUInt16Bits((Half)0.0f);
        input[1] = BitConverter.HalfToUInt16Bits(Half.PositiveInfinity);
        input[2] = BitConverter.HalfToUInt16Bits(Half.NegativeInfinity);
        input[3] = BitConverter.HalfToUInt16Bits(Half.Epsilon);      // smallest subnormal
        input[4] = BitConverter.HalfToUInt16Bits(Half.MaxValue);
        input[5] = BitConverter.HalfToUInt16Bits(Half.MinValue);
        input[6] = BitConverter.HalfToUInt16Bits((Half)(-0.0f));

        using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        inputBuffer.Upload<ushort>(input);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();

        var actual = new float[size];
        outputBuffer.Download<float>(actual);
        for (int i = 0; i < size; i++)
        {
            // FP16 -> FP32 is exact for every input, so this is a bit-for-bit
            // comparison, not a tolerance check.
            float expected = (float)BitConverter.UInt16BitsToHalf(input[i]);
            Assert.Equal(
                BitConverter.SingleToInt32Bits(expected),
                BitConverter.SingleToInt32Bits(actual[i]));
        }
    }
#endif

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
