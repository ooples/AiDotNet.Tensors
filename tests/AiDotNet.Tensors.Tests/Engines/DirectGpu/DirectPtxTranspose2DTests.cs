using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape shared-tile 2D transpose (issue #845).
/// The emitter, shape-domain, and fail-closed assertions run without a GPU; the
/// driver correctness assertion is skipped unless an admitted SM86 device is
/// present. The specialization stays disabled by default and fails closed until
/// three clean promotion runs clear the release gate.
/// </summary>
public class DirectPtxTranspose2DTests
{
    [Fact]
    public void TransposeEmitter_StagesOneTileAndIsPointerOnly()
    {
        string ptx = PtxFusedTranspose2DF32Kernel.EmitPtx(8, 6, 1024, 1024);
        Assert.Contains(".maxntid 32, 8, 1", ptx);
        Assert.Contains("exact-shape rows=1024 columns=1024 tile=32x32", ptx);
        Assert.Contains("strategy=shared-tile-cp-async", ptx);
        Assert.Contains("op=transpose2d-f32", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        // 32 rows / 8 thread-rows = 4 staged elements per thread, each way.
        // Staging uses cp.async, so global->shared happens in ONE instruction
        // per element with no register round-trip and no separate store.
        Assert.Equal(4, Count(ptx, "cp.async.ca.shared.global"));
        Assert.DoesNotContain("ld.global.ca.f32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("st.shared.f32", ptx, StringComparison.Ordinal);
        Assert.Equal(4, Count(ptx, "ld.shared.f32"));
        Assert.Equal(4, Count(ptx, "st.global.f32"));
        // The four copies commit as one group and are awaited once.
        Assert.Equal(1, Count(ptx, "cp.async.commit_group"));
        Assert.Equal(1, Count(ptx, "cp.async.wait_group 0"));
        // Exactly one barrier: stage the whole tile, then drain it.
        Assert.Equal(1, Count(ptx, "bar.sync"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void TransposeEmitter_PadsTheTileRowStrideToAvoidBankConflicts()
    {
        string ptx = PtxFusedTranspose2DF32Kernel.EmitPtx(8, 6, 1024, 1024);
        // 32 rows x 33 floats: the odd stride is what makes the transposed
        // shared read touch 32 distinct banks instead of one bank 32 times.
        Assert.Equal(33, PtxFusedTranspose2DF32Kernel.TileStride);
        Assert.Contains(".shared .align 4 .b32 tile[1056];", ptx);
        Assert.Contains("mad.lo.u32 %r9, %r8, 33, %r0;", ptx);
        Assert.Contains("mad.lo.u32 %r15, %r0, 33, %r14;", ptx);
        Assert.Equal(32 * 33 * sizeof(float), PtxFusedTranspose2DF32Kernel.SharedBytes);
    }

    [Fact]
    public void TransposeEmitter_BakesBothExtentsSoTheStridesDiffer()
    {
        // Input row stride is `columns`, output row stride is `rows`. A
        // rectangular shape proves the emitter does not confuse the two.
        string ptx = PtxFusedTranspose2DF32Kernel.EmitPtx(8, 6, 512, 2048);
        Assert.Contains("mad.lo.u32 %r7, %r6, 2048, %r4;", ptx);   // read:  * columns
        Assert.Contains("mad.lo.u32 %r13, %r12, 512, %r10;", ptx); // write: * rows
    }

    [Fact]
    public void TransposeShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedTranspose2DF32Kernel.IsSupportedShape(512, 512));
        Assert.True(PtxFusedTranspose2DF32Kernel.IsSupportedShape(1024, 4096));
        Assert.True(PtxFusedTranspose2DF32Kernel.IsSupportedShape(4096, 2048));
        Assert.False(PtxFusedTranspose2DF32Kernel.IsSupportedShape(1000, 1024));
        Assert.False(PtxFusedTranspose2DF32Kernel.IsSupportedShape(1024, 1023));
        // 256 is a whole tile but outside the measured set: still fails closed.
        Assert.False(PtxFusedTranspose2DF32Kernel.IsSupportedShape(256, 256));
        Assert.False(PtxFusedTranspose2DF32Kernel.IsPromotedShape(1024, 1024));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedTranspose2DF32Kernel.EmitPtx(8, 6, 1000, 1024));
    }

    [Fact]
    public void TransposeArchitectureGate_FailsClosedOutsideSm86()
    {
        Assert.True(DirectPtxArchitecture.HasValidatedTranspose2D(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedTranspose2D(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedTranspose2D(8, 7));
        Assert.False(DirectPtxArchitecture.HasValidatedTranspose2D(8, 9));
        Assert.False(DirectPtxArchitecture.HasValidatedTranspose2D(9, 0));
    }

    [SkippableFact]
    public void BackendTranspose_UnsupportedShapeAndDisabledGateFailClosed()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.Transpose2DExperimentOverride;
        try
        {
            using var backend = new CudaBackend();
            using var input = backend.AllocateBuffer(new float[1024 * 1024]);
            using var output = backend.AllocateBuffer(1024 * 1024);

            DirectPtxFeatureGate.TestOverride = false;
            Assert.False(backend.TryDirectPtxTranspose2D(input, output, 1024, 1024));
            Assert.Equal("transpose2d-feature-disabled", backend.DirectPtxLastError);

            DirectPtxFeatureGate.TestOverride = true;
            DirectPtxFeatureGate.Transpose2DExperimentOverride = true;
            Skip.IfNot(backend.IsDirectPtxTranspose2DEnabled, "Requires a GA10x/SM86 CUDA backend.");
            Assert.False(backend.TryDirectPtxTranspose2D(input, output, 1000, 1024));
            Assert.Equal("transpose2d-shape-not-implemented", backend.DirectPtxLastError);

            // A transpose cannot run in place: the alias guard must reject it.
            Assert.False(backend.TryDirectPtxTranspose2D(input, input, 1024, 1024));
            Assert.Equal("transpose2d-alias-not-supported", backend.DirectPtxLastError);
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.Transpose2DExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void BackendTranspose_PrewarmCaptureAndModuleLifetimeContractsHold()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previousGate = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.Transpose2DExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.Transpose2DExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxTranspose2DEnabled, "Requires a GA10x/SM86 CUDA backend.");
            const int rows = 512, columns = 512;
            using var input = backend.AllocateBuffer(new float[rows * columns]);
            using var output = backend.AllocateBuffer(rows * columns);

            Assert.True(backend.PrewarmDirectPtxTranspose2D(rows, columns), backend.DirectPtxLastError);
            bool captured = true;
            IntPtr graph = backend.CaptureGraph(() =>
                captured &= backend.TryDirectPtxTranspose2D(input, output, rows, columns));
            Assert.True(captured, backend.DirectPtxLastError);
            Assert.NotEqual(IntPtr.Zero, graph);
            Assert.Equal(1, backend.DirectPtxTranspose2DPinnedKernelCount);
            try { backend.LaunchCapturedGraph(graph); }
            finally { backend.DestroyCapturedGraph(graph); }
            backend.Synchronize();
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previousGate;
            DirectPtxFeatureGate.Transpose2DExperimentOverride = previousExperiment;
        }
    }

    [SkippableTheory]
    [InlineData(512, 512)]
    [InlineData(1024, 512)]
    [InlineData(512, 2048)]
    public void DriverOnlyTranspose_IsBitExactAndUsesNoLocalMemory(int rows, int columns)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedTranspose2D(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in transpose specialization is admitted only on SM86.");
        using var kernel = new PtxFusedTranspose2DF32Kernel(runtime, rows, columns);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(PtxFusedTranspose2DF32Kernel.SharedBytes, kernel.Audit.Function.StaticSharedBytes);
        Assert.Equal("transpose2d-f32", kernel.Blueprint.Operation);

        var random = RandomHelper.CreateSeededRandom(20260722 + rows * 31 + columns);
        float[] input = Enumerable.Range(0, rows * columns)
            .Select(_ => (float)((random.NextDouble() * 2.0 - 1.0) * 64.0)).ToArray();

        using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        inputBuffer.Upload<float>(input);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();

        var actual = new float[rows * columns];
        outputBuffer.Download<float>(actual);
        // A transpose only moves elements, so this is bit-exact: no tolerance.
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
                Assert.Equal(input[r * columns + c], actual[c * rows + r]);
    }

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
