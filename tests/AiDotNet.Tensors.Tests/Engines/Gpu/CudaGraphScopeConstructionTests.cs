using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Verifies that <see cref="CudaGraphScope"/> is constructible with any
/// <see cref="IDirectGpuBackend"/> — in particular the real
/// <see cref="CudaBackend"/>, which does not implement the tighter
/// <c>IGpuBatchExecution</c> interface.
///
/// Reference: Issue #171. Before this PR, <c>CudaGraphScope</c> required
/// <c>IGpuBatchExecution</c>, which <c>CudaBackend</c> does not implement,
/// so no caller could construct the type. Option A relaxes the constructor
/// to <see cref="IDirectGpuBackend"/>.
/// </summary>
public class CudaGraphScopeConstructionTests
{
    /// <summary>
    /// Metadata-level acceptance check (runs anywhere, no GPU required):
    /// verifies the public constructor signature is <c>(IDirectGpuBackend, IntPtr)</c>.
    /// This is the exact signature the issue calls for — without requiring a live
    /// CUDA driver to exercise.
    /// </summary>
    [Fact]
    public void Constructor_PublicSignature_TakesIDirectGpuBackend()
    {
        var ctor = typeof(CudaGraphScope).GetConstructors(BindingFlags.Public | BindingFlags.Instance)
            .Single();

        var parameters = ctor.GetParameters();
        Assert.Equal(2, parameters.Length);
        Assert.Equal(typeof(IDirectGpuBackend), parameters[0].ParameterType);
        Assert.Equal(typeof(IntPtr), parameters[1].ParameterType);
    }

    /// <summary>
    /// Verifies the relaxed constructor still guards against null.
    /// No backend instance required — the null path throws before use.
    /// </summary>
    [Fact]
    public void Constructor_RejectsNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new CudaGraphScope((IDirectGpuBackend)null!, new IntPtr(1)));
    }

    /// <summary>
    /// Integration test covering issue #171's full acceptance criterion:
    /// <c>BeginCapture</c> → forward op → <c>EndCapture</c> → <c>Replay</c>
    /// with verified output equivalence vs. eager execution, using the real
    /// <see cref="CudaBackend"/> passed to the relaxed constructor.
    ///
    /// The "forward op" is a device-to-device async memcpy on the captured
    /// stream — the simplest real stream-submitted operation that (a) gets
    /// recorded into the graph during capture, and (b) has a deterministic,
    /// bit-exact expected output that can be compared against eager execution
    /// without depending on any floating-point-sensitive kernel.
    ///
    /// Skipped when CUDA is unavailable on the host (CI without GPU, non-CUDA
    /// dev machines).
    /// </summary>
    [SkippableFact]
    public void Constructor_WithRealCudaBackend_CaptureReplayProducesSameOutputAsEager()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable,
            "CUDA driver not available on this machine");
        Skip.IfNot(CudaNativeBindings.SupportsGraphCapture,
            "Installed CUDA driver predates graph capture (requires 10.0+)");

        using var cudaBackend = new CudaBackend();
        Skip.IfNot(cudaBackend.IsAvailable, "CudaBackend failed to initialize");

        // Graph capture requires a user-created stream — the default stream
        // (IntPtr.Zero) is rejected by cuStreamBeginCapture, which is why the
        // constructor's stream guard exists.
        var streamResult = CudaNativeBindings.cuStreamCreate(out IntPtr stream, 0);
        Skip.IfNot(streamResult == CudaResult.Success,
            $"cuStreamCreate failed: {streamResult}");

        // Use 16 uint32 elements (64 bytes) as the test payload — large enough
        // to catch any per-element bugs, small enough to stay on any GPU.
        const int elementCount = 16;
        const ulong byteCount = elementCount * sizeof(uint);
        const uint pattern = 0xDEADBEEFu;

        IntPtr srcDevice = IntPtr.Zero;
        IntPtr dstEagerDevice = IntPtr.Zero;
        IntPtr dstCapturedDevice = IntPtr.Zero;

        try
        {
            // Allocate three device buffers: src (source pattern), dstEager
            // (direct-copy reference), dstCaptured (filled via graph replay).
            Assert.Equal(CudaResult.Success, CuBlasNative.cuMemAlloc(out srcDevice, byteCount));
            Assert.Equal(CudaResult.Success, CuBlasNative.cuMemAlloc(out dstEagerDevice, byteCount));
            Assert.Equal(CudaResult.Success, CuBlasNative.cuMemAlloc(out dstCapturedDevice, byteCount));

            // Fill source with the known pattern (sync, pre-capture).
            Assert.Equal(CudaResult.Success,
                CuBlasNative.cuMemsetD32(srcDevice, pattern, elementCount));
            // Zero both destinations so a failed copy would show as all-zeros.
            Assert.Equal(CudaResult.Success,
                CuBlasNative.cuMemsetD32(dstEagerDevice, 0u, elementCount));
            Assert.Equal(CudaResult.Success,
                CuBlasNative.cuMemsetD32(dstCapturedDevice, 0u, elementCount));

            // ── Eager path: direct sync memcpy on default stream ──────────
            // Using the sync D2H path to roundtrip: stage src -> host -> dstEager
            // would defeat the comparison. Instead, we use cuMemcpyDtoDAsync
            // with stream=Zero (synchronous default stream), which is the eager
            // analogue of the captured async copy below.
            Assert.Equal(CudaResult.Success,
                CudaNativeBindings.cuMemcpyDtoDAsync(
                    dstEagerDevice, srcDevice, byteCount, IntPtr.Zero));
            // Default stream is synchronous in the legacy mode this test uses,
            // but synchronize explicitly to be robust against per-thread-default
            // stream behaviour.
            CudaNativeBindings.cuStreamSynchronize(IntPtr.Zero);

            // ── Captured path: construct scope with real CudaBackend ─────
            // THIS is the line the issue's acceptance criterion calls out —
            // passing CudaBackend (IAsyncGpuBackend : IDirectGpuBackend) where
            // previously IGpuBatchExecution was demanded.
            using var scope = new CudaGraphScope(cudaBackend, stream);
            Assert.True(scope.IsSupported);

            scope.BeginCapture();
            Assert.True(scope.IsCapturing);

            // Forward op recorded into the graph: async D2D memcpy on the
            // captured stream. CUDA records this as a memcpy node; it does not
            // execute during capture.
            Assert.Equal(CudaResult.Success,
                CudaNativeBindings.cuMemcpyDtoDAsync(
                    dstCapturedDevice, srcDevice, byteCount, stream));

            scope.EndCapture();
            Assert.True(scope.HasGraph);

            // Re-zero the captured destination so any "pattern" we see after
            // replay must have been written by Replay, not by a lingering
            // side-effect of capture.
            Assert.Equal(CudaResult.Success,
                CuBlasNative.cuMemsetD32(dstCapturedDevice, 0u, elementCount));

            scope.Replay();
            // Replay() internally synchronises the captured stream.

            // ── Compare outputs ──────────────────────────────────────────
            uint[] eagerHost = new uint[elementCount];
            uint[] capturedHost = new uint[elementCount];
            CopyDeviceToHost(dstEagerDevice, eagerHost, byteCount);
            CopyDeviceToHost(dstCapturedDevice, capturedHost, byteCount);

            // Sanity: the eager path wrote the pattern.
            for (int i = 0; i < elementCount; i++)
                Assert.Equal(pattern, eagerHost[i]);

            // Acceptance criterion: captured/replayed output is bit-identical
            // to the eager reference.
            Assert.Equal(eagerHost, capturedHost);
        }
        finally
        {
            if (srcDevice != IntPtr.Zero) CuBlasNative.cuMemFree(srcDevice);
            if (dstEagerDevice != IntPtr.Zero) CuBlasNative.cuMemFree(dstEagerDevice);
            if (dstCapturedDevice != IntPtr.Zero) CuBlasNative.cuMemFree(dstCapturedDevice);
            CudaNativeBindings.cuStreamDestroy(stream);
        }
    }

    /// <summary>
    /// Synchronous device-to-host copy for uint32 buffers, used by the
    /// integration test to verify bit-identical output.
    /// </summary>
    private static void CopyDeviceToHost(IntPtr deviceBuffer, uint[] host, ulong byteCount)
    {
        var handle = System.Runtime.InteropServices.GCHandle.Alloc(host,
            System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            var hostPtr = handle.AddrOfPinnedObject();
            var result = CudaNativeBindings.cuMemcpyDtoHAsync(
                hostPtr, deviceBuffer, byteCount, IntPtr.Zero);
            Assert.Equal(CudaResult.Success, result);
            CudaNativeBindings.cuStreamSynchronize(IntPtr.Zero);
        }
        finally
        {
            handle.Free();
        }
    }
}
