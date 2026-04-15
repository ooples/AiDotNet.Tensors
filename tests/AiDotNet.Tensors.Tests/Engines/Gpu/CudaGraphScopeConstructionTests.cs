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
    /// Integration test (per issue #171 acceptance criterion):
    /// <c>BeginCapture</c> → <c>EndCapture</c> → <c>Replay</c> with the real
    /// <see cref="CudaBackend"/> passed to the relaxed constructor.
    ///
    /// Skipped when CUDA is unavailable on the host (CI without GPU, non-CUDA
    /// dev machines). When a CUDA driver with graph-capture support (10.0+) is
    /// present, the test:
    /// <list type="number">
    ///   <item>Creates a user stream via <c>cuStreamCreate</c> (graph capture
    ///         rejects the default stream).</item>
    ///   <item>Constructs <see cref="CudaGraphScope"/> with the real
    ///         <see cref="CudaBackend"/> — the exact call that #171 unblocks.</item>
    ///   <item>Captures an empty graph and replays it.</item>
    /// </list>
    ///
    /// Capturing an empty graph is an intentionally minimal integration test —
    /// it verifies the entire capture → instantiate → replay pipeline is
    /// reachable via the relaxed constructor without depending on specific
    /// kernel infrastructure (covered by the broader GPU correctness suite).
    /// Output equivalence of real forward ops vs. eager is verified in those
    /// other tests; this test specifically proves the constructor-relaxation
    /// lets CUDA users wire up graph capture at all.
    /// </summary>
    [SkippableFact]
    public void Constructor_WithRealCudaBackend_CanCaptureAndReplay()
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

        try
        {
            // The acceptance criterion: this line must compile and run with
            // CudaBackend passed directly. Before this PR, CudaBackend did not
            // satisfy the constructor's IGpuBatchExecution parameter.
            using var scope = new CudaGraphScope(cudaBackend, stream);
            Assert.True(scope.IsSupported);

            scope.BeginCapture();
            Assert.True(scope.IsCapturing);

            // Empty graph: minimal valid capture. No kernels submitted — this
            // test exists to prove the lifecycle is reachable via the relaxed
            // constructor, not to re-verify kernel correctness.
            scope.EndCapture();
            Assert.False(scope.IsCapturing);
            Assert.True(scope.HasGraph);

            scope.Replay();
            // Replay internally calls cuStreamSynchronize, so completion is
            // implicit. The assertion is "returns without throwing".
        }
        finally
        {
            CudaNativeBindings.cuStreamDestroy(stream);
        }
    }
}
