using System;
using System.Linq;
using System.Text;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Tests for the issue #840 softmax-family direct-PTX kernels.</summary>
[Collection("DirectGpuSerial")]
public class DirectPtxSoftmaxTests
{
    private sealed class TrackedResource : IDisposable
    {
        internal bool IsDisposed { get; private set; }
        internal bool ThrowOnDispose { get; init; }

        public void Dispose()
        {
            IsDisposed = true;
            if (ThrowOnDispose) throw new InvalidOperationException("cleanup failed");
        }
    }

    [SkippableFact]
    public void StandaloneRuntime_OrdersSynchronousUploadsBeforeLaunches()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Assert.Equal(0u, runtime.StreamFlags);
    }

    [Fact]
    public void SoftmaxCoverageManifest_AssignsEveryScopedApiExactlyOnce()
    {
        Assert.Equal(10, DirectPtxSoftmaxCoverageManifest.All.Count);
        string[] names = DirectPtxSoftmaxCoverageManifest.All.Select(c => c.Api).ToArray();
        Assert.Equal(names.Length, names.Distinct(StringComparer.Ordinal).Count());
        Assert.All(DirectPtxSoftmaxCoverageManifest.All, cell =>
        {
            Assert.False(string.IsNullOrWhiteSpace(cell.ExistingCudaImplementation));
            Assert.False(string.IsNullOrWhiteSpace(cell.Semantics));
            Assert.False(string.IsNullOrWhiteSpace(cell.PhysicalLayout));
            Assert.False(string.IsNullOrWhiteSpace(cell.DTypes));
            Assert.False(string.IsNullOrWhiteSpace(cell.DirectPtxAssignment));
            Assert.Equal(["HIP", "Metal", "OpenCL", "Vulkan", "WebGPU"],
                cell.PeerBackends.Select(peer => peer.Backend));
            Assert.All(cell.PeerBackends, peer => Assert.True(peer.IsAccountedFor));
        });
        // The row-softmax kernel owns both the general Softmax and SoftmaxRows entry points.
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.Softmax").Status);
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.SoftmaxRows").Status);
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.LogSoftmax").Status);
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.LogSumExpAxis").Status);
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.SoftmaxBackward").Status);
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.LogSumExpBackward").Status);
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.MaskedFillKernel").Status);
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.MaskedFillBackward").Status);
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.TaylorSoftmax").Status);
        Assert.Equal(DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            DirectPtxSoftmaxCoverageManifest.Get("CudaBackend.Sparsemax").Status);
        // The whole softmax family now has a direct-PTX owner.
        Assert.DoesNotContain(DirectPtxSoftmaxCoverageManifest.All,
            c => c.Status == DirectPtxSoftmaxCoverageStatus.PlannedDirectPtx);
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxSoftmaxCoverageManifest.Get("UnassignedSoftmaxApi"));
    }

    [Fact]
    public void SoftmaxPostLoadInitialization_DisposesAndPreservesPrimaryFailure()
    {
        var resource = new TrackedResource { ThrowOnDispose = true };
        var expected = new InvalidOperationException("resource validation failed");

        var actual = Assert.Throws<InvalidOperationException>(() =>
            DirectPtxResourceInitialization.Complete<TrackedResource, int>(
                resource, _ => throw expected));

        Assert.Same(expected, actual);
        Assert.True(resource.IsDisposed);
    }

    [Fact]
    public void SoftmaxPostLoadInitialization_TransfersSuccessfulOwnership()
    {
        var resource = new TrackedResource();

        var loaded = DirectPtxResourceInitialization.Complete(resource, _ => 42);

        Assert.Same(resource, loaded.Resource);
        Assert.Equal(42, loaded.Value);
        Assert.False(resource.IsDisposed);
        loaded.Resource.Dispose();
        Assert.True(resource.IsDisposed);
    }

    [Fact]
    public void SoftmaxCoverageManifest_TracksEveryMissingPeerDispatch()
    {
        string[] fullyNativeApis =
        [
            "CudaBackend.Softmax",
            "CudaBackend.SoftmaxRows",
            "CudaBackend.SoftmaxBackward",
            "CudaBackend.LogSumExpBackward",
            "CudaBackend.MaskedFillKernel",
            "CudaBackend.MaskedFillBackward"
        ];
        Assert.All(fullyNativeApis, api =>
            Assert.All(DirectPtxSoftmaxCoverageManifest.Get(api).PeerBackends,
                peer => Assert.False(string.IsNullOrWhiteSpace(peer.NativeImplementation))));

        string[] variantApis =
        [
            "CudaBackend.LogSoftmax",
            "CudaBackend.LogSumExpAxis",
            "CudaBackend.Sparsemax",
            "CudaBackend.TaylorSoftmax"
        ];
        var expectedIssues = new System.Collections.Generic.Dictionary<string, int>
        {
            ["HIP"] = 914,
            ["Metal"] = 915,
            ["OpenCL"] = 916,
            ["WebGPU"] = 917
        };
        Assert.All(variantApis, api =>
        {
            DirectPtxSoftmaxCoverageCell cell = DirectPtxSoftmaxCoverageManifest.Get(api);
            DirectPtxPeerBackendCoverage vulkan = Assert.Single(cell.PeerBackends,
                peer => peer.Backend == "Vulkan");
            Assert.False(string.IsNullOrWhiteSpace(vulkan.NativeImplementation));
            Assert.All(expectedIssues, expected =>
            {
                DirectPtxPeerBackendCoverage peer = Assert.Single(cell.PeerBackends,
                    candidate => candidate.Backend == expected.Key);
                Assert.Equal(expected.Value, peer.FollowUpIssue);
            });
        });
    }

    [Fact]
    public void RowKernelScaffolding_CentralizesAuditedReductionAndShapePolicy()
    {
        Assert.Equal(256, PtxRowShape.BlockThreads);
        Assert.True(PtxRowShape.IsSupported(64, 256));
        Assert.True(PtxRowShape.IsSupported(2048, 4096));
        Assert.False(PtxRowShape.IsSupported(63, 256));
        Assert.False(PtxRowShape.IsSupported(64, 768));
        Assert.False(PtxRowShape.IsPromoted(128, 2048));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxRowShape.Validate(63, 256, "Test row operation"));

        var ptx = new StringBuilder();
        PtxRowReduce.Emit(ptx, "add.rn.f32", "%f0");
        string emitted = ptx.ToString();
        Assert.Equal(2, Count(emitted, "bar.sync 0"));
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            Assert.Equal(2, Count(emitted,
                $"shfl.sync.down.b32 %r11, %r10, {offset}, 31, 0xffffffff"));
        }
        Assert.Contains("setp.lt.u32 %p3, %r0, 8", emitted);
        Assert.Contains("@%p3 shfl.sync.down.b32", emitted);
        Assert.Contains("@%p3 st.shared.f32 [%rd19], %f10", emitted);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxRowReduce.Emit(new StringBuilder(), "mul.rn.f32", "%f0"));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxRowReduce.Emit(new StringBuilder(), "add.rn.f32", "%f1"));

        Assert.Equal(PtxRowShape.BlockThreads, PtxElementwiseShape.BlockThreads);
        Assert.Equal(PtxRowShape.BlockThreads / 32, PtxRowReduce.WarpCount);
        Assert.Equal(32, PtxRowReduce.SharedBytes);
        Assert.Equal(8, PtxElementwiseShape.VectorWidth);
        Assert.Equal(1, PtxElementwiseShape.VectorGridBlocks(256));
        Assert.Equal(1, PtxElementwiseShape.VectorGridBlocks(1280));
        Assert.Equal(2, PtxElementwiseShape.VectorGridBlocks(2304));
        Assert.Equal(1, PtxElementwiseShape.VectorGridBlocks(2304, 512));
        Assert.False(PtxElementwiseShape.RequiresBoundsGuard(2048));
        Assert.True(PtxElementwiseShape.RequiresBoundsGuard(1280));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxElementwiseShape.VectorGridBlocks(256, 0));
        Assert.True(PtxElementwiseShape.IsSupported(PtxElementwiseShape.BlockThreads));
        Assert.True(PtxElementwiseShape.IsSupported(PtxElementwiseShape.MaxCount));
        Assert.False(PtxElementwiseShape.IsSupported(PtxElementwiseShape.BlockThreads + 1));
        Assert.False(PtxElementwiseShape.IsSupported(PtxElementwiseShape.MaxCount + 256));
        Assert.False(PtxElementwiseShape.IsPromoted(1024));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxElementwiseShape.Validate(257, "Test elementwise operation"));
    }

    [Fact]
    public void SparsemaxEmitter_FindsThresholdByBisection()
    {
        string ptx = PtxSparsemaxKernel.EmitPtx(8, 6, 64, 512);
        Assert.Contains(PtxSparsemaxKernel.EntryPoint, ptx);
        Assert.Contains("LOAD_LOOP:", ptx);
        Assert.Contains("BISECT_LOOP:", ptx);
        Assert.Contains("SUM_LOOP:", ptx);
        Assert.Contains("OUT_LOOP:", ptx);
        Assert.Contains("setp.gt.f32 %p1, %f3, 0f3F800000", ptx);       // S(mid) > 1
        Assert.Contains("@!%p1 mov.f32 %f6, %f7", ptx);                 // negated-predicate bracket update
        Assert.DoesNotContain("ex2.approx.f32", ptx);                   // no transcendental
        Assert.Equal(6, Count(ptx, "bar.sync 0"));                      // register max + bisection reduction
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxSparsemaxKernel.IsSupportedShape(128, 2048));
        Assert.False(PtxSparsemaxKernel.IsPromotedShape(128, 2048));
    }

    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(128, 512)]
    public void DriverOnlySparsemax_MatchesSortedOracle(int m, int n)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in sparsemax specialization is measured on GA10x/SM86.");
        using var kernel = new PtxSparsemaxKernel(runtime, m, n);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(PtxRowReduce.Strategy, kernel.Blueprint.Semantics["reduction"]);

        var random = RandomHelper.CreateSeededRandom(20265400 + m + n);
        float[] zHost = Values(random, m * n, 3.0f);
        float[] expected = SparsemaxReference(zHost, m, n);

        using var z = runtime.AllocateBytes((nuint)(zHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        z.Upload<float>(zHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(z, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[m * n];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, $"sparsemax {m}x{n}");
    }

    [Fact]
    public void TaylorSoftmaxEmitter_NormalizesPositivePolynomial()
    {
        string ptx = PtxTaylorSoftmaxKernel.EmitPtx(8, 6, 64, 512);
        Assert.Contains(PtxTaylorSoftmaxKernel.EntryPoint, ptx);
        Assert.Contains("LOAD_LOOP:", ptx);
        Assert.Contains("OUT_LOOP:", ptx);
        Assert.Contains("fma.rn.f32 %f5, %f5, 0f3F000000, %f1", ptx);   // 0.5 x^2 + x
        Assert.Contains("rcp.approx.f32", ptx);
        Assert.DoesNotContain("ex2.approx.f32", ptx);                   // polynomial, no exp
        Assert.DoesNotContain("max.f32", ptx);                          // strictly positive -> no max shift
        Assert.Equal(3, Count(ptx, "bar.sync 0"));                      // one register/warp sum reduction
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxTaylorSoftmaxKernel.IsSupportedShape(128, 2048));
        Assert.False(PtxTaylorSoftmaxKernel.IsPromotedShape(128, 2048));
    }

    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(128, 512)]
    public void DriverOnlyTaylorSoftmax_MatchesOracle(int m, int n)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in Taylor-softmax specialization is measured on GA10x/SM86.");
        using var kernel = new PtxTaylorSoftmaxKernel(runtime, m, n);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(PtxRowReduce.Strategy, kernel.Blueprint.Semantics["reduction"]);

        var random = RandomHelper.CreateSeededRandom(20265300 + m + n);
        float[] xHost = Values(random, m * n, 2.0f);
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        {
            double sum = 0;
            for (int col = 0; col < n; col++)
            {
                double xv = xHost[row * n + col];
                sum += 1.0 + xv + 0.5 * xv * xv;
            }
            for (int col = 0; col < n; col++)
            {
                double xv = xHost[row * n + col];
                expected[row * n + col] = (float)((1.0 + xv + 0.5 * xv * xv) / sum);
            }
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        x.Upload<float>(xHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[m * n];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, $"taylor-softmax {m}x{n}");
    }

    [Fact]
    public void MaskedFillEmitter_IsElementwiseSelect()
    {
        string ptx = PtxMaskedFillKernel.EmitPtx(8, 6, 16384);
        Assert.Contains(PtxMaskedFillKernel.EntryPoint, ptx);
        Assert.Contains("ld.param.f32 %f8, [fill];", ptx);
        Assert.Equal(4, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Contains("setp.neu.f32 %p0, %f4, 0f00000000", ptx);      // mask.x != 0
        Assert.Contains("selp.f32 %f12, %f8, %f3, %p3", ptx);           // fill : input.w
        Assert.Equal(2, Count(ptx, "st.global.cg.v4.f32"));
        Assert.DoesNotContain("bra.uni MASKED_FILL_DONE", ptx, StringComparison.Ordinal);
        Assert.Contains("bra.uni MASKED_FILL_DONE",
            PtxMaskedFillKernel.EmitPtx(8, 6, 1280), StringComparison.Ordinal);
        Assert.Equal(0, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxMaskedFillKernel.IsSupportedCount(16384));
        Assert.False(PtxMaskedFillKernel.IsSupportedCount(100));        // not a multiple of 256
        Assert.False(PtxMaskedFillKernel.IsPromotedCount(16384));
    }

    [Fact]
    public void MaskedFillBackwardEmitter_GatesGradientAtMaskedPositions()
    {
        string ptx = PtxMaskedFillBackwardKernel.EmitPtx(8, 6, 16384);
        Assert.Contains(PtxMaskedFillBackwardKernel.EntryPoint, ptx);
        Assert.Equal(4, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Contains("setp.neu.f32 %p0, %f4, 0f00000000", ptx);
        Assert.Contains("selp.f32 %f12, 0f00000000, %f3, %p3", ptx);    // 0 : gradOutput.w
        Assert.Equal(2, Count(ptx, "st.global.wt.v4.f32"));
        Assert.DoesNotContain("bra.uni MASKED_FILL_BACKWARD_DONE", ptx, StringComparison.Ordinal);
        Assert.Contains("bra.uni MASKED_FILL_BACKWARD_DONE",
            PtxMaskedFillBackwardKernel.EmitPtx(8, 6, 1280), StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxMaskedFillBackwardKernel.IsSupportedCount(16384));
        Assert.False(PtxMaskedFillBackwardKernel.IsPromotedCount(16384));
    }

    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(128, 512)]
    public void DriverOnlyMaskedFillAndBackward_MatchOracle(int m, int n)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in masked-fill specialization is measured on GA10x/SM86.");
        const float fill = -1e9f;
        var random = RandomHelper.CreateSeededRandom(20265200 + m + n);
        float[] inHost = Values(random, m * n, 2.0f);
        float[] maskHost = new float[m * n];
        for (int i = 0; i < maskHost.Length; i++) maskHost[i] = random.NextDouble() < 0.3 ? 1f : 0f;

        var expFill = new float[m * n];
        var expBwd = new float[m * n];
        for (int i = 0; i < inHost.Length; i++)
        {
            bool masked = maskHost[i] != 0f;
            expFill[i] = masked ? fill : inHost[i];
            expBwd[i] = masked ? 0f : inHost[i];   // treat inHost as gradOutput for the backward
        }

        using var fwd = new PtxMaskedFillKernel(runtime, m * n, fill);
        using var bwd = new PtxMaskedFillBackwardKernel(runtime, m * n);
        Assert.Equal(0, fwd.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, bwd.Audit.Function.LocalBytesPerThread);

        using var inBuf = runtime.AllocateBytes((nuint)(inHost.Length * sizeof(float)));
        using var maskBuf = runtime.AllocateBytes((nuint)(maskHost.Length * sizeof(float)));
        using var outFill = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        using var outBwd = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        inBuf.Upload<float>(inHost);
        maskBuf.Upload<float>(maskHost);
        fwd.Launch(
            DirectPtxTensorView.CreateOwned(inBuf, fwd.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(maskBuf, fwd.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outFill, fwd.Blueprint.Tensors[2]));
        bwd.Launch(
            DirectPtxTensorView.CreateOwned(inBuf, bwd.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(maskBuf, bwd.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(outBwd, bwd.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actualFill = new float[m * n];
        var actualBwd = new float[m * n];
        outFill.Download<float>(actualFill);
        outBwd.Download<float>(actualBwd);
        AssertVectorClose(actualFill, expFill, 0f, $"masked-fill {m}x{n}");
        AssertVectorClose(actualBwd, expBwd, 0f, $"masked-fill-backward {m}x{n}");
    }

    [Fact]
    public void LogSumExpBackwardEmitter_ReusesSuppliedLogPartition()
    {
        string ptx = PtxLogSumExpBackwardKernel.EmitPtx(8, 6, 64, 512);
        Assert.Contains(PtxLogSumExpBackwardKernel.EntryPoint, ptx);
        Assert.Contains("OUT_LOOP:", ptx);
        Assert.Equal(1, Count(ptx, "ex2.approx.f32"));                 // one elementwise broadcast pass
        Assert.Contains("ld.param.u64 %rd1, [lse_ptr]", ptx);
        Assert.Contains("ld.global.nc.f32 %f0, [%rd8]", ptx);           // supplied lse[m]
        Assert.Contains("ld.global.nc.f32 %f1, [%rd9]", ptx);           // per-row dY[m]
        Assert.Contains("sub.rn.f32 %f2, %f2, %f0", ptx);               // x - supplied lse
        Assert.DoesNotContain("LOAD_LOOP:", ptx);
        Assert.DoesNotContain("SUM_LOOP:", ptx);
        Assert.DoesNotContain("rcp.approx.f32", ptx);
        Assert.DoesNotContain("bar.sync", ptx);
        Assert.DoesNotContain(".shared", ptx);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxLogSumExpBackwardKernel.IsSupportedShape(128, 2048));
        Assert.False(PtxLogSumExpBackwardKernel.IsPromotedShape(128, 2048));
    }

    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(128, 512)]
    public void DriverOnlyLogSumExpBackward_MatchesOracle(int m, int n)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in log-sum-exp-backward specialization is measured on GA10x/SM86.");
        using var kernel = new PtxLogSumExpBackwardKernel(runtime, m, n);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);

        var random = RandomHelper.CreateSeededRandom(20265100 + m + n);
        float[] xHost = Values(random, m * n, 3.0f);
        float[] dyHost = Values(random, m, 1.0f);
        var lseHost = new float[m];
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        {
            double max = double.NegativeInfinity;
            for (int col = 0; col < n; col++) max = Math.Max(max, xHost[row * n + col]);
            double sum = 0;
            for (int col = 0; col < n; col++) sum += Math.Exp(xHost[row * n + col] - max);
            lseHost[row] = (float)(max + Math.Log(sum));
            for (int col = 0; col < n; col++)
                expected[row * n + col] =
                    (float)(Math.Exp(xHost[row * n + col] - lseHost[row]) * dyHost[row]);
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var lse = runtime.AllocateBytes((nuint)(lseHost.Length * sizeof(float)));
        using var dy = runtime.AllocateBytes((nuint)(dyHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        x.Upload<float>(xHost);
        lse.Upload<float>(lseHost);
        dy.Upload<float>(dyHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(lse, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(dy, kernel.Blueprint.Tensors[2]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[3]));
        runtime.Synchronize();
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);
        var actual = new float[m * n];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, $"logsumexp-backward {m}x{n}");
    }

    [Fact]
    public void SoftmaxBackwardEmitter_IsExactJacobianVectorProduct()
    {
        string ptx = PtxSoftmaxBackwardKernel.EmitPtx(8, 6, 64, 512);
        Assert.Contains(PtxSoftmaxBackwardKernel.EntryPoint, ptx);
        Assert.Contains("LOAD_LOOP:", ptx);
        Assert.Contains("OUT_LOOP:", ptx);
        Assert.DoesNotContain("SUM_LOOP:", ptx);                       // dot folded into the load pass
        Assert.Contains("fma.rn.f32 %f0, %f2, %f1, %f0", ptx);         // dot += dY*S
        Assert.DoesNotContain("ex2.approx.f32", ptx);                  // exact identity, no transcendental
        Assert.DoesNotContain("lg2.approx.f32", ptx);
        Assert.Equal(3, Count(ptx, "bar.sync 0"));                     // one register/warp reduction
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxSoftmaxBackwardKernel.IsSupportedShape(128, 2048));
        Assert.False(PtxSoftmaxBackwardKernel.IsPromotedShape(128, 2048));
    }

    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(128, 512)]
    public void DriverOnlySoftmaxBackward_MatchesOracle(int m, int n)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in softmax-backward specialization is measured on GA10x/SM86.");
        using var kernel = new PtxSoftmaxBackwardKernel(runtime, m, n);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(PtxRowReduce.Strategy, kernel.Blueprint.Semantics["reduction"]);

        var random = RandomHelper.CreateSeededRandom(20265000 + m + n);
        // S is a valid softmax distribution per row; dY is arbitrary upstream gradient.
        float[] sHost = new float[m * n];
        float[] dyHost = Values(random, m * n, 0.5f);
        for (int row = 0; row < m; row++)
        {
            var logits = Values(random, n, 2.0f);
            double max = double.NegativeInfinity;
            for (int col = 0; col < n; col++) max = Math.Max(max, logits[col]);
            double sum = 0;
            for (int col = 0; col < n; col++) sum += Math.Exp(logits[col] - max);
            for (int col = 0; col < n; col++) sHost[row * n + col] = (float)(Math.Exp(logits[col] - max) / sum);
        }
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        {
            double dot = 0;
            for (int col = 0; col < n; col++) dot += (double)dyHost[row * n + col] * sHost[row * n + col];
            for (int col = 0; col < n; col++)
                expected[row * n + col] = (float)(sHost[row * n + col] * (dyHost[row * n + col] - dot));
        }

        using var s = runtime.AllocateBytes((nuint)(sHost.Length * sizeof(float)));
        using var dy = runtime.AllocateBytes((nuint)(dyHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        s.Upload<float>(sHost);
        dy.Upload<float>(dyHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(s, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(dy, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
        var actual = new float[m * n];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, $"softmax-backward {m}x{n}");
    }

    [Fact]
    public void LogSumExpEmitter_ReducesRowToSingleLogPartition()
    {
        string ptx = PtxLogSumExpKernel.EmitPtx(8, 6, 64, 512);
        Assert.Contains(PtxLogSumExpKernel.EntryPoint, ptx);
        Assert.Contains("LOAD_LOOP:", ptx);
        Assert.Contains("SUM_LOOP:", ptx);
        Assert.DoesNotContain("OUT_LOOP:", ptx);                       // single write, not a full row
        Assert.Contains("ex2.approx.f32", ptx);
        Assert.Contains("lg2.approx.f32", ptx);
        Assert.Contains("setp.ne.u32 %p2, %r0, 0", ptx);               // thread-0 guard
        Assert.Equal(1, Count(ptx, "st.global.f32"));                  // one value per row
        // Two hierarchical reductions minus the final post-load barrier.
        Assert.Equal(5, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxLogSumExpKernel.IsSupportedShape(128, 2048));
        Assert.False(PtxLogSumExpKernel.IsPromotedShape(128, 2048));
    }

    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(128, 1024)]
    public void DriverOnlyLogSumExp_MatchesOracle(int m, int n)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in log-sum-exp specialization is measured on GA10x/SM86.");
        using var kernel = new PtxLogSumExpKernel(runtime, m, n);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(PtxRowReduce.Strategy, kernel.Blueprint.Semantics["reduction"]);

        var random = RandomHelper.CreateSeededRandom(20264900 + m + n);
        float[] xHost = Values(random, m * n, 3.0f);
        var expected = new float[m];
        for (int row = 0; row < m; row++)
        {
            double max = double.NegativeInfinity;
            for (int col = 0; col < n; col++) max = Math.Max(max, xHost[row * n + col]);
            double sum = 0;
            for (int col = 0; col < n; col++) sum += Math.Exp(xHost[row * n + col] - max);
            expected[row] = (float)(max + Math.Log(sum));
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(m * sizeof(float)));
        x.Upload<float>(xHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[m];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 3e-3f, $"logsumexp {m}x{n}");
    }

    [Fact]
    public void LogSoftmaxEmitter_SubtractsTreeReducedLogPartition()
    {
        string ptx = PtxLogSoftmaxKernel.EmitPtx(8, 6, 64, 512);
        Assert.Contains(PtxLogSoftmaxKernel.EntryPoint, ptx);
        Assert.Contains("LOAD_LOOP:", ptx);
        Assert.Contains("SUM_LOOP:", ptx);
        Assert.Contains("OUT_LOOP:", ptx);
        Assert.Contains("ex2.approx.f32", ptx);                        // exp-sum
        Assert.Contains("lg2.approx.f32", ptx);                        // log-partition
        Assert.Contains("sub.rn.f32 %f1, %f1, %f4", ptx);              // x - logZ
        Assert.DoesNotContain("rcp.approx.f32", ptx);                  // no division
        Assert.Equal(6, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxLogSoftmaxKernel.IsSupportedShape(128, 2048));
        Assert.False(PtxLogSoftmaxKernel.IsPromotedShape(128, 2048));
    }

    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(128, 512)]
    public void DriverOnlyLogSoftmax_MatchesOracle(int m, int n)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in log-softmax specialization is measured on GA10x/SM86.");
        using var kernel = new PtxLogSoftmaxKernel(runtime, m, n);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(PtxRowReduce.Strategy, kernel.Blueprint.Semantics["reduction"]);

        var random = RandomHelper.CreateSeededRandom(20264800 + m + n);
        float[] xHost = Values(random, m * n, 3.0f);
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        {
            double max = double.NegativeInfinity;
            for (int col = 0; col < n; col++) max = Math.Max(max, xHost[row * n + col]);
            double sum = 0;
            for (int col = 0; col < n; col++) sum += Math.Exp(xHost[row * n + col] - max);
            double logZ = max + Math.Log(sum);
            for (int col = 0; col < n; col++)
                expected[row * n + col] = (float)(xHost[row * n + col] - logZ);
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        x.Upload<float>(xHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[m * n];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 3e-3f, $"log-softmax {m}x{n}");
    }

    [Fact]
    public void SoftmaxEmitter_IsSinglePassStableRowReduction()
    {
        string ptx = PtxSoftmaxKernel.EmitPtx(8, 6, 64, 512);
        Assert.Contains(PtxSoftmaxKernel.EntryPoint, ptx);
        Assert.DoesNotContain("row_sh", ptx);                          // L1 + final-output staging
        Assert.Contains(".shared .align 16 .b8 red[32]", ptx);
        Assert.DoesNotContain("_LOOP:", ptx);                          // baked N is fully unrolled
        Assert.Contains("max.f32 %f10", ptx);                          // warp-hierarchical max
        Assert.Equal(2, Count(ptx, "ex2.approx.f32"));                 // once per 256-column slice
        Assert.Contains("ld.global.ca.f32", ptx);                      // repeated row pass stays cacheable
        Assert.Equal(4, Count(ptx, "st.global.f32"));                  // stage + normalize each slice
        Assert.Contains("rcp.approx.f32", ptx);                        // 1/sumExp
        // Two reductions, each: two-level warp reduction + post-load barrier.
        Assert.Equal(6, Count(ptx, "bar.sync 0"));
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.True(PtxSoftmaxKernel.IsSupportedShape(128, 2048));
        Assert.False(PtxSoftmaxKernel.IsSupportedShape(63, 2048));
        Assert.False(PtxSoftmaxKernel.IsPromotedShape(128, 2048));
    }

    [SkippableTheory]
    [InlineData(64, 256)]
    [InlineData(128, 512)]
    [InlineData(64, 1024)]
    public void DriverOnlySoftmax_MatchesOracle(int m, int n)
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in softmax specialization is measured on GA10x/SM86.");
        using var kernel = new PtxSoftmaxKernel(runtime, m, n);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(PtxRowReduce.Strategy, kernel.Blueprint.Semantics["reduction"]);

        var random = RandomHelper.CreateSeededRandom(20264700 + m + n);
        float[] xHost = Values(random, m * n, 3.0f);   // wide range exercises stability
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        {
            double max = double.NegativeInfinity;
            for (int col = 0; col < n; col++) max = Math.Max(max, xHost[row * n + col]);
            double sum = 0;
            for (int col = 0; col < n; col++) sum += Math.Exp(xHost[row * n + col] - max);
            for (int col = 0; col < n; col++)
                expected[row * n + col] = (float)(Math.Exp(xHost[row * n + col] - max) / sum);
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        x.Upload<float>(xHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();
        var actual = new float[m * n];
        output.Download<float>(actual);
        AssertVectorClose(actual, expected, 2e-3f, $"softmax {m}x{n}");
    }

    [SkippableFact]
    public void Backend_Softmax_ThreeWayParityAndAudit()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxSoftmaxEnabled, "Requires a validated Ampere CUDA backend.");
            const int m = 64, n = 256;

            Assert.True(backend.PrewarmDirectPtxSoftmax(m, n), backend.DirectPtxLastError);
            Assert.True(backend.TryGetDirectPtxSoftmaxAudit(m, n, out DirectPtxKernelAudit audit));
            Assert.Equal(0, audit.Function.LocalBytesPerThread);
            Assert.Equal(64, audit.PtxSha256.Length);

            var random = RandomHelper.CreateSeededRandom(20265500);
            float[] xHost = Values(random, m * n, 3.0f);
            var expected = new float[m * n];
            for (int row = 0; row < m; row++)
            {
                double max = double.NegativeInfinity;
                for (int col = 0; col < n; col++) max = Math.Max(max, xHost[row * n + col]);
                double sum = 0;
                for (int col = 0; col < n; col++) sum += Math.Exp(xHost[row * n + col] - max);
                for (int col = 0; col < n; col++)
                    expected[row * n + col] = (float)(Math.Exp(xHost[row * n + col] - max) / sum);
            }

            using var inBuf = backend.AllocateBuffer(xHost);
            using var outBuf = backend.AllocateBuffer(m * n);

            long before = backend.DirectPtxSoftmaxDispatchCount;
            backend.Softmax(inBuf, outBuf, m, n);
            backend.Synchronize();
            Assert.Equal(before, backend.DirectPtxSoftmaxDispatchCount);
            float[] incumbent = backend.DownloadBuffer(outBuf);
            AssertVectorClose(incumbent, expected, 2e-3f, "incumbent CUDA softmax");

            DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
            backend.Softmax(inBuf, outBuf, m, n);   // public route flows through the fail-closed guard
            backend.Synchronize();
            Assert.True(backend.DirectPtxSoftmaxDispatchCount > before, backend.DirectPtxLastError);
            float[] directPtx = backend.DownloadBuffer(outBuf);
            AssertVectorClose(directPtx, expected, 2e-3f, "direct-PTX softmax");
            AssertVectorClose(directPtx, incumbent, 2e-3f, "direct-PTX vs incumbent softmax");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void Backend_SoftmaxFamily_PrewarmedKernelCapturesAndReplays()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxSoftmaxEnabled, "Requires a validated Ampere CUDA backend.");
            const int m = 64, n = 256;
            using var input = backend.AllocateBuffer(new float[m * n]);
            using var output = backend.AllocateBuffer(m);

            Assert.True(backend.TryDirectPtxLogSumExp(input, output, m, n), backend.DirectPtxLastError);
            backend.Synchronize();

            bool captureLaunch = false;
            IntPtr graph = backend.CaptureGraph(() =>
                captureLaunch = backend.TryDirectPtxLogSumExp(input, output, m, n));
            Assert.True(captureLaunch, backend.DirectPtxLastError);
            Assert.NotEqual(IntPtr.Zero, graph);
            try
            {
                backend.LaunchCapturedGraph(graph);
                float expected = MathF.Log(n);
                Assert.All(backend.DownloadBuffer(output), value =>
                    Assert.InRange(value, expected - 2e-3f, expected + 2e-3f));
            }
            finally
            {
                backend.DestroyCapturedGraph(graph);
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void Backend_SoftmaxVariants_ThreeWayParity()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxSoftmaxEnabled, "Requires a validated Ampere CUDA backend.");
            const int m = 64, n = 256;
            var random = RandomHelper.CreateSeededRandom(20265600);
            float[] xHost = Values(random, m * n, 2.0f);
            using var inBuf = backend.AllocateBuffer(xHost);
            using var outBuf = backend.AllocateBuffer(m * n);

            // Log-softmax: x - logsumexp.
            var logExpected = new float[m * n];
            for (int row = 0; row < m; row++)
            {
                double max = double.NegativeInfinity;
                for (int col = 0; col < n; col++) max = Math.Max(max, xHost[row * n + col]);
                double sum = 0;
                for (int col = 0; col < n; col++) sum += Math.Exp(xHost[row * n + col] - max);
                double logZ = max + Math.Log(sum);
                for (int col = 0; col < n; col++) logExpected[row * n + col] = (float)(xHost[row * n + col] - logZ);
            }
            long beforeLog = backend.DirectPtxLogSoftmaxDispatchCount;
            backend.LogSoftmax(inBuf, outBuf, m, n);
            backend.Synchronize();
            Assert.Equal(beforeLog, backend.DirectPtxLogSoftmaxDispatchCount);
            float[] incumbentLog = backend.DownloadBuffer(outBuf);
            AssertVectorClose(incumbentLog, logExpected, 3e-3f, "incumbent CUDA log-softmax");

            DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
            backend.LogSoftmax(inBuf, outBuf, m, n);
            backend.Synchronize();
            Assert.True(backend.DirectPtxLogSoftmaxDispatchCount > beforeLog, backend.DirectPtxLastError);
            Assert.True(backend.TryGetDirectPtxLogSoftmaxAudit(m, n, out DirectPtxKernelAudit logAudit));
            Assert.Equal(0, logAudit.Function.LocalBytesPerThread);
            float[] directPtxLog = backend.DownloadBuffer(outBuf);
            AssertVectorClose(directPtxLog, logExpected, 3e-3f, "direct-PTX log-softmax");
            AssertVectorClose(directPtxLog, incumbentLog, 3e-3f, "direct-PTX vs incumbent log-softmax");

            // Taylor softmax: (1+x+x^2/2) normalized.
            var taylorExpected = new float[m * n];
            for (int row = 0; row < m; row++)
            {
                double sum = 0;
                for (int col = 0; col < n; col++) { double v = xHost[row * n + col]; sum += 1.0 + v + 0.5 * v * v; }
                for (int col = 0; col < n; col++) { double v = xHost[row * n + col]; taylorExpected[row * n + col] = (float)((1.0 + v + 0.5 * v * v) / sum); }
            }
            DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
            long beforeTaylor = backend.DirectPtxTaylorSoftmaxDispatchCount;
            backend.TaylorSoftmax(inBuf, outBuf, m, n);
            backend.Synchronize();
            Assert.Equal(beforeTaylor, backend.DirectPtxTaylorSoftmaxDispatchCount);
            float[] incumbentTaylor = backend.DownloadBuffer(outBuf);
            AssertVectorClose(incumbentTaylor, taylorExpected, 2e-3f, "incumbent CUDA Taylor softmax");

            DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
            backend.TaylorSoftmax(inBuf, outBuf, m, n);
            backend.Synchronize();
            Assert.True(backend.DirectPtxTaylorSoftmaxDispatchCount > beforeTaylor, backend.DirectPtxLastError);
            float[] directPtxTaylor = backend.DownloadBuffer(outBuf);
            AssertVectorClose(directPtxTaylor, taylorExpected, 2e-3f, "direct-PTX Taylor softmax");
            AssertVectorClose(directPtxTaylor, incumbentTaylor, 2e-3f, "direct-PTX vs incumbent Taylor softmax");

            // Sparsemax dispatches and projects onto the simplex (rows sum to 1).
            using var sparseBuf = backend.AllocateBuffer(m * n);
            float[] sparseExpected = SparsemaxReference(xHost, m, n);
            DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
            long beforeSparse = backend.DirectPtxSparsemaxDispatchCount;
            backend.Sparsemax(inBuf, sparseBuf, m, n);
            backend.Synchronize();
            Assert.Equal(beforeSparse, backend.DirectPtxSparsemaxDispatchCount);
            float[] incumbentSparse = backend.DownloadBuffer(sparseBuf);
            AssertVectorClose(incumbentSparse, sparseExpected, 2e-3f, "incumbent CUDA sparsemax");

            DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
            backend.Sparsemax(inBuf, sparseBuf, m, n);
            backend.Synchronize();
            Assert.True(backend.DirectPtxSparsemaxDispatchCount > beforeSparse, backend.DirectPtxLastError);
            var sparse = backend.DownloadBuffer(sparseBuf);
            AssertVectorClose(sparse, sparseExpected, 2e-3f, "direct-PTX sparsemax");
            AssertVectorClose(sparse, incumbentSparse, 2e-3f, "direct-PTX vs incumbent sparsemax");
            for (int row = 0; row < m; row++)
            {
                double rowSum = 0;
                int exactZeros = 0;
                for (int col = 0; col < n; col++)
                {
                    int index = row * n + col;
                    Assert.True(sparse[index] >= 0f, "sparsemax output must be non-negative");
                    if (sparseExpected[index] == 0f)
                    {
                        Assert.Equal(0f, sparse[index]);
                        exactZeros++;
                    }
                    rowSum += sparse[index];
                }
                Assert.True(Math.Abs(rowSum - 1.0) < 3e-3, $"sparsemax row {row} sums to {rowSum}, expected 1");
                Assert.True(exactZeros > 0, $"sparsemax row {row} produced no exact zeros");
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void Backend_SoftmaxBackwardReductionAndMasking_ThreeWayParity()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        bool? previous = DirectPtxFeatureGate.TestOverride;
        bool previousExperiment = DirectPtxFeatureGate.SoftmaxExperimentOverride;
        DirectPtxFeatureGate.TestOverride = true;
        DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
        try
        {
            using var backend = new CudaBackend();
            Skip.IfNot(backend.IsDirectPtxSoftmaxEnabled, "Requires a validated Ampere CUDA backend.");
            const int m = 64, n = 256, size = m * n;
            var random = RandomHelper.CreateSeededRandom(20265700);

            // ---- SoftmaxBackward: dX = S*(dY - sum(dY*S)) ----
            float[] sHost = new float[size];
            float[] dyHost = Values(random, size, 0.5f);
            for (int row = 0; row < m; row++)
            {
                var logits = Values(random, n, 2.0f);
                double mx = double.NegativeInfinity;
                for (int c = 0; c < n; c++) mx = Math.Max(mx, logits[c]);
                double sm = 0;
                for (int c = 0; c < n; c++) sm += Math.Exp(logits[c] - mx);
                for (int c = 0; c < n; c++) sHost[row * n + c] = (float)(Math.Exp(logits[c] - mx) / sm);
            }
            var sbExpected = new float[size];
            for (int row = 0; row < m; row++)
            {
                double dot = 0;
                for (int c = 0; c < n; c++) dot += (double)dyHost[row * n + c] * sHost[row * n + c];
                for (int c = 0; c < n; c++) sbExpected[row * n + c] = (float)(sHost[row * n + c] * (dyHost[row * n + c] - dot));
            }
            using (var sBuf = backend.AllocateBuffer(sHost))
            using (var dyBuf = backend.AllocateBuffer(dyHost))
            using (var dxBuf = backend.AllocateBuffer(size))
            {
                long before = backend.DirectPtxSoftmaxBackwardDispatchCount;
                backend.SoftmaxBackward(dyBuf, sBuf, dxBuf, m, n);
                backend.Synchronize();
                Assert.Equal(before, backend.DirectPtxSoftmaxBackwardDispatchCount);
                float[] incumbent = backend.DownloadBuffer(dxBuf);
                AssertVectorClose(incumbent, sbExpected, 2e-3f, "incumbent CUDA softmax backward");

                DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
                backend.SoftmaxBackward(dyBuf, sBuf, dxBuf, m, n);
                backend.Synchronize();
                Assert.True(backend.DirectPtxSoftmaxBackwardDispatchCount > before, backend.DirectPtxLastError);
                float[] directPtx = backend.DownloadBuffer(dxBuf);
                AssertVectorClose(directPtx, sbExpected, 2e-3f, "direct-PTX softmax backward");
                AssertVectorClose(directPtx, incumbent, 2e-3f, "direct-PTX vs incumbent softmax backward");
            }

            // ---- LogSumExpAxis: [M,N] -> [M] ----
            float[] xHost = Values(random, size, 3.0f);
            var lseExpected = new float[m];
            for (int row = 0; row < m; row++)
            {
                double mx = double.NegativeInfinity;
                for (int c = 0; c < n; c++) mx = Math.Max(mx, xHost[row * n + c]);
                double sm = 0;
                for (int c = 0; c < n; c++) sm += Math.Exp(xHost[row * n + c] - mx);
                lseExpected[row] = (float)(mx + Math.Log(sm));
            }
            using (var xBuf = backend.AllocateBuffer(xHost))
            using (var lseBuf = backend.AllocateBuffer(m))
            {
                DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
                long before = backend.DirectPtxLogSumExpDispatchCount;
                backend.LogSumExpAxis(xBuf, lseBuf, m, n);
                backend.Synchronize();
                Assert.Equal(before, backend.DirectPtxLogSumExpDispatchCount);
                float[] incumbentLse = backend.DownloadBuffer(lseBuf);
                AssertVectorClose(incumbentLse, lseExpected, 3e-3f, "incumbent CUDA logsumexp");

                DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
                backend.LogSumExpAxis(xBuf, lseBuf, m, n);
                backend.Synchronize();
                Assert.True(backend.DirectPtxLogSumExpDispatchCount > before, backend.DirectPtxLastError);
                float[] directPtxLse = backend.DownloadBuffer(lseBuf);
                AssertVectorClose(directPtxLse, lseExpected, 3e-3f, "direct-PTX logsumexp");
                AssertVectorClose(directPtxLse, incumbentLse, 3e-3f, "direct-PTX vs incumbent logsumexp");

                // ---- LogSumExpBackward: dX = softmax(x) * dY[m] ----
                float[] dLseHost = Values(random, m, 1.0f);
                var lseBwdExpected = new float[size];
                for (int row = 0; row < m; row++)
                {
                    for (int c = 0; c < n; c++)
                        lseBwdExpected[row * n + c] =
                            (float)(Math.Exp(xHost[row * n + c] - lseExpected[row]) * dLseHost[row]);
                }
                using var dLseBuf = backend.AllocateBuffer(dLseHost);
                using var oracleLseBuf = backend.AllocateBuffer(lseExpected);
                using var dxBuf = backend.AllocateBuffer(size);
                DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
                long beforeB = backend.DirectPtxLogSumExpBackwardDispatchCount;
                backend.LogSumExpBackward(dLseBuf, xBuf, oracleLseBuf, dxBuf, m, n);
                backend.Synchronize();
                Assert.Equal(beforeB, backend.DirectPtxLogSumExpBackwardDispatchCount);
                float[] incumbentBwd = backend.DownloadBuffer(dxBuf);
                AssertVectorClose(incumbentBwd, lseBwdExpected, 2e-3f, "incumbent CUDA logsumexp backward");

                DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
                backend.LogSumExpBackward(dLseBuf, xBuf, oracleLseBuf, dxBuf, m, n);
                backend.Synchronize();
                Assert.True(backend.DirectPtxLogSumExpBackwardDispatchCount > beforeB, backend.DirectPtxLastError);
                float[] directPtxBwd = backend.DownloadBuffer(dxBuf);
                AssertVectorClose(directPtxBwd, lseBwdExpected, 2e-3f, "direct-PTX logsumexp backward");
                AssertVectorClose(directPtxBwd, incumbentBwd, 2e-3f, "direct-PTX vs incumbent logsumexp backward");

                // The API contract consumes the supplied lse; prove the PTX route does too.
                var corruptedLseHost = new float[m];
                var corruptedExpected = new float[size];
                for (int row = 0; row < m; row++)
                {
                    corruptedLseHost[row] = lseExpected[row] + 0.75f;
                    for (int col = 0; col < n; col++)
                        corruptedExpected[row * n + col] =
                            (float)(Math.Exp(xHost[row * n + col] - corruptedLseHost[row]) * dLseHost[row]);
                }
                using var corruptedLseBuf = backend.AllocateBuffer(corruptedLseHost);
                using var corruptedDxBuf = backend.AllocateBuffer(size);
                long beforeCorrupted = backend.DirectPtxLogSumExpBackwardDispatchCount;
                backend.LogSumExpBackward(
                    dLseBuf, xBuf, corruptedLseBuf, corruptedDxBuf, m, n);
                backend.Synchronize();
                Assert.True(
                    backend.DirectPtxLogSumExpBackwardDispatchCount > beforeCorrupted,
                    backend.DirectPtxLastError);
                AssertVectorClose(
                    backend.DownloadBuffer(corruptedDxBuf), corruptedExpected, 2e-3f,
                    "direct-PTX logsumexp-backward supplied-lse contract");
            }

            // ---- MaskedFill / MaskedFillBackward (flat size) ----
            float[] inFlat = Values(random, size, 2.0f);
            float[] maskFlat = new float[size];
            for (int i = 0; i < size; i++) maskFlat[i] = random.NextDouble() < 0.3 ? 1f : 0f;
            const float fill = -1e9f;
            using (var inBuf = backend.AllocateBuffer(inFlat))
            using (var maskBuf = backend.AllocateBuffer(maskFlat))
            using (var outFill = backend.AllocateBuffer(size))
            using (var outBwd = backend.AllocateBuffer(size))
            {
                DirectPtxFeatureGate.SoftmaxExperimentOverride = false;
                long beforeF = backend.DirectPtxMaskedFillDispatchCount;
                long beforeBwd = backend.DirectPtxMaskedFillBackwardDispatchCount;
                backend.MaskedFillKernel(inBuf, maskBuf, outFill, fill, size);
                backend.MaskedFillBackward(inBuf, maskBuf, outBwd, size);
                backend.Synchronize();
                Assert.Equal(beforeF, backend.DirectPtxMaskedFillDispatchCount);
                Assert.Equal(beforeBwd, backend.DirectPtxMaskedFillBackwardDispatchCount);
                float[] incumbentFill = backend.DownloadBuffer(outFill);
                float[] incumbentBwd = backend.DownloadBuffer(outBwd);
                AssertMaskedFillResults(inFlat, maskFlat, incumbentFill, incumbentBwd, fill,
                    "incumbent CUDA");

                DirectPtxFeatureGate.SoftmaxExperimentOverride = true;
                backend.MaskedFillKernel(inBuf, maskBuf, outFill, fill, size);
                backend.MaskedFillBackward(inBuf, maskBuf, outBwd, size);
                backend.Synchronize();
                Assert.True(backend.DirectPtxMaskedFillDispatchCount > beforeF, backend.DirectPtxLastError);
                Assert.True(backend.DirectPtxMaskedFillBackwardDispatchCount > beforeBwd, backend.DirectPtxLastError);
                float[] directPtxFill = backend.DownloadBuffer(outFill);
                float[] directPtxBwd = backend.DownloadBuffer(outBwd);
                AssertMaskedFillResults(inFlat, maskFlat, directPtxFill, directPtxBwd, fill,
                    "direct-PTX");
                AssertVectorClose(directPtxFill, incumbentFill, 0f,
                    "direct-PTX vs incumbent masked fill");
                AssertVectorClose(directPtxBwd, incumbentBwd, 0f,
                    "direct-PTX vs incumbent masked-fill backward");
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    private static void AssertMaskedFillResults(
        float[] input, float[] mask, float[] fillOutput, float[] backwardOutput,
        float fill, string implementation)
    {
        Assert.Equal(input.Length, fillOutput.Length);
        Assert.Equal(input.Length, backwardOutput.Length);
        for (int i = 0; i < input.Length; i++)
        {
            bool masked = mask[i] != 0f;
            float expectedFill = masked ? fill : input[i];
            float expectedBackward = masked ? 0f : input[i];
            Assert.True(fillOutput[i] == expectedFill,
                $"{implementation} masked-fill index {i}: expected {expectedFill}, got {fillOutput[i]}");
            Assert.True(backwardOutput[i] == expectedBackward,
                $"{implementation} masked-fill backward index {i}: expected {expectedBackward}, got {backwardOutput[i]}");
        }
    }

    private static float[] Values(Random random, int count, float magnitude)
    {
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)((random.NextDouble() * 2.0 - 1.0) * magnitude);
        return data;
    }

    private static float[] SparsemaxReference(float[] input, int rows, int cols)
    {
        var expected = new float[input.Length];
        for (int row = 0; row < rows; row++)
        {
            // Sorted closed form: tau = (sum_{i<=k} z_(i) - 1) / k.
            var sorted = new double[cols];
            for (int col = 0; col < cols; col++) sorted[col] = input[row * cols + col];
            Array.Sort(sorted);
            Array.Reverse(sorted);
            double cumulative = 0;
            double tau = 0;
            for (int index = 0; index < cols; index++)
            {
                cumulative += sorted[index];
                if (1.0 + (index + 1) * sorted[index] > cumulative)
                    tau = (cumulative - 1.0) / (index + 1);
            }
            for (int col = 0; col < cols; col++)
                expected[row * cols + col] =
                    (float)Math.Max(input[row * cols + col] - tau, 0.0);
        }
        return expected;
    }

    private static void AssertVectorClose(float[] actual, float[] expected, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= tolerance,
                $"{what}: index {i} expected {expected[i]} actual {actual[i]} (tol {tolerance}).");
    }

    private static int Count(string text, string value)
    {
        int count = 0, index = 0;
        while ((index = text.IndexOf(value, index, StringComparison.Ordinal)) >= 0)
        {
            count++;
            index += value.Length;
        }
        return count;
    }
}
