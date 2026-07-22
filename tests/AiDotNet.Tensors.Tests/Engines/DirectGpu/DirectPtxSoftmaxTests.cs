using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>Tests for the issue #840 softmax-family direct-PTX kernels.</summary>
public class DirectPtxSoftmaxTests
{
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
        Assert.Equal(20, Count(ptx, "bar.sync 0"));                     // max reduction + bisection-body reduction
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

        var random = RandomHelper.CreateSeededRandom(20265400 + m + n);
        float[] zHost = Values(random, m * n, 3.0f);
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        {
            // Reference sorted closed form: tau = (sum_{i<=k} z_(i) - 1) / k.
            var sorted = new double[n];
            for (int col = 0; col < n; col++) sorted[col] = zHost[row * n + col];
            Array.Sort(sorted);
            Array.Reverse(sorted);
            double cum = 0, tau = 0;
            int k = 0;
            for (int j = 0; j < n; j++)
            {
                cum += sorted[j];
                if (1.0 + (j + 1) * sorted[j] > cum) { k = j + 1; tau = (cum - 1.0) / (j + 1); }
            }
            for (int col = 0; col < n; col++)
                expected[row * n + col] = (float)Math.Max(zHost[row * n + col] - tau, 0.0);
        }

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
        Assert.Equal(10, Count(ptx, "bar.sync 0"));                     // single sum reduction
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
        Assert.Contains("ld.param.f32 %f2, [fill];", ptx);
        Assert.Contains("setp.neu.f32 %p0, %f1, 0f00000000", ptx);      // mask != 0
        Assert.Contains("selp.f32 %f3, %f2, %f0, %p0", ptx);            // fill : input
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
        Assert.Contains("setp.neu.f32 %p0, %f1, 0f00000000", ptx);
        Assert.Contains("selp.f32 %f3, 0f00000000, %f0, %p0", ptx);     // 0 : gradOutput
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
    public void LogSumExpBackwardEmitter_BroadcastsPerRowScalarGradient()
    {
        string ptx = PtxLogSumExpBackwardKernel.EmitPtx(8, 6, 64, 512);
        Assert.Contains(PtxLogSumExpBackwardKernel.EntryPoint, ptx);
        Assert.Contains("LOAD_LOOP:", ptx);
        Assert.Contains("SUM_LOOP:", ptx);
        Assert.Contains("OUT_LOOP:", ptx);
        Assert.Contains("ex2.approx.f32", ptx);
        Assert.Contains("rcp.approx.f32", ptx);
        Assert.Contains("ld.global.nc.f32 %f5, [%rd15]", ptx);          // per-row scalar dY[m]
        Assert.Contains("mul.rn.f32 %f4, %f4, %f5", ptx);               // fold into inv
        Assert.Equal(20, Count(ptx, "bar.sync 0"));
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
        var expected = new float[m * n];
        for (int row = 0; row < m; row++)
        {
            double max = double.NegativeInfinity;
            for (int col = 0; col < n; col++) max = Math.Max(max, xHost[row * n + col]);
            double sum = 0;
            for (int col = 0; col < n; col++) sum += Math.Exp(xHost[row * n + col] - max);
            for (int col = 0; col < n; col++)
                expected[row * n + col] = (float)(Math.Exp(xHost[row * n + col] - max) / sum * dyHost[row]);
        }

        using var x = runtime.AllocateBytes((nuint)(xHost.Length * sizeof(float)));
        using var dy = runtime.AllocateBytes((nuint)(dyHost.Length * sizeof(float)));
        using var output = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        x.Upload<float>(xHost);
        dy.Upload<float>(dyHost);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(x, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(dy, kernel.Blueprint.Tensors[1]),
            DirectPtxTensorView.CreateOwned(output, kernel.Blueprint.Tensors[2]));
        runtime.Synchronize();
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
        Assert.Equal(10, Count(ptx, "bar.sync 0"));                    // one reduction
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
        // Two reductions minus the final post-load barrier (no pass reads shared after sumExp).
        Assert.Equal(19, Count(ptx, "bar.sync 0"));
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
        Assert.Equal(20, Count(ptx, "bar.sync 0"));
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
        Assert.Contains(".shared .align 16 .b8 row_sh[2048]", ptx);   // N=512 row cache
        Assert.Contains(".shared .align 16 .b8 red[1024]", ptx);
        Assert.Contains("LOAD_LOOP:", ptx);
        Assert.Contains("SUM_LOOP:", ptx);
        Assert.Contains("OUT_LOOP:", ptx);
        Assert.Contains("max.f32 %f10", ptx);                          // tree-reduced max
        Assert.Contains("ex2.approx.f32", ptx);
        Assert.Contains("rcp.approx.f32", ptx);                        // 1/sumExp
        // Two reductions, each: store-barrier + 8 tree-halving barriers + post-load barrier.
        Assert.Equal(20, Count(ptx, "bar.sync 0"));
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
    public void Backend_DirectPtxSoftmax_PrewarmsDispatchesAndAudits()
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
            backend.Softmax(inBuf, outBuf, m, n);   // public route flows through the fail-closed guard
            backend.Synchronize();
            Assert.True(backend.DirectPtxSoftmaxDispatchCount > before, backend.DirectPtxLastError);
            AssertVectorClose(backend.DownloadBuffer(outBuf), expected, 2e-3f, "backend softmax via direct-ptx");
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    [SkippableFact]
    public void Backend_DirectPtxSoftmaxFamily_RoutesDispatchThroughPublicMethods()
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
            var random = RandomHelper.CreateSeededRandom(20265600);
            float[] xHost = Values(random, m * n, 2.0f);
            using var inBuf = backend.AllocateBuffer(xHost);
            using var outBuf = backend.AllocateBuffer(m * n);

            // Log-softmax: x - logsumexp.
            long beforeLog = backend.DirectPtxLogSoftmaxDispatchCount;
            backend.LogSoftmax(inBuf, outBuf, m, n);
            backend.Synchronize();
            Assert.True(backend.DirectPtxLogSoftmaxDispatchCount > beforeLog, backend.DirectPtxLastError);
            Assert.True(backend.TryGetDirectPtxLogSoftmaxAudit(m, n, out DirectPtxKernelAudit logAudit));
            Assert.Equal(0, logAudit.Function.LocalBytesPerThread);
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
            AssertVectorClose(backend.DownloadBuffer(outBuf), logExpected, 3e-3f, "backend log-softmax route");

            // Taylor softmax: (1+x+x^2/2) normalized.
            long beforeTaylor = backend.DirectPtxTaylorSoftmaxDispatchCount;
            backend.TaylorSoftmax(inBuf, outBuf, m, n);
            backend.Synchronize();
            Assert.True(backend.DirectPtxTaylorSoftmaxDispatchCount > beforeTaylor, backend.DirectPtxLastError);
            var taylorExpected = new float[m * n];
            for (int row = 0; row < m; row++)
            {
                double sum = 0;
                for (int col = 0; col < n; col++) { double v = xHost[row * n + col]; sum += 1.0 + v + 0.5 * v * v; }
                for (int col = 0; col < n; col++) { double v = xHost[row * n + col]; taylorExpected[row * n + col] = (float)((1.0 + v + 0.5 * v * v) / sum); }
            }
            AssertVectorClose(backend.DownloadBuffer(outBuf), taylorExpected, 2e-3f, "backend taylor-softmax route");

            // Sparsemax dispatches and projects onto the simplex (rows sum to 1).
            long beforeSparse = backend.DirectPtxSparsemaxDispatchCount;
            backend.Sparsemax(inBuf, outBuf, m, n);
            backend.Synchronize();
            Assert.True(backend.DirectPtxSparsemaxDispatchCount > beforeSparse, backend.DirectPtxLastError);
            var sparse = backend.DownloadBuffer(outBuf);
            for (int row = 0; row < m; row++)
            {
                double rowSum = 0;
                for (int col = 0; col < n; col++)
                {
                    Assert.True(sparse[row * n + col] >= -1e-4f, "sparsemax output must be non-negative");
                    rowSum += sparse[row * n + col];
                }
                Assert.True(Math.Abs(rowSum - 1.0) < 3e-3, $"sparsemax row {row} sums to {rowSum}, expected 1");
            }
        }
        finally
        {
            DirectPtxFeatureGate.TestOverride = previous;
            DirectPtxFeatureGate.SoftmaxExperimentOverride = previousExperiment;
        }
    }

    private static float[] Values(Random random, int count, float magnitude)
    {
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)((random.NextDouble() * 2.0 - 1.0) * magnitude);
        return data;
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
