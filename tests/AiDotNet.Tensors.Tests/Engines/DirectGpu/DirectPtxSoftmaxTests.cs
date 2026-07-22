using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
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
        Assert.Throws<System.Collections.Generic.KeyNotFoundException>(() =>
            DirectPtxSoftmaxCoverageManifest.Get("UnassignedSoftmaxApi"));
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
