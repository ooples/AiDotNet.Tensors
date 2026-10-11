using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1150: a full reduction, or a few long rows, is split across blocks on CUDA (sum_axis_rows_split) and then reduced
/// again. Values are small integers, so every float partial sum is exact (below 2^24) and the sums must equal the
/// reference exactly -- a dropped or double-counted element shows. Repeated runs are bit-identical.
/// </summary>
[Collection("VulkanGlobalState")]
public class SplitRowSumTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public SplitRowSumTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private IEngine CudaOrSkip()
    {
        Skip.If(!_fixture.IsAvailable, "No DirectGpu backend available.");
        DirectGpuTensorEngine gpu = _fixture.Engine!;
        Skip.IfNot(gpu.GetBackend() is Tensors.Engines.DirectGpu.CUDA.CudaBackend, "The split row sum is CUDA's.");
        return gpu;
    }

    private static Tensor<float> Values(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        int n = shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<float>(Enumerable.Range(0, n).Select(_ => (float)rng.Next(-7, 9)).ToArray(), shape);
    }

    private static void AssertExact(double expected, float got, string what)
        => Assert.True(expected == got, $"{what}: expected {expected}, got {got}");

    [SkippableTheory]
    [InlineData(1_000_003)]   // not a multiple of any chunk length
    [InlineData(65_536)]      // exactly the split threshold
    public void FullReduction_MatchesDoubleReference(int n)
    {
        var gpu = CudaOrSkip();
        var x = Values(n, n);
        double expected = x.ToArray().Sum(v => (double)v);
        GpuLaunchProbe.Reset();
        var cuda = (Tensors.Engines.DirectGpu.CUDA.CudaBackend)((DirectGpuTensorEngine)gpu).GetBackend()!;
        long splits = cuda.SplitRowSumLaunches;
        AssertExact(expected, gpu.ReduceSum(x, null).ToArray()[0], $"sum of {n}");
        Assert.True(cuda.SplitRowSumLaunches > splits, "the full sum did not take the split-row path");
        Assert.Empty(GpuLaunchProbe.Fallbacks);
        float mean = gpu.ReduceMean(x, new[] { 0 }, keepDims: false).ToArray()[0];
        Assert.True(Math.Abs(expected / n - mean) <= 1e-6 * Math.Max(1, Math.Abs(expected / n)), $"mean of {n}: expected {expected / n}, got {mean}");
    }

    [SkippableFact]
    public void FewLongRows_MatchDoubleReference()
    {
        var gpu = CudaOrSkip();
        const int rows = 3, cols = 200_001;
        var x = Values(7, rows, cols);
        var got = gpu.ReduceSum(x, new[] { 1 }).ToArray();
        var data = x.ToArray();
        for (int r = 0; r < rows; r++)
            AssertExact(Enumerable.Range(0, cols).Sum(c => (double)data[r * cols + c]), got[r], $"row {r}");
    }

    [SkippableFact]
    public void SplitSum_IsDeterministic()
    {
        var gpu = CudaOrSkip();
        var x = Values(9, 2_500_000);
        float first = gpu.ReduceSum(x, null).ToArray()[0];
        for (int i = 0; i < 3; i++)
            Assert.Equal(BitConverter.ToInt32(BitConverter.GetBytes(first), 0),
                BitConverter.ToInt32(BitConverter.GetBytes(gpu.ReduceSum(x, null).ToArray()[0]), 0));
    }
}
