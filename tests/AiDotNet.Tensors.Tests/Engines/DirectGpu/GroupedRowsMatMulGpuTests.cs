#if NETCOREAPP
// GpuCpuCorrectnessFixture (GpuCpuCorrectnessTests.cs) is compiled only off .NET Framework.
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Ref = AiDotNet.Tensors.Tests.Engines.GroupedRowsMatMulTests;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The device jagged-rows grouped GEMM (IGroupedGemmBackend, implemented by every backend) and its backward against the
/// double-precision reference: an unbalanced routed load with empty groups and rows past the last offset, plain and
/// transposed b, with and without an [N] or [G, N] bias. No case may fall back to the host.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GroupedRowsMatMulGpuTests : IClassFixture<GpuCpuCorrectnessFixture>
{
    private readonly GpuCpuCorrectnessFixture _fixture;

    public GroupedRowsMatMulGpuTests(GpuCpuCorrectnessFixture fixture) => _fixture = fixture;

    private static void AssertNoHostFallback() =>
        Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, f => f.Contains("x GroupedRows"));

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Forward_MatchesReference(bool transposeB)
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        int k = 24, n = 20, g = Ref.UnbalancedEnds.Length;
        var a = Ref.Random(1, Ref.UnbalancedRows, k);
        var b = transposeB ? Ref.Random(2, g, n, k) : Ref.Random(2, g, k, n);
        var y = _fixture.Gpu.GroupedRowsMatMul(a, b, Ref.UnbalancedEnds, transposeB);
        Ref.AssertClose(Ref.Reference(a, b, Ref.UnbalancedEnds, transposeB), y, 1e-4, "y");
        AssertNoHostFallback();
    }

    [SkippableTheory]
    [InlineData(0, false)]
    [InlineData(0, true)]
    [InlineData(1, false)]
    [InlineData(2, true)]
    public void Gradients_MatchReference(int biasRank, bool transposeB)
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        var e = _fixture.Gpu;
        int k = 10, n = 9, g = Ref.UnbalancedEnds.Length;
        var a = Ref.Random(5, Ref.UnbalancedRows, k);
        var b = transposeB ? Ref.Random(6, g, n, k) : Ref.Random(6, g, k, n);
        var bias = biasRank == 0 ? null : biasRank == 1 ? Ref.Random(13, n) : Ref.Random(13, g, n);
        var w = Ref.Random(7, Ref.UnbalancedRows, n);
        Tensor<float> y, ga, gb;
        Tensor<float>? gbias = null;
        using (var tape = new GradientTape<float>())
        {
            y = e.GroupedRowsMatMul(a, b, Ref.UnbalancedEnds, transposeB, bias);
            var loss = e.ReduceSum(e.TensorMultiply(y, w), null);
            var grads = tape.ComputeGradients(loss, bias is null ? new[] { a, b } : new[] { a, b, bias });
            ga = grads[a]; gb = grads[b];
            if (bias is not null) gbias = grads[bias];
        }
        Ref.AssertClose(Ref.Reference(a, b, Ref.UnbalancedEnds, transposeB, bias), y, 1e-4, "y");
        var (da, db) = Ref.ReferenceGrads(a, b, w, Ref.UnbalancedEnds, transposeB);
        Ref.AssertClose(da, ga, 1e-4, "da");
        Ref.AssertClose(db, gb, 1e-4, "db");
        if (bias is not null) Ref.AssertClose(Ref.ReferenceBiasGrad(w, Ref.UnbalancedEnds, biasRank == 2), gbias!, 1e-4, "dbias");
        AssertNoHostFallback();
    }

    // No routed rows: the weight gradient is a zero [G, K, N] without uploading the empty inputs.
    [SkippableFact]
    public void WeightGrad_WithNoRows_IsZero()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        var w = _fixture.Gpu.GroupedRowsWeightGrad(new Tensor<float>(new[] { 0, 3 }), new Tensor<float>(new[] { 0, 2 }), new[] { 0 });
        Assert.Equal(new[] { 1, 3, 2 }, w.Shape.ToArray());
        Assert.All(w.ToArray(), x => Assert.Equal(0f, x));
    }

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void PublicGroupedMatMul_MatchesReference(int biasRank)
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        int k = 16, n = 12, g = Ref.UnbalancedEnds.Length;
        var a = Ref.Random(3, Ref.UnbalancedRows, k); var b = Ref.Random(4, g, k, n);
        var bias = biasRank == 0 ? null : biasRank == 1 ? Ref.Random(12, n) : Ref.Random(12, g, n);
        var y = _fixture.Gpu.TensorGroupedMatMul(a, b, new Tensor<int>(Ref.UnbalancedEnds, new[] { g }), bias);
        Ref.AssertClose(Ref.Reference(a, b, Ref.UnbalancedEnds, false, bias), y, 1e-4, "y");
        AssertNoHostFallback();
    }
}
#endif
