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
/// The device jagged-rows grouped GEMM (per-group GEMMs on buffer views) and its backward against the double-precision
/// reference: an unbalanced routed load with empty groups and rows past the last offset, plain and transposed b.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GroupedRowsMatMulGpuTests : IClassFixture<GpuCpuCorrectnessFixture>
{
    private readonly GpuCpuCorrectnessFixture _fixture;

    public GroupedRowsMatMulGpuTests(GpuCpuCorrectnessFixture fixture) => _fixture = fixture;

    // Where views are free at any offset (CUDA, HIP) the device path must run without a host fallback.
    private void AssertNoHostFallback()
    {
        if (!_fixture.Gpu.UsesNativeGroupedRows) return;
        Assert.DoesNotContain(GpuLaunchProbe.Fallbacks, f => f.Contains("x GroupedRows"));
    }

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
    [InlineData(false)]
    [InlineData(true)]
    public void Gradients_MatchReference(bool transposeB)
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        var e = _fixture.Gpu;
        int k = 10, n = 9, g = Ref.UnbalancedEnds.Length;
        var a = Ref.Random(5, Ref.UnbalancedRows, k);
        var b = transposeB ? Ref.Random(6, g, n, k) : Ref.Random(6, g, k, n);
        var w = Ref.Random(7, Ref.UnbalancedRows, n);
        Tensor<float> ga, gb;
        using (var tape = new GradientTape<float>())
        {
            var loss = e.ReduceSum(e.TensorMultiply(e.GroupedRowsMatMul(a, b, Ref.UnbalancedEnds, transposeB), w), null);
            var grads = tape.ComputeGradients(loss, new[] { a, b });
            ga = grads[a]; gb = grads[b];
        }
        var (da, db) = Ref.ReferenceGrads(a, b, w, Ref.UnbalancedEnds, transposeB);
        Ref.AssertClose(da, ga, 1e-4, "da");
        Ref.AssertClose(db, gb, 1e-4, "db");
        AssertNoHostFallback();
    }

    [SkippableFact]
    public void PublicGroupedMatMul_MatchesReference()
    {
        Skip.IfNot(_fixture.IsGpuReady, "No DirectGpu backend available on this system.");
        int k = 16, n = 12, g = Ref.UnbalancedEnds.Length;
        var a = Ref.Random(3, Ref.UnbalancedRows, k); var b = Ref.Random(4, g, k, n);
        var y = _fixture.Gpu.TensorGroupedMatMul(a, b, new Tensor<int>(Ref.UnbalancedEnds, new[] { g }));
        Ref.AssertClose(Ref.Reference(a, b, Ref.UnbalancedEnds, false), y, 1e-4, "y");
    }
}
#endif
