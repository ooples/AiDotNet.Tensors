using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// TensorBroadcastSubtract on <c>double</c> returns only FLOAT-precision results.
/// </summary>
/// <remarks>
/// <para>
/// Found via the gradcheck sweep: the analytical gradient is exactly 1 (correct — the derivative of
/// a - b w.r.t. a is unambiguously 1), while central finite differences gave 1.0133. The gradient
/// was right and the FORWARD was wrong, the same shape of defect as the FastLogDouble256 bug.
/// </para>
/// <para>
/// Measured: for a[0]-b[0] the engine returns -0.2453465461730957 where the exact double result is
/// -0.24534654647360865 — agreement to only ~8 significant digits, i.e. float32. That ~1e-8 rounding
/// per element, differenced with eps 1e-6, is what produces the ~1.3% gradient error.
/// </para>
/// <para>
/// The float fast paths in <c>TensorBroadcastSubtract</c> and <c>ApplyBroadcastChannelOp</c> are both
/// correctly gated on <c>typeof(T) == typeof(float)</c>, so the downcast is somewhere else on the
/// double path and is NOT yet located. This test pins the contract; it fails until that is fixed.
/// </para>
/// </remarks>
public class BroadcastSubtractDiag
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();
    public BroadcastSubtractDiag(ITestOutputHelper o) => _out = o;

    [Fact]
    public void TensorBroadcastSubtract_Double_KeepsDoublePrecision()
    {
        var rng = new Random(1234);
        var a = new Tensor<double>([6]);
        var b = new Tensor<double>([6]);
        for (int i = 0; i < 6; i++) a[i] = 0.35 + rng.NextDouble() * 0.6;
        for (int i = 0; i < 6; i++) b[i] = 0.35 + rng.NextDouble() * 0.6;

        var r = _engine.TensorBroadcastSubtract(a, b);

        double worst = 0;
        for (int i = 0; i < r.Length; i++)
        {
            double exact = a[i] - b[i];
            double rel = Math.Abs(r[i] - exact) / Math.Max(1e-300, Math.Abs(exact));
            worst = Math.Max(worst, rel);
            _out.WriteLine($"r[{i}]={r[i]:G17}  exact={exact:G17}  rel={rel:E3}");
        }

        Assert.True(worst < 1e-14,
            $"TensorBroadcastSubtract<double> worst relative error {worst:E3} — subtraction is exact in " +
            "floating point, so any error above rounding means the double path is computing at float " +
            "precision.");
    }
}
