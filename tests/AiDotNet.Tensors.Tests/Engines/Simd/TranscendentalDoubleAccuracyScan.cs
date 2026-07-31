using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Measures how far each double-precision transcendental engine op is from System.Math, to
/// establish the blast radius of the FastLogDouble256 accuracy defect across its siblings.
/// </summary>
/// <remarks>
/// Reports rather than asserting per-op, so one bad kernel does not mask the others. Length 64
/// exercises the widest vectorized path.
/// </remarks>
public class TranscendentalDoubleAccuracyScan
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public TranscendentalDoubleAccuracyScan(ITestOutputHelper output) => _out = output;

    [Fact]
    public void ScanDoubleTranscendentals()
    {
        const int n = 64;
        var rng = new Random(20260730);

        // Domain-safe inputs: positive and modest, valid for log/sqrt and non-saturating for the
        // sigmoidal ops.
        var x = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) x[i] = 0.05 + rng.NextDouble() * 3.0;

        var cases = new (string Name, Func<Tensor<double>, Tensor<double>> Op, Func<double, double> Exact)[]
        {
            ("TensorLog",  t => _engine.TensorLog(t),  Math.Log),
            ("TensorExp",  t => _engine.TensorExp(t),  Math.Exp),
            ("TensorSqrt", t => _engine.TensorSqrt(t), Math.Sqrt),
            ("TensorSin",  t => _engine.TensorSin(t),  Math.Sin),
            ("TensorCos",  t => _engine.TensorCos(t),  Math.Cos),
            ("TensorSinh", t => _engine.TensorSinh(t), Math.Sinh),
            ("TensorCosh", t => _engine.TensorCosh(t), Math.Cosh),
            ("Tanh",       t => _engine.Tanh(t),       Math.Tanh),
            ("Sigmoid",    t => _engine.Sigmoid(t),    v => 1.0 / (1.0 + Math.Exp(-v))),
        };

        var rows = new List<(string Name, double Worst)>();

        foreach (var (name, op, exact) in cases)
        {
            double worst = 0;
            try
            {
                var got = op(x);
                for (int i = 0; i < n; i++)
                {
                    double e = exact(x[i]);
                    double rel = Math.Abs(got[i] - e) / Math.Max(1e-300, Math.Abs(e));
                    if (rel > worst) worst = rel;
                }
            }
            catch (Exception ex)
            {
                _out.WriteLine($"{name,-12} threw {ex.GetType().Name}");
                continue;
            }

            rows.Add((name, worst));
        }

        _out.WriteLine("double-precision accuracy vs System.Math (length 64, worst relative error):");
        foreach (var (name, worst) in rows.OrderByDescending(r => r.Worst))
            _out.WriteLine($"  {name,-12} {worst:E3}   {(worst > 1e-12 ? "<-- FAILS double precision" : "ok")}");

        var bad = rows.Where(r => r.Worst > 1e-12).ToList();
        Assert.True(bad.Count == 0,
            $"{bad.Count} of {rows.Count} double transcendentals exceed 1e-12 relative error against " +
            $"System.Math:\n" +
            string.Join("\n", bad.OrderByDescending(b => b.Worst)
                                 .Select(b => $"  {b.Name}: {b.Worst:E3}")));
    }
}
