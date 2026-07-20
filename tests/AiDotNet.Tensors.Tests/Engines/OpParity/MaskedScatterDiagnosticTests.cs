using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// DIAGNOSTIC ONLY — inspects the recorded tape entry and the produced gradient for MaskedScatter /
/// IndexFill / IndexCopy on each engine. Both GPU overrides bail on IsTapeActive and run the CpuEngine base,
/// and the recorded tapes are byte-identical, so identical code should produce identical gradients. It does
/// not, and every hypothesis so far has been refuted by inspection — so this prints the intermediates.
/// </summary>
[Collection("OpParity")]
public class MaskedScatterDiagnosticTests
{
    private readonly OpParityFixture _fx;
    private readonly ITestOutputHelper _out;
    public MaskedScatterDiagnosticTests(OpParityFixture fx, ITestOutputHelper output)
    { _fx = fx; _out = output; }

    private static string Fmt(Tensor<float> t, int n = 10)
    {
        var parts = new List<string>();
        for (int i = 0; i < Math.Min(n, t.Length); i++) parts.Add(t[i].ToString("G6"));
        return string.Join(", ", parts) + (t.Length > n ? ", …" : "");
    }

    private void Probe(string label, IEngine engine, Func<IEngine, Tensor<float>> run, Tensor<float> leafToWatch)
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            using var tape = new GradientTape<float>();
            var output = run(engine);

            var weight = new Tensor<float>(output.Shape.ToArray());
            for (int i = 0; i < weight.Length; i++) weight[i] = 0.19f + 0.013f * (i % 13);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, weight), null);

            _out.WriteLine($"  [{label}] entries={tape.EntryCount}");
            for (int i = 0; i < tape.EntryCount; i++)
            {
                ref var e = ref tape.Entries[i];
                if (e.SavedState is not null)
                {
                    foreach (var s in e.SavedState)
                    {
                        if (s is Tensor<Bit> bt)
                        {
                            int trues = 0;
                            for (int k = 0; k < bt.Length; k++) if ((bool)bt[k]) trues++;
                            _out.WriteLine($"    #{i} {e.OperationName}: mask len={bt.Length} trueCount={trues}");
                        }
                        else if (s is Tensor<int> it)
                        {
                            var vals = new List<string>();
                            for (int k = 0; k < Math.Min(8, it.Length); k++) vals.Add(it[k].ToString());
                            _out.WriteLine($"    #{i} {e.OperationName}: idx len={it.Length} [{string.Join(",", vals)}]");
                        }
                    }
                }
            }

            _out.WriteLine($"    output      : {Fmt(output)}");
            var grads = tape.ComputeGradients(loss, new[] { leafToWatch });
            if (grads.TryGetValue(leafToWatch, out var g) && g is not null)
                _out.WriteLine($"    d/d(watched): {Fmt(g)}");
            else
                _out.WriteLine($"    d/d(watched): <ABSENT>");
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [SkippableFact]
    public void MaskedScatter()
    {
        Skip.If(!_fx.GpuReady, "No DirectGpu backend available.");
        var gpu = _fx.Gpu;
        Skip.If(gpu is null, "GPU engine unavailable.");

        var bits = new Bit[24];
        for (int i = 0; i < bits.Length; i++) bits[i] = (Bit)(i % 2 == 1);

        _out.WriteLine("### TensorMaskedScatter[4,6;checker]");
        foreach (var (label, eng) in new[] { ("CPU", _fx.Cpu), ("GPU", (IEngine)gpu!) })
        {
            var dest = OpParityRegistryProbe.Rand(1900, 24, new[] { 4, 6 });
            var src = OpParityRegistryProbe.Rand(1901, 12, new[] { 12 });
            Probe(label, eng,
                e => e.TensorMaskedScatter(dest, new Tensor<Bit>((Bit[])bits.Clone(), new[] { 4, 6 }), src),
                dest);
        }
    }
}

/// <summary>Deterministic tensor builder mirroring OpInput.Rand, without needing registry internals.</summary>
internal static class OpParityRegistryProbe
{
    public static Tensor<float> Rand(int seed, int count, int[] shape)
    {
        var rng = new Random(seed);
        var d = new float[count];
        for (int i = 0; i < count; i++) d[i] = (float)(-1.0 + rng.NextDouble() * 2.0);
        return new Tensor<float>(d, shape);
    }
}
