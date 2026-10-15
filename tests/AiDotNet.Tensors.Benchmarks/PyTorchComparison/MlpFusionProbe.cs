using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks.PyTorchComparison;

/// <summary>
/// Issue #436 dev probe: where the AIsEval MLP (128 x 784-512-128-10, ReLU) spends its time, and what a row-striped
/// whole-MLP dispatch would buy. Configurations are interleaved round by round (each round times every configuration
/// once), so drift on the shared machine hits them all alike. Run: <c>--aiseval-mlp-probe</c>.
/// </summary>
internal static class MlpFusionProbe
{
    private const int M = 128;
    private static readonly int[] Dims = { 784, 512, 128, 10 };

    public static void Run()
    {
        var rng = new Random(436);
        float[] Rand(int n, float s) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1) * s).ToArray();
        var x = new Tensor<float>(Rand(M * Dims[0], 1f), new[] { M, Dims[0] });
        var w = new List<Tensor<float>>();
        var b = new List<Tensor<float>?>();
        for (int l = 0; l < 3; l++)
        {
            w.Add(new Tensor<float>(Rand(Dims[l] * Dims[l + 1], 1f / (float)Math.Sqrt(Dims[l])), new[] { Dims[l], Dims[l + 1] }));
            b.Add(new Tensor<float>(Rand(Dims[l + 1], 0.1f), new[] { Dims[l + 1] }));
        }
        var engine = new CpuEngine();
        float[] xa = x.ToArray();
        float[][] wa = w.Select(t => t.ToArray()).ToArray();
        float[][] ba = b.Select(t => t!.ToArray()).ToArray();
        bool savedStripes = CpuEngine.MlpRowStripesEnabled;
        CpuEngine.MlpRowStripesEnabled = false;
        float[] reference = engine.MlpForward(x, w, b, FusedActivationType.ReLU, FusedActivationType.None).ToArray();
        CpuEngine.MlpRowStripesEnabled = true;
        float[] striped = engine.MlpForward(x, w, b, FusedActivationType.ReLU, FusedActivationType.None).ToArray();
        CpuEngine.MlpRowStripesEnabled = savedStripes;
        {
            double err = 0, big = 0;
            for (int i = 0; i < reference.Length; i++) { err = Math.Max(err, Math.Abs(striped[i] - reference[i])); big = Math.Max(big, Math.Abs(reference[i])); }
            Console.WriteLine($"row stripes vs per-layer MlpForward: max relative difference {err / big:G3}");
        }

        var configs = new List<(string Name, Action Body)>();
        int saved = CpuParallelSettings.MaxDegreeOfParallelism;
        foreach (int dop in new[] { 16, 32, 64, Environment.ProcessorCount })
            foreach (bool stripesOn in new[] { false, true })
            {
                int d = dop;
                bool on = stripesOn;
                configs.Add(($"MlpForward {(on ? "row stripes" : "per-layer")} dop={d}", () =>
                {
                    CpuParallelSettings.MaxDegreeOfParallelism = d;
                    CpuEngine.MlpRowStripesEnabled = on;
                    try { engine.MlpForward(x, w, b, FusedActivationType.ReLU, FusedActivationType.None).Dispose(); }
                    finally
                    {
                        CpuParallelSettings.MaxDegreeOfParallelism = saved;
                        CpuEngine.MlpRowStripesEnabled = savedStripes;
                    }
                }));
            }

        // Row stripes: one dispatch, each participant runs its rows through all three layers (no inter-layer barrier).
        foreach (int stripes in new[] { 8, 16, 32 })
        {
            int s = stripes;
            int rows = (M + s - 1) / s;
            var outBuf = new float[M * Dims[3]];
            configs.Add(($"row-stripe x{s} (SgemmSequential)", () =>
                PersistentParallelExecutor.Instance.Execute(s, s, chunk =>
                {
                    int r0 = chunk * rows, r = Math.Min(rows, M - r0);
                    if (r <= 0) return;
                    float[] h1 = new float[r * Dims[1]], h2 = new float[r * Dims[2]];
                    SimdGemm.SgemmSequential(xa.AsSpan(r0 * Dims[0], r * Dims[0]), wa[0], h1, r, Dims[0], Dims[1]);
                    BiasRelu(h1, ba[0], r, Dims[1], relu: true);
                    SimdGemm.SgemmSequential(h1, wa[1], h2, r, Dims[1], Dims[2]);
                    BiasRelu(h2, ba[1], r, Dims[2], relu: true);
                    var o = outBuf.AsSpan(r0 * Dims[3], r * Dims[3]);
                    SimdGemm.SgemmSequential(h2, wa[2], o, r, Dims[2], Dims[3]);
                    BiasRelu(o, ba[2], r, Dims[3], relu: false);
                })));
            // Correctness of the prototype.
            configs[configs.Count - 1].Body();
            double err = 0, big = 0;
            for (int i = 0; i < reference.Length; i++) { err = Math.Max(err, Math.Abs(outBuf[i] - reference[i])); big = Math.Max(big, Math.Abs(reference[i])); }
            Console.WriteLine($"row-stripe x{s}: max relative difference vs MlpForward {err / big:G3}");
        }

        // Per-layer cost of the pieces, single-threaded, for the arithmetic floor.
        var h1s = new float[M * Dims[1]];
        var h2s = new float[M * Dims[2]];
        var os = new float[M * Dims[3]];
        configs.Add(("L0 128x784x512 sequential", () => SimdGemm.SgemmSequential(xa, wa[0], h1s, M, Dims[0], Dims[1])));
        configs.Add(("L1 128x512x128 sequential", () => SimdGemm.SgemmSequential(h1s, wa[1], h2s, M, Dims[1], Dims[2])));
        configs.Add(("L2 128x128x10 sequential", () => SimdGemm.SgemmSequential(h2s, wa[2], os, M, Dims[2], Dims[3])));
        configs.Add(("empty dispatch x16", () => PersistentParallelExecutor.Instance.Execute(16, 16, _ => { })));
        configs.Add(("empty dispatch x32", () => PersistentParallelExecutor.Instance.Execute(32, 32, _ => { })));

        const int rounds = 7, iters = 200;
        var medians = new double[configs.Count][];
        for (int c = 0; c < configs.Count; c++) medians[c] = new double[rounds];
        foreach (var (_, body) in configs) for (int i = 0; i < 20; i++) body();
        var sw = new Stopwatch();
        var t = new double[iters];
        for (int r = 0; r < rounds; r++)
            for (int c = 0; c < configs.Count; c++)
            {
                for (int i = 0; i < iters; i++) { sw.Restart(); configs[c].Body(); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
                Array.Sort(t);
                medians[c][r] = t[iters / 2];
            }
        Console.WriteLine($"{"configuration",-40} {"median ms",10} {"min round",10} {"max round",10}");
        for (int c = 0; c < configs.Count; c++)
        {
            var m = medians[c].OrderBy(v => v).ToArray();
            Console.WriteLine($"{configs[c].Name,-40} {m[rounds / 2],10:F4} {m[0],10:F4} {m[rounds - 1],10:F4}");
        }
    }

    private static void BiasRelu(Span<float> h, float[] bias, int rows, int cols, bool relu)
    {
        int vw = System.Numerics.Vector<float>.Count;
        var zero = System.Numerics.Vector<float>.Zero;
        for (int r = 0; r < rows; r++)
        {
            var row = h.Slice(r * cols, cols);
            int j = 0;
            for (; j + vw <= cols; j += vw)
            {
                var v = new System.Numerics.Vector<float>(row.Slice(j)) + new System.Numerics.Vector<float>(bias, j);
                (relu ? System.Numerics.Vector.Max(v, zero) : v).CopyTo(row.Slice(j));
            }
            for (; j < cols; j++) { float v = row[j] + bias[j]; row[j] = relu && v < 0 ? 0 : v; }
        }
    }
}
