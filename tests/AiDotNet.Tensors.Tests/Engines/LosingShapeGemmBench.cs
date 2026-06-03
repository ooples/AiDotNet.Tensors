// Per-shape GEMM bake-off on the exact AIsEval losing shapes: managed (RyuJIT
// codegen) vs the raw-machine-code asm-JIT kernel vs OpenBLAS. Answers whether
// RyuJIT codegen is the gap and whether the asm-JIT closes it at these shapes.
// Category=Performance => excluded from the normal test run.

using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

[Trait("Category", "Performance")]
public class LosingShapeGemmBench
{
    private readonly ITestOutputHelper _out;
    public LosingShapeGemmBench(ITestOutputHelper o) => _out = o;

    [Fact]
    public unsafe void GemmBakeOff()
    {
        // (label, M, K, N)
        var shapes = new (string, int, int, int)[]
        {
            ("mlp-L1 b32",      32, 784, 512),
            ("mlp-L1 b128",    128, 784, 512),
            ("mlp-L2 b128",    128, 512, 128),
            ("lstm-recur b32",  32,  64, 256),
            ("lstm-recur b128",128,  64, 256),
            ("xfmr-FFN b32",  1024,  64, 128),
            ("xfmr-QKV b32",  1024,  64, 192),
            ("xfmr-FFN b8",    256,  64, 128),
        };
        var rng = new Random(11);
        bool jitEnabled = Environment.GetEnvironmentVariable("AIDOTNET_JIT_GEMM") == "1";
        bool hasBlas = BlasProvider.HasRawSgemm;
        _out.WriteLine($"JIT enabled={jitEnabled} (JitGemmAvx2.Available={JitGemmAvx2.Available})  OpenBLAS={hasBlas}");
        _out.WriteLine($"{"shape",-18}{"managed",12}{"asmJIT",12}{"OpenBLAS",12}   (GFLOP/s)   verdict");

        foreach (var (label, M, K, N) in shapes)
        {
            var A = RandF(M * K, rng);
            var B = RandF(K * N, rng);
            var C = new float[M * N];
            var Cj = new float[M * N];
            double flop = 2.0 * M * N * K;

            double Min(int iters, Action f) { for (int i = 0; i < 100; i++) f(); double m = 1e30; for (int i = 0; i < iters; i++) { var s = Stopwatch.GetTimestamp(); f(); double u = (Stopwatch.GetTimestamp() - s) * 1e6 / Stopwatch.Frequency; if (u < m) m = u; } return m; }

            double tMgd = Min(2000, () => SimdGemm.SgemmAddInternal(A, K, false, B, N, false, C, M, K, N, allowParallel: true, clearedOutput: false));

            // Time the JIT panel kernel directly (force it, bypass the auto-tuner) so we
            // see its real throughput vs managed regardless of the tuner's verdict.
            bool jitWon = JitGemmAvx2.Available && M >= 6 && N >= 16;
            double tJit = jitWon ? Min(2000, () => JitGemmAvx2.RunJit(A, B, Cj, M, N, K)) : double.NaN;

            double tBlas = double.NaN;
            if (hasBlas)
            {
                var Cb = new float[M * N];
                fixed (float* pa = A, pb = B, pc = Cb)
                {
                    float* la = pa, lb = pb, lc = pc;
                    tBlas = Min(2000, () => BlasProvider.SgemmRaw(M, N, K, la, K, lb, N, lc, N));
                }
            }

            string G(double us) => double.IsNaN(us) ? "  --" : $"{flop / us / 1e3,7:F0}";
            _out.WriteLine($"{label,-18}{tMgd,9:F1}us {(double.IsNaN(tJit) ? "  --" : tJit.ToString("F1") + "us"),11} {(double.IsNaN(tBlas) ? "  --" : tBlas.ToString("F1") + "us"),11}   M:{G(tMgd)} J:{G(tJit)} B:{G(tBlas)}  jitWon={jitWon}");
        }
        Assert.True(true);
    }

    private static float[] RandF(int n, Random r) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1); return a; }
}
