#if !NET471 // JitGemmAvx2 is the AVX2 runtime-emit kernel — net5+/x86 only (see JitGemmAvx2.cs).
// Correctness of the 6×N asm panel kernel (JitGemmAvx2): the runtime-emitted
// machine code must match a managed reference GEMM across shapes and edges
// (M%6 != 0, N%16 != 0, various K). Requires AIDOTNET_JIT_GEMM=1 so the panel
// path actually engages; otherwise the test no-ops with a note.

using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class JitPanelKernelTests
{
    private readonly ITestOutputHelper _out;
    public JitPanelKernelTests(ITestOutputHelper o) => _out = o;

    [SkippableFact]
    public void PanelKernel_MatchesReference_AcrossShapesAndEdges()
    {
        Skip.IfNot(JitGemmAvx2.Available, "JIT kernel unavailable on this platform.");
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_JIT_GEMM") == "1",
            "Set AIDOTNET_JIT_GEMM=1 to exercise the JIT panel path.");

        var rng = new Random(20260603);
        // Mix of aligned and edge shapes; include the AIsEval losing shapes.
        var shapes = new (int M, int K, int N)[]
        {
            (6, 64, 16), (12, 64, 32), (128, 64, 256), (1024, 64, 128), (1024, 64, 192),
            (32, 784, 512), (128, 512, 128),
            (7, 64, 17), (13, 70, 33), (130, 64, 257), (5, 64, 16), (6, 64, 15), (100, 96, 100),
        };

        int tested = 0;
        foreach (var (M, K, N) in shapes)
        {
            var A = RandF(M * K, rng);
            var B = RandF(K * N, rng);
            var got = new float[M * N];
            var want = new float[M * N];

            // Reference = the trusted managed FMA kernel (same accumulation style as the
            // JIT), so a correct panel kernel matches to ~1e-5; an indexing/encoding bug
            // shows up as O(1) error. (Naive triple-loop would differ by FMA rounding.)
            SimdGemm.SgemmAddInternal(A, K, false, B, N, false, want, M, K, N, allowParallel: false, clearedOutput: false);
            // Force the JIT panel path (bypass the auto-tuner) so correctness is always
            // exercised regardless of which kernel is faster.
            JitGemmAvx2.RunJit(A, B, got, M, N, K);

            double maxAbs = 0, maxMag = 0;
            for (int i = 0; i < want.Length; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(got[i] - want[i]));
                maxMag = Math.Max(maxMag, Math.Abs(want[i]));
            }
            double rel = maxAbs / Math.Max(1e-6, maxMag);
            _out.WriteLine($"[{M}x{K}x{N}] maxAbs={maxAbs:E2} relToMax={rel:E2}");
            Assert.True(rel < 1e-4, $"shape {M}x{K}x{N}: relToMax {rel} exceeds 1e-4 (likely a kernel bug)");
            tested++;
        }
        _out.WriteLine($"verified {tested} shapes via the JIT panel path");
    }

    private static void Reference(float[] a, float[] b, float[] c, int M, int K, int N)
    {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                for (int k = 0; k < K; k++) acc += a[i * K + k] * b[k * N + j];
                c[i * N + j] = acc;
            }
    }

    private static float[] RandF(int n, Random r) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1); return a; }
}
#endif
