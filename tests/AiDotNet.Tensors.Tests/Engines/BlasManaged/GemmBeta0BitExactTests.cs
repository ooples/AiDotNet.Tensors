using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// beta=0 (write-first, skip output-zeroing) correctness suite.
///
/// <para>
/// The core contract of <see cref="BlasOptions{T}.BetaZero"/> is that a pure
/// <c>C := op(A)·op(B)</c> is computed <b>bit-for-bit identically</b> to the historical
/// zero-then-accumulate path (<c>BetaZero=false</c>), just without the redundant <c>C.Clear()</c>
/// memset. Writing X is identical to <c>0 + X</c> (the accumulator starts at <c>+0.0</c> either
/// way), so every path must agree to the last bit.
/// </para>
///
/// <para>
/// Each case runs the SAME <see cref="BlasManagedLib.Gemm{T}"/> twice — once with
/// <c>BetaZero=false</c> into a zero-initialized C (the reference / current behavior) and once
/// with <c>BetaZero=true</c> into a C whose backing buffer is POISONED with a large sentinel —
/// then asserts:
/// <list type="number">
///   <item>the [m, n] output tile is byte-for-byte identical between the two runs (bit-exact,
///         and — because the beta=0 buffer was poisoned — proof that beta=0 OVERWRITES every
///         tile element rather than reading stale C);</item>
///   <item>the beta=0 run leaves every buffer byte OUTSIDE the [m, n] tile (ldc padding, leading
///         and trailing offset) untouched (still the sentinel) — beta=0 must not clobber memory
///         the caller owns outside the logical tile;</item>
///   <item>the beta=1 (BetaZero=false) run also fully overwrites a poisoned buffer's tile
///         (clear-then-accumulate semantics preserved).</item>
/// </list>
/// </para>
///
/// <para>
/// The matrix spans thin-M / thin-N / K=1 (GEMV / outer-product edges), the N-BEATS m=96/256
/// forward shapes, square small/medium, non-square and a large shape; K spanning one-panel and
/// multi-panel (1, 64, 96, 256, 512); both dtypes (float, double); all four transpose combos
/// (NN, NT, TN, TT); contiguous AND ldc-padded/offset C; and each routable strategy forced via
/// <see cref="PackingMode"/> so every path's beta=0 handling is covered. Runs on net10.0 (AVX2
/// microkernels + thin-M direct) and net471 (scalar fallbacks).
/// </para>
/// </summary>
// Serialize against global GEMM-path mutators (determinism flag / stats) so a concurrent
// toggle can't change the reduction order mid-assertion.
[Collection("BlasManaged-Stats-Serial")]
public class GemmBeta0BitExactTests
{
    // Distinctive, large-but-finite sentinel poison. Large so that any path that erroneously
    // READS a "cleared" C (instead of overwriting) produces a wildly wrong value → mismatch;
    // finite so comparisons stay well-defined (no NaN games).
    private const float PoisonF = 3.0e30f;
    private const double PoisonD = 3.0e30;

    private static readonly PackingMode[] Modes =
    {
        PackingMode.Auto,
        PackingMode.DisableAutotune,   // the mode CpuEngine's pure-matmul float path uses
        PackingMode.ForceStreaming,
        PackingMode.ForcePackBoth,
        PackingMode.ForcePackAOnly,
    };

    // ── shape list: (m, n, k). n is kept a multiple of 8 on the shapes meant to exercise the
    //    thin-M direct kernel; the rest cover edges / non-alignment. ──────────────────────────
    public static TheoryData<int, int, int> Shapes()
    {
        var d = new TheoryData<int, int, int>();
        // thin-M (below the m>=64 direct-kernel gate → tuned strategy / streaming)
        d.Add(1, 16, 8); d.Add(2, 32, 64); d.Add(5, 24, 96); d.Add(8, 64, 128);
        // thin-N
        d.Add(96, 1, 64); d.Add(128, 4, 96); d.Add(256, 8, 128);
        // K=1 (outer product / GEMV edge)
        d.Add(1, 8, 1); d.Add(64, 64, 1); d.Add(96, 32, 1);
        // N-BEATS forward shapes (m=96 / m=256, n%8==0 → thin-M direct on Auto/DisableAutotune)
        d.Add(96, 256, 256); d.Add(256, 256, 256); d.Add(256, 512, 512); d.Add(256, 128, 96);
        // square small / medium
        d.Add(16, 16, 16); d.Add(64, 64, 64); d.Add(128, 128, 128);
        // non-square + multi-panel K
        d.Add(96, 80, 512); d.Add(200, 72, 300); d.Add(130, 130, 130);
        // larger (exercises GotoGemm / PackBoth cache-blocking under beta=0)
        d.Add(384, 384, 384);
        return d;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Beta0_IsBitExact_ToBeta1_AndOverwritesTile(int m, int n, int k)
    {
        foreach (var mode in Modes)
        {
            // Transposes: NN, NT, TN, TT.
            for (int t = 0; t < 4; t++)
            {
                bool transA = (t & 1) != 0;
                bool transB = (t & 2) != 0;

                // C layouts: contiguous (ldc=n, no offset) AND padded+offset.
                CheckOne<float>(m, n, k, transA, transB, mode, ldcPad: 0, lead: 0, trail: 0, seed: 1234);
                CheckOne<double>(m, n, k, transA, transB, mode, ldcPad: 0, lead: 0, trail: 0, seed: 1234);
                CheckOne<float>(m, n, k, transA, transB, mode, ldcPad: 3, lead: 5, trail: 7, seed: 5678);
                CheckOne<double>(m, n, k, transA, transB, mode, ldcPad: 3, lead: 5, trail: 7, seed: 5678);
            }
        }
    }

    private static void CheckOne<T>(
        int m, int n, int k, bool transA, bool transB, PackingMode mode,
        int ldcPad, int lead, int trail, int seed) where T : unmanaged
    {
        bool isDouble = typeof(T) == typeof(double);

        // Stored (possibly transposed) operand layouts + leading dims.
        int lda = transA ? m : k;              // A: !transA=[m,k] lda=k ; transA=[k,m] lda=m
        int ldb = transB ? k : n;              // B: !transB=[k,n] ldb=n ; transB=[n,k] ldb=k
        int aRows = transA ? k : m;
        int bRows = transB ? n : k;
        int aLen = aRows * lda;
        int bLen = bRows * ldb;

        int ldc = n + ldcPad;
        int cTileSpan = (m - 1) * ldc + n;     // last touched element of the logical tile
        int cBufLen = lead + m * ldc + trail;  // full backing buffer (tile occupies [lead, lead+m*ldc))

        var a = new T[aLen];
        var b = new T[bLen];
        FillRandom(a, seed);
        FillRandom(b, seed ^ 0x5bd1e995);

        // Reference: BetaZero=false into a ZERO buffer (current behavior).
        var cRefBuf = new T[cBufLen];   // default-zeroed
        var cRefSpan = new Span<T>(cRefBuf, lead, cBufLen - lead);
        BlasManagedLib.Gemm<T>(a, lda, transA, b, ldb, transB, cRefSpan, ldc, m, n, k,
            new BlasOptions<T> { PackingMode = mode, BetaZero = false });

        // beta=0: BetaZero=true into a fully POISONED buffer.
        var cBetaBuf = new T[cBufLen];
        Poison(cBetaBuf);
        var cBetaSpan = new Span<T>(cBetaBuf, lead, cBufLen - lead);
        BlasManagedLib.Gemm<T>(a, lda, transA, b, ldb, transB, cBetaSpan, ldc, m, n, k,
            new BlasOptions<T> { PackingMode = mode, BetaZero = true });

        string ctx = $"[{typeof(T).Name} {m}x{n}x{k} tA={transA} tB={transB} mode={mode} pad={ldcPad} lead={lead}]";

        // (1) tile bit-exact between beta=0 and beta=1, cell by cell (respecting ldc stride).
        for (int r = 0; r < m; r++)
        {
            for (int col = 0; col < n; col++)
            {
                int idx = lead + r * ldc + col;
                AssertBitEqual(cRefBuf[idx], cBetaBuf[idx], $"tile bit mismatch at ({r},{col}) {ctx}");
            }
        }

        // (2) beta=0 must NOT touch anything outside the [m,n] tile: leading offset, per-row
        //     ldc padding, and the trailing region all remain the poison sentinel.
        for (int i = 0; i < lead; i++)
            AssertIsPoison(cBetaBuf[i], $"beta=0 clobbered LEADING offset at {i} {ctx}");
        for (int r = 0; r < m; r++)
            for (int col = n; col < ldc; col++)
                AssertIsPoison(cBetaBuf[lead + r * ldc + col], $"beta=0 clobbered ldc PADDING at row {r} col {col} {ctx}");
        for (int i = lead + cTileSpan; i < cBufLen; i++)
            AssertIsPoison(cBetaBuf[i], $"beta=0 clobbered TRAILING region at {i} {ctx}");

        // (3) beta=1 (BetaZero=false) still fully overwrites a POISONED buffer's tile
        //     (clear-then-accumulate contract preserved — no stale C leaks through).
        var cBeta1Buf = new T[cBufLen];
        Poison(cBeta1Buf);
        var cBeta1Span = new Span<T>(cBeta1Buf, lead, cBufLen - lead);
        BlasManagedLib.Gemm<T>(a, lda, transA, b, ldb, transB, cBeta1Span, ldc, m, n, k,
            new BlasOptions<T> { PackingMode = mode, BetaZero = false });
        for (int r = 0; r < m; r++)
            for (int col = 0; col < n; col++)
            {
                int idx = lead + r * ldc + col;
                AssertBitEqual(cRefBuf[idx], cBeta1Buf[idx], $"beta=1 poisoned-buffer tile mismatch at ({r},{col}) {ctx}");
            }
    }

    /// <summary>
    /// Independent correctness anchor: on a handful of small shapes, confirm the beta=0 result
    /// actually equals A·B computed by a naive double-accumulation reference (guards against the
    /// beta=0 and beta=1 paths being identically wrong).
    /// </summary>
    [Theory]
    [InlineData(64, 64, 64, false, false)]
    [InlineData(96, 32, 128, false, false)]
    [InlineData(72, 40, 96, true, false)]
    [InlineData(64, 48, 80, false, true)]
    [InlineData(56, 56, 56, true, true)]
    public void Beta0_Matches_NaiveReference(int m, int n, int k, bool transA, bool transB)
    {
        var rng = new Random(99);
        int lda = transA ? m : k;
        int ldb = transB ? k : n;
        var a = new float[(transA ? k : m) * lda];
        var b = new float[(transB ? n : k) * ldb];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        var c = new float[m * n];
        Poison(c);
        BlasManagedLib.Gemm<float>(a, lda, transA, b, ldb, transB, c, n, m, n, k,
            new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune, BetaZero = true });

        double maxAbs = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int p = 0; p < k; p++)
                {
                    double av = transA ? a[p * lda + i] : a[i * lda + p];
                    double bv = transB ? b[j * ldb + p] : b[p * ldb + j];
                    acc += av * bv;
                }
                maxAbs = Math.Max(maxAbs, Math.Abs(c[i * n + j] - acc));
            }
        Assert.True(maxAbs < 1e-3, $"beta=0 {m}x{n}x{k} tA={transA} tB={transB}: maxAbsDelta={maxAbs} vs naive ref");
    }

    // ── helpers ──────────────────────────────────────────────────────────────────────────────

    private static void FillRandom<T>(T[] arr, int seed) where T : unmanaged
    {
        var rng = new Random(seed);
        Span<T> span = arr;
        if (typeof(T) == typeof(float))
        {
            var f = MemoryMarshal.Cast<T, float>(span);
            for (int i = 0; i < f.Length; i++) f[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        else
        {
            var dbl = MemoryMarshal.Cast<T, double>(span);
            for (int i = 0; i < dbl.Length; i++) dbl[i] = rng.NextDouble() * 2 - 1;
        }
    }

    private static void Poison<T>(T[] arr) where T : unmanaged
    {
        Span<T> span = arr;
        if (typeof(T) == typeof(float))
            MemoryMarshal.Cast<T, float>(span).Fill(PoisonF);
        else
            MemoryMarshal.Cast<T, double>(span).Fill(PoisonD);
    }

    private static void AssertBitEqual<T>(T expected, T actual, string msg) where T : unmanaged
    {
        // Raw-bit comparison (portable across net10.0 / net471): reinterpret to an integer of
        // the same width. Distinguishes ±0.0 and catches any ULP difference.
        if (typeof(T) == typeof(float))
        {
            uint e = BitConverter.ToUInt32(BitConverter.GetBytes((float)(object)expected!), 0);
            uint a = BitConverter.ToUInt32(BitConverter.GetBytes((float)(object)actual!), 0);
            Assert.True(e == a, $"{msg} (expected bits {e:X8}, got {a:X8})");
        }
        else
        {
            ulong e = BitConverter.ToUInt64(BitConverter.GetBytes((double)(object)expected!), 0);
            ulong a = BitConverter.ToUInt64(BitConverter.GetBytes((double)(object)actual!), 0);
            Assert.True(e == a, $"{msg} (expected bits {e:X16}, got {a:X16})");
        }
    }

    private static void AssertIsPoison<T>(T value, string msg) where T : unmanaged
    {
        if (typeof(T) == typeof(float))
            Assert.True((float)(object)value! == PoisonF, msg);
        else
            Assert.True((double)(object)value! == PoisonD, msg);
    }
}
