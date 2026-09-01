// Copyright (c) AiDotNet. All rights reserved.
#if NET5_0_OR_GREATER
// These entry points and their block kernels are AVX2/FMA intrinsics, which SimdGemm declares
// inside #if NET5_0_OR_GREATER.
using System;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines
{
    /// <summary>
    /// The 4-row block kernels must not write past C when n is not a multiple of 8.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Three block kernels walk columns eight at a time -- <c>SgemmTransABlock</c>,
    /// <c>DgemmDirectBlockRange</c> and <c>DgemmTransABlock</c> -- and each used to finish every
    /// step with a full-width store. C rows are only n wide, so when n was not a multiple of 8 the
    /// final step wrote <c>8 - n % 8</c> elements past the end of every row it touched.
    /// </para>
    /// <para>
    /// Three of those four overruns were invisible: they landed in the following row and the next
    /// store overwrote them, so the arithmetic came out right and every existing assertion passed.
    /// The fourth is the last row of the block, and for the final block it lands past the end of C
    /// itself -- a write into the GC heap.
    /// </para>
    /// <para>
    /// Nothing fails at the call. The process dies at some later, unrelated allocation with
    /// <c>Internal CLR error (0x80131506)</c> / <c>ExecutionEngineException</c>, reported against
    /// whatever code happened to trigger the collection. It was found as a test-host death two
    /// classes away from any GEMM.
    /// </para>
    /// <para>
    /// The headers declared <c>n % 8 == 0</c>, and the INTERNAL dispatch does gate on it -- but
    /// these public entry points bypass that gate. <c>SgemmDirectParallelMIntoTransA</c> validates
    /// operand LENGTHS and lets any n through; <c>DgemmDirectParallelMInto</c> does not validate at
    /// all. A public method that silently corrupts the heap for n = 5 is a defect whichever way the
    /// contract is read, and every sibling here already handles arbitrary n, as does BLAS.
    /// </para>
    /// <para>
    /// An over-WRITE is directly testable, unlike an over-read: C is allocated with a
    /// sentinel-filled margin and the margin is checked afterwards.
    /// </para>
    /// </remarks>
    public class GemmColumnTailWriteBoundsTests
    {
        private const float SentinelF = -987654.5f;
        private const double SentinelD = -987654.5;

        /// <summary>A margin no kernel may touch; one full vector is the most a store can spill.</summary>
        private const int Margin = 8;

        /// <summary>Column counts around the 8-wide step, and m on both sides of the 4-row block.</summary>
        /// <remarks>
        /// m must reach 4 for the vectorized block to run at all; below that everything falls to the
        /// scalar edge path, which was never the problem. m = 13 exercises three full blocks plus the
        /// scalar remainder, so both paths run in one case.
        /// </remarks>
        public static TheoryData<int, int, int> Shapes()
        {
            var data = new TheoryData<int, int, int>();
            for (int n = 1; n <= 20; n++)
            {
                data.Add(4, 3, n);     // exactly one block -- the case that overruns C itself
                data.Add(8, 5, n);     // two blocks
                data.Add(13, 7, n);    // three blocks plus a scalar edge
            }
            return data;
        }

        // ------------------------------------------------------------------ FP32, A transposed

        [Theory]
        [MemberData(nameof(Shapes))]
        public void SgemmTransA_DoesNotWritePastC(int m, int k, int n)
        {
            var (a, b) = FloatOperands(k * m, k * n);
            var c = FilledF(m * n + Margin);

            SimdGemm.SgemmDirectParallelMIntoTransA(a, b, c.AsSpan(0, m * n), m, k, n);

            AssertMarginF(c, m, k, n, "SgemmDirectParallelMIntoTransA");
        }

        [Theory]
        [MemberData(nameof(Shapes))]
        public void SgemmTransA_MatchesReference(int m, int k, int n)
        {
            var (a, b) = FloatOperands(k * m, k * n);
            var c = new float[m * n];

            SimdGemm.SgemmDirectParallelMIntoTransA(a, b, c, m, k, n);

            // A is [k,m] at lda=m; B is [k,n] at ldb=n.
            AssertProduct(m, n, k, (i, p) => a[p * m + i], (p, j) => b[p * n + j], i => c[i], m, k, n);
        }

        // ------------------------------------------------------------------ FP64, A transposed

        [Theory]
        [MemberData(nameof(Shapes))]
        public void DgemmTransA_DoesNotWritePastC(int m, int k, int n)
        {
            var (a, b) = DoubleOperands(k * m, k * n);
            var c = FilledD(m * n + Margin);

            SimdGemm.DgemmDirectParallelMIntoTransA(a, b, c.AsSpan(0, m * n), m, k, n);

            AssertMarginD(c, m, k, n, "DgemmDirectParallelMIntoTransA");
        }

        [Theory]
        [MemberData(nameof(Shapes))]
        public void DgemmTransA_MatchesReference(int m, int k, int n)
        {
            var (a, b) = DoubleOperands(k * m, k * n);
            var c = new double[m * n];

            SimdGemm.DgemmDirectParallelMIntoTransA(a, b, c, m, k, n);

            AssertProduct(m, n, k, (i, p) => a[p * m + i], (p, j) => b[p * n + j], i => c[i], m, k, n);
        }

        // ------------------------------------------------------------- FP64, no transpose

        [Theory]
        [MemberData(nameof(Shapes))]
        public void DgemmDirect_DoesNotWritePastC(int m, int k, int n)
        {
            var (a, b) = DoubleOperands(m * k, k * n);
            var c = FilledD(m * n + Margin);

            SimdGemm.DgemmDirectParallelMInto(a, b, c.AsSpan(0, m * n), m, k, n);

            AssertMarginD(c, m, k, n, "DgemmDirectParallelMInto");
        }

        [Theory]
        [MemberData(nameof(Shapes))]
        public void DgemmDirect_MatchesReference(int m, int k, int n)
        {
            var (a, b) = DoubleOperands(m * k, k * n);
            var c = new double[m * n];

            SimdGemm.DgemmDirectParallelMInto(a, b, c, m, k, n);

            // A is [m,k] at lda=k; B is [k,n] at ldb=n.
            AssertProduct(m, n, k, (i, p) => a[i * k + p], (p, j) => b[p * n + j], i => c[i], m, k, n);
        }

        // ------------------------------------------------------------------------- helpers

        private static float[] FilledF(int length)
        {
            var c = new float[length];
            for (int i = 0; i < c.Length; i++) c[i] = SentinelF;
            return c;
        }

        private static double[] FilledD(int length)
        {
            var c = new double[length];
            for (int i = 0; i < c.Length; i++) c[i] = SentinelD;
            return c;
        }

        private static (float[] A, float[] B) FloatOperands(int aLen, int bLen)
        {
            var rng = new Random(20260829);
            var a = new float[aLen];
            var b = new float[bLen];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
            return (a, b);
        }

        private static (double[] A, double[] B) DoubleOperands(int aLen, int bLen)
        {
            var rng = new Random(20260830);
            var a = new double[aLen];
            var b = new double[bLen];
            for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
            return (a, b);
        }

        private static void AssertMarginF(float[] c, int m, int k, int n, string entry)
        {
            for (int i = 0; i < Margin; i++)
            {
                int at = m * n + i;
                Assert.True(c[at] == SentinelF, Message(entry, at, c[at], m, k, n));
            }
        }

        private static void AssertMarginD(double[] c, int m, int k, int n, string entry)
        {
            for (int i = 0; i < Margin; i++)
            {
                int at = m * n + i;
                Assert.True(c[at] == SentinelD, Message(entry, at, c[at], m, k, n));
            }
        }

        private static string Message(string entry, int at, double actual, int m, int k, int n) =>
            $"{entry}: m={m} k={k} n={n} (tail {n % 8}) wrote {actual} at C[{at}], past the end of "
                + $"the {m * n}-element output. The column loop steps 8 at a time and must mask its "
                + "final store; a full-width store there writes into the GC heap, and the process "
                + "dies at some later collection rather than here.";

        private static void AssertProduct(
            int m, int n, int k,
            Func<int, int, double> a, Func<int, int, double> b, Func<int, double> c,
            int mm, int kk, int nn)
        {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double want = 0;
                    for (int p = 0; p < k; p++) want += a(i, p) * b(p, j);

                    double got = c(i * n + j);
                    double tol = 1e-4 * Math.Max(1.0, Math.Abs(want));
                    Assert.True(
                        Math.Abs(want - got) <= tol,
                        $"m={mm} k={kk} n={nn} at [{i},{j}]: expected {want:G9}, got {got:G9}. "
                            + "Masking the tail store must not change the values it does write.");
                }
        }
    }
}
#endif
