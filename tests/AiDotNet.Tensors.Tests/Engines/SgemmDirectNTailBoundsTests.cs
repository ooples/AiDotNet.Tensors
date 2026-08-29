// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines
{
    /// <summary>
    /// The N-tail direct kernels must read B only within the operand the caller supplied, and must
    /// still compute the same product while doing so.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>DirectKernelMxNMasked</c> and <c>DirectKernelMxNMaskedStore</c> are entered when the
    /// column count leaves a tail, so <c>ncActual</c> is between 1 and Nr-1. Both used to load a
    /// full 16-wide B row on every k step regardless of that, masking only the store. On the last k
    /// row the final address touched is <c>(k-1)*ldb + j + 15</c>, while the operand check
    /// guarantees only <c>(k-1)*ldb + n</c> and <c>j + ncActual == n</c> -- an over-read of
    /// <c>16 - ncActual</c> floats with no slack to absorb it. It surfaced as an
    /// AccessViolationException that killed a Linux CI host inside a backward-pass GEMM, on a shard
    /// that reported 250 tests executed and 0 failed.
    /// </para>
    /// <para>
    /// The over-read is not directly observable from managed code: the extra lanes are discarded by
    /// the masked store, so results are unchanged and the read only faults when the overhang happens
    /// to cross onto an unmapped page. What IS checkable, and what this fix's correctness rests on,
    /// is that switching those loads to masked ones leaves the product identical for every tail
    /// width -- so these tests walk the whole range the masked loads changed.
    /// </para>
    /// <para>
    /// THE TWO KERNELS NEED TWO DIFFERENT ENTRY POINTS, which is the part that is easy to get wrong.
    /// <c>SgemmDirect</c> selects between them on <c>clearedOutput</c>: the store kernel overwrites
    /// C, the plain one accumulates into it. <see cref="SimdGemm.SgemmDirectParallelMInto"/> passes
    /// <c>clearedOutput: true</c> unconditionally, so a test written only against it exercises
    /// <c>DirectKernelMxNMaskedStore</c> and never touches <c>DirectKernelMxNMasked</c> at all.
    /// <see cref="AccumulatePath_NTail_MatchesReference"/> covers the other half through
    /// <c>SgemmAdd</c>.
    /// </para>
    /// </remarks>
    public class SgemmDirectNTailBoundsTests
    {
        /// <summary>Every tail width the masked loads changed, on both lanes of the tile.</summary>
        /// <remarks>
        /// <c>ncActual</c> is <c>n % 16</c>, so each width is generated twice at different n to keep
        /// the tail from only ever being the first block. Both an M-full and an M-tail shape run, so
        /// the M-edge call sites (which pass <c>ncActual: Nr</c> and must stay unmasked) execute
        /// beside the tail ones.
        /// </remarks>
        public static TheoryData<int, int, int> StoreTailShapes()
        {
            var data = new TheoryData<int, int, int>();
            for (int tail = 1; tail <= 15; tail++)
            {
                data.Add(6, 5, 16 + tail);    // one full M tile, tail in the second N block
                data.Add(13, 9, 48 + tail);   // two M tiles plus an M tail, later N block
            }

            // Exact multiples of Nr: no tail at all, so the masked kernels must not be entered and
            // the result must still be right. Guards against a fix that "works" by always masking.
            data.Add(6, 5, 16);
            data.Add(13, 9, 64);
            return data;
        }

        [Theory]
        [MemberData(nameof(StoreTailShapes))]
        public void StorePath_NTail_MatchesReference(int m, int k, int n)
        {
            var (a, b) = MakeOperands(m, k, n, seed: 20260829);

            var actual = new float[m * n];
            SimdGemm.SgemmDirectParallelMInto(a, b, actual, m, k, n);

            AssertMatches(Reference(a, b, m, k, n, seed: null), actual, m, k, n, "store");
        }

        /// <summary>
        /// The accumulate kernel, <c>DirectKernelMxNMasked</c>, which the store-path test cannot
        /// reach.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Only ONE tail width is reachable here, and it is not an arbitrary choice. The dispatch in
        /// <c>SgemmAddInternal</c> admits the <c>SgemmDirect</c> branch only when <c>n % 8 == 0</c>,
        /// and a tail needs <c>n % 16 != 0</c>; together those force <c>n % 16 == 8</c>, so
        /// <c>ncActual</c> is 8. The other widths are unreachable through the public accumulate API
        /// rather than merely untested.
        /// </para>
        /// <para>
        /// Eight is also the worst case of the bug. At <c>ncActual == 8</c> the row holds exactly the
        /// eight columns lane 0 reads, so the old unconditional <c>Avx.LoadVector256(pB + 8)</c> for
        /// lane 1 lay ENTIRELY past the end of the row -- the full 8-float over-read, not a partial
        /// one. The fix supplies <c>Vector256&lt;float&gt;.Zero</c> there instead.
        /// </para>
        /// <para>
        /// Shapes are kept small so the work stays under <c>ParallelDirectWorkThreshold</c> and m
        /// under <c>ParallelDirectMinM</c>, which is what keeps the call in <c>SgemmDirect</c>
        /// instead of the parallel variant.
        /// </para>
        /// </remarks>
        public static TheoryData<int, int, int> AccumulateTailShapes()
        {
            var data = new TheoryData<int, int, int>();
            foreach (int n in new[] { 24, 40, 56 })   // n % 16 == 8
            {
                data.Add(6, 5, n);
                data.Add(13, 9, n);
                data.Add(20, 17, n);
            }
            return data;
        }

        [Theory]
        [MemberData(nameof(AccumulateTailShapes))]
        public void AccumulatePath_NTail_MatchesReference(int m, int k, int n)
        {
            Assert.Equal(8, n % 16);   // the case under test is genuinely a tail of 8

            var (a, b) = MakeOperands(m, k, n, seed: 20260830);

            // Pre-filled C: SgemmAdd is C += A*B, so a zero C would not distinguish accumulate from
            // overwrite and the test would pass against the wrong kernel.
            var c = new float[m * n];
            var rng = new Random(99);
            for (int i = 0; i < c.Length; i++) c[i] = (float)(rng.NextDouble() * 2 - 1);
            var seeded = (float[])c.Clone();

            SimdGemm.SgemmAdd(a, b, c, m, k, n);

            AssertMatches(Reference(a, b, m, k, n, seeded), c, m, k, n, "accumulate");
        }

        /// <summary>Exactly sized operands: no slack after the last element, which is the condition
        /// the over-read had no room for.</summary>
        private static (float[] A, float[] B) MakeOperands(int m, int k, int n, int seed)
        {
            var rng = new Random(seed);
            var a = new float[m * k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
            return (a, b);
        }

        private static double[] Reference(
            float[] a, float[] b, int m, int k, int n, IReadOnlyList<float>? seed)
        {
            var expected = new double[m * n];
            if (seed is not null)
            {
                for (int i = 0; i < expected.Length; i++) expected[i] = seed[i];
            }

            for (int i = 0; i < m; i++)
                for (int p = 0; p < k; p++)
                {
                    double av = a[i * k + p];
                    for (int j = 0; j < n; j++)
                        expected[i * n + j] += av * b[p * n + j];
                }

            return expected;
        }

        private static void AssertMatches(
            double[] expected, float[] actual, int m, int k, int n, string path)
        {
            for (int i = 0; i < expected.Length; i++)
            {
                double want = expected[i];
                double got = actual[i];
                double tol = 1e-4 * Math.Max(1.0, Math.Abs(want));
                Assert.True(
                    Math.Abs(want - got) <= tol,
                    $"{path} path, m={m} k={k} n={n} (tail {n % 16}) index {i}: expected {want:G9}, "
                        + $"got {got:G9}. The N-tail kernels must produce the same product with "
                        + "masked B loads as with full-width ones -- the masked lanes contribute "
                        + "nothing to any stored column.");
            }
        }
    }
}
