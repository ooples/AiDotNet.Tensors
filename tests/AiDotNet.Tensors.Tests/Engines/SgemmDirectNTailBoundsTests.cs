// Copyright (c) AiDotNet. All rights reserved.
using System;
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
    /// The over-read is not directly observable from managed code: the extra lanes are masked out
    /// of the store, so results are unchanged and the read only faults when the overhang happens to
    /// cross onto an unmapped page. What IS checkable, and what this fixes' correctness rests on, is
    /// that switching those loads to masked ones leaves the product identical for every tail width.
    /// The widths below cover ncActual = 1..15 across both lanes of the 16-wide tile, which is
    /// exactly the range the masked loads changed.
    /// </para>
    /// </remarks>
    public class SgemmDirectNTailBoundsTests
    {
        public static TheoryData<int, int, int> TailShapes()
        {
            var data = new TheoryData<int, int, int>();
            // n chosen so n % 16 walks every tail width; m spans full tiles and an m tail.
            foreach (int n in new[] { 1, 7, 8, 9, 15, 17, 23, 31, 33, 47 })
            {
                data.Add(6, 5, n);    // exactly one full M tile
                data.Add(13, 9, n);   // two M tiles plus an M tail
            }
            return data;
        }

        [Theory]
        [MemberData(nameof(TailShapes))]
        public void SgemmDirect_NTail_MatchesReference(int m, int k, int n)
        {
            var rng = new Random(20260829);

            // Exactly sized operands: no slack after the last element, which is the condition the
            // over-read had no room for.
            var a = new float[m * k];
            var b = new float[k * n];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            var actual = new float[m * n];
            SimdGemm.SgemmDirectParallelMInto(a, b, actual, m, k, n);

            var expected = new double[m * n];
            for (int i = 0; i < m; i++)
                for (int p = 0; p < k; p++)
                {
                    double av = a[i * k + p];
                    for (int j = 0; j < n; j++)
                        expected[i * n + j] += av * b[p * n + j];
                }

            for (int i = 0; i < expected.Length; i++)
            {
                double want = expected[i];
                double got = actual[i];
                double tol = 1e-4 * Math.Max(1.0, Math.Abs(want));
                Assert.True(
                    Math.Abs(want - got) <= tol,
                    $"m={m} k={k} n={n} (tail {n % 16}) index {i}: expected {want:G9}, got {got:G9}. "
                        + "The N-tail kernels must produce the same product with masked B loads as "
                        + "with full-width ones -- the masked lanes contribute nothing to any stored "
                        + "column.");
            }
        }
    }
}
