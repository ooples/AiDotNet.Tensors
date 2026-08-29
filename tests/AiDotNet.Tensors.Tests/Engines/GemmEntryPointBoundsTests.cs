// Copyright (c) AiDotNet. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines
{
    /// <summary>
    /// Drives every public span-taking GEMM entry point at a column count that is not a multiple of
    /// a SIMD vector, and checks it does not write past the output it was given.
    /// </summary>
    /// <remarks>
    /// <para>
    /// WHY THIS CANNOT BE DONE AT THE ENGINE LEVEL. The obvious place to catch a bad vector tail is
    /// the op-parity registry, and it does not work: the thin-M fast path that reaches these kernels
    /// refuses to route unless the column count is already a multiple of eight —
    /// </para>
    /// <code>
    /// if (ldc != n || (n &amp; 7) != 0) return false;
    /// </code>
    /// <para>
    /// — so no <c>IEngine</c> op can reach them with a tail at all, however its shapes are chosen.
    /// The kernels are reachable ONLY by a direct caller of these public entry points, which is
    /// precisely the gap the two shipped bugs lived in: surface the library exposes but never itself
    /// calls at those shapes. <c>OpTailShapeSweepTests</c>'s engine-level sweep is the
    /// complement to this file, not a substitute for it.
    /// </para>
    /// <para>
    /// AN OVER-WRITE IS TESTABLE. The output is allocated with a sentinel margin the kernel has no
    /// right to touch, and the margin is checked afterwards, so a full-width store into a row that is
    /// not full-width fails HERE rather than corrupting the GC heap and killing the process at some
    /// later, unrelated allocation. (An over-READ leaves no trace and needs a guard page; see
    /// SgemmDirectGuardPageTests.)
    /// </para>
    /// <para>
    /// The inventory is reflected, not listed, so this cannot quietly go stale:
    /// <see cref="EveryPublicGemmEntryPoint_HasABoundsDriver"/> fails by name when a new public
    /// span-taking entry point appears without one.
    /// </para>
    /// </remarks>
    public class GemmEntryPointBoundsTests
    {
        private const float SentinelF = -987654.5f;
        private const double SentinelD = -987654.5;
        private const int Margin = 8;

        /// <summary>Shapes whose column count leaves a tail in both the 8-float and 4-double kernels.</summary>
        public static TheoryData<int, int, int> AwkwardShapes()
        {
            var data = new TheoryData<int, int, int>();
            foreach (var (m, k) in new[] { (4, 3), (8, 5), (13, 7) })
                foreach (int n in new[] { 13, 21, 26 })   // none divisible by 4 or 8
                    data.Add(m, k, n);
            return data;
        }

        // ------------------------------------------------------------------ the drivers

        private delegate void BoundsDriver(int m, int k, int n);

        /// <summary>One driver per public span-taking entry point; the gate below enforces the set.</summary>
        private static readonly IReadOnlyDictionary<string, BoundsDriver> Drivers =
            new Dictionary<string, BoundsDriver>(StringComparer.Ordinal)
            {
                ["Sgemm"] = (m, k, n) => RunF(m, k, n, m * k, k * n,
                    (a, b, c) => SimdGemm.Sgemm(a, b, c, m, k, n)),
                ["SgemmSequential"] = (m, k, n) => RunF(m, k, n, m * k, k * n,
                    (a, b, c) => SimdGemm.SgemmSequential(a, b, c, m, k, n)),
                ["SgemmAdd"] = (m, k, n) => RunF(m, k, n, m * k, k * n,
                    (a, b, c) => SimdGemm.SgemmAdd(a, b, c, m, k, n)),
                ["SgemmWithCachedB"] = (m, k, n) => RunF(m, k, n, m * k, k * n,
                    (a, b, c) => SimdGemm.SgemmWithCachedB(a, b, c, m, k, n)),
                ["SgemmWithInt8CachedB"] = (m, k, n) => RunF(m, k, n, m * k, k * n,
                    (a, b, c) => SimdGemm.SgemmWithInt8CachedB(a, b, c, m, k, n)),
                ["SgemmDirectParallelMInto"] = (m, k, n) => RunF(m, k, n, m * k, k * n,
                    (a, b, c) => SimdGemm.SgemmDirectParallelMInto(a, b, c, m, k, n)),
                ["SgemmDirectParallelMOverwrite"] = (m, k, n) => RunF(m, k, n, m * k, k * n,
                    (a, b, c) => SimdGemm.SgemmDirectParallelMOverwrite(a, b, c, m, k, n)),
                // A is [k, m] at lda = m.
                ["SgemmDirectParallelMIntoTransA"] = (m, k, n) => RunF(m, k, n, k * m, k * n,
                    (a, b, c) => SimdGemm.SgemmDirectParallelMIntoTransA(a, b, c, m, k, n)),
                // B is [n, k] at ldb = k.
                ["SgemmDirectParallelMIntoTransB"] = (m, k, n) => RunF(m, k, n, m * k, n * k,
                    (a, b, c) => SimdGemm.SgemmDirectParallelMIntoTransB(a, b, c, m, k, n)),

                ["DgemmDirectParallelMInto"] = (m, k, n) => RunD(m, k, n, m * k, k * n,
                    (a, b, c) => SimdGemm.DgemmDirectParallelMInto(a, b, c, m, k, n)),
                ["DgemmDirectParallelMIntoTransA"] = (m, k, n) => RunD(m, k, n, k * m, k * n,
                    (a, b, c) => SimdGemm.DgemmDirectParallelMIntoTransA(a, b, c, m, k, n)),
                ["DgemmDirectParallelMIntoTransB"] = (m, k, n) => RunD(m, k, n, m * k, n * k,
                    (a, b, c) => SimdGemm.DgemmDirectParallelMIntoTransB(a, b, c, m, k, n)),

                // Quantized / half-precision B. Same output contract, different operand types --
                // which is exactly why the inventory is reflected rather than listed: these live in
                // partial-class files (SimdGemm.Fp16Weight.cs, .Int8Int8.cs, .Int8RowScaled.cs) and a
                // file-scoped search of SimdGemm.cs does not see them at all.
                ["SgemmFp16WeightB"] = (m, k, n) => WithGuardedOutputF(m, n, c =>
                {
                    var a = RandF(m * k, 771);
                    var b = new Half[k * n];
                    var rng = new Random(772);
                    for (int i = 0; i < b.Length; i++) b[i] = (Half)(rng.NextDouble() * 2 - 1);
                    SimdGemm.SgemmFp16WeightB(a, b, c, m, k, n);
                }),
                ["SgemmA8W8RowScaledCachedB"] = (m, k, n) => WithGuardedOutputF(m, n, c =>
                    SimdGemm.SgemmA8W8RowScaledCachedB(
                        RandF(m * k, 773), RandI8(n * k, 774), RandF(n, 775), c, m, k, n)),
                ["SgemmWithInt8RowScaledCachedB"] = (m, k, n) => WithGuardedOutputF(m, n, c =>
                    SimdGemm.SgemmWithInt8RowScaledCachedB(
                        RandF(m * k, 776), RandI8(n * k, 777), RandF(n, 778), c, m, k, n)),
            };

        [Theory]
        [MemberData(nameof(AwkwardShapes))]
        public void EveryEntryPoint_AtAnAwkwardColumnCount_StaysInsideItsOutput(int m, int k, int n)
        {
            var failures = new List<string>();
            foreach (var (name, driver) in Drivers.OrderBy(p => p.Key, StringComparer.Ordinal))
            {
                try
                {
                    driver(m, k, n);
                }
                catch (ArgumentException)
                {
                    // A documented precondition this shape does not satisfy is a legitimate answer;
                    // silently corrupting memory is not. Rejecting is the behaviour under test.
                }
                catch (BoundsViolationException ex)
                {
                    failures.Add($"{name}: {ex.Message}");
                }
            }

            Assert.True(
                failures.Count == 0,
                $"m={m} k={k} n={n} (tail {n % 8}): {failures.Count} GEMM entry point(s) wrote past "
                    + "the end of the output they were given. A column loop that steps by a vector "
                    + "width must mask its final store.\n  " + string.Join("\n  ", failures));
        }

        [Fact]
        public void EveryPublicGemmEntryPoint_HasABoundsDriver()
        {
            var discovered = typeof(SimdGemm)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(TakesSpansAndDims)
                .Select(x => x.Name)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(x => x, StringComparer.Ordinal)
                .ToList();

            var missing = discovered.Where(x => !Drivers.ContainsKey(x)).ToList();

            Assert.True(
                missing.Count == 0,
                "These public GEMM entry points take spans and (m, k, n) but have no bounds driver, "
                    + "so nothing checks that they stay inside the output at a column count that is "
                    + "not a multiple of a vector. Add a driver to "
                    + $"{nameof(GemmEntryPointBoundsTests)}.{nameof(Drivers)}:\n  "
                    + string.Join("\n  ", missing));

            // And the reflection must actually be finding things — a filter that silently matched
            // nothing would make the assertion above vacuous forever.
            Assert.True(
                discovered.Count >= 10,
                $"Only {discovered.Count} public span-taking GEMM entry points were discovered; the "
                    + "reflection filter has probably gone stale.");
        }

        private static bool TakesSpansAndDims(MethodInfo m)
        {
            var ps = m.GetParameters();
            bool span = ps.Any(p =>
                p.ParameterType == typeof(Span<float>) || p.ParameterType == typeof(ReadOnlySpan<float>) ||
                p.ParameterType == typeof(Span<double>) || p.ParameterType == typeof(ReadOnlySpan<double>));
            if (!span) return false;

            var names = ps.Select(p => p.Name).ToList();
            return names.Contains("m") && names.Contains("k") && names.Contains("n");
        }

        // ------------------------------------------------------------------ margin machinery

        private sealed class BoundsViolationException : Exception
        {
            public BoundsViolationException(string message) : base(message) { }
        }

        private static float[] RandF(int length, int seed)
        {
            var rng = new Random(seed);
            var a = new float[length];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            return a;
        }

        private static sbyte[] RandI8(int length, int seed)
        {
            var rng = new Random(seed);
            var a = new sbyte[length];
            for (int i = 0; i < a.Length; i++) a[i] = (sbyte)rng.Next(-127, 128);
            return a;
        }

        /// <summary>Allocates C with a sentinel margin, runs the call, then checks the margin.</summary>
        private static void WithGuardedOutputF(int m, int n, Action<Span<float>> call)
        {
            var c = new float[m * n + Margin];
            for (int i = 0; i < c.Length; i++) c[i] = SentinelF;

            call(c.AsSpan(0, m * n));

            for (int i = 0; i < Margin; i++)
            {
                int at = m * n + i;
                if (c[at] != SentinelF)
                {
                    throw new BoundsViolationException(
                        $"wrote {c[at]} at index {at}, {i + 1} element(s) past the {m * n}-element output");
                }
            }
        }

        private static void RunF(
            int m, int k, int n, int aLen, int bLen,
            Action<float[], float[], Span<float>> call)
        {
            var rng = new Random(20260829);
            var a = new float[aLen];
            var b = new float[bLen];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

            var c = new float[m * n + Margin];
            for (int i = 0; i < c.Length; i++) c[i] = SentinelF;

            call(a, b, c.AsSpan(0, m * n));

            for (int i = 0; i < Margin; i++)
            {
                int at = m * n + i;
                if (c[at] != SentinelF)
                {
                    throw new BoundsViolationException(
                        $"wrote {c[at]} at index {at}, {i + 1} element(s) past the {m * n}-element output");
                }
            }
        }

        private static void RunD(
            int m, int k, int n, int aLen, int bLen,
            Action<double[], double[], Span<double>> call)
        {
            var rng = new Random(20260830);
            var a = new double[aLen];
            var b = new double[bLen];
            for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

            var c = new double[m * n + Margin];
            for (int i = 0; i < c.Length; i++) c[i] = SentinelD;

            call(a, b, c.AsSpan(0, m * n));

            for (int i = 0; i < Margin; i++)
            {
                int at = m * n + i;
                if (c[at] != SentinelD)
                {
                    throw new BoundsViolationException(
                        $"wrote {c[at]} at index {at}, {i + 1} element(s) past the {m * n}-element output");
                }
            }
        }
    }
}
#endif
