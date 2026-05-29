// Copyright (c) AiDotNet. All rights reserved.
// Pins the deterministic-AND-parallel contract that ManagedBlas relies on to
// exceed PyTorch (whose deterministic mode sacrifices parallelism). The audit
// (ManagedBlas determinism plan, Phase 0) established that in deterministic mode
// EVERY managed GEMM parallel path splits work over disjoint-write output tiles
// (M / N / MN_2D) and performs each output element's full K-reduction on a single
// thread in fixed order — so the result is bit-identical no matter how many
// threads (i.e. how many partitions) run. The non-associative K-axis split is
// gated off in deterministic mode by AxisSelector.
//
// This test makes that contract executable: the SAME GEMM run at NumThreads
// 1 / 2 / 4 / 16 (which changes the M/N partition boundaries) must produce a
// BIT-IDENTICAL result. If a future change introduces a cross-thread K-reduction
// (or a thread-count-dependent partition of the reduction) into a
// deterministic-mode path, the exact-equality assertion fails.
//
// Scope: bit-exact across THREAD COUNTS on one machine/ISA. Cross-ISA bit-exact
// (AVX-512 vs AVX2 vs scalar vs Neon, which reduce at different vector widths) is
// out of scope — PyTorch makes no such guarantee either.

using System;
using Xunit;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

// Toggles process-wide deterministic mode; serialize against the other
// determinism-sensitive tests.
[Collection("BlasManaged-Stats-Serial")]
public class DeterministicParallelGemmContractTests
{
    // Partition counts to compare. 1 forces the serial path (AxisSelector → None);
    // 2/4/16 force progressively finer M/N partitions. All must agree bit-for-bit.
    private static readonly int[] ThreadCounts = { 1, 2, 4, 16 };

    public static TheoryData<int, int, int> Shapes => new()
    {
        // (m, n, k) covering the strategy regimes the audit enumerated:
        { 128, 128, 16 },   // small-K  → StreamingStrategy (no packing)
        { 256, 256, 256 },  // square   → PackBothStrategy (M-axis / 2D)
        { 512, 16,  512 },  // tall-skinny (N small) → M-axis
        { 16,  512, 256 },  // M-small / N-large → N-axis
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Gemm_Float_BitIdentical_AcrossThreadCounts(int m, int n, int k)
        => AssertBitIdenticalAcrossThreadCounts<float>(m, n, k, seed: 11);

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Gemm_Double_BitIdentical_AcrossThreadCounts(int m, int n, int k)
        => AssertBitIdenticalAcrossThreadCounts<double>(m, n, k, seed: 22);

    private static void AssertBitIdenticalAcrossThreadCounts<T>(int m, int n, int k, int seed)
        where T : unmanaged
    {
        // Clear any thread-local determinism override first, else SetDeterministicMode
        // (process-wide) wouldn't guarantee this thread actually runs deterministic
        // (the override wins). Capture/restore it separately from the process-wide value.
        bool? beforeThreadDet = BlasProvider.GetThreadLocalDeterministicMode();
        if (beforeThreadDet is not null) BlasProvider.SetThreadLocalDeterministicMode(null);
        bool beforeDet = BlasProvider.IsDeterministicMode;
        int beforeMax = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            // Deterministic mode is the contract under test (it excludes the
            // non-associative K-axis split). Ensure the executor is actually
            // allowed to run multi-threaded so the parallel partitions execute
            // in parallel (not collapsed to serial by a pinned MaxDegree).
            BlasProvider.SetDeterministicMode(true);
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(4, Environment.ProcessorCount);

            var a = RandomArray<T>(m * k, seed);
            var b = RandomArray<T>(k * n, seed + 1);

            T[]? reference = null;
            int referenceThreads = 0;
            foreach (int threads in ThreadCounts)
            {
                var c = new T[m * n];
                // NumThreads drives `procs` inside the strategies → the M/N
                // partition boundaries. 1 → serial; >1 → that many partitions.
                var options = new BlasOptions<T> { NumThreads = threads };
                BlasManagedLib.Gemm<T>(a, k, false, b, n, false, c, n, m, n, k, options);

                if (reference is null)
                {
                    reference = c;
                    referenceThreads = threads;
                    continue;
                }

                for (int i = 0; i < c.Length; i++)
                {
                    // Bitwise comparison, not value-equality: the contract is BIT-identical.
                    // .Equals treats +0.0 == -0.0 (and would also mishandle NaN), which could
                    // mask a deterministic-drift regression.
                    if (!BitIdentical(reference[i], c[i]))
                    {
                        Assert.Fail(
                            $"Deterministic GEMM drifted with thread-count change " +
                            $"({referenceThreads} → {threads}) for {typeof(T).Name} " +
                            $"[{m}x{n}x{k}] at index {i}: {reference[i]} vs {c[i]}. " +
                            "A deterministic-mode path introduced a thread-count-dependent " +
                            "reduction (cross-thread K split?) — the disjoint-write contract broke.");
                    }
                }
            }
        }
        finally
        {
            BlasProvider.SetDeterministicMode(beforeDet);
            BlasProvider.SetThreadLocalDeterministicMode(beforeThreadDet);
            CpuParallelSettings.MaxDegreeOfParallelism = beforeMax;
        }
    }

    /// <summary>Bit-for-bit equality (reinterpreted integer bits), so +0.0/-0.0 and
    /// NaN payloads count as different — the contract is bit-identical, not value-equal.</summary>
    private static bool BitIdentical<T>(T x, T y) where T : unmanaged
    {
        if (typeof(T) == typeof(float))
            return FloatBits((float)(object)x) == FloatBits((float)(object)y);
        if (typeof(T) == typeof(double))
            return BitConverter.DoubleToInt64Bits((double)(object)x) == BitConverter.DoubleToInt64Bits((double)(object)y);
        throw new NotSupportedException($"Unsupported element type {typeof(T).Name}.");
    }

    // BitConverter.SingleToInt32Bits is net5+; GetBytes→ToInt32 is the net471-safe
    // equivalent (allocates a 4-byte buffer, fine for a test).
    private static int FloatBits(float f) => BitConverter.ToInt32(BitConverter.GetBytes(f), 0);

    private static T[] RandomArray<T>(int len, int seed) where T : unmanaged
    {
        var rng = new Random(seed);
        var arr = new T[len];
        if (typeof(T) == typeof(float))
        {
            var f = (float[])(object)arr;
            for (int i = 0; i < len; i++) f[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }
        else if (typeof(T) == typeof(double))
        {
            var d = (double[])(object)arr;
            for (int i = 0; i < len; i++) d[i] = rng.NextDouble() * 2.0 - 1.0;
        }
        else
        {
            throw new NotSupportedException($"Unsupported element type {typeof(T).Name}.");
        }
        return arr;
    }
}
