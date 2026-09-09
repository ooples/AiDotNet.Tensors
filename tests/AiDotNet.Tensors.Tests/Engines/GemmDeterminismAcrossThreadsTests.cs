using System;
using System.Threading;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The same GEMM, run twice in one process, must return bit-identical results.
///
/// <para><b>What is already covered, and what is not.</b> <c>DeterministicByDefaultTests</c> asserts the
/// determinism CONTRACT at the level of flags and options — that <c>Deterministic</c> defaults to true, that
/// <c>BlasProvider.IsDeterministicMode</c> follows it, that a plan is reused. Nothing there runs a GEMM and
/// compares numbers, so a dispatcher that honours every flag and still returns different arithmetic on two
/// calls would pass all of it.</para>
///
/// <para><b>Why that gap is worth closing.</b> <c>BlasManaged</c> chooses between kernels on properties that
/// are not fixed by the caller: whether pre-packed operands were supplied, the thread budget, and which ISA
/// the CPU reports (<c>Avx.IsSupported</c>, <c>Avx2.IsSupported &amp;&amp; Nr == 16</c>). Those kernels block
/// the reduction differently, and floating-point addition is not associative, so two "numerically equivalent"
/// routes legitimately disagree in the last bits. The GotoGemm path documents itself as "computed by one
/// thread in fixed K order ⇒ thread-count-independent, Deterministic/DisableAutotune-contract safe", which
/// says plainly that being thread-count-independent is a property some paths have and not others.</para>
///
/// <para><b>What prompted it.</b> A consumer's Clone() test compares a model against its clone through a
/// 22-step denoising sampler, where a last-bit disagreement compounds. It fails intermittently on shared CI
/// and passes on idle machines — the signature of a run-to-run difference rather than a clone defect. These
/// tests check the mechanism directly, on whatever CPU they run on, instead of relying on a long sampler to
/// amplify it into view somewhere else.</para>
/// </summary>
/// <summary>Serialised: the thread-budget test mutates the process-wide ThreadPool ceiling.</summary>
[CollectionDefinition("GemmDeterminismSerial", DisableParallelization = true)]
public sealed class GemmDeterminismSerialCollection { }

[Collection("GemmDeterminismSerial")]
public sealed class GemmDeterminismAcrossThreadsTests
{
    [Theory]
    [InlineData(64, 64, 64)]
    [InlineData(128, 96, 112)]
    // Large enough to clear GotoGemmFp32.ParallelMinWork, so the parallel regime is actually exercised
    // rather than every case quietly taking one small-shape path.
    [InlineData(256, 256, 256)]
    public void The_same_gemm_twice_in_one_process_is_bit_identical(int m, int k, int n)
    {
        var a = Fill(m * k, seed: 11);
        var b = Fill(k * n, seed: 29);

        var first = new float[m * n];
        var second = new float[m * n];

#pragma warning disable CS0618 // Sgemm forwards to BlasManaged.Gemm - exactly the dispatcher under test.
        SimdGemm.Sgemm(a, b, first, m, k, n);
        SimdGemm.Sgemm(a, b, second, m, k, n);
#pragma warning restore CS0618

        AssertBitIdentical(first, second, $"repeat call at {m}x{k}x{n}");
    }

    [Theory]
    [InlineData(192, 160, 176)]
    [InlineData(256, 256, 256)]
    public void A_gemm_is_bit_identical_regardless_of_the_thread_budget(int m, int k, int n)
    {
        // THE HYPOTHESIS THIS EXISTS TO TEST. A parallel reduction whose accumulation order follows the thread
        // count is not deterministic between a quiet machine and a loaded one - and a loaded shared CI runner
        // is exactly where the downstream symptom appears. If this fails, the determinism contract is not
        // being honoured by whichever kernel these shapes select, and the Clone() divergence downstream needs
        // no clone-specific explanation at all.
        var a = Fill(m * k, seed: 7);
        var b = Fill(k * n, seed: 13);

        var baseline = RunWithMaxThreads(a, b, m, k, n, threads: 1);

        foreach (var threads in new[] { 2, 4, Environment.ProcessorCount })
        {
            if (threads <= 1)
            {
                continue;
            }

            var candidate = RunWithMaxThreads(a, b, m, k, n, threads);
            AssertBitIdentical(baseline, candidate, $"{m}x{k}x{n} at {threads} threads vs 1");
        }
    }

    /// <summary>
    /// Runs the GEMM with the process's worker-thread ceiling lowered, then restores it.
    /// </summary>
    /// <remarks>
    /// The thread budget is changed around the call rather than passed in, because the dispatcher decides its
    /// own parallelism from the default (all-core) budget - which is the configuration production uses and the
    /// one whose determinism is being claimed.
    /// </remarks>
    private static float[] RunWithMaxThreads(float[] a, float[] b, int m, int k, int n, int threads)
    {
        ThreadPool.GetMaxThreads(out var workers, out var io);
        ThreadPool.GetMinThreads(out var minWorkers, out var minIo);
        var result = new float[m * n];

        try
        {
            ThreadPool.SetMinThreads(Math.Min(minWorkers, threads), minIo);
            ThreadPool.SetMaxThreads(Math.Max(1, threads), io);

#pragma warning disable CS0618
            SimdGemm.Sgemm(a, b, result, m, k, n);
#pragma warning restore CS0618
        }
        finally
        {
            ThreadPool.SetMinThreads(minWorkers, minIo);
            ThreadPool.SetMaxThreads(workers, io);
        }

        return result;
    }

    /// <summary>
    /// Bit-identical, not "close".
    /// </summary>
    /// <remarks>
    /// A tolerance here would defeat the purpose: the question is not whether two runs are both approximately
    /// correct - they are - but whether they are the SAME, because a consumer comparing a model to its clone
    /// through an iterative sampler compounds any difference at all.
    /// </remarks>
    private static void AssertBitIdentical(float[] expected, float[] actual, string context)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            if (Bits(expected[i]) != Bits(actual[i]))
            {
                Assert.Fail(
                    $"{context}: element {i} differs, {expected[i]:R} vs {actual[i]:R} "
                    + $"(delta {Math.Abs((double)expected[i] - actual[i]):E3}). The dispatcher selected a "
                    + "different accumulation order for two runs of the same multiplication.");
            }
        }
    }

    /// <summary>
    /// Raw bits of a float, portably.
    /// </summary>
    /// <remarks>
    /// Not BitConverter.SingleToInt32Bits: this project multi-targets net471, where that overload does not
    /// exist. GetBytes allocates, which does not matter in a test and does matter less than the build failing
    /// on one framework nobody ran locally.
    /// </remarks>
    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    /// <summary>Deterministic, spread across several magnitudes so reordering a sum actually shows.</summary>
    /// <remarks>
    /// Values all of one magnitude can sum to the same float in any order and would hide a real reordering.
    /// </remarks>
    private static float[] Fill(int count, int seed)
    {
        var rng = new Random(seed);
        var values = new float[count];
        for (var i = 0; i < count; i++)
        {
            var scale = MathF.Pow(10f, rng.Next(-3, 4));
            values[i] = (float)((rng.NextDouble() - 0.5) * 2.0) * scale;
        }

        return values;
    }
}
