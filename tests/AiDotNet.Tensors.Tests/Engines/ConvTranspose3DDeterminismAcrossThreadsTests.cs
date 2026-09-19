using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>Serialised: these tests mutate the process-wide <c>CpuParallelSettings.MaxDegreeOfParallelism</c>.</summary>
[CollectionDefinition("ConvTranspose3DDeterminismSerial", DisableParallelization = true)]
public sealed class ConvTranspose3DDeterminismSerialCollection { }

/// <summary>
/// <c>ConvTranspose3D</c> and its kernel gradient must return bit-identical results whatever the
/// thread budget is, and however the tasks happen to finish.
///
/// <para><b>What was wrong.</b> Both paths used to parallelise over an axis that does NOT partition the
/// output — the forward over <c>(b, ic)</c> while <c>outputIdx</c> depends on <c>oc</c>, the kernel gradient
/// over <c>(b, ic)</c> while <c>kernelIdx</c> depends on <c>oc</c> and not on <c>b</c>. Every task therefore
/// needed its own full-size accumulator, and those accumulators were merged into the shared result under a
/// lock — so the order the partial sums were added in was whatever order the tasks happened to complete in.
/// Floating-point addition is not associative, so the same call could return different bits on two runs of
/// one process, and different bits again on a host with a different core count.</para>
///
/// <para><b>Why bit-identity and not a tolerance.</b> Both answers are equally "correct" to within rounding;
/// the property under test is that they are the SAME. A consumer comparing a model against its clone through
/// an iterative denoising sampler compounds a last-bit disagreement over every step, which is how this
/// surfaced: intermittently, only on loaded shared CI, never on an idle machine. A tolerance here would pass
/// against the exact defect the test exists to catch.</para>
/// </summary>
[Collection("ConvTranspose3DDeterminismSerial")]
public sealed class ConvTranspose3DDeterminismAcrossThreadsTests : IDisposable
{
    // The engine is pinned to CPU because the partitioning under test is CpuEngine's. PR #333's
    // [ModuleInitializer] promotes Current to a GPU engine when a GPU is present, and on such a host
    // this would otherwise silently exercise a different implementation and assert nothing.
    private readonly IEngine _priorEngine;
    private readonly int _priorMaxDop;

    public ConvTranspose3DDeterminismAcrossThreadsTests()
    {
        _priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        _priorMaxDop = CpuParallelSettings.MaxDegreeOfParallelism;
    }

    public void Dispose()
    {
        CpuParallelSettings.MaxDegreeOfParallelism = _priorMaxDop;
        AiDotNetEngine.Current = _priorEngine;
    }

    // Thread budgets, then REPEATS of earlier budgets. The repeats matter: a merge ordered by completion
    // can return a different answer for the same budget on a later run, and a strictly ascending sweep
    // would never ask that question.
    private static readonly int[] ThreadBudgets = { 1, 2, 3, 4, 1, 4, 3, 2 };

    [Fact]
    public void ConvTranspose3D_is_bit_identical_regardless_of_the_thread_budget()
    {
        // 4 * 16 * 18^3 = 373,248 output elements, comfortably past the serial grain gate described on the
        // gradient test below, so the parallel path is the one actually being measured.
        var input = Fill(new[] { 4, 16, 16, 16, 16 }, seed: 777);
        var kernel = Fill(new[] { 16, 16, 3, 3, 3 }, seed: 778);

        float[]? baseline = null;
        foreach (var dop in ThreadBudgets)
        {
            CpuParallelSettings.MaxDegreeOfParallelism = dop;
            var result = Flatten(AiDotNetEngine.Current.ConvTranspose3D(
                input, kernel, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }, new[] { 0, 0, 0 }));

            if (baseline is null)
            {
                baseline = result;
                continue;
            }

            AssertBitIdentical(baseline, result, $"ConvTranspose3D at MaxDegreeOfParallelism {dop} vs 1");
        }
    }

    [Fact]
    public void ConvTranspose3DBackwardKernel_is_bit_identical_regardless_of_the_thread_budget()
    {
        // THE SHAPE IS PART OF THE TEST. CpuParallelSettings.ParallelForOrSerial runs SERIALLY whenever the
        // reported work is below PersistentParallelExecutor.DefaultSerialGrainSize (32 * 1024), and the work
        // this path reports is the gradient buffer's length. A smaller kernel never reaches the parallel gate
        // at all: it would report "deterministic" without having run in parallel even once — a dead control
        // that passes against the unfixed engine. 32*32*4*4*4 = 65,536 clears the gate; the 16*16*3*3*3 =
        // 6,912 kernel used by the forward test above does NOT. Do not shrink these.
        var kernelShape = new[] { 32, 32, 4, 4, 4 };
        var input = Fill(new[] { 2, 32, 12, 12, 12 }, seed: 779);
        var gradOutput = Fill(new[] { 2, 32, 15, 15, 15 }, seed: 780);

        float[]? baseline = null;
        foreach (var dop in ThreadBudgets)
        {
            CpuParallelSettings.MaxDegreeOfParallelism = dop;
            var result = Flatten(AiDotNetEngine.Current.ConvTranspose3DBackwardKernel(
                gradOutput, input, kernelShape, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }));

            if (baseline is null)
            {
                baseline = result;
                continue;
            }

            AssertBitIdentical(baseline, result, $"ConvTranspose3DBackwardKernel at MaxDegreeOfParallelism {dop} vs 1");
        }
    }

    private static void AssertBitIdentical(IReadOnlyList<float> expected, IReadOnlyList<float> actual, string context)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            if (Bits(expected[i]) != Bits(actual[i]))
            {
                Assert.Fail(
                    $"{context}: element {i} differs, {expected[i]:R} vs {actual[i]:R} "
                    + $"(delta {Math.Abs((double)expected[i] - actual[i]):E3}). The accumulation order followed "
                    + "the thread budget or the order the tasks completed in, not the data.");
            }
        }
    }

    /// <summary>
    /// Raw bits of a float, portably.
    /// </summary>
    /// <remarks>
    /// Not BitConverter.SingleToInt32Bits: this project multi-targets net471, where that overload does not
    /// exist.
    /// </remarks>
    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    private static float[] Flatten(Tensor<float> tensor)
    {
        var values = new float[tensor.Length];
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = tensor[i];
        }

        return values;
    }

    /// <summary>Deterministic, spread across several magnitudes so reordering a sum actually shows.</summary>
    /// <remarks>
    /// Values all of one magnitude can sum to the same float in any order and would hide a real reordering.
    /// </remarks>
    private static Tensor<float> Fill(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<float>(shape);
        for (var i = 0; i < tensor.Length; i++)
        {
            var scale = (float)Math.Pow(10.0, rng.Next(-3, 4));
            tensor[i] = (float)((rng.NextDouble() - 0.5) * 2.0) * scale;
        }

        return tensor;
    }
}
