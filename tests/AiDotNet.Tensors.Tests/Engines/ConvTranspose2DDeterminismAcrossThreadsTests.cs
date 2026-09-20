using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <c>ConvTranspose2D</c>'s GENERIC element path must return bit-identical results whatever the
/// thread budget is, and however the tasks happen to finish — the same contract
/// <see cref="ConvTranspose3DDeterminismAcrossThreadsTests"/> pins for 3D.
///
/// <para><b>Why the element type is <see cref="BFloat16"/> and not float.</b> <c>ConvTranspose2D</c>
/// has three implementations: a gather-shaped float path, a gather-shaped double path, and the
/// generic fallback every other element type takes. The first two were already partitioned over
/// <c>(b, oc)</c> and were never affected. Only the generic branch parallelised over <c>(b, ic)</c> —
/// an axis that does NOT partition the output, because <c>outputIdx</c> depends on <c>oc</c> and not
/// on <c>ic</c> — so every task carried its own full-size accumulator and those accumulators were
/// merged under a lock in whatever order the tasks completed. A float test cannot reach that code at
/// all and would assert nothing about it.</para>
///
/// <para><b>Why bit-identity and not a tolerance.</b> Both accumulation orders are equally correct to
/// within rounding; the property under test is that they are the SAME. A tolerance would pass against
/// the exact defect the test exists to catch. <c>BFloat16</c> keeps 8 mantissa bits, so a reordered
/// sum shows up immediately, and <see cref="BFloat16.RawValue"/> compares the bits exactly on every
/// target framework — unlike <c>Half</c>, which does not exist on net471.</para>
/// </summary>
[Collection("ConvTransposeDeterminismSerial")]
public sealed class ConvTranspose2DDeterminismAcrossThreadsTests : IDisposable
{
    // The engine is pinned to CPU because the partitioning under test is CpuEngine's. PR #333's
    // [ModuleInitializer] promotes Current to a GPU engine when a GPU is present, and on such a host
    // this would otherwise silently exercise a different implementation and assert nothing.
    private readonly IEngine _priorEngine;
    private readonly int _priorMaxDop;
    private readonly bool _priorDeterministicReductions;

    public ConvTranspose2DDeterminismAcrossThreadsTests()
    {
        _priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        _priorMaxDop = CpuParallelSettings.MaxDegreeOfParallelism;

        // Pinned off, not merely assumed off — see the same note on the 3D tests.
        _priorDeterministicReductions = CpuParallelSettings.DeterministicReductions;
        CpuParallelSettings.DeterministicReductions = false;
    }

    public void Dispose()
    {
        CpuParallelSettings.DeterministicReductions = _priorDeterministicReductions;
        CpuParallelSettings.MaxDegreeOfParallelism = _priorMaxDop;
        AiDotNetEngine.Current = _priorEngine;
    }

    // Thread budgets, then REPEATS of earlier budgets. The repeats matter: a merge ordered by
    // completion can return a different answer for the same budget on a later run, and a strictly
    // ascending sweep would never ask that question.
    private static readonly int[] ThreadBudgets = { 1, 2, 3, 4, 1, 4, 3, 2 };

    [Fact]
    public void ConvTranspose2D_generic_path_is_bit_identical_regardless_of_the_thread_budget()
    {
        // THE SHAPE IS PART OF THE TEST. CpuParallelSettings.ParallelForOrSerial runs SERIALLY when
        // the reported work is below PersistentParallelExecutor.DefaultSerialGrainSize (32 * 1024),
        // and the work this path reports is batch*inChannels*height*width. 4*16*32*32 = 65,536
        // clears the gate with room to spare; halve any of those and the test would report
        // "deterministic" without having run in parallel even once. Do not shrink them.
        var input = Fill(new[] { 4, 16, 32, 32 }, seed: 781);
        var kernel = Fill(new[] { 16, 4, 3, 3 }, seed: 782);

        ushort[]? baseline = null;
        foreach (var dop in ThreadBudgets)
        {
            CpuParallelSettings.MaxDegreeOfParallelism = dop;
            var result = Flatten(AiDotNetEngine.Current.ConvTranspose2D(
                input, kernel, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 0, 0 }));

            if (baseline is null)
            {
                baseline = result;
                continue;
            }

            AssertBitIdentical(baseline, result, $"ConvTranspose2D at MaxDegreeOfParallelism {dop} vs 1");
        }
    }

    private static void AssertBitIdentical(IReadOnlyList<ushort> expected, IReadOnlyList<ushort> actual, string context)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (var i = 0; i < expected.Count; i++)
        {
            if (expected[i] != actual[i])
            {
                Assert.Fail(
                    $"{context}: element {i} differs, raw 0x{expected[i]:X4} vs 0x{actual[i]:X4} "
                    + $"({BFloat16.ToFloat(BFloat16.FromRawBits(expected[i])):R} vs "
                    + $"{BFloat16.ToFloat(BFloat16.FromRawBits(actual[i])):R}). The accumulation order "
                    + "followed the thread budget or the order the tasks completed in, not the data.");
            }
        }
    }

    private static ushort[] Flatten(Tensor<BFloat16> tensor)
    {
        var values = new ushort[tensor.Length];
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = tensor[i].RawValue;
        }

        return values;
    }

    /// <summary>Deterministic, spread across several magnitudes so reordering a sum actually shows.</summary>
    /// <remarks>
    /// Values all of one magnitude can sum to the same number in any order and would hide a real
    /// reordering. The range stays well inside BFloat16's, so nothing saturates to infinity — which
    /// would agree across every thread budget and hide the defect just as effectively.
    /// </remarks>
    private static Tensor<BFloat16> Fill(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<BFloat16>(shape);
        for (var i = 0; i < tensor.Length; i++)
        {
            var scale = (float)Math.Pow(10.0, rng.Next(-3, 2));
            tensor[i] = (BFloat16)((float)((rng.NextDouble() - 0.5) * 2.0) * scale);
        }

        return tensor;
    }
}
