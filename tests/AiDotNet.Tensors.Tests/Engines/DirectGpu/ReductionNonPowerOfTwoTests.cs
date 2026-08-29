// Copyright (c) AiDotNet. All rights reserved.
#if !NETFRAMEWORK
using System;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu
{
    /// <summary>
    /// GPU reductions must be exact when the element count is not a power of two.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The work-group size is clamped to the element count, so a 31-element reduction runs with a
    /// local size of 31. The classic tree — <c>for (stride = localSize / 2; stride &gt; 0; stride
    /// &gt;&gt;= 1)</c> — folds indices 0..29 and never touches index 30, so the last element is
    /// dropped entirely.
    /// </para>
    /// <para>
    /// SUM IS THE TEST THAT MATTERS, and Min alone would not have caught the whole class. Min and
    /// Max are idempotent: folding the same element twice, or reading one that has already been
    /// folded away, changes nothing. Addition is not. A tree whose partner bound is the ORIGINAL
    /// work-group size rather than the shrinking ACTIVE count double-counts — at a local size of 6,
    /// the second step merges lane 1 with lane 3, which is outside the live range and still holds
    /// its original value, so that value lands in the total twice. Min survives that; Sum reports a
    /// number that is simply wrong.
    /// </para>
    /// <para>
    /// So the lengths below are odd and small-odd on purpose, and the expected values are exact
    /// integers held exactly in float, which makes any double-count or dropped lane a hard
    /// mismatch rather than a tolerance question.
    /// </para>
    /// </remarks>
    public class ReductionNonPowerOfTwoTests
    {
        public static TheoryData<int> Lengths()
        {
            var data = new TheoryData<int>();
            // 3, 5, 6, 7 exercise the small local sizes where a mis-bounded partner is reachable in
            // the second step; the larger odd ones exercise multi-pass reductions.
            foreach (int n in new[] { 1, 2, 3, 5, 6, 7, 9, 15, 17, 31, 33, 63, 255, 257, 1025 })
                data.Add(n);
            return data;
        }

        [SkippableTheory]
        [MemberData(nameof(Lengths))]
        public void Sum_IsExact_AtAnyLength(int length)
        {
            using var engine = TryCreateGpuEngine();
            Skip.If(engine is null, "No direct GPU backend is available on this host.");
            var backend = engine!.GetBackend();
            Skip.If(backend is null, "No direct GPU backend is available on this host.");

            // 1, 2, 3, ... n : every element distinct, so a dropped or doubled lane cannot cancel
            // out, and the total is exact in float for these sizes.
            var values = new float[length];
            for (int i = 0; i < length; i++) values[i] = i + 1;
            double expected = (double)length * (length + 1) / 2.0;

            using var buffer = backend!.AllocateBuffer(values);
            float actual = backend.Sum(buffer, length);

            Assert.True(
                Math.Abs(actual - expected) < 1e-3,
                $"Sum over {length} elements returned {actual}, expected {expected}. A tree reduction "
                    + "that halves without bounding its partner against the ACTIVE count either drops "
                    + "the last lane or folds an already-consumed one twice; both show up here and "
                    + "neither shows up in Min or Max, which are idempotent.");
        }

        [SkippableTheory]
        [MemberData(nameof(Lengths))]
        public void MinAndMax_SeeEveryLane_AtAnyLength(int length)
        {
            using var engine = TryCreateGpuEngine();
            Skip.If(engine is null, "No direct GPU backend is available on this host.");
            var backend = engine!.GetBackend();
            Skip.If(backend is null, "No direct GPU backend is available on this host.");

            // The extremes sit at the LAST index, which is the lane a truncated tree drops.
            var low = new float[length];
            var high = new float[length];
            for (int i = 0; i < length; i++) { low[i] = 10f; high[i] = 10f; }
            low[length - 1] = -5f;
            high[length - 1] = 99f;

            using var lowBuffer = backend!.AllocateBuffer(low);
            using var highBuffer = backend.AllocateBuffer(high);

            Assert.True(
                backend.Min(lowBuffer, length) == -5f,
                $"Min over {length} elements missed the minimum at index {length - 1}.");
            Assert.True(
                backend.Max(highBuffer, length) == 99f,
                $"Max over {length} elements missed the maximum at index {length - 1}.");
        }

        private static DirectGpuTensorEngine? TryCreateGpuEngine()
        {
            try
            {
                var engine = new DirectGpuTensorEngine();
                if (!engine.IsGpuAvailable)
                {
                    engine.Dispose();
                    return null;
                }

                return engine;
            }
            catch (Exception)
            {
                return null;
            }
        }
    }
}
#endif
