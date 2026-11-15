using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class CudaArenaRangeReviewTests
{
    [Fact]
    public void Ten_thousand_reused_slots_require_linear_metadata()
    {
        const int count = 10000;
        var offsets = new long[count];
        var sizes = Enumerable.Repeat(256L, count).ToArray();
        var occupancy = new CudaStepArena.Block(IntPtr.Zero, IntPtr.Zero, 0, count);
        for (int i = 0; i < count; i++)
        {
            Assert.True(occupancy.TryClaim(i, 0, 256));
            Assert.False(occupancy.TryClaim((i + 1) % count, 0, 256));
            occupancy.Release(i);
            occupancy.SlotLive[i] = 0;
        }
    }

    [Fact]
    public void Occupancy_matches_live_intervals_through_random_returns()
    {
        var rng = new Random(23);
        var offsets = Enumerable.Range(0, 120).Select(_ => (long)rng.Next(50) * 256).ToArray();
        var sizes = Enumerable.Range(0, 120).Select(_ => (long)rng.Next(1, 10) * 256).ToArray();
        var live = new bool[120];
        var occupancy = new CudaStepArena.Block(IntPtr.Zero, IntPtr.Zero, 0, live.Length);
        for (int n = 0; n < 10000; n++)
        {
            int i = rng.Next(live.Length);
            if (live[i]) { occupancy.Release(i); occupancy.SlotLive[i] = 0; live[i] = false; continue; }
            bool expected = !Enumerable.Range(0, live.Length).Any(j => live[j]
                && offsets[j] < offsets[i] + sizes[i] && offsets[i] < offsets[j] + sizes[j]);
            Assert.Equal(expected, occupancy.TryClaim(i, offsets[i], offsets[i] + sizes[i]));
            live[i] = expected;
        }
    }
}
