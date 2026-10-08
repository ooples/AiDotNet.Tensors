using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// An arena that is never Reset (a training loop without a GradientTape inside an arena opened around the whole fit)
/// must stop growing once its scratch growth since the last Reset passes the budget, handing callers to ordinary
/// allocation instead of retaining every buffer; after a Reset it serves from its ring again.
/// </summary>
[Collection("TensorArenaGlobalState")]
public class TensorArenaGrowthCapTests
{
    [Fact]
    public void GrowthWithoutReset_StopsAtTheBudget_AndResumesAfterReset()
    {
        long saved = TensorArena.UnresetGrowthCapBytes;
        TensorArena.UnresetGrowthCapBytes = 1 << 20;                       // 1 MiB budget
        try
        {
            using var arena = TensorArena.Create();
            int served = 0, declined = 0;
            for (int i = 0; i < 100; i++)                                  // 100 x 64 KiB = 6.25 MiB requested
            {
                var t = arena.TryRentTensor<float>(16384, new[] { 16384 });
                if (t is null) declined++; else served++;
            }
            Assert.True(served <= 17, $"arena kept growing past the budget: served {served} 64 KiB tensors");
            Assert.True(declined >= 80, $"expected the arena to decline once over budget, declined {declined}");
            Assert.True(arena.PeakBackingBytes <= (1 << 20) + 65536 * 2, $"peak {arena.PeakBackingBytes:N0} bytes");

            arena.Reset();
            Assert.NotNull(arena.TryRentTensor<float>(16384, new[] { 16384 }));   // ring reuse after Reset
        }
        finally
        {
            TensorArena.UnresetGrowthCapBytes = saved;
        }
    }
}

/// <summary>The growth-cap test lowers a process-wide budget, so it runs alone.</summary>
[CollectionDefinition("TensorArenaGlobalState", DisableParallelization = true)]
public sealed class TensorArenaGlobalStateCollection { }
