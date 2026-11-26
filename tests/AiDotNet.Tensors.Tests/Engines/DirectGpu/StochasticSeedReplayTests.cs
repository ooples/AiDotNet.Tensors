using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// An activation-checkpoint recompute replays the seeds its forward drew. It used to rewind the engine's shared seed
/// counter and put it back afterwards, so an op another thread ran on the engine meanwhile drew a seed the step had
/// already used, and its own advance was lost when the counter was put back. The replay is now the recomputing
/// thread's only.
/// </summary>
public sealed class StochasticSeedReplayTests
{
    [SkippableFact]
    public void A_replay_repeats_the_seeds_on_its_thread_and_leaves_other_threads_and_the_sequence_alone()
    {
        DirectGpuTensorEngine engine;
        try { engine = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU engine can be constructed here"); return; }
        using (engine)
        {
            long start = engine.StochasticSeedPosition;
            var first = new List<ulong> { engine.NextStochasticSeedForTests(), engine.NextStochasticSeedForTests() };
            long reached = engine.StochasticSeedPosition;

            ulong otherThreadSeed = 0;
            var replayed = new List<ulong>();
            using (engine.ReplayStochasticSeedsFrom(start))
            {
                replayed.Add(engine.NextStochasticSeedForTests());
                var other = new Thread(() => otherThreadSeed = engine.NextStochasticSeedForTests());
                other.Start();
                other.Join();
                replayed.Add(engine.NextStochasticSeedForTests());
            }

            Assert.Equal(first, replayed);
            Assert.DoesNotContain(otherThreadSeed, first);
            // The other thread's draw advanced the sequence, and the replay neither rewound nor reset it.
            Assert.Equal(reached + 1, engine.StochasticSeedPosition);
            Assert.DoesNotContain(engine.NextStochasticSeedForTests(), first);
        }
    }
}
