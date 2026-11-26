using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// An activation-checkpoint recompute replays the seeds its forward drew. It used to rewind the engine's shared seed
/// counter and put it back afterwards, so an op another thread ran on the engine meanwhile drew a seed the step had
/// already used, and its own advance was lost when the counter was put back. Then it replayed the positions that
/// followed the one recorded before the forward, which another thread drawing during the forward shifts: the forward
/// drew p+1 and p+3, the replay p+1 and p+2. The forward now records the seeds its thread drew and the recompute
/// replays exactly those, on its thread only.
/// </summary>
public sealed class StochasticSeedReplayTests
{
    private static DirectGpuTensorEngine? TryEngine()
    {
        try { return new DirectGpuTensorEngine(); }
        catch { return null; }
    }

    private static ulong DrawOnAnotherThread(DirectGpuTensorEngine engine)
    {
        ulong seed = 0;
        var other = new Thread(() => seed = engine.NextStochasticSeedForTests());
        other.Start();
        other.Join();
        return seed;
    }

    [SkippableFact]
    public void A_replay_repeats_the_seeds_on_its_thread_and_leaves_other_threads_and_the_sequence_alone()
    {
        var engine = TryEngine();
        Skip.If(engine is null, "No GPU engine can be constructed here");
        using (engine)
        {
            var recorded = new List<long>();
            var first = new List<ulong>();
            using (engine!.RecordStochasticSeeds(recorded))
            {
                first.Add(engine.NextStochasticSeedForTests());
                first.Add(engine.NextStochasticSeedForTests());
            }
            long reached = engine.StochasticSeedPosition;
            ulong otherThreadSeed = 0;
            var replayed = new List<ulong>();
            using (engine.ReplayStochasticSeeds(recorded))
            {
                replayed.Add(engine.NextStochasticSeedForTests());
                otherThreadSeed = DrawOnAnotherThread(engine);
                replayed.Add(engine.NextStochasticSeedForTests());
            }
            Assert.Equal(first, replayed);
            Assert.DoesNotContain(otherThreadSeed, first);
            // The other thread's draw advanced the sequence, and the replay neither rewound nor reset it.
            Assert.Equal(reached + 1, engine.StochasticSeedPosition);
            Assert.DoesNotContain(engine.NextStochasticSeedForTests(), first);
        }
    }

    [SkippableFact]
    public void Another_threads_draw_during_the_forward_does_not_shift_what_the_recompute_replays()
    {
        var engine = TryEngine();
        Skip.If(engine is null, "No GPU engine can be constructed here");
        using (engine)
        {
            var recorded = new List<long>();
            var forward = new List<ulong>();
            ulong interleaved;
            using (engine!.RecordStochasticSeeds(recorded))
            {
                forward.Add(engine.NextStochasticSeedForTests());
                interleaved = DrawOnAnotherThread(engine);   // between the forward's two draws
                forward.Add(engine.NextStochasticSeedForTests());
            }
            var replayed = new List<ulong>();
            using (engine.ReplayStochasticSeeds(recorded))
            {
                replayed.Add(engine.NextStochasticSeedForTests());
                replayed.Add(engine.NextStochasticSeedForTests());
            }
            Assert.Equal(forward, replayed);
            Assert.DoesNotContain(interleaved, replayed);
        }
    }

    [SkippableFact]
    public void A_recompute_that_draws_more_seeds_than_its_forward_throws()
    {
        var engine = TryEngine();
        Skip.If(engine is null, "No GPU engine can be constructed here");
        using (engine)
        {
            var recorded = new List<long>();
            using (engine!.RecordStochasticSeeds(recorded)) engine.NextStochasticSeedForTests();
            using (engine.ReplayStochasticSeeds(recorded))
            {
                engine.NextStochasticSeedForTests();
                Assert.Throws<System.InvalidOperationException>(() => engine.NextStochasticSeedForTests());
            }
        }
    }

    [SkippableFact]
    public void A_recompute_that_draws_fewer_seeds_than_its_forward_is_refused()
    {
        var engine = TryEngine();
        Skip.If(engine is null, "No GPU engine can be constructed here");
        using (engine)
        {
            var recorded = new List<long>();
            using (engine!.RecordStochasticSeeds(recorded))
            {
                engine.NextStochasticSeedForTests();
                engine.NextStochasticSeedForTests();
            }
            using (var replay = engine.ReplayStochasticSeeds(recorded))
            {
                engine.NextStochasticSeedForTests();
                Assert.Throws<System.InvalidOperationException>(() => replay.EnsureAllDrawn());
            }
        }
    }
}
