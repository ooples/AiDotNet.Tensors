// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Distributed;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Distributed;

/// <summary>
/// Acceptance tests for issue #215 — ProcessGroup, DDP, FSDP, DTensor,
/// Pipeline, RPC, ElasticLauncher.
///
/// <para>Tests use the in-process backend so the full multi-rank
/// behavior is exercised inside a single test process. Each rank runs
/// on a dedicated <see cref="Task"/>; collective ops resolve through
/// shared barriers in <see cref="InProcessGroupCoordinator"/>. Behaviour
/// is bit-identical to the network backend per the in-process backend's
/// contract.</para>
/// </summary>
public class DistributedTrainingTests
{
    // ── ProcessGroup core ───────────────────────────────────────────────

    [Fact]
    public void AllReduce_Sum_AcrossFourRanks()
    {
        RunRanks(4, (rank, group) =>
        {
            var t = new Tensor<float>(new float[] { rank, rank * 2 }, new[] { 2 });
            group.AllReduce(t, ReduceOp.Sum);
            // Sum of ranks 0..3 = 6; sum of 2*ranks = 12.
            Assert.Equal(6f, t.AsSpan()[0]);
            Assert.Equal(12f, t.AsSpan()[1]);
        });
    }

    [Fact]
    public void AllReduce_Avg_AcrossFourRanks()
    {
        RunRanks(4, (rank, group) =>
        {
            var t = new Tensor<float>(new float[] { rank * 4f }, new[] { 1 });
            group.AllReduce(t, ReduceOp.Avg);
            // (0 + 4 + 8 + 12) / 4 = 6.
            Assert.Equal(6f, t.AsSpan()[0]);
        });
    }

    [Fact]
    public void Broadcast_FromRoot_ReachesAllRanks()
    {
        RunRanks(3, (rank, group) =>
        {
            var t = new Tensor<float>(new[] { 4 });
            if (rank == 1) for (int i = 0; i < 4; i++) t.AsWritableSpan()[i] = 100 + i;
            group.Broadcast(t, root: 1);
            Assert.Equal(new[] { 100f, 101, 102, 103 }, t.AsSpan().ToArray());
        });
    }

    [Fact]
    public void AllGather_EachRankReceivesAllInputs()
    {
        RunRanks(3, (rank, group) =>
        {
            var input = new Tensor<float>(new[] { (float)rank }, new[] { 1 });
            var output = new List<Tensor<float>>(3);
            for (int i = 0; i < 3; i++) output.Add(new Tensor<float>(new[] { 1 }));
            group.AllGather(input, output);
            for (int r = 0; r < 3; r++)
                Assert.Equal((float)r, output[r].AsSpan()[0]);
        });
    }

    [Fact]
    public void ReduceScatter_PerRankSliceReductions()
    {
        // Each rank provides a list of WorldSize tensors, one per dest.
        // Rank R ends up with the SUM of slot[R] across all ranks.
        RunRanks(3, (rank, group) =>
        {
            var input = new List<Tensor<float>>(3);
            for (int r = 0; r < 3; r++)
            {
                var t = new Tensor<float>(new[] { 2 });
                t.AsWritableSpan()[0] = rank * 10 + r;
                t.AsWritableSpan()[1] = rank * 10 + r + 100;
                input.Add(t);
            }
            var output = new Tensor<float>(new[] { 2 });
            group.ReduceScatter(input, output, ReduceOp.Sum);

            // Rank R's slot collects sum of r0[R] + r1[R] + r2[R]:
            //   slot[R][0] = (0 + 10 + 20) + R*3 = 30 + 3R
            //   slot[R][1] = same + 300 = 330 + 3R
            Assert.Equal(30f + 3 * rank, output.AsSpan()[0]);
            Assert.Equal(330f + 3 * rank, output.AsSpan()[1]);
        });
    }

    [Fact]
    public void SendRecv_PointToPoint()
    {
        RunRanks(2, (rank, group) =>
        {
            var t = new Tensor<float>(new[] { 3 });
            if (rank == 0)
            {
                for (int i = 0; i < 3; i++) t.AsWritableSpan()[i] = 7 + i;
                group.Send(t, dst: 1);
            }
            else
            {
                group.Recv(t, src: 0);
                Assert.Equal(new[] { 7f, 8, 9 }, t.AsSpan().ToArray());
            }
        });
    }

    [Fact]
    public void Barrier_NoCrash_AllRanksSync()
    {
        RunRanks(4, (rank, group) =>
        {
            // Just call Barrier on every rank — must not deadlock or throw.
            group.Barrier();
            group.Barrier();
        });
    }

    [Fact]
    public void NewGroup_OnlyContainsListedRanks()
    {
        RunRanks(4, (rank, group) =>
        {
            var sub = group.NewGroup(new[] { 0, 2 });
            if (rank == 0 || rank == 2)
            {
                Assert.NotNull(sub);
                Assert.Equal(2, sub!.WorldSize);
                sub.Dispose();
            }
            else
            {
                Assert.Null(sub);
            }
        });
    }

    // ── DDP ─────────────────────────────────────────────────────────────

    [Fact]
    public void DDP_AveragesGradientsAcrossRanks()
    {
        RunRanks(4, (rank, group) =>
        {
            var ddp = new DistributedDataParallel<float>(group);
            var p = new Tensor<float>(new[] { 4 });
            ddp.RegisterParameter(p);

            // Each rank's "gradient" is rank-dependent.
            var grad = new Tensor<float>(new float[] { rank, rank, rank, rank }, new[] { 4 });
            ddp.ReduceGradients(new[] { grad });

            // Mean of [0, 1, 2, 3] = 1.5.
            for (int i = 0; i < 4; i++)
                Assert.Equal(1.5f, grad.AsSpan()[i]);
        });
    }

    [Fact]
    public void DDP_BucketsParameters_ByByteCap()
    {
        RunRanks(2, (rank, group) =>
        {
            var ddp = new DistributedDataParallel<float>(
                group, new DdpOptions { BucketSizeBytes = 16 /* 4 floats */ });
            // Three params of length 3 each — total 9 floats = 36 bytes.
            // With cap=16 bytes, expect 3 buckets (one per param, since
            // 12 bytes fits but 24 doesn't).
            for (int i = 0; i < 3; i++) ddp.RegisterParameter(new Tensor<float>(new[] { 3 }));

            var grads = new[] {
                new Tensor<float>(new[] { 3 }), new Tensor<float>(new[] { 3 }), new Tensor<float>(new[] { 3 })
            };
            ddp.ReduceGradients(grads);
            Assert.True(ddp.BucketBoundaries.Count >= 1,
                "DDP must produce at least one bucket.");
        });
    }

    [Fact]
    public void DDP_StaticGraph_FreezesParameterRegistration()
    {
        RunRanks(2, (rank, group) =>
        {
            var ddp = new DistributedDataParallel<float>(
                group, new DdpOptions { StaticGraph = true });
            ddp.RegisterParameter(new Tensor<float>(new[] { 4 }));
            ddp.ReduceGradients(new[] { new Tensor<float>(new[] { 4 }) });

            // Plan now frozen — re-register must throw.
            Assert.Throws<InvalidOperationException>(() =>
                ddp.RegisterParameter(new Tensor<float>(new[] { 4 })));
        });
    }

    // ── FSDP ────────────────────────────────────────────────────────────

    [Fact]
    public void FSDP_FullShard_ParametersSharded_GatherReplicates()
    {
        RunRanks(2, (rank, group) =>
        {
            var fsdp = new FullyShardedDataParallel<float>(group);
            // Full param: 8 floats. With WorldSize=2, each rank holds 4.
            var fullParam = new Tensor<float>(new float[] { 0, 1, 2, 3, 4, 5, 6, 7 }, new[] { 8 });
            var sharded = fsdp.RegisterParameter(fullParam);

            Assert.Equal(4, sharded.LocalShard.Length);
            // Rank 0 holds [0..3], rank 1 holds [4..7].
            for (int i = 0; i < 4; i++)
                Assert.Equal((float)(rank * 4 + i), sharded.LocalShard.AsSpan()[i]);

            // GatherParameter all-gathers to recover the full tensor.
            var gathered = fsdp.GatherParameter(sharded);
            for (int i = 0; i < 8; i++)
                Assert.Equal((float)i, gathered.AsSpan()[i]);
        });
    }

    [Fact]
    public void FSDP_NoShard_BehavesLikeReplicated()
    {
        RunRanks(2, (rank, group) =>
        {
            var fsdp = new FullyShardedDataParallel<float>(group,
                new FsdpOptions { Strategy = ShardingStrategy.NoShard });
            var fullParam = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 3 });
            var sharded = fsdp.RegisterParameter(fullParam);
            Assert.Equal(3, sharded.LocalShard.Length); // full size
        });
    }

    [Fact]
    public void FSDP_ReduceScatterGradients_PerRankSlice()
    {
        RunRanks(2, (rank, group) =>
        {
            var fsdp = new FullyShardedDataParallel<float>(group);
            var fullParam = new Tensor<float>(new[] { 8 });
            var sharded = fsdp.RegisterParameter(fullParam);

            // Each rank computes a "full" grad of [r, r, r, r, r, r, r, r].
            var fullGrad = new Tensor<float>(new[] { 8 });
            for (int i = 0; i < 8; i++) fullGrad.AsWritableSpan()[i] = rank;

            var localGrad = fsdp.ReduceScatterGradients(sharded, fullGrad);
            // Reduce-scatter avg over ranks 0,1 = 0.5 in every slot.
            Assert.Equal(4, localGrad.Length);
            for (int i = 0; i < 4; i++)
                Assert.Equal(0.5f, localGrad.AsSpan()[i]);
        });
    }

    // ── DTensor + DeviceMesh ────────────────────────────────────────────

    [Fact]
    public void DeviceMesh_LocalCoordinates_MatchShape()
    {
        RunRanks(6, (rank, group) =>
        {
            var mesh = new DeviceMesh(group, new[] { 2, 3 });
            var coords = mesh.LocalCoordinates();
            Assert.Equal(2, coords.Length);
            // Row-major: rank 4 = (1, 1).
            int reconstructed = coords[0] * 3 + coords[1];
            Assert.Equal(rank, reconstructed);
        });
    }

    [Fact]
    public void DTensor_PartialToReplicate_DoesAllReduce()
    {
        RunRanks(2, (rank, group) =>
        {
            var mesh = new DeviceMesh(group, new[] { 2 });
            // Each rank holds a partial — when we redistribute to Replicate,
            // an all-reduce sums them.
            var local = new Tensor<float>(new float[] { rank + 1 }, new[] { 1 });
            var dt = DTensor<float>.FromLocal(mesh, local, new[] { Placement.Partial });
            var replicated = dt.Redistribute(new[] { Placement.Replicate });
            // Sum of 1 + 2 = 3, on every rank.
            Assert.Equal(3f, replicated.Local.AsSpan()[0]);
        });
    }

    // ── Pipeline ────────────────────────────────────────────────────────

    [Fact]
    public void GPipe_ForwardBackwardThroughTwoStages()
    {
        // Stage 0: out = in + 1.   Stage 1: out = in * 2.
        // With micro-batch [3], stage 0 emits 4, stage 1 emits 8.
        RunRanks(2, (rank, group) =>
        {
            PipelineStage<float> stage = rank == 0
                ? new AdditionStage(0, 2)
                : new MultiplyStage(1, 2);
            var sched = new GPipeSchedule<float>(group, stage);

            IReadOnlyList<Tensor<float>>? micros = null;
            IReadOnlyList<Tensor<float>>? grads = null;
            if (rank == 0)
            {
                micros = new[]
                {
                    new Tensor<float>(new float[] { 3 }, new[] { 1 }),
                    new Tensor<float>(new float[] { 5 }, new[] { 1 }),
                };
            }
            if (rank == 1)
            {
                grads = new[]
                {
                    new Tensor<float>(new float[] { 1 }, new[] { 1 }),
                    new Tensor<float>(new float[] { 1 }, new[] { 1 }),
                };
            }
            var outputs = sched.Run(micros, grads);

            if (rank == 1)
            {
                Assert.Equal(2, outputs.Length);
                // (3+1)*2 = 8, (5+1)*2 = 12.
                Assert.Equal(8f, outputs[0].AsSpan()[0]);
                Assert.Equal(12f, outputs[1].AsSpan()[0]);
            }
        });
    }

    // ── RPC ─────────────────────────────────────────────────────────────

    [Fact]
    public void TensorRpc_RpcSync_DispatchesToTargetRank()
    {
        RunRanks(2, (rank, group) =>
        {
            using var rpc = new TensorRpc<float>(group);
            // Every rank registers an "add_one" handler that adds 1 to every element.
            rpc.Register("add_one", args =>
            {
                var input = args[0];
                var result = new Tensor<float>((int[])input._shape.Clone());
                var dst = result.AsWritableSpan();
                var src = input.AsSpan();
                for (int i = 0; i < src.Length; i++) dst[i] = src[i] + 1;
                return result;
            });
            group.Barrier(); // ensure both ranks have registered before any invokes

            if (rank == 0)
            {
                var arg = new Tensor<float>(new float[] { 5, 7 }, new[] { 2 });
                var result = rpc.RpcSync(dst: 1, "add_one", arg);
                Assert.Equal(6f, result.AsSpan()[0]);
                Assert.Equal(8f, result.AsSpan()[1]);
            }
            // Second barrier prevents rank 1 from falling through the
            // `using var rpc = …` scope (which calls Dispose →
            // UnregisterRank) before rank 0's RpcSync completes. Without
            // this, the test races: rank 1's Dispose can drop its
            // 'add_one' registration before rank 0 looks it up,
            // surfacing as "No RPC handler registered on rank 1".
            group.Barrier();
        });
    }

    // ── Elastic launcher ────────────────────────────────────────────────

    [Fact]
    public void ElasticLauncher_FileBackend_AssignsConsistentRanks()
    {
        var dir = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), $"aidotnet-rendezvous-{Guid.NewGuid():N}");
        var launcher = new ElasticLauncher(new ElasticLauncherOptions
        {
            WorldSize = 4,
            Backend = RendezvousBackend.File,
            Endpoint = dir,
        });
        try
        {
            // Capture each worker's assignment under a lock + verify the
            // distinct-set property: 4 workers must collectively report
            // ranks 0,1,2,3 with world_size=4.
            var assignedRanks = new System.Collections.Concurrent.ConcurrentBag<int>();
            launcher.SpawnInProcessWorkers(a =>
            {
                Assert.Equal(4, a.WorldSize);
                Assert.Equal(dir, a.Endpoint);
                assignedRanks.Add(a.Rank);
            }).Wait();

            var sortedRanks = assignedRanks.OrderBy(r => r).ToArray();
            Assert.Equal(new[] { 0, 1, 2, 3 }, sortedRanks);
        }
        finally
        {
            launcher.CleanupRendezvous();
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    /// <summary>
    /// Spawns <paramref name="worldSize"/> threads, hands each one a
    /// rank handle, runs <paramref name="body"/> on every rank in
    /// parallel, and propagates the first exception. Every test in this
    /// file goes through here.
    /// </summary>
    private static void RunRanks(int worldSize, Action<int, IProcessGroup> body)
    {
        var groups = InProcessGroup.Create(worldSize);
        var tasks = new Task[worldSize];
        var errors = new Exception?[worldSize];
        for (int r = 0; r < worldSize; r++)
        {
            int rank = r;
            tasks[r] = Task.Run(() =>
            {
                try { body(rank, groups[rank]); }
                catch (Exception ex) { errors[rank] = ex; }
                finally { groups[rank].Dispose(); }
            });
        }
        // Hard-fail on timeout — earlier versions of this helper let
        // deadlocked workers silently report "passed" because the lambda
        // hadn't thrown by the time WaitAll returned.
        bool completed = Task.WaitAll(tasks, TimeSpan.FromSeconds(30));
        if (!completed)
            throw new Xunit.Sdk.XunitException(
                $"Distributed test timed out after 30s — at least one rank deadlocked. " +
                $"Tasks status: [{string.Join(", ", tasks.Select(t => t.Status))}].");
        for (int r = 0; r < worldSize; r++)
            if (errors[r] is { } e) throw new Xunit.Sdk.XunitException(
                $"Rank {r} threw: {e.Message}\n{e.StackTrace}");
    }

    private sealed class AdditionStage : PipelineStage<float>
    {
        public AdditionStage(int idx, int total) : base(idx, total) { }
        public override Tensor<float> Forward(Tensor<float> input)
        {
            var output = new Tensor<float>((int[])input._shape.Clone());
            var dst = output.AsWritableSpan();
            var src = input.AsSpan();
            for (int i = 0; i < src.Length; i++) dst[i] = src[i] + 1;
            return output;
        }
        public override Tensor<float> Backward(Tensor<float> gradOutput) => gradOutput;
    }

    private sealed class MultiplyStage : PipelineStage<float>
    {
        public MultiplyStage(int idx, int total) : base(idx, total) { }
        public override Tensor<float> Forward(Tensor<float> input)
        {
            var output = new Tensor<float>((int[])input._shape.Clone());
            var dst = output.AsWritableSpan();
            var src = input.AsSpan();
            for (int i = 0; i < src.Length; i++) dst[i] = src[i] * 2;
            return output;
        }
        public override Tensor<float> Backward(Tensor<float> gradOutput) => gradOutput;
    }
}
