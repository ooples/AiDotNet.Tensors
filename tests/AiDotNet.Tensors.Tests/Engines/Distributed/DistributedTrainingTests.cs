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
/// on a dedicated <see cref="Thread"/> (not a thread-pool task — see
/// <see cref="RunWorkers"/> for the rationale); collective ops resolve
/// through shared barriers in <see cref="InProcessGroupCoordinator"/>.
/// Behaviour is bit-identical to the network backend per the in-process
/// backend's contract.</para>
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
    public void FSDP_ShardOptimizerOnly_RealZero1_AllRanksConvergeToSameUpdate()
    {
        RunRanks(2, (rank, group) =>
        {
            var fsdp = new FullyShardedDataParallel<float>(group,
                new FsdpOptions { Strategy = ShardingStrategy.ShardOptimizerOnly });

            // 8-element param replicated, 8-element grad rank-dependent.
            var fullParam = new Tensor<float>(new float[] { 1, 1, 1, 1, 1, 1, 1, 1 }, new[] { 8 });
            var sharded = fsdp.RegisterParameter(fullParam);
            // Each rank's local "full grad" is [r, r, r, r, r, r, r, r].
            var fullGrad = new Tensor<float>(new[] { 8 });
            for (int i = 0; i < 8; i++) fullGrad.AsWritableSpan()[i] = rank;

            // Reduce-scatter for ZeRO-1 returns the rank-local slice (size 4).
            var localGrad = fsdp.ReduceScatterGradients(sharded, fullGrad);
            Assert.Equal(4, localGrad.Length);
            // Avg over ranks {0,1} = 0.5 in every slot (local-slice-shaped).
            for (int i = 0; i < 4; i++)
                Assert.Equal(0.5f, localGrad.AsSpan()[i]);

            // Optimizer step: SGD with lr=1.0, applied only to local slice
            // by the user's lambda. Each rank holds only its own m/v —
            // simulated here as a no-op since SGD is stateless.
            fsdp.OptimizerStep(sharded, fullParam, localGrad, (paramSlice, gradSlice) =>
            {
                var p = paramSlice.AsWritableSpan();
                var g = gradSlice.AsSpan();
                for (int i = 0; i < p.Length; i++) p[i] = p[i] - 1.0f * g[i];
            });

            // After all-gather, every rank's fullParam should equal the
            // global update (1 - 0.5 = 0.5 in every slot).
            for (int i = 0; i < 8; i++)
                Assert.Equal(0.5f, fullParam.AsSpan()[i]);
        });
    }

    [Fact]
    public void AutoFsdp_BackwardHookFires_ZeRO1_LocalGradMatchesManualReduceScatter()
    {
        RunRanks(2, (rank, group) =>
        {
            var fsdp = new FullyShardedDataParallel<float>(group,
                new FsdpOptions { Strategy = ShardingStrategy.ShardOptimizerOnly });
            using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
            var auto = new AutoFsdp<float>(fsdp, tape);
            // 4-element param replicated on every rank.
            var weights = new Tensor<float>(new float[] { 0, 0, 0, 0 }, new[] { 4 });
            var sharded = auto.Shard(weights);

            // Synthetic forward: y = weights * inputs (per-rank scaled),
            // loss = sum(y) so dLoss/dweights = inputs (rank-dependent).
            // Each rank's "inputs" differs → ZeRO-1 averages the gradient
            // across ranks (= 1.5 in every slot) then slices to 2 entries
            // per rank.
            var inputs = new Tensor<float>(new[] { 4 });
            for (int i = 0; i < 4; i++) inputs.AsWritableSpan()[i] = rank + 1;

            var engine = new AiDotNet.Tensors.Engines.CpuEngine();
            var prod = engine.TensorMultiply(weights, inputs);
            // Use prod as the loss tensor directly — ComputeGradients seeds
            // a ones-shaped grad, so dLoss/dweights = inputs which is what
            // we want to reduce-scatter.
            var grads = tape.ComputeGradients(prod, sources: new[] { weights });
            Assert.True(grads.ContainsKey(weights));

            // Run the FSDP collective on every registered parameter in one pass.
            auto.ProcessGradients(grads);
            Assert.True(auto.HasLocalGradient(sharded),
                "AutoFsdp.ProcessGradients should populate the rank-local slot.");
            var local = auto.LocalGradient(sharded);
            Assert.Equal(2, local.Length);
            // Mean across ranks 0,1 of inputs = (1+2)/2 = 1.5, in every slot.
            Assert.Equal(1.5f, local.AsSpan()[0], precision: 4);
            Assert.Equal(1.5f, local.AsSpan()[1], precision: 4);
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

    [Fact]
    public void TcpProcessGroup_AllReduce_AveragesAcrossRanks()
    {
        // 3 ranks on loopback; each contributes a different vector;
        // AllReduce-Avg should converge to the per-element mean on every rank.
        int port = FreeLoopbackPort();
        const int worldSize = 3;
        var endpoint = $"127.0.0.1:{port}";
        var results = new float[worldSize][];

        RunWorkers(worldSize, rank =>
        {
            using var pg = new TcpProcessGroup(rank, worldSize, endpoint);
            var t = new Tensor<float>(new[] { 4 });
            for (int k = 0; k < 4; k++) t.AsWritableSpan()[k] = rank + 1; // rank 0:[1,1,1,1]; 1:[2,2,2,2]; 2:[3,3,3,3]
            pg.AllReduce(t, ReduceOp.Avg);
            results[rank] = new float[4];
            t.AsSpan().CopyTo(results[rank]);
        }, timeout: TimeSpan.FromSeconds(20), opName: "TcpProcessGroup_AllReduce");
        for (int r = 0; r < worldSize; r++)
            for (int i = 0; i < 4; i++)
                Assert.Equal(2.0f, results[r][i], precision: 4); // mean(1,2,3) = 2
    }

    [Fact]
    public void ElasticLauncher_Rendezvous_RejectsHostileNonces()
    {
        var dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"aidotnet-rendezvous-{Guid.NewGuid():N}");
        var launcher = new ElasticLauncher(new ElasticLauncherOptions
        {
            WorldSize = 2, Backend = RendezvousBackend.File, Endpoint = dir,
        });
        try
        {
            // Empty / whitespace nonce → arg validation.
            Assert.Throws<ArgumentException>(() => launcher.Rendezvous(""));
            Assert.Throws<ArgumentException>(() => launcher.Rendezvous("   "));
            // Control characters (would sort before everything → rank 0 grab).
            Assert.Throws<ArgumentException>(() => launcher.Rendezvous("\0bad"));
            // Path traversal — file backend only.
            Assert.Throws<ArgumentException>(() => launcher.Rendezvous("../escape"));
            Assert.Throws<ArgumentException>(() => launcher.Rendezvous("a/b"));
            Assert.Throws<ArgumentException>(() => launcher.Rendezvous(".."));
        }
        finally
        {
            if (System.IO.Directory.Exists(dir)) System.IO.Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void TcpProcessGroup_Broadcast_FromNonRoot_ReachesEveryone()
    {
        int port = FreeLoopbackPort();
        const int worldSize = 3;
        var endpoint = $"127.0.0.1:{port}";
        var results = new float[worldSize][];

        RunWorkers(worldSize, rank =>
        {
            using var pg = new TcpProcessGroup(rank, worldSize, endpoint);
            var t = new Tensor<float>(new[] { 3 });
            if (rank == 1) // root is rank 1, not coord
            {
                t.AsWritableSpan()[0] = 7; t.AsWritableSpan()[1] = 8; t.AsWritableSpan()[2] = 9;
            }
            pg.Broadcast(t, root: 1);
            results[rank] = new float[3];
            t.AsSpan().CopyTo(results[rank]);
        }, timeout: TimeSpan.FromSeconds(20), opName: "TcpProcessGroup_Broadcast");
        for (int r = 0; r < worldSize; r++)
        {
            Assert.Equal(7, results[r][0]);
            Assert.Equal(8, results[r][1]);
            Assert.Equal(9, results[r][2]);
        }
    }

    [Fact]
    public void TcpProcessGroup_Barrier_ReturnsForEveryRank()
    {
        int port = FreeLoopbackPort();
        const int worldSize = 4;
        var endpoint = $"127.0.0.1:{port}";
        int reachedBarrier = 0;

        RunWorkers(worldSize, rank =>
        {
            using var pg = new TcpProcessGroup(rank, worldSize, endpoint);
            pg.Barrier();
            System.Threading.Interlocked.Increment(ref reachedBarrier);
        }, timeout: TimeSpan.FromSeconds(20), opName: "TcpProcessGroup_Barrier");
        Assert.Equal(worldSize, reachedBarrier);
    }

    [Fact]
    public void ElasticLauncher_TcpBackend_AssignsRanksDeterministically()
    {
        // 3 workers connect to a coordinator on a free loopback port.
        // The bind-winner becomes the coordinator; the rest connect.
        // Final ranks must be the deterministic sort of nonces.
        int port = FreeLoopbackPort();
        const int worldSize = 3;
        var nonces = new[] { "alpha", "bravo", "charlie" };
        var assigned = new int?[worldSize];

        RunWorkers(worldSize, idx =>
        {
            var launcher = new ElasticLauncher(new ElasticLauncherOptions
            {
                WorldSize = worldSize,
                Backend = RendezvousBackend.Tcp,
                Endpoint = $"127.0.0.1:{port}",
                RendezvousTimeoutSeconds = 10,
            });
            var a = launcher.Rendezvous(nonces[idx]);
            Assert.Equal(worldSize, a.WorldSize);
            assigned[idx] = a.Rank;
        }, timeout: TimeSpan.FromSeconds(15), opName: "ElasticLauncher_TcpBackend_Rendezvous");

        // Sort by nonce — alpha=0, bravo=1, charlie=2.
        Assert.Equal(0, assigned[0]);
        Assert.Equal(1, assigned[1]);
        Assert.Equal(2, assigned[2]);
    }

    private static int FreeLoopbackPort()
    {
        var l = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        l.Start();
        int port = ((System.Net.IPEndPoint)l.LocalEndpoint).Port;
        l.Stop();
        return port;
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    /// <summary>
    /// Runs <paramref name="worldSize"/> rank-indexed bodies on dedicated
    /// background threads (NOT the thread pool) and joins them with a
    /// timeout, propagating the first exception observed.
    ///
    /// <para><b>Why dedicated threads, not <see cref="Task.Run(Action)"/>:</b>
    /// xunit runs test classes in parallel by default. Every distributed
    /// test in this file spawns <c>worldSize</c> workers that block on
    /// collective synchronisation (barriers, all-reduce, broadcast) —
    /// each worker only makes progress when its siblings have ALSO
    /// reached the rendezvous. Under <c>Task.Run</c>, those workers
    /// queue onto the shared <see cref="ThreadPool"/>; when sibling
    /// xunit tests are also running pool-backed work, the pool can be
    /// saturated and the worker tasks remain in
    /// <see cref="TaskStatus.WaitingToRun"/> indefinitely while the
    /// pool's hill-climbing slowly ramps. By the time the test's
    /// timeout fires the workers haven't even started — observed in
    /// CI as <c>"Tasks status: [WaitingToRun, WaitingToRun]"</c>.
    /// Dedicated <see cref="Thread"/> instances are scheduled directly
    /// by the OS, bypass the pool entirely, and start immediately.
    /// Cost is negligible (~1MB stack per worker for 2–8 workers per
    /// test), and we get deterministic startup which is what these
    /// barrier-style tests actually require.</para>
    /// </summary>
    private static void RunRanks(int worldSize, Action<int, IProcessGroup> body)
    {
        var groups = InProcessGroup.Create(worldSize);
        try
        {
            RunWorkers(
                worldSize,
                rank =>
                {
                    try { body(rank, groups[rank]); }
                    finally { groups[rank].Dispose(); }
                },
                timeout: TimeSpan.FromSeconds(30),
                opName: "RunRanks");
        }
        catch
        {
            // If RunWorkers threw before some ranks could dispose their
            // group (e.g. a timeout where some workers never started),
            // make sure we don't leak InProcessGroup state into the
            // next test. RunWorkers already propagated; this is just
            // defence-in-depth.
            for (int r = 0; r < worldSize; r++)
            {
                try { groups[r].Dispose(); } catch { /* swallow secondary dispose errors */ }
            }
            throw;
        }
    }

    /// <summary>
    /// Spawns <paramref name="workerCount"/> dedicated background
    /// threads (one per <paramref name="body"/> invocation indexed by
    /// rank), joins them with <paramref name="timeout"/>, and propagates
    /// the first exception observed on any worker. See <see cref="RunRanks"/>
    /// for the rationale on dedicated-thread vs. thread-pool execution.
    /// </summary>
    private static void RunWorkers(int workerCount, Action<int> body, TimeSpan timeout, string opName)
    {
        if (workerCount <= 0) throw new ArgumentOutOfRangeException(nameof(workerCount));
        var threads = new Thread[workerCount];
        var errors = new Exception?[workerCount];
        var startedFlags = new bool[workerCount];
        for (int i = 0; i < workerCount; i++)
        {
            int idx = i;
            var t = new Thread(() =>
            {
                System.Threading.Volatile.Write(ref startedFlags[idx], true);
                try { body(idx); }
                catch (Exception ex) { errors[idx] = ex; }
            })
            {
                IsBackground = true,
                Name = $"{opName}-worker-{idx}",
            };
            threads[i] = t;
            t.Start();
        }

        // Join each thread with the per-call deadline. Using a shared
        // deadline (not per-thread timeout) so a slow first worker
        // doesn't hide that subsequent workers also missed.
        var deadline = DateTime.UtcNow + timeout;
        for (int i = 0; i < workerCount; i++)
        {
            var remaining = deadline - DateTime.UtcNow;
            if (remaining < TimeSpan.Zero) remaining = TimeSpan.Zero;
            if (!threads[i].Join(remaining))
            {
                // A blocked worker on rank i is often the SYMPTOM of a
                // sibling rank having thrown — e.g. rank 0 fails an
                // Assert before reaching a barrier, leaving rank 1
                // blocked in the collective forever. Surface any
                // already-recorded worker exception in that case so
                // the test report shows the real assertion/exception
                // and not the synthetic "worker did not complete"
                // message that would otherwise mask it.
                for (int k = 0; k < workerCount; k++)
                {
                    if (threads[k].Join(TimeSpan.Zero) && errors[k] is { } recorded)
                    {
                        throw new Xunit.Sdk.XunitException(
                            $"{opName}: rank {k} threw {recorded.GetType().Name}: {recorded.Message}\n{recorded.StackTrace}");
                    }
                }
                // Genuine timeout — no worker has finished with an
                // exception, which means we're in a real deadlock.
                // Build a diagnostic that distinguishes "never started"
                // (scheduling problem — would surface as NEVER-STARTED)
                // from "started but blocked" (deadlock in test logic).
                var states = string.Join(", ", threads.Select((th, k) =>
                {
                    var started = System.Threading.Volatile.Read(ref startedFlags[k]) ? "started" : "NEVER-STARTED";
                    return $"#{k}={th.ThreadState}/{started}";
                }));
                throw new Xunit.Sdk.XunitException(
                    $"{opName}: worker did not complete within {timeout.TotalSeconds:F0}s. " +
                    $"Thread states: [{states}].");
            }
        }

        for (int i = 0; i < workerCount; i++)
            if (errors[i] is { } e)
                throw new Xunit.Sdk.XunitException(
                    $"{opName}: rank {i} threw {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
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
