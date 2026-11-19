#if NET5_0_OR_GREATER
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class GotoGemmWorkerWorkspaceTests
{
    [SkippableTheory]
    [InlineData(13, 35, 259, 12, 32, 128, 1)]
    [InlineData(13, 35, 259, 12, 32, 128, 2)]
    [InlineData(13, 35, 259, 12, 32, 128, 0)]
    [InlineData(121, 259, 515, 60, 128, 256, 1)]
    [InlineData(121, 259, 515, 60, 128, 256, 2)]
    [InlineData(121, 259, 515, 60, 128, 256, 0)]
    public unsafe void ReusedWorkspacePreservesOriginalTileBitsAndDestinationBounds(
        int m, int n, int k, int mc, int nc, int kc, int maxWorkers)
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The FP32 machine kernel is unavailable.");
        const int offset = 7;
        int lda = k + 3, ldb = n + 5, ldc = n + 9;
        var a = new float[offset + m * lda + 7];
        var b = new float[offset + k * ldb + 7];
        var expected = new float[offset + m * ldc + 7];
        Array.Fill(expected, 12345f);
        var actual = (float[])expected.Clone();
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++) a[offset + i * lda + p] = ((i * 13 + p * 7) % 41 - 20) * .013f;
        for (int p = 0; p < k; p++)
            for (int j = 0; j < n; j++) b[offset + p * ldb + j] = ((p * 3 + j * 11) % 47 - 23) * .009f;
        int previousDop = CpuParallelSettings.MaxDegreeOfParallelism;
        bool previousDeterministic = CpuParallelSettings.DeterministicReductions;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 4;
            CpuParallelSettings.DeterministicReductions = true;
            fixed (float* pa = a, pb = b, pe = expected, pc = actual)
            {
                // Independent old entry point still acquires a fresh pair for each tile.
                for (int jc = 0; jc < n; jc += nc)
                    for (int ic = 0; ic < m; ic += mc)
                        GotoGemmFp32.RunTile(pa + offset, lda, pb + offset, ldb, pe + offset, ldc,
                            ic, jc, Math.Min(mc, m - ic), Math.Min(nc, n - jc), k, mc, nc, kc);
                GotoGemmFp32.RunParallel(pa + offset, lda, pb + offset, ldb, pc + offset, ldc,
                    m, n, k, mc, nc, kc, maxWorkers);
            }
            Assert.Equal(expected, actual);
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double oracle = 0;
                    for (int p = 0; p < k; p++) oracle += (double)a[offset + i * lda + p] * b[offset + p * ldb + j];
                    Assert.True(Math.Abs(actual[offset + i * ldc + j] - oracle) < 2e-5 * Math.Max(1, Math.Abs(oracle)));
                }
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = previousDop;
            CpuParallelSettings.DeterministicReductions = previousDeterministic;
        }
    }

    [SkippableFact]
    public unsafe void OneWorkerBorrowsOncePerCallRatherThanOncePerTile()
    {
        Skip.IfNot(GotoGemmFp32.IsAvailable, "The FP32 machine kernel is unavailable.");
        Skip.IfNot(CpuStorageAllocationDiagnostics.Enabled, "Launch with AIDOTNET_BUFFER_ALLOC_DIAG=1.");
        const int m = 25, n = 129, k = 19, mc = 12, nc = 32, kc = 16;
        var a = new float[m * k];
        var b = new float[k * n];
        var c = new float[m * n];
        static long Count(string tag)
        {
            var tags = (Dictionary<string, object>)CpuStorageAllocationDiagnostics.Snapshot()["tags"];
            return tags.TryGetValue(tag, out var value)
                ? Convert.ToInt64(((Dictionary<string, object>)value)["count"]) : 0;
        }
        const string tagA = "workspace.blas_goto_tile_packA", tagB = "workspace.blas_goto_tile_packB";
        long aBefore = Count(tagA), bBefore = Count(tagB);
        fixed (float* pa = a, pb = b, pc = c)
            for (int repeat = 0; repeat < 2; repeat++)
                GotoGemmFp32.RunParallel(pa, k, pb, n, pc, n, m, n, k, mc, nc, kc, maxWorkers: 1);
        Assert.Equal(2, Count(tagA) - aBefore);
        Assert.Equal(2, Count(tagB) - bBefore);
        // The former entry point performs 15 pairs for each of these two calls.
        Assert.Equal(15, ((m + mc - 1) / mc) * ((n + nc - 1) / nc));
    }

    [Fact]
    public void SecondRentFailureReturnsTheAlreadyAcquiredPanel()
    {
        var pool = new TrackingPool(failSecondRent: true);
        Assert.Throws<InvalidOperationException>(() => GotoGemmFp32.TileWorkspace.Rent(12, 32, 16, pool));
        Assert.Equal(2, pool.RentCalls);
        Assert.Equal(1, pool.ReturnCalls);
        Assert.Equal(0, pool.Outstanding);
    }

    [Fact]
    public void WorkspaceReturnIsIdempotentAndSizesAreCheckedBeforeBorrow()
    {
        var pool = new TrackingPool();
        var workspace = GotoGemmFp32.TileWorkspace.Rent(12, 32, 16, pool);
        Assert.True(workspace.PackA.Length >= (12 + 6) * 16 + 8);
        Assert.True(workspace.PackB.Length >= 16 * 32);
        workspace.Dispose();
        workspace.Dispose();
        Assert.Equal(2, pool.ReturnCalls);
        Assert.Equal(0, pool.Outstanding);
        Assert.Throws<OverflowException>(() => GotoGemmFp32.TileWorkspace.Rent(int.MaxValue, 32, 16, pool));
        Assert.Equal(2, pool.RentCalls);
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(true, true)]
    [InlineData(false, true)]
    public void ParticipantLocalsBorrowOnceAndReturnExactlyOnce(bool cooperative, bool failBody)
    {
        int previousDop = CpuParallelSettings.MaxDegreeOfParallelism;
        bool previousPool = CpuParallelSettings.UseCooperativePool;
        bool previousDeterministic = CpuParallelSettings.DeterministicReductions;
        var locals = new ConcurrentBag<Local>();
        var pool = new TrackingPool();
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 4;
            CpuParallelSettings.UseCooperativePool = cooperative;
            CpuParallelSettings.DeterministicReductions = true;
            void Run() => CpuParallelSettings.ParallelForIndependentWithLocal(
                0, 64, 8L * 1024 * 1024,
                () => { var local = new Local(GotoGemmFp32.TileWorkspace.Rent(12, 32, 16, pool)); locals.Add(local); return local; },
                (i, local) =>
                {
                    Assert.Equal(0, Volatile.Read(ref local.Finalized));
                    if (failBody && i == 3) throw new InvalidOperationException("Injected body failure");
                    Interlocked.Increment(ref local.Iterations);
                },
                local => { Interlocked.Increment(ref local.Finalized); local.Workspace.Dispose(); });
            if (failBody) Assert.ThrowsAny<Exception>(Run);
            else Run();
            Assert.NotEmpty(locals);
            Assert.All(locals, local => Assert.Equal(1, local.Finalized));
            int iterations = 0;
            foreach (var local in locals) iterations += local.Iterations;
            if (!failBody) Assert.Equal(64, iterations);
            Assert.InRange(locals.Count, 1, 4);
            Assert.Equal(2 * locals.Count, pool.RentCalls);
            Assert.Equal(pool.RentCalls, pool.ReturnCalls);
            Assert.Equal(0, pool.Outstanding);
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = previousDop;
            CpuParallelSettings.UseCooperativePool = previousPool;
            CpuParallelSettings.DeterministicReductions = previousDeterministic;
        }
    }

    private sealed class Local
    {
        internal readonly GotoGemmFp32.TileWorkspace Workspace;
        internal int Finalized, Iterations;
        internal Local(GotoGemmFp32.TileWorkspace workspace) => Workspace = workspace;
    }

    private sealed class TrackingPool : ArrayPool<float>
    {
        private readonly ConcurrentDictionary<float[], byte> _live = new();
        private readonly bool _failSecondRent;
        internal int RentCalls, ReturnCalls;
        internal int Outstanding => _live.Count;
        internal TrackingPool(bool failSecondRent = false) => _failSecondRent = failSecondRent;
        public override float[] Rent(int minimumLength)
        {
            int call = Interlocked.Increment(ref RentCalls);
            if (_failSecondRent && call == 2) throw new InvalidOperationException("Injected second rent failure");
            var result = new float[minimumLength];
            Assert.True(_live.TryAdd(result, 0));
            return result;
        }
        public override void Return(float[] array, bool clearArray = false)
        {
            Assert.True(_live.TryRemove(array, out _));
            Interlocked.Increment(ref ReturnCalls);
        }
    }
}
#endif
