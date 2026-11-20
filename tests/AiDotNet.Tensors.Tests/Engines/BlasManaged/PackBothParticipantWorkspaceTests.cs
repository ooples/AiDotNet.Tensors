using System.Buffers;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasMgd = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public sealed class PackBothParticipantWorkspaceTests : IDisposable
{
    private readonly int _priorDegree = CpuParallelSettings.MaxDegreeOfParallelism;
    private readonly bool _priorDiagnostics = PackBothParticipantWorkspace.DiagnosticsEnabled;
    private readonly bool _priorPool = CpuParallelSettings.UseCooperativePool;

    public PackBothParticipantWorkspaceTests()
    {
        CpuParallelSettings.MaxDegreeOfParallelism = 8;
        PackBothParticipantWorkspace.DiagnosticsEnabled = true;
        PackBothParticipantWorkspace.ResetDiagnostics();
    }

    public void Dispose()
    {
        CpuParallelSettings.MaxDegreeOfParallelism = _priorDegree;
        CpuParallelSettings.UseCooperativePool = _priorPool;
        PackBothParticipantWorkspace.DiagnosticsEnabled = _priorDiagnostics;
    }

    [Fact]
    public void SingleRegionRetainsOneAPackAcrossEveryKPanelAndRowBlock()
    {
        var tile = BlasMgd.GetEvolutionTuningTile<float>(PackingMode.ForcePackBoth, false);
        int mc = tile.Mr * 2, m = mc * 9, n = tile.Nr * 2 + 3;
        const int k = 65, kc = 16;
        var a = Data(m * k, 3);
        var b = Data(k * n, 5);
        var c = new float[m * n];
        PackBothStrategy.Run<float>(a, k, false, b, n, false, c, n, m, n, k,
            mc, tile.Nr * 3, kc, tile.Mr, tile.Nr,
            new BlasOptions<float> { NumThreads = 1, ParallelismAxis = ParallelismAxis.M });
        CheckOracle(a, k, false, b, n, false, c, n, m, n, k);
        var stats = PackBothParticipantWorkspace.SnapshotDiagnostics();
        Assert.Equal(1, stats["owners_created"]);
        Assert.Equal(1, stats["rents"]); // old loop:9ICblocks ×5Kpanels =45successful private-A requests
        Assert.Equal(stats["rents"], stats["returns"]);
        Assert.Equal(0, stats["live_backing_bytes"]);
        Assert.Equal((long)mc * kc * sizeof(float), stats["peak_recorded_live_requested_bytes"]);
        Assert.InRange(stats["peak_recorded_live_backing_bytes"], stats["peak_recorded_live_requested_bytes"],
            2 * stats["peak_recorded_live_requested_bytes"]);
    }

    [Fact]
    public void NAxisRetainsOneBPackAcrossColumnBlocksOrRefusesUnsupportedHardware()
    {
        const int m = 6, n = 8192, k = 64;
        var a = Data(m * k, 7);
        var b = Data(k * n, 11);
        var c = new float[m * n];
        void Run() => PackBothStrategy.Run<float>(a, k, false, b, n, false, c, n, m, n, k,
            mc: 12, nc: n, kc: 64, mr: 6, nr: 16,
            new BlasOptions<float> { NumThreads = 1, ParallelismAxis = ParallelismAxis.N });
        if (!MachineKernelGemm.IsFp32PanelAvailable)
        {
            Assert.Throws<NotSupportedException>(Run);
            return;
        }
        Run();
        CheckOracle(a, k, false, b, n, false, c, n, m, n, k);
        var stats = PackBothParticipantWorkspace.SnapshotDiagnostics();
        Assert.Equal(1, stats["rents"]); // old loop:8Nblocks, each separately rented
        Assert.Equal(1, stats["owners_created"]);
        Assert.Equal(1, stats["returns"]);
        Assert.Equal(0, stats["live_backing_bytes"]);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void FloatAllTransposeModesAndRaggedTailsMatchFp64Oracle(bool transA, bool transB)
    {
        const int m = 25, n = 37, k = 43, ldc = n + 5;
        int lda = (transA ? m : k) + 3, ldb = (transB ? k : n) + 2;
        var a = Data((transA ? k : m) * lda, 13);
        var b = Data((transB ? n : k) * ldb, 17);
        var c = Enumerable.Repeat(777f, m * ldc).ToArray();
        BlasMgd.Gemm<float>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k,
            new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth, NumThreads = 2,
                Mc = 96, Nc = 64, Kc = 16, ParallelismAxis = ParallelismAxis.M });
        CheckOracle(a, lda, transA, b, ldb, transB, c, ldc, m, n, k);
        for (int row = 0; row < m; row++)
            for (int col = n; col < ldc; col++) Assert.Equal(777f, c[row * ldc + col]);
        Assert.Equal(0, PackBothParticipantWorkspace.SnapshotDiagnostics()["live_backing_bytes"]);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void DoubleAllTransposeModesAndRaggedTailsMatchIndependentOracle(bool transA, bool transB)
    {
        const int m = 25, n = 37, k = 43, ldc = n + 5;
        int lda = (transA ? m : k) + 3, ldb = (transB ? k : n) + 2;
        var a = Data((transA ? k : m) * lda, 19).Select(value => (double)value).ToArray();
        var b = Data((transB ? n : k) * ldb, 23).Select(value => (double)value).ToArray();
        var c = Enumerable.Repeat(777.0, m * ldc).ToArray();
        BlasMgd.Gemm<double>(a, lda, transA, b, ldb, transB, c, ldc, m, n, k,
            new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth, NumThreads = 2,
                Mc = 96, Nc = 64, Kc = 16, ParallelismAxis = ParallelismAxis.M });
        for (int row = 0; row < m; row++)
            for (int col = 0; col < ldc; col++)
            {
                if (col >= n) { Assert.Equal(777.0, c[row * ldc + col]); continue; }
                double expected = 0;
                for (int p = 0; p < k; p++)
                    expected += a[transA ? p * lda + row : row * lda + p]
                              * b[transB ? col * ldb + p : p * ldb + col];
                Assert.InRange(Math.Abs(c[row * ldc + col] - expected), 0, 2e-12 * Math.Max(1, Math.Abs(expected)));
            }
    }

    [Fact]
    public void OperandViewsCanShareBackingAndKeepInputAndOutputPaddingUntouched()
    {
        const int m = 48, n = 35, k = 67, ldc = 40, aOffset = 7, bOffset = 19, cOffset = 5000;
        var storage = Data(cOffset + m * ldc + 9, 29);
        var original = storage.ToArray();
        var result = new float[m * ldc];
        BlasMgd.Gemm<float>(storage.AsSpan(aOffset, m * k), k, false,
            storage.AsSpan(bOffset, k * n), n, false, storage.AsSpan(cOffset, m * ldc), ldc,
            m, n, k, new BlasOptions<float> { PackingMode = PackingMode.ForcePackBoth,
                NumThreads = 2, Mc = 96, Nc = 64, Kc = 16, ParallelismAxis = ParallelismAxis.M });
        storage.AsSpan(cOffset, result.Length).CopyTo(result);
        CheckOracle(original.AsSpan(aOffset, m * k).ToArray(), k, false,
            original.AsSpan(bOffset, k * n).ToArray(), n, false, result, ldc, m, n, k);
        for (int i = 0; i < storage.Length; i++)
            if (i < cOffset || i >= cOffset + m * ldc || (i - cOffset) % ldc >= n)
                Assert.Equal(original[i], storage[i]);
    }

    [Fact]
    public void NestedGemmKeepsSeparateOwnersAndHonorsSerialInnerBudget()
    {
        var tile = BlasMgd.GetEvolutionTuningTile<float>(PackingMode.ForcePackBoth, false);
        int mc = tile.Mr * 2, m = mc * 5, n = tile.Nr * 2;
        const int k = 65;
        var a = Data(m * k, 31);
        var b = Data(k * n, 37);
        var outputs = new[] { new float[m * n], new float[m * n] };
        CpuParallelSettings.ParallelForRegion(2, replica =>
            PackBothStrategy.Run<float>(a, k, false, b, n, false, outputs[replica], n, m, n, k,
                mc, n, 16, tile.Mr, tile.Nr,
                new BlasOptions<float> { NumThreads = 8, ParallelismAxis = ParallelismAxis.M }));
        foreach (var output in outputs) CheckOracle(a, k, false, b, n, false, output, n, m, n, k);
        var stats = PackBothParticipantWorkspace.SnapshotDiagnostics();
        Assert.Equal(2, stats["owners_created"]);
        Assert.Equal(2, stats["rents"]);
        Assert.Equal(2, stats["owners_disposed"]);
        Assert.Equal(0, stats["live_backing_bytes"]);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ActualParticipantOwnersReturnEveryLeaseOnBodyOrGrowthFailure(bool cooperative, bool rentFailure)
    {
        CpuParallelSettings.UseCooperativePool = cooperative;
        var pool = new TrackingPool { ThrowOnRent = rentFailure ? 2 : int.MaxValue };
        int bodyFailures = 0;
        void Run() => CpuParallelSettings.ParallelForIndependentWithLocal(0, 64, 64L * 1024 * 1024,
            () => new PackBothParticipantWorkspace("workspace.test", pool), (index, workspace) =>
            {
                workspace.ForPanel((index % 3 + 1) * 32).Fill((byte)index);
                if (!rentFailure && index == 4)
                {
                    Interlocked.Increment(ref bodyFailures);
                    throw new InvalidOperationException("Deliberate body failure.");
                }
            }, static workspace => workspace.Dispose(), maximumDegreeOfParallelism: 2);
        Assert.ThrowsAny<Exception>(Run);
        // The selected failure, not an unrelated one before any lease existed, must have ended the dispatch.
        Assert.Equal(rentFailure ? 1 : 0, pool.RentFailures);
        Assert.Equal(rentFailure ? 0 : 1, bodyFailures);
        Assert.True(pool.SuccessfulRents >= 1, "The failure must follow at least one successful rent.");
        var stats = PackBothParticipantWorkspace.SnapshotDiagnostics();
        Assert.Equal(stats["owners_created"], stats["owners_disposed"]);
        Assert.InRange(stats["owners_created"], 1, 2);
        Assert.Equal(stats["rents"], stats["returns"]);
        Assert.True(stats["rents"] >= 1);
        Assert.Equal(0, pool.Outstanding);
        Assert.Equal(0, stats["live_backing_bytes"]);
    }

    [Fact]
    public void LazyGrowthReturnsOldLeaseBeforeRentAndEmptyDispatchBorrowsNothing()
    {
        var pool = new TrackingPool { ThrowOnRent = 2 };
        var workspace = new PackBothParticipantWorkspace("workspace.test", pool);
        workspace.ForPanel(32).Fill(1);
        Assert.Throws<InvalidOperationException>(() => { workspace.ForPanel(64); });
        Assert.Equal(0, pool.Outstanding);
        Assert.Equal(1, pool.ReturnCalls);
        workspace.Dispose();
        workspace.Dispose();
        Assert.Equal(1, pool.ReturnCalls);
        Assert.Equal(0, PackBothParticipantWorkspace.SnapshotDiagnostics()["live_backing_bytes"]);
        PackBothParticipantWorkspace.ResetDiagnostics();
        using (var growing = new PackBothParticipantWorkspace("workspace.test", new TrackingPool()))
        {
            growing.ForPanel(32);
            growing.ForPanel(96);
            var stats = PackBothParticipantWorkspace.SnapshotDiagnostics();
            Assert.Equal(2, stats["rents"]);
            Assert.Equal(1, stats["returns"]);
            Assert.Equal(96, stats["peak_recorded_live_requested_bytes"]);
            Assert.Equal(128, stats["peak_recorded_live_backing_bytes"]); //96requested+32test-pool padding, never32+96
        }
        PackBothParticipantWorkspace.ResetDiagnostics();
        CpuParallelSettings.ParallelForIndependentWithLocal(0, 0, 100000,
            () => new PackBothParticipantWorkspace("workspace.test", pool), (_, _) => { }, static value => value.Dispose());
        Assert.Equal(0, PackBothParticipantWorkspace.SnapshotDiagnostics()["owners_created"]);
    }

    private static float[] Data(int length, int salt)
        => Enumerable.Range(0, length).Select(i => ((i * salt) % 43 - 21) * .003125f).ToArray();

    private static void CheckOracle(float[] a, int lda, bool transA, float[] b, int ldb, bool transB,
        float[] c, int ldc, int m, int n, int k)
    {
        for (int row = 0; row < m; row++)
            for (int col = 0; col < n; col++)
            {
                double expected = 0;
                for (int p = 0; p < k; p++)
                    expected += (double)a[transA ? p * lda + row : row * lda + p]
                                      * b[transB ? col * ldb + p : p * ldb + col];
                Assert.InRange(Math.Abs(c[row * ldc + col] - expected), 0, 2e-5 * Math.Max(1, Math.Abs(expected)));
            }
    }

    private sealed class TrackingPool : ArrayPool<byte>
    {
        private readonly ConcurrentDictionary<byte[], byte> _live = new();
        private int _rentCalls, _returnCalls, _rentFailures, _successfulRents;
        internal int ThrowOnRent = int.MaxValue;
        internal int ReturnCalls => _returnCalls;
        internal int Outstanding => _live.Count;
        internal int RentFailures => Volatile.Read(ref _rentFailures);
        internal int SuccessfulRents => Volatile.Read(ref _successfulRents);
        public override byte[] Rent(int minimumLength)
        {
            if (Interlocked.Increment(ref _rentCalls) == ThrowOnRent)
            {
                Interlocked.Increment(ref _rentFailures);
                throw new InvalidOperationException("Deliberate rent failure.");
            }
            Interlocked.Increment(ref _successfulRents);
            var array = new byte[minimumLength + 32];
            Assert.True(_live.TryAdd(array, 0));
            return array;
        }
        public override void Return(byte[] array, bool clearArray = false)
        {
            Assert.True(_live.TryRemove(array, out _));
            Interlocked.Increment(ref _returnCalls);
        }
    }
}
