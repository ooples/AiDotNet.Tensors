using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using Xunit;
using BlasMgd = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public sealed class GemmSerialDispatchTests
{
    [Theory]
    [InlineData(17)]
    [InlineData(64)]
    [InlineData(67)]
    public void SerialFloatPreservesJoinedPackingOrderAndPaddedOffsets(int k) => Compare<float>(k);

    [Theory]
    [InlineData(17)]
    [InlineData(64)]
    [InlineData(67)]
    public void SerialDoublePreservesJoinedPackingOrderAndPaddedOffsets(int k) => Compare<double>(k);

    private static unsafe void Compare<T>(int k) where T : unmanaged
    {
        var tile = BlasMgd.GetEvolutionTuningTile<T>(PackingMode.ForcePackBoth, false);
        int mr = tile.Mr, nr = tile.Nr, mc = mr * 2, m = mr * 5, n = nr * 2 + 3;
        int lda = k + 3, ldb = n + 5, ldc = n + 7;
        const int aOffset = 5, bOffset = 7, cOffset = 11, kc = 16;
        T Convert(double value) => typeof(T) == typeof(float) ? (T)(object)(float)value : (T)(object)value;
        T[] Data(int length, int salt) => Enumerable.Range(0, length).Select(i => Convert(Math.Sin(i * salt + .31) * .07)).ToArray();
        var a = Data(aOffset + m * lda + 9, 3);
        var b = Data(bOffset + k * ldb + 13, 5);
        var expected = Data(cOffset + m * ldc + 17, 7);
        var actual = (T[])expected.Clone();
        var aBefore = (T[])a.Clone(); var bBefore = (T[])b.Clone();
        fixed (T* pa = a, pb = b, pe = expected, pc = actual)
        {
            var options = new BlasOptions<T> { NumThreads = 1 };
            PackBothStrategy.RunParallelSingleRegionJoinedUnsafe(pa + aOffset, a.Length - aOffset, lda,
                pb + bOffset, b.Length - bOffset, ldb, pe + cOffset, expected.Length - cOffset, ldc,
                m, n, k, mc, kc, mr, nr, in options, sizeof(T));
            PackBothStrategy.RunSingleRegionSerialUnsafe(pa + aOffset, a.Length - aOffset, lda,
                pb + bOffset, b.Length - bOffset, ldb, pc + cOffset, actual.Length - cOffset, ldc,
                m, n, k, mc, kc, mr, nr, sizeof(T));
        }
        Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())));
        Assert.Equal(aBefore, a); Assert.Equal(bBefore, b);
    }

    [Fact]
    public static unsafe void WarmSerialRegionAvoidsJoinedDispatchObjectAllocations()
    {
        var tile = BlasMgd.GetEvolutionTuningTile<float>(PackingMode.ForcePackBoth, false);
        int m = tile.Mr * 4, n = tile.Nr * 2, k = 67, mc = tile.Mr * 2, kc = 16;
        var a = new float[m * k]; var b = new float[k * n]; var c = new float[m * n];
        bool prior = PackBothParticipantWorkspace.DiagnosticsEnabled;
        PackBothParticipantWorkspace.DiagnosticsEnabled = false;
        try
        {
            fixed (float* pa = a, pb = b, pc = c)
            {
                var options = new BlasOptions<float> { NumThreads = 1 };
                for (int i = 0; i < 20; i++)
                {
                    PackBothStrategy.RunSingleRegionSerialUnsafe(pa, a.Length, k, pb, b.Length, n, pc, c.Length, n,
                        m, n, k, mc, kc, tile.Mr, tile.Nr, sizeof(float));
                    PackBothStrategy.RunParallelSingleRegionJoinedUnsafe(pa, a.Length, k, pb, b.Length, n, pc, c.Length, n,
                        m, n, k, mc, kc, tile.Mr, tile.Nr, in options, sizeof(float));
                }
                long start = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 50; i++)
                    PackBothStrategy.RunSingleRegionSerialUnsafe(pa, a.Length, k, pb, b.Length, n, pc, c.Length, n,
                        m, n, k, mc, kc, tile.Mr, tile.Nr, sizeof(float));
                long serialBytes = GC.GetAllocatedBytesForCurrentThread() - start;
                start = GC.GetAllocatedBytesForCurrentThread();
                for (int i = 0; i < 50; i++)
                    PackBothStrategy.RunParallelSingleRegionJoinedUnsafe(pa, a.Length, k, pb, b.Length, n, pc, c.Length, n,
                        m, n, k, mc, kc, tile.Mr, tile.Nr, in options, sizeof(float));
                long joinedBytes = GC.GetAllocatedBytesForCurrentThread() - start;
                Assert.True(joinedBytes > 0 && serialBytes * 4 < joinedBytes,
                    $"serial={serialBytes}B joined={joinedBytes}B over50calls");
            }
        }
        finally { PackBothParticipantWorkspace.DiagnosticsEnabled = prior; }
    }

    [Fact]
    public static void DispatcherProfilesReuseImmutableShapesAndPreserveEveryCacheKeyDimension()
    {
        var first = AutotuneDispatcher.DispatcherShape<float>(30, 35, 67, false, false, 6, 16, false, true);
        Assert.Same(first, AutotuneDispatcher.DispatcherShape<float>(30, 35, 67, false, false, 6, 16, false, true));
        Assert.Equal(BlasManagedAutotune.EncodeShape<float>(30, 35, 67, false, false, 6, 16, false, true), first);
        var dimensions = first.Dimensions; dimensions[0] = -1;
        Assert.Equal(30, first.Dimensions[0]);
        Assert.NotEqual(first, AutotuneDispatcher.DispatcherShape<double>(30, 35, 67, false, false, 6, 16, false, true));
        Assert.NotEqual(first, AutotuneDispatcher.DispatcherShape<float>(30, 35, 67, true, false, 6, 16, false, true));
        Assert.NotEqual(first, AutotuneDispatcher.DispatcherShape<float>(30, 35, 67, false, true, 6, 16, false, true));
        Assert.NotEqual(first, AutotuneDispatcher.DispatcherShape<float>(30, 35, 67, false, false, 6, 16, true, true));
        Assert.NotEqual(first, AutotuneDispatcher.DispatcherShape<float>(30, 35, 67, false, false, 6, 16, false, false));
    }
}
