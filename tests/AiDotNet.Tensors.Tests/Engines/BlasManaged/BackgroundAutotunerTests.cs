using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class BackgroundAutotunerTests
{
    private static SightingTracker.ShapeId Id(int m, int n, int k)
        => new(m, n, k, Fp64: false, TransA: false, TransB: false);

    [Fact]
    public void Sighting_FirstIsOne_SecondTriggers()
    {
        var t = new SightingTracker(capacity: 16);
        Assert.False(t.RecordAndShouldMeasure(Id(64, 64, 64)));   // 1st
        Assert.True(t.RecordAndShouldMeasure(Id(64, 64, 64)));    // 2nd → measure
    }

    [Fact]
    public void Sighting_Dedup_OnlyOnceWhileInFlight()
    {
        var t = new SightingTracker(capacity: 16);
        var id = Id(96, 96, 96);
        t.RecordAndShouldMeasure(id);
        Assert.True(t.RecordAndShouldMeasure(id));   // 2nd → measure, marks in-flight
        Assert.False(t.RecordAndShouldMeasure(id));  // 3rd while in-flight → no duplicate
        t.MarkDone(id);
        Assert.True(t.RecordAndShouldMeasure(id));   // after done, eligible again
    }

    [Fact]
    public void Sighting_LRU_EvictsBeyondCapacity()
    {
        var t = new SightingTracker(capacity: 2);
        t.RecordAndShouldMeasure(Id(1, 1, 1));
        t.RecordAndShouldMeasure(Id(2, 2, 2));
        t.RecordAndShouldMeasure(Id(3, 3, 3)); // evicts id(1)
        Assert.False(t.RecordAndShouldMeasure(Id(1, 1, 1))); // evicted → "1st" again
    }

    [Fact]
    public void Transposed_And_NonTransposed_AreDistinctShapes()
    {
        var t = new SightingTracker(capacity: 16);
        var nt = new SightingTracker.ShapeId(64, 64, 64, false, false, false);
        var tt = new SightingTracker.ShapeId(64, 64, 64, false, false, true);
        t.RecordAndShouldMeasure(nt);
        Assert.False(t.RecordAndShouldMeasure(tt)); // transB variant is a separate 1st sighting
    }

    [Fact]
    public void ShouldMeasureSize_SkipsLargeShapes()
    {
        // Work ceiling ~8M: large shapes get offline pre-warm, not live background sweep (G3).
        Assert.False(BackgroundAutotuner.ShouldMeasureSize(1024, 3072, 768));
        Assert.True(BackgroundAutotuner.ShouldMeasureSize(96, 128, 64));
    }

    [Fact]
    public void Measure_PopulatesCache_TransAware()
    {
        // A transB shape's measurement must store under the transB key (so a transB
        // call's learned-cache consult actually hits it).
        var id = new SightingTracker.ShapeId(80, 80, 96, Fp64: true, TransA: false, TransB: true);
        BackgroundAutotuner.MeasureNowForTest(id);
        var shape = BlasManagedAutotune.EncodeShape<double>(80, 80, 96, false, true, 0, 0, false,
            AiDotNet.Tensors.Helpers.BlasProvider.IsDeterministicMode);
        Assert.NotNull(BlasManagedAutotune.TryLookupStrategy(shape));
    }
}
