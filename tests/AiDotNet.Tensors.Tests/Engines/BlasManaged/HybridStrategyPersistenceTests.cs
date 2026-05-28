using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Stats-Serial")]
public class HybridStrategyPersistenceTests
{
    [Fact]
    public void KernelVersion_IsStable_NonEmpty()
    {
        Assert.False(string.IsNullOrEmpty(BlasKernelVersion.Current));
        Assert.Equal(BlasKernelVersion.Current, BlasKernelVersion.Current); // stable per process
    }

    [Fact]
    public void StoreStrategy_RoundTrips_PackingMode()
    {
        var shape = BlasManagedAutotune.EncodeShape<float>(64, 64, 64, false, false, 8, 8, false, false);
        BlasManagedAutotune.StoreStrategy(shape, PackingMode.ForceStreaming, ParallelismAxis.M,
            mc: 64, nc: 64, kc: 64, threadCount: 8, BlasKernelVersion.Current);
        var got = BlasManagedAutotune.TryLookupStrategy(shape);
        Assert.NotNull(got);
        Assert.Equal(PackingMode.ForceStreaming, got!.Value.Mode);
    }

    [Fact]
    public void TryLookupStrategy_IgnoresKernelVersionMismatch()
    {
        var shape = BlasManagedAutotune.EncodeShape<float>(48, 48, 48, false, false, 8, 8, false, false);
        BlasManagedAutotune.StoreStrategy(shape, PackingMode.ForcePackBoth, ParallelismAxis.M,
            64, 64, 64, 8, "stale-version-token");
        Assert.Null(BlasManagedAutotune.TryLookupStrategy(shape)); // mismatched version → miss
    }

    [Fact]
    public void SeedFromShipped_DoesNotOverwriteLocalLearnedEntry()
    {
        var shape = BlasManagedAutotune.EncodeShape<float>(123, 45, 67, false, false, 0, 0, false, false);
        BlasManagedAutotune.StoreStrategy(shape, PackingMode.ForceStreaming, ParallelismAxis.M,
            64, 64, 64, 8, BlasKernelVersion.Current);
        // A (hypothetical) shipped PackBoth entry must NOT override the local learned one.
        BlasManagedAutotune.SeedFromShippedIfAbsent(shape, PackingMode.ForcePackBoth, ParallelismAxis.M,
            64, 64, 64, 8);
        Assert.Equal(PackingMode.ForceStreaming, BlasManagedAutotune.TryLookupStrategy(shape)!.Value.Mode);
    }

    [Fact]
    public void SeedFromShipped_SeedsWhenAbsent()
    {
        var shape = BlasManagedAutotune.EncodeShape<float>(124, 46, 68, false, false, 0, 0, false, false);
        BlasManagedAutotune.SeedFromShippedIfAbsent(shape, PackingMode.ForcePackBoth, ParallelismAxis.M,
            64, 64, 64, 8);
        Assert.Equal(PackingMode.ForcePackBoth, BlasManagedAutotune.TryLookupStrategy(shape)!.Value.Mode);
    }
}
