using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// #1715 stage 3 — the WRITABLE param-IO zero-copy path through WeightRegistry. In training mode a
/// foundation streaming weight materializes by aliasing its bytes to a writable slice of the param-IO
/// mmap store (no resident GC copy), mutations persist through the mapping, and that slice is the
/// weight's canonical storage (Materialize / ReleaseToPool no-op on it). This is what stops the
/// GetParameters → mutate → SetParameters round-trip from materializing the whole model resident → OOM.
///
/// [Collection("WeightRegistry")] serializes against the other tests that mutate the process-global
/// registry via Configure/Reset.
/// </summary>
[Collection("WeightRegistry")]
public class MmapParamIoAliasTests : IDisposable
{
    public void Dispose() => WeightRegistry.Reset();

    [Fact]
    public void TrainingMaterialize_WritablyAliasesStore_RoundTripsMutation_NoResidentCopy()
    {
        WeightRegistry.Configure(new GpuOffloadOptions
        {
            EnableZeroCopyMmapResidency = true,
            StreamingStoreDtype = StreamingStoreDtype.FullPrecision, // native encoding ⇒ aliasable
            StreamingPoolMaxResidentBytes = 64,                       // tiny ⇒ the weight is paged out
        });
        WeightRegistry.SetStreamingExecutionTraining(true);           // !inferenceMode ⇒ writable path

        var original = new float[256];
        for (int i = 0; i < original.Length; i++) original[i] = i * 0.25f - 7f;
        var w = new Tensor<float>((float[])original.Clone(), new[] { original.Length });
        w.Lifetime = WeightLifetime.Streaming;
        WeightRegistry.RegisterWeight(w);
        Assert.True(w.StreamingPoolHandle >= 0);

        WeightRegistry.ReleaseToPool(w);   // drop resident _data → bytes live only in the pool
        WeightRegistry.Materialize(w);     // training-mode materialize → writable mmap alias

        Assert.True(w.IsWritableMmapAliased, "training materialize must writably-alias the store slice");
        for (int i = 0; i < original.Length; i++) Assert.Equal(original[i], w[i]); // reads the slice, correct

        // Mutate through the weight's storage; the change lands in the mmap slice (its canonical storage).
        var span = w.DataVector.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = original[i] * 3f + 2f;

        // Re-materialize must be a NO-OP (already the canonical alias) — never read stale pool bytes.
        WeightRegistry.Materialize(w);
        Assert.True(w.IsWritableMmapAliased);
        for (int i = 0; i < original.Length; i++) Assert.Equal(original[i] * 3f + 2f, w[i]);

        // ReleaseToPool must be a NO-OP on a writable alias — the storage stays the canonical mmap slice
        // (dropping it would discard the mutation and a re-materialize would read stale bytes).
        WeightRegistry.ReleaseToPool(w);
        Assert.True(w.IsWritableMmapAliased);
        Assert.Equal(original.Length, w.DataVector.Length);
        for (int i = 0; i < original.Length; i++) Assert.Equal(original[i] * 3f + 2f, w[i]);
    }
}
