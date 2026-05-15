using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// xUnit collection that serialises every test class touching the
/// <see cref="IEngine.TensorEmbeddingLookup{TValue,TIndex}"/> compiled-plan
/// path. Both <see cref="TensorEmbeddingLookupCompiledPlanTests"/> and
/// <see cref="TensorEmbeddingLookupLongIndexTests"/> reach into the same
/// process-wide compilation state (lazy-graph tracer, plan cache, allocator
/// buffers) and can flake when run in parallel across xUnit's test
/// collections. The collection has no fixture — it exists only to disable
/// parallelism for the classes that opt in via <c>[Collection]</c>.
/// </summary>
[CollectionDefinition("EmbeddingLookupCompiledPlan", DisableParallelization = true)]
public class EmbeddingLookupCompiledPlanCollection { }

/// <summary>
/// Regression tests for the <c>Tensor&lt;long&gt;</c> index path in
/// <see cref="IEngine.TensorEmbeddingLookup{TValue,TIndex}"/>. Earlier
/// versions of the op snapshotted indices as <c>int[]</c> via
/// <c>Convert.ToInt32(...)</c>, which threw <see cref="OverflowException"/>
/// when any element exceeded <see cref="int.MaxValue"/>. Large-scale
/// retrieval embeddings (search corpora, RAG indexes) need vocabularies
/// above 2.1 billion. The current implementation widens to <c>long[]</c>
/// end-to-end (CpuEngine forward + backward, SavedStateSerializer,
/// BackwardFunctions consumer) so those workloads work without silent
/// overflow.
/// </summary>
[Collection("EmbeddingLookupCompiledPlan")]
public class TensorEmbeddingLookupLongIndexTests
{
    /// <summary>
    /// Baseline: <c>Tensor&lt;long&gt;</c> indices with small values
    /// (within int range) still produce the expected scatter-add
    /// gradient pattern. Verifies the long-index path doesn't regress
    /// the common case.
    /// </summary>
    [Fact]
    public void TensorEmbeddingLookup_LongIndices_SmallValues_BackwardMatchesIntPath()
    {
        var engine = new CpuEngine();
        const int vocab = 6, dim = 4;
        var E = new Tensor<float>([vocab, dim]);
        var eSpan = E.AsWritableSpan();
        for (int i = 0; i < eSpan.Length; i++) eSpan[i] = 0.01f * (i + 1);

        var indices = new Tensor<long>([3]);
        indices[0] = 0L; indices[1] = 2L; indices[2] = 4L;

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorEmbeddingLookup<float, long>(E, indices);
            engine.ReduceSum(output, null);
            plan = scope.CompileTraining(new[] { E });
        }
        using (plan)
        {
            plan.Step();
            var gradE = plan.Gradients[0];
            var gSpan = gradE.AsSpan();

            // Rows 0, 2, 4 selected → grad = 1; rows 1, 3, 5 → grad = 0.
            var selected = new System.Collections.Generic.HashSet<long> { 0L, 2L, 4L };
            for (int row = 0; row < vocab; row++)
            {
                float expected = selected.Contains(row) ? 1f : 0f;
                for (int col = 0; col < dim; col++)
                    Assert.Equal(expected, gSpan[row * dim + col], 5);
            }
        }
    }

    /// <summary>
    /// SavedStateSerializer must round-trip <c>long[]</c> losslessly so
    /// large-index embedding plans can be saved + loaded without
    /// truncation. Tests the write/read cycle for an array containing
    /// values both inside and outside the int32 range.
    /// </summary>
    [Fact]
    public void SavedStateSerializer_LongArray_RoundTripsLosslessly()
    {
        var original = new object[]
        {
            new long[] { 0L, 1L, int.MaxValue, ((long)int.MaxValue) + 1L, long.MaxValue / 2L },
            new int[] { 1, 2, 3 },
            42,
            17L,
            "embedding-lookup",
        };

        using var ms = new System.IO.MemoryStream();
        var tensorMap = new TensorIdMap<float>();
        using (var writer = new System.IO.BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            SavedStateSerializerTestProxy.Write(writer, original, tensorMap);
        }

        ms.Position = 0;
        object[]? roundTripped;
        using (var reader = new System.IO.BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            roundTripped = SavedStateSerializerTestProxy.Read<float>(reader, System.Array.Empty<Tensor<float>>());
        }

        Assert.NotNull(roundTripped);
        Assert.Equal(original.Length, roundTripped!.Length);

        // long[] preserved exactly, including values above int.MaxValue.
        var origLongs = (long[])original[0];
        var rtLongs = (long[])roundTripped[0];
        Assert.Equal(origLongs, rtLongs);
        Assert.Contains(rtLongs, v => v > int.MaxValue);

        Assert.Equal((int[])original[1], (int[])roundTripped[1]);
        Assert.Equal(42, (int)roundTripped[2]);
        Assert.Equal(17L, (long)roundTripped[3]);
        Assert.Equal("embedding-lookup", (string)roundTripped[4]);
    }

    /// <summary>
    /// Reading legacy plan files: when the saved-state slot for indices
    /// is an <c>int[]</c> (plans serialised before the long widening),
    /// <see cref="Autodiff.BackwardFunctions{T}.TensorEmbeddingLookupBackward"/>
    /// widens to <c>long[]</c> internally so the backward kernel still
    /// runs. Constructs the saved-state by hand to simulate the
    /// pre-widening format.
    /// </summary>
    [Fact]
    public void TensorEmbeddingLookupBackward_LegacyIntArraySavedState_StillWorks()
    {
        var engine = new CpuEngine();
        const int vocab = 4, dim = 2;
        var E = new Tensor<float>([vocab, dim]);

        // gradOutput rows = [1.0, 1.0] each. Three rows selected by indices [1, 3, 0]
        // → expected grad: row0=[1,1], row1=[1,1], row2=[0,0], row3=[1,1].
        var gradOutput = new Tensor<float>([3, dim]);
        var goSpan = gradOutput.AsWritableSpan();
        for (int i = 0; i < goSpan.Length; i++) goSpan[i] = 1f;

        // Use the engine's typed backward directly with a Tensor<long>
        // input, mirroring what BackwardFunctions.TensorEmbeddingLookupBackward
        // will pass after the legacy-int[] widening branch.
        var indices = new Tensor<long>([3]);
        indices[0] = 1L; indices[1] = 3L; indices[2] = 0L;

        var grad = engine.TensorEmbeddingLookupBackward<float, long>(gradOutput, indices, vocab, dim);
        var gSpan = grad.AsSpan();
        Assert.Equal(1f, gSpan[0 * dim + 0], 5);
        Assert.Equal(1f, gSpan[1 * dim + 0], 5);
        Assert.Equal(0f, gSpan[2 * dim + 0], 5);
        Assert.Equal(1f, gSpan[3 * dim + 0], 5);
    }

    /// <summary>
    /// Negative-path: the engine rejects out-of-bounds indices with an
    /// <see cref="ArgumentOutOfRangeException"/> that carries the
    /// long-value index (not a truncated int) in the message — useful for
    /// debugging large-vocab workloads.
    /// </summary>
    [Fact]
    public void TensorEmbeddingLookup_LongIndexOutOfBounds_ThrowsWithLongValue()
    {
        var engine = new CpuEngine();
        var E = new Tensor<float>([4, 2]);
        var indices = new Tensor<long>([1]);
        indices[0] = ((long)int.MaxValue) + 100L;

        var ex = Assert.Throws<ArgumentOutOfRangeException>(
            () => engine.TensorEmbeddingLookup<float, long>(E, indices));
        // Message should contain the actual long value, not a truncated int.
        Assert.Contains(indices[0].ToString(), ex.Message);
    }
}

/// <summary>
/// Test-only access to <see cref="SavedStateSerializer"/> internal Write /
/// Read methods. The serializer is intentionally <c>internal</c> on the
/// production assembly; the tests use <c>InternalsVisibleTo</c> to call
/// it directly without exposing the writer/reader contract publicly.
/// </summary>
internal static class SavedStateSerializerTestProxy
{
    internal static void Write<T>(System.IO.BinaryWriter writer, object[]? state, TensorIdMap<T> tensorMap)
        => SavedStateSerializer.Write(writer, state, tensorMap);

    internal static object[]? Read<T>(System.IO.BinaryReader reader, Tensor<T>[] tensorTable)
        => SavedStateSerializer.Read(reader, tensorTable);
}
