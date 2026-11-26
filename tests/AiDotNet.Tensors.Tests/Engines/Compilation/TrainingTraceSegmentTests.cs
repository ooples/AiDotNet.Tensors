// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.IO;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Activation-checkpoint segment bookkeeping of a trace: a segment opens only inside a training trace, segments do not
/// nest, and a closed segment lets the next one open. Also the plan serializer's refusal of a saved-state entry it
/// cannot write.
/// </summary>
public class TrainingTraceSegmentTests
{
    [Fact]
    public void A_segment_opens_only_in_a_training_trace_and_segments_do_not_nest()
    {
        Assert.False(GraphMode.IsTrainingTrace);
        using (GraphMode.BeginCheckpointSegment()) { }   // outside a training trace: does nothing

        var parameter = new Tensor<float>(new float[] { 1, 2 }, new[] { 2 });
        using var scope = GraphMode.EnableTraining(new[] { parameter });
        Assert.True(GraphMode.IsTrainingTrace);
        Assert.Empty(scope.CheckpointSegments);

        using (GraphMode.BeginCheckpointSegment())
        {
            var ex = Assert.Throws<InvalidOperationException>(() => GraphMode.BeginCheckpointSegment());
            Assert.Contains("do not nest", ex.Message);
        }
        using (GraphMode.BeginCheckpointSegment()) { }   // the closed segment lets the next one open
    }

    [Fact]
    public void A_saved_state_entry_of_an_unsupported_type_fails_at_save_time()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        var ex = Assert.Throws<NotSupportedException>(
            () => SavedStateSerializer.Write(writer, new object[] { new Uri("http://localhost/") }, new TensorIdMap<float>()));
        Assert.Contains(typeof(Uri).FullName!, ex.Message);
    }
}
