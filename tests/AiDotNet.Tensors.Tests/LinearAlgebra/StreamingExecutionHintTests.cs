using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// The streaming store's training/inference hint can be read back, so a caller that sets it for a scoped operation
/// can restore it. Before, it could only be written.
/// </summary>
[Collection("WeightRegistry")]
public class StreamingExecutionHintTests
{
    [Fact]
    public void The_hint_reads_back_what_was_set_and_is_unknown_after_a_reset()
    {
        try
        {
            WeightRegistry.SetStreamingExecutionTraining(false);
            Assert.False(WeightRegistry.StreamingExecutionTraining);
            WeightRegistry.SetStreamingExecutionTraining(true);
            Assert.True(WeightRegistry.StreamingExecutionTraining);
            WeightRegistry.SetStreamingExecutionTraining(null);
            Assert.Null(WeightRegistry.StreamingExecutionTraining);
            WeightRegistry.SetStreamingExecutionTraining(false);
            WeightRegistry.Reset();
            Assert.Null(WeightRegistry.StreamingExecutionTraining);
        }
        finally
        {
            WeightRegistry.SetStreamingExecutionTraining(null);
            WeightRegistry.Reset();
        }
    }
}
