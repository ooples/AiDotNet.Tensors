using System;
using System.IO;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Host-side guards on the compiled GPU step's device paths (AiDotNet#1804): what a GPU ReduceMax leaves in its
/// saved state must survive a plan save, a recorded slice must be validated before a kernel sees it, and the
/// transient arena's size classes must cover what they are asked for.
/// </summary>
public class CompiledGpuStepGuardTests
{
    [Fact]
    public void A_device_argmax_in_a_float_plans_saved_state_round_trips()
    {
        // After a GPU ReduceMax the step's saved state is { Tensor<int> indices, int[] axes }: not a Tensor<float>,
        // so not in the plan's tensor table. Saving the plan threw NotSupportedException before.
        var indices = new Tensor<int>(new[] { 3, 0, 2, 1, 1, 4 }, new[] { 2, 3 });
        var state = new object[] { indices, new[] { 1 } };
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            SavedStateSerializerTestProxy.Write(writer, state, new TensorIdMap<float>());
        ms.Position = 0;
        object[]? read;
        using (var reader = new BinaryReader(ms, Encoding.UTF8, leaveOpen: true))
            read = SavedStateSerializerTestProxy.Read(reader, Array.Empty<Tensor<float>>());
        Assert.NotNull(read);
        var back = Assert.IsType<Tensor<int>>(read![0]);
        Assert.Equal(new[] { 2, 3 }, back.Shape.ToArray());
        Assert.Equal(indices.ToArray(), back.ToArray());
        Assert.Equal(new[] { 1 }, Assert.IsType<int[]>(read[1]));
    }

    [Fact]
    public void A_saved_int_tensor_whose_shape_overflows_is_rejected_as_corrupt()
    {
        // [65536]^4 wraps to 0 in int, matching an empty value list; the tensor constructor then threw an
        // ArgumentException the plan loader does not treat as a corrupt file.
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(1);                                    // one saved-state entry
            writer.Write(PlanFormatConstants.TagInt32Tensor);
            writer.Write(4);
            for (int i = 0; i < 4; i++) writer.Write(65536);
            writer.Write(0);                                    // no values
        }
        ms.Position = 0;
        using var reader = new BinaryReader(ms, Encoding.UTF8, leaveOpen: true);
        Assert.Throws<InvalidDataException>(() => SavedStateSerializerTestProxy.Read(reader, Array.Empty<Tensor<float>>()));
    }

    [Theory]
    [InlineData(-1, 0)]
    [InlineData(3, 2)]
    [InlineData(0, 1)]
    public void A_saved_int_tensor_with_invalid_shape_or_value_count_is_rejected(int dimension, int valueCount)
    {
        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(1);
            writer.Write(PlanFormatConstants.TagInt32Tensor);
            writer.Write(1);
            writer.Write(dimension);
            writer.Write(valueCount);
            for (int i = 0; i < valueCount; i++) writer.Write(i);
        }
        ms.Position = 0;
        using var reader = new BinaryReader(ms, Encoding.UTF8, leaveOpen: true);
        Assert.Throws<InvalidDataException>(() => SavedStateSerializerTestProxy.Read(reader, Array.Empty<Tensor<float>>()));
    }

    [Theory]
    [InlineData(int.MaxValue, 1)]   // start + length overflows to negative and passed a start + length check
    [InlineData(3, 2)]
    [InlineData(-1, 1)]
    public void A_recorded_narrow_outside_its_dimension_is_refused_when_recorded(int start, int length)
    {
        // The replay hands start/length to a device slice kernel, which would read past the input buffer.
        var engine = new CpuEngine();
        var w = new Tensor<float>([4, 3]);
        using var scope = GraphMode.EnableTraining(new[] { w });
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.TensorNarrow(w, 0, start, length));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(1000)]
    [InlineData(1 << 20)]
    [InlineData((1 << 20) + 1)]
    [InlineData(32 * 256 * 256)]
    [InlineData(int.MaxValue / 2)]
    public void A_transient_size_class_covers_its_request_within_an_eighth(int length)
    {
        long sizeClass = DirectGpuTensorEngine.TransientSizeClass(length);
        Assert.True(sizeClass >= length, $"class {sizeClass} < request {length}");
        Assert.True(sizeClass <= Math.Max(16, length + length / 8 + 1), $"class {sizeClass} wastes more than an eighth of {length}");
    }
}
