using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class CompressedMomentBufferValidationTests
{
    private sealed class Buffer : IGpuBuffer
    {
        internal Buffer(long bytes) => SizeInBytes = bytes;
        public int Size => 1024; // Element count must never stand in for byte capacity.
        public long SizeInBytes { get; }
        public IntPtr Handle => IntPtr.Zero;
        public void Dispose() { }
    }

    [Theory]
    [InlineData(1L, 2L)]
    [InlineData(7L, 8L)]
    [InlineData(2147483647L, 4294967294L)]
    public void UndersizedByteCapacity_IsRejected(long capacity, long required)
    {
        var error = Assert.Throws<ArgumentException>(() =>
            CompressedMomentBufferValidation.RequireCapacity(new Buffer(capacity), required, "moment"));
        Assert.Equal("moment", error.ParamName);
    }

    [Theory]
    [InlineData(0L)]
    [InlineData(8L)]
    [InlineData(4294967294L)]
    public void ExactByteCapacity_IsAccepted(long bytes)
        => CompressedMomentBufferValidation.RequireCapacity(new Buffer(bytes), bytes, "moment");

    [Theory]
    [InlineData("param")]
    [InlineData("gradient")]
    [InlineData("mQuant")]
    [InlineData("vQuant")]
    [InlineData("mScales")]
    [InlineData("vScales")]
    public void Int8Layout_RejectsEachUndersizedBuffer(string shortBuffer)
    {
        IGpuBuffer State(string name, long bytes) => new Buffer(bytes - (name == shortBuffer ? 1 : 0));
        var error = Assert.Throws<ArgumentException>(() => CompressedMomentBufferValidation.ValidateInt8(4, 2, 2,
            State("param", 16), State("gradient", 16), State("mQuant", 4), State("vQuant", 4),
            State("mScales", 8), State("vScales", 8)));
        Assert.Equal(shortBuffer, error.ParamName);
    }

    [Theory]
    [InlineData(0, 2, 1, "paramLength")]
    [InlineData(4, 0, 2, "blockSize")]
    [InlineData(4, 2, 3, "numBlocks")]
    public void Int8Layout_RejectsInvalidGeometry(int length, int blockSize, int blocks, string parameter)
    {
        var valid = new Buffer(16);
        var error = Assert.ThrowsAny<ArgumentException>(() => CompressedMomentBufferValidation.ValidateInt8(
            length, blockSize, blocks, valid, valid, valid, valid, valid, valid));
        Assert.Equal(parameter, error.ParamName);
    }

    [Fact]
    public void Int8Layout_UsesWideByteArithmetic()
    {
        var floats = new Buffer((long)int.MaxValue * sizeof(float));
        var quantized = new Buffer(int.MaxValue); var scale = new Buffer(sizeof(float));
        CompressedMomentBufferValidation.ValidateInt8(int.MaxValue, int.MaxValue, 1,
            floats, floats, quantized, quantized, scale, scale);
    }
}
