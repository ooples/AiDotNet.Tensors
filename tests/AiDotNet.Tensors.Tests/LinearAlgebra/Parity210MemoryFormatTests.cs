using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210MemoryFormatTests
{
    [Fact]
    public void MemoryFormat_Defaults_ToContiguous()
    {
        var x = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        Assert.Equal(MemoryFormat.Contiguous, x.MemoryFormat);
    }

    [Fact]
    public void MemoryFormat_IsSettable()
    {
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        x.MemoryFormat = MemoryFormat.ChannelsLast;
        Assert.Equal(MemoryFormat.ChannelsLast, x.MemoryFormat);
    }

    [Fact]
    public void PreserveLayoutFrom_CopiesFormat()
    {
        var src = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 }) { MemoryFormat = MemoryFormat.ChannelsLast };
        var dst = new Tensor<float>(new[] { 1f, 2f }, new[] { 2 });
        dst.PreserveLayoutFrom(src);
        Assert.Equal(MemoryFormat.ChannelsLast, dst.MemoryFormat);
    }

    [Fact]
    public void PreserveLayoutFrom_Returns_This_ForChaining()
    {
        var src = new Tensor<float>(new[] { 1f }, new[] { 1 }) { MemoryFormat = MemoryFormat.ChannelsLast3D };
        var dst = new Tensor<float>(new[] { 1f }, new[] { 1 });
        var returned = dst.PreserveLayoutFrom(src);
        Assert.Same(dst, returned);
        Assert.Equal(MemoryFormat.ChannelsLast3D, returned.MemoryFormat);
    }

    [Fact]
    public void PreserveLayoutFrom_NullSource_Throws()
    {
        var dst = new Tensor<float>(new[] { 1f }, new[] { 1 });
        Assert.Throws<System.ArgumentNullException>(() => dst.PreserveLayoutFrom(null!));
    }
}
