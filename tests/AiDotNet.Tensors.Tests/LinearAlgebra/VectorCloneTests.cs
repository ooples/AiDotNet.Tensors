using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Vector.Clone is the copy behind every copy-on-write detach. It used to spread the vector's yield enumerator into
/// a collection expression - an element-at-a-time walk into a growing buffer, copied again by the constructor - which
/// took ~750 us for 65K floats and allocated several times the payload.
/// </summary>
public class VectorCloneTests
{
    [Fact]
    public void Clone_CopiesValuesIndependently()
    {
        var source = new Vector<float>(1000);
        for (int i = 0; i < source.Length; i++) source[i] = i * 0.5f;

        var clone = source.Clone();
        Assert.Equal(source.Length, clone.Length);
        for (int i = 0; i < source.Length; i++) Assert.Equal(source[i], clone[i]);

        clone[3] = -1f;
        source[4] = -2f;
        Assert.Equal(1.5f, source[3]);
        Assert.Equal(2f, clone[4]);
    }

#if NET5_0_OR_GREATER
    [Fact]
    public void Clone_AllocatesAboutOnePayload()
    {
        const int n = 65536;
        var source = new Vector<float>(n);
        for (int i = 0; i < n; i++) source[i] = i;
        source.Clone(); // JIT and first-use allocations out of the measurement

        long before = GC.GetAllocatedBytesForCurrentThread();
        var clone = source.Clone();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(n, clone.Length);
        long payload = (long)n * sizeof(float);
        // One block copy plus object headers.
        Assert.True(allocated <= payload + 4096, $"Clone of {payload} bytes allocated {allocated} bytes.");
    }
#endif
}
