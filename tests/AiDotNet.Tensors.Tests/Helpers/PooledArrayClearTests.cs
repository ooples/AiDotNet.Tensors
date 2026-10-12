using System;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// <see cref="PooledArrayClear"/> zeroes exactly [start, start + length): serially below
/// <see cref="PooledArrayClear.ParallelMinBytes"/>, split across workers above it.
/// </summary>
public class PooledArrayClearTests
{
    [Theory]
    [InlineData(1000, 10, 900)]          // small: serial path
    [InlineData(1 << 20, 0, 1 << 20)]    // 4 MB of floats, whole array: parallel path
    [InlineData(3_000_001, 17, 2_999_000)] // ragged range inside a large array: parallel path
    public void Float_ZeroesExactlyTheRange(int size, int start, int length)
    {
        var data = Enumerable.Range(0, size).Select(i => i + 1f).ToArray();
        PooledArrayClear.Clear(data, start, length);
        for (int i = 0; i < size; i++)
        {
            bool inside = i >= start && i < start + length;
            Assert.True(inside ? data[i] == 0f : data[i] == i + 1f, $"index {i} (range [{start}, {start + length}))");
        }
    }

    [Fact]
    public void ReferenceType_ClearsReferencesAboveTheParallelThreshold()
    {
        // 8-byte references: 200k elements is 1.6 MB, above the 1 MB threshold.
        var data = Enumerable.Range(0, 200_000).Select(i => (object)i).ToArray();
        PooledArrayClear.Clear(data, 5, data.Length - 10);
        for (int i = 0; i < data.Length; i++)
            Assert.True(i >= 5 && i < data.Length - 5 ? data[i] is null : data[i] is int, $"index {i}");
    }
}
