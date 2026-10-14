using System;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// <see cref="PooledArrayClear"/> zeroes exactly [start, start + length): serially below
/// <see cref="PooledArrayClear.ParallelMinBytes"/>, split across workers at and above it. Each case asserts the dispatch
/// the clear actually ran (<see cref="PooledArrayClear.LastClearChunks"/>) as well as the zeroed range, since a serial
/// clear produces the same range. Pins MaxDegreeOfParallelism, so it shares the serial CpuParallelSettings collection.
/// </summary>
[Collection("CpuParallelSettings")]
public class PooledArrayClearTests : IDisposable
{
    private const int Workers = 4;
    private readonly int _priorMaxDop = CpuParallelSettings.MaxDegreeOfParallelism;

    public PooledArrayClearTests() => CpuParallelSettings.MaxDegreeOfParallelism = Workers;

    public void Dispose() => CpuParallelSettings.MaxDegreeOfParallelism = _priorMaxDop;

    // The split the clear must take: one piece below the byte threshold, else the elementwise chunking.
    private static int ExpectedChunks(int length, int elementBytes)
        => (long)length * elementBytes < PooledArrayClear.ParallelMinBytes
            ? 1
            : Math.Max(1, Math.Min(Workers, length / CpuParallelSettings.ElementwiseGrain));

    [SkippableTheory]
    [InlineData(1000, 10, 900)]                    // small: serial
    [InlineData(300_000, 7, (1 << 18) - 1)]        // 4 bytes short of 1 MiB: serial
    [InlineData(300_000, 7, 1 << 18)]              // exactly 1 MiB: split
    [InlineData(1 << 20, 0, 1 << 20)]              // 4 MiB, whole array: split
    [InlineData(3_000_001, 17, 2_999_000)]         // ragged range inside a large array: split
    public void Float_ZeroesExactlyTheRange_WithTheExpectedDispatch(int size, int start, int length)
    {
        int expected = ExpectedChunks(length, sizeof(float));
        // An AIDOTNET_ELEMENTWISE_GRAIN override can make even a large clear one chunk; the range check still holds.
        if ((long)length * sizeof(float) >= PooledArrayClear.ParallelMinBytes)
            Skip.If(expected < 2, $"AIDOTNET_ELEMENTWISE_GRAIN={CpuParallelSettings.ElementwiseGrain} leaves {length} elements unsplit.");
        var data = Enumerable.Range(0, size).Select(i => i + 1f).ToArray();
        PooledArrayClear.Clear(data, start, length);
        Assert.Equal(expected, PooledArrayClear.LastClearChunks);
        for (int i = 0; i < size; i++)
        {
            bool inside = i >= start && i < start + length;
            Assert.True(inside ? data[i] == 0f : data[i] == i + 1f, $"index {i} (range [{start}, {start + length}))");
        }
    }

    [SkippableFact]
    public void ReferenceType_ClearsReferencesAboveTheParallelThreshold()
    {
        // 8-byte references (on 64-bit): 200k elements is 1.6 MB, above the 1 MB threshold.
        var data = Enumerable.Range(0, 200_000).Select(i => (object)i).ToArray();
        int length = data.Length - 10;
        int expected = ExpectedChunks(length, IntPtr.Size);
        Skip.If(expected < 2, $"{length} references of {IntPtr.Size} bytes stay below the split on this process.");
        PooledArrayClear.Clear(data, 5, length);
        Assert.Equal(expected, PooledArrayClear.LastClearChunks);
        for (int i = 0; i < data.Length; i++)
            Assert.True(i >= 5 && i < data.Length - 5 ? data[i] is null : data[i] is int, $"index {i}");
    }
}