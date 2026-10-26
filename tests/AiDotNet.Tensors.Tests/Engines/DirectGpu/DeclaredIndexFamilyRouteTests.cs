using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The index family through declared programs (#1441) against the hand-written kernels on the device: IndexWrite,
/// index copy / fill / add, ScatterReduce's sum, max and min, int32 take-along-dim, the linear take, float-id
/// IndexSelect and ScatterAddBackward. Each declared call must launch its program -- a decline fails.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeclaredIndexFamilyRouteTests
{
    private static IGpuBuffer Ints(CudaBackend backend, int[] data)
    {
        var buffer = backend.AllocateByteBuffer(data.Length * sizeof(int));
        var bytes = new byte[data.Length * sizeof(int)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        backend.UploadByteBuffer(buffer, bytes);
        return buffer;
    }

    private static void AssertClose(float[] expected, float[] actual, string what, float tolerance = 1e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i]) || float.IsInfinity(expected[i]))
            {
                Assert.True(expected[i].Equals(actual[i]), $"{what} index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
                continue;
            }
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1f, Math.Abs(expected[i])),
                $"{what} index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
        }
    }

    private static CudaBackend OpenOrSkip()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        var backend = CudaBackend.CreateOrThrow();
        if (!backend.DeclaredKernelsEnabled)
        {
            backend.Dispose();
            Skip.If(true, "The declared route is off for this process or this architecture has no released codegen kernels.");
        }
        return backend;
    }

    private static float[] Values(Random random, int count)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)(random.NextDouble() * 6 - 3);
        return v;
    }

    // Runs call on a fresh copy of seed, first hand-written then declared.
    private static (float[] Hand, float[] Declared) Both(CudaBackend backend, float[] seed, Action<IGpuBuffer> call)
    {
        var results = new float[2][];
        foreach (bool declared in new[] { false, true })
        {
            using var buffer = backend.AllocateBuffer(seed);
            backend.DeclaredKernelsSwitch = declared;
            long before = backend.DeclaredProgramLaunches;
            call(buffer);
            if (declared)
                Assert.True(backend.DeclaredProgramLaunches == before + 1, "The declared route declined on a device that supports it.");
            results[declared ? 1 : 0] = backend.DownloadBuffer(buffer);
        }
        backend.DeclaredKernelsSwitch = null;
        return (results[0], results[1]);
    }

    [SkippableTheory]
    [InlineData(2, 9, 5, 12)]
    [InlineData(4, 64, 16, 100)]
    public void IndexWritesAndAdds_MatchTheHandWrittenKernels(int outer, int indexLength, int inner, int destination)
    {
        using var backend = OpenOrSkip();
        var random = RandomHelper.CreateSeededRandom(1450 + indexLength);
        var seed = Values(random, outer * destination * inner);
        var source = Values(random, outer * indexLength * inner);
        using var src = backend.AllocateBuffer(source);
        // Duplicates and out-of-range indices: IndexWrite's own kernel resolves duplicates (the last wins).
        var dup = new int[indexLength];
        for (int j = 0; j < indexLength; j++) dup[j] = j % 7 == 6 ? destination + 3 : random.Next(destination - 1);
        using var dupBuf = Ints(backend, dup);
        // Unique indices (a permutation prefix): copy / fill race on duplicates in the hand-written kernels.
        var unique = new int[indexLength];
        var perm = new int[destination];
        for (int t = 0; t < destination; t++) perm[t] = t;
        for (int t = destination - 1; t > 0; t--) { int r = random.Next(t + 1); (perm[t], perm[r]) = (perm[r], perm[t]); }
        for (int j = 0; j < indexLength; j++) unique[j] = j < destination - 1 ? perm[j] : -1;
        using var uniqueBuf = Ints(backend, unique);

        var w0 = Both(backend, seed, o => backend.IndexWrite(o, dupBuf, src, 0f, 0, outer, indexLength, inner, destination));
        Assert.Equal(w0.Hand, w0.Declared);
        var w1 = Both(backend, seed, o => backend.IndexWrite(o, dupBuf, src, 2.5f, 1, outer, indexLength, inner, destination));
        Assert.Equal(w1.Hand, w1.Declared);
        var copy = Both(backend, seed, o => backend.Parity210IndexCopy(o, uniqueBuf, src, outer, destination, inner, indexLength));
        Assert.Equal(copy.Hand, copy.Declared);
        var fill = Both(backend, seed, o => backend.Parity210IndexFill(o, uniqueBuf, -4.25f, outer, destination, inner, indexLength));
        Assert.Equal(fill.Hand, fill.Declared);
        var add = Both(backend, seed, o => backend.Parity210IndexAdd(o, dupBuf, src, outer, destination, inner, indexLength));
        AssertClose(add.Hand, add.Declared, "index add");
    }

    [SkippableTheory]
    [InlineData(2, 9, 5, 3)]
    [InlineData(3, 200, 40, 8)]
    public void ScatterReduce_MatchesTheHandWrittenKernel(int outer, int sourceDim, int destinationDim, int inner)
    {
        using var backend = OpenOrSkip();
        var random = RandomHelper.CreateSeededRandom(1451 + sourceDim);
        var seed = Values(random, outer * destinationDim * inner);
        var source = Values(random, outer * sourceDim * inner);
        source[1] = float.NaN;
        var index = new int[source.Length];
        for (int k = 0; k < index.Length; k++) index[k] = k % 9 == 8 ? -1 : random.Next(destinationDim);
        using var src = backend.AllocateBuffer(source);
        using var idx = Ints(backend, index);
        foreach (int mode in new[] { 0, 2, 3 })
        {
            var r = Both(backend, seed, o => backend.ScatterReduce(o, src, idx, outer, sourceDim, destinationDim, inner, mode));
            if (mode == 0)
            {
                // A cell the NaN reaches is NaN in both; the sums otherwise differ only in the atomic's order.
                for (int k = 0; k < r.Hand.Length; k++)
                    if (float.IsNaN(r.Hand[k])) Assert.True(float.IsNaN(r.Declared[k]), "sum index " + k);
                    else Assert.True(Math.Abs(r.Hand[k] - r.Declared[k]) <= 1e-4f * Math.Max(1f, Math.Abs(r.Hand[k])),
                        $"sum index {k}: hand-written {r.Hand[k]:G9}, declared {r.Declared[k]:G9}");
            }
            else Assert.Equal(r.Hand, r.Declared);
        }
    }

    [SkippableFact]
    public void Gathers_MatchTheHandWrittenKernels()
    {
        using var backend = OpenOrSkip();
        var random = RandomHelper.CreateSeededRandom(1452);
        const int outer = 4, axisOut = 64, inner = 80, axisIn = 50;   // 20480 outputs, past the gathers' floor
        var input = Values(random, outer * axisIn * inner);
        using var inBuf = backend.AllocateBuffer(input);
        var take = new int[outer * axisOut * inner];
        for (int k = 0; k < take.Length; k++) take[k] = k % 13 == 12 ? axisIn + 2 : random.Next(axisIn);
        using var takeIdx = Ints(backend, take);
        var t1 = Both(backend, new float[take.Length], o => backend.Parity210TakeAlongDim(inBuf, takeIdx, o, outer, axisOut, inner, axisIn));
        Assert.Equal(t1.Hand, t1.Declared);

        var linear = new int[20000];
        for (int k = 0; k < linear.Length; k++) linear[k] = k % 11 == 10 ? -3 : random.Next(input.Length);
        using var linIdx = Ints(backend, linear);
        var t2 = Both(backend, new float[linear.Length], o => backend.Parity210TakeLinear(inBuf, linIdx, o, linear.Length, input.Length));
        Assert.Equal(t2.Hand, t2.Declared);

        // index_select by float ids, and ScatterAddBackward by int ids, over rows of width 80.
        const int rows = outer * axisIn, width = inner, picks = 300;
        var fids = new float[picks];
        var iids = new int[picks];
        for (int k = 0; k < picks; k++) { iids[k] = random.Next(rows); fids[k] = iids[k] + 0.75f; }
        using var fidBuf = backend.AllocateBuffer(fids);
        using var iidBuf = Ints(backend, iids);
        var s1 = Both(backend, new float[picks * width], o => backend.IndexSelect(inBuf, fidBuf, o, picks, width));
        Assert.Equal(s1.Hand, s1.Declared);
        var s2 = Both(backend, new float[picks * width], o => backend.ScatterAddBackward(inBuf, iidBuf, o, picks, width));
        Assert.Equal(s2.Hand, s2.Declared);
    }
}