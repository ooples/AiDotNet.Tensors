using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The rest of the row-wise scatter family through its declared programs (#1441) against the hand-written kernels on
/// the device: the scatter maximum with its argmax, ScatterAddRows, the four backward passes and the activation-layout
/// ScatterMean. Indices include rows outside the groups and an empty last group; values include infinities. Each call
/// must launch the declared program -- a decline fails rather than skips.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DeclaredScatterRowsFamilyRouteTests
{
    public static TheoryData<int, int, int> Shapes => new()
    {
        { 1, 1, 3 },
        { 300, 33, 7 },
        { 1000, 16, 37 },
        { 4096, 8, 256 },
    };

    private static (int[] Ids, float[] Values) Data(int sourceRows, int innerSize, int groups)
    {
        var random = RandomHelper.CreateSeededRandom(1441 + sourceRows);
        var ids = new int[sourceRows];
        for (int i = 0; i < sourceRows; i++)
            ids[i] = i % 11 == 10 ? -1 : (groups > 1 ? random.Next(groups - 1) : 0);   // the last group stays empty
        if (sourceRows == 1) ids[0] = Math.Min(1, groups - 1);
        var values = new float[sourceRows * innerSize];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 6 - 3);
        if (values.Length > 5) { values[3] = float.NegativeInfinity; values[5] = float.PositiveInfinity; }
        return (ids, values);
    }

    private static IGpuBuffer Ints(CudaBackend backend, int[] data)
    {
        var buffer = backend.AllocateByteBuffer(data.Length * sizeof(int));
        var bytes = new byte[data.Length * sizeof(int)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        backend.UploadByteBuffer(buffer, bytes);
        return buffer;
    }

    private static int[] DownloadInts(CudaBackend backend, IGpuBuffer buffer, int count)
    {
        var bytes = backend.DownloadByteBuffer(buffer, count * sizeof(int));
        var ints = new int[count];
        Buffer.BlockCopy(bytes, 0, ints, 0, bytes.Length);
        return ints;
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i]) || float.IsInfinity(expected[i]))
            {
                Assert.True(expected[i].Equals(actual[i]), $"{what} index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
                continue;
            }
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                $"{what} index {i}: hand-written {expected[i]:G9}, declared {actual[i]:G9}");
        }
    }

    private static CudaBackend OpenOrSkip()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        var backend = CudaBackend.CreateOrThrow();
        backend.DeclaredScatterRowsSwitch = true;
        if (!backend.DeclaredKernelsEnabled)
        {
            backend.Dispose();
            Skip.If(true, "The declared route is off for this process or this architecture has no released codegen kernels.");
        }
        return backend;
    }

    // Runs the hand-written then the declared call, asserting the declared one launched exactly `launches` programs.
    private static void Both(CudaBackend backend, Action call, int launches = 1)
    {
        backend.DeclaredScatterRowsSwitch = false;
        call();
        backend.DeclaredScatterRowsSwitch = true;
        long before = backend.DeclaredProgramLaunches;
        call();
        Assert.True(backend.DeclaredProgramLaunches == before + launches,
            "The declared route declined on a device that supports it.");
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void ForwardMaxAddAndMean_MatchTheHandWrittenKernels(int sourceRows, int innerSize, int groups)
    {
        using var backend = OpenOrSkip();
        var (ids, values) = Data(sourceRows, innerSize, groups);
        using var source = backend.AllocateBuffer(values);
        using var indices = Ints(backend, ids);
        int cells = groups * innerSize;

        // Scatter maximum with argmax.
        using var maxHand = backend.AllocateBuffer(cells);
        using var argHand = backend.AllocateBuffer(cells);
        using var maxDecl = backend.AllocateBuffer(cells);
        using var argDecl = backend.AllocateBuffer(cells);
        backend.DeclaredScatterRowsSwitch = false;
        backend.ScatterMaxWithArgmaxRows(source, indices, maxHand, argHand, sourceRows, innerSize, groups);
        backend.DeclaredScatterRowsSwitch = true;
        long before = backend.DeclaredProgramLaunches;
        backend.ScatterMaxWithArgmaxRows(source, indices, maxDecl, argDecl, sourceRows, innerSize, groups);
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        Assert.Equal(backend.DownloadBuffer(maxHand), backend.DownloadBuffer(maxDecl));
        Assert.Equal(backend.DownloadBuffer(argHand), backend.DownloadBuffer(argDecl));

        // ScatterAddRows.
        using var addHand = backend.AllocateBuffer(cells);
        using var addDecl = backend.AllocateBuffer(cells);
        backend.DeclaredScatterRowsSwitch = false;
        backend.ScatterAddRows(source, indices, addHand, sourceRows, innerSize, groups);
        backend.DeclaredScatterRowsSwitch = true;
        before = backend.DeclaredProgramLaunches;
        backend.ScatterAddRows(source, indices, addDecl, sourceRows, innerSize, groups);
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        AssertClose(backend.DownloadBuffer(addHand), backend.DownloadBuffer(addDecl), "scatter add rows");

        // Activation-layout ScatterMean with int32 counts.
        using var meanHand = backend.AllocateBuffer(cells);
        using var meanDecl = backend.AllocateBuffer(cells);
        using var countsHand = backend.AllocateByteBuffer(groups * sizeof(int));
        using var countsDecl = backend.AllocateByteBuffer(groups * sizeof(int));
        backend.DeclaredScatterRowsSwitch = false;
        backend.ScatterMean(source, indices, meanHand, countsHand, sourceRows * innerSize, groups, innerSize);
        backend.DeclaredScatterRowsSwitch = true;
        before = backend.DeclaredProgramLaunches;
        backend.ScatterMean(source, indices, meanDecl, countsDecl, sourceRows * innerSize, groups, innerSize);
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        Assert.Equal(DownloadInts(backend, countsHand, groups), DownloadInts(backend, countsDecl, groups));
        AssertClose(backend.DownloadBuffer(meanHand), backend.DownloadBuffer(meanDecl), "scatter mean");
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void BackwardPasses_MatchTheHandWrittenKernels(int sourceRows, int innerSize, int groups)
    {
        using var backend = OpenOrSkip();
        var (ids, values) = Data(sourceRows, innerSize, groups);
        var random = RandomHelper.CreateSeededRandom(1442 + sourceRows);
        int cells = groups * innerSize, rowCells = sourceRows * innerSize;
        var gradGroups = new float[cells];
        for (int i = 0; i < cells; i++) gradGroups[i] = (float)(random.NextDouble() * 4 - 2);
        var gradRows = new float[rowCells];
        for (int i = 0; i < rowCells; i++) gradRows[i] = (float)(random.NextDouble() * 4 - 2);

        using var source = backend.AllocateBuffer(values);
        using var indices = Ints(backend, ids);
        using var gradOut = backend.AllocateBuffer(gradGroups);
        using var gradRowBuf = backend.AllocateBuffer(gradRows);

        // Add backward.
        using var addHand = backend.AllocateBuffer(rowCells);
        using var addDecl = backend.AllocateBuffer(rowCells);
        backend.DeclaredScatterRowsSwitch = false;
        backend.ScatterAddBackwardRows(gradOut, indices, addHand, sourceRows, innerSize, groups);
        backend.DeclaredScatterRowsSwitch = true;
        long before = backend.DeclaredProgramLaunches;
        backend.ScatterAddBackwardRows(gradOut, indices, addDecl, sourceRows, innerSize, groups);
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        Assert.Equal(backend.DownloadBuffer(addHand), backend.DownloadBuffer(addDecl));

        // Mean backward, with int32 counts including a zero.
        var counts = new int[groups];
        foreach (int g in ids) if (g >= 0 && g < groups) counts[g]++;
        using var countsBuf = Ints(backend, counts);
        using var meanHand = backend.AllocateBuffer(rowCells);
        using var meanDecl = backend.AllocateBuffer(rowCells);
        backend.DeclaredScatterRowsSwitch = false;
        backend.ScatterMeanBackwardRows(gradOut, indices, countsBuf, meanHand, sourceRows, innerSize, groups);
        backend.DeclaredScatterRowsSwitch = true;
        before = backend.DeclaredProgramLaunches;
        backend.ScatterMeanBackwardRows(gradOut, indices, countsBuf, meanDecl, sourceRows, innerSize, groups);
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        AssertClose(backend.DownloadBuffer(meanHand), backend.DownloadBuffer(meanDecl), "scatter mean backward");

        // Max backward, from the forward's argmax as int32 (-1 for the empty group).
        using var maxOut = backend.AllocateBuffer(cells);
        using var argF = backend.AllocateBuffer(cells);
        backend.DeclaredScatterRowsSwitch = false;
        backend.ScatterMaxWithArgmaxRows(source, indices, maxOut, argF, sourceRows, innerSize, groups);
        var argmax = Array.ConvertAll(backend.DownloadBuffer(argF), v => (int)v);
        using var argmaxBuf = Ints(backend, argmax);
        using var maxHand = backend.AllocateBuffer(rowCells);
        using var maxDecl = backend.AllocateBuffer(rowCells);
        backend.ScatterMaxBackwardRows(gradOut, argmaxBuf, maxHand, sourceRows, innerSize, groups);
        backend.DeclaredScatterRowsSwitch = true;
        before = backend.DeclaredProgramLaunches;
        backend.ScatterMaxBackwardRows(gradOut, argmaxBuf, maxDecl, sourceRows, innerSize, groups);
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        AssertClose(backend.DownloadBuffer(maxHand), backend.DownloadBuffer(maxDecl), "scatter max backward");

        // Softmax backward, from the forward softmax.
        using var soft = backend.AllocateBuffer(rowCells);
        backend.DeclaredScatterRowsSwitch = false;
        backend.ScatterSoftmaxRows(source, indices, soft, sourceRows, innerSize, groups);
        using var softHand = backend.AllocateBuffer(rowCells);
        using var softDecl = backend.AllocateBuffer(rowCells);
        backend.ScatterSoftmaxBackwardRows(gradRowBuf, soft, indices, softHand, sourceRows, innerSize, groups);
        backend.DeclaredScatterRowsSwitch = true;
        before = backend.DeclaredProgramLaunches;
        backend.ScatterSoftmaxBackwardRows(gradRowBuf, soft, indices, softDecl, sourceRows, innerSize, groups);
        Assert.Equal(before + 1, backend.DeclaredProgramLaunches);
        var softDeclared = backend.DownloadBuffer(softDecl);
        AssertClose(backend.DownloadBuffer(softHand), softDeclared, "scatter softmax backward");
        for (int i = 0; i < sourceRows; i++)
            if (ids[i] < 0 || ids[i] >= groups)
                for (int j = 0; j < innerSize; j++) Assert.Equal(0f, softDeclared[i * innerSize + j]);
    }
}