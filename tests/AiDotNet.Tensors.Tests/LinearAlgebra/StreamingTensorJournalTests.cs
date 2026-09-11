using System;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

[Collection("EngineCurrentGlobalState")]
public sealed class StreamingTensorJournalTests
{
    [Fact]
    public void OversizedDeferredPayload_IsRejectedBeforeMaterialization()
    {
        using var tensor = Tensor<double>.CreateDeferred(new[] { int.MaxValue / sizeof(double) + 1 });
        using var journal = new StreamingTensorJournal<double>(32);
        Assert.Null(tensor.GetLiveBackingArrayOrNull());

        Assert.Throws<NotSupportedException>(() => journal.Append(tensor));

        Assert.Null(tensor.GetLiveBackingArrayOrNull());
        Assert.Equal(0, journal.Count);
        Assert.Equal(0, journal.GetReport().DiskWriteBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailedAppend_PoisonsOnlyWhenRangeCleanupAlsoFails(bool failCleanup)
    {
        bool failWrite = true;
        var observed = new System.Collections.Generic.List<StreamingTensorJournalFileOperation>();
        using var journal = new StreamingTensorJournal<float>(32, null, operation =>
        {
            observed.Add(operation);
            if (operation == StreamingTensorJournalFileOperation.Write && failWrite)
                throw new IOException("Injected backing-store write failure.");
            if (operation == StreamingTensorJournalFileOperation.ReleaseRange && failCleanup)
                throw new IOException("Injected backing-range cleanup failure.");
        });
        var retained = journal.Append(new Tensor<float>(new float[] { 7 }, new[] { 1 }));
        var oversized = new Tensor<float>(Enumerable.Range(0, 16).Select(i => (float)i).ToArray(), new[] { 16 });

        IOException original = Assert.Throws<IOException>(() => journal.Append(oversized));
        Assert.Contains("write failure", original.Message);
        Assert.Equal(new[] { StreamingTensorJournalFileOperation.Write, StreamingTensorJournalFileOperation.ReleaseRange }, observed);
        if (failCleanup)
        {
            Assert.Null(retained.Segments[0].ResidentData);
            Assert.Throws<InvalidOperationException>(() => journal.Read(retained));
            Assert.Throws<InvalidOperationException>(() => journal.Remove(retained));
            Assert.Throws<InvalidOperationException>(() => journal.Append(oversized));
            Assert.Throws<InvalidOperationException>(() => journal.GetReport());
        }
        else
        {
            Assert.Equal(new[] { 7f }, journal.Read(retained).ToArray());
            failWrite = false;
            var recovered = journal.Append(oversized);
            Assert.Equal(oversized.ToArray(), journal.Read(recovered).ToArray());
            Assert.Equal(2, journal.Count);
        }
    }

    [Fact]
    public void FloatJournal_RoundTripsBitIdenticallyAfterSpill()
    {
        using var journal = new StreamingTensorJournal<float>(maxResidentBytes: 16);
        int[] firstBits =
        {
            0, unchecked((int)0x80000000), 1, unchecked((int)0x80000001),
            0x7f800000, unchecked((int)0xff800000), 0x7fc01234, unchecked((int)0xffc05678),
        };
        int[] secondBits =
        {
            0x3f000000, unchecked((int)0xbf000000), 0x7f7fffff, unchecked((int)0xff7fffff),
            0x00800000, unchecked((int)0x80800000), 0x00012345, unchecked((int)0x80012345),
        };
        var first = new Tensor<float>(firstBits.Select(SingleFromBits).ToArray(), new[] { 8 });
        var second = new Tensor<float>(secondBits.Select(SingleFromBits).ToArray(), new[] { 8 });

        var firstRecord = journal.Append(first);
        var secondRecord = journal.Append(second);
        Tensor<float> firstReplay = journal.Read(firstRecord);
        Tensor<float> secondReplay = journal.Read(secondRecord);

        Assert.Equal(firstBits, firstReplay.ToArray().Select(SingleBits));
        Assert.Equal(secondBits, secondReplay.ToArray().Select(SingleBits));
        StreamingTensorJournalReport report = journal.GetReport();
        Assert.True(report.DiskWriteBytes > 0);
        Assert.InRange(report.ResidentBytes, 0, report.MaxResidentBytes);
        Assert.InRange(report.ResidentBytesPeak, 0, report.MaxResidentBytes);
    }

    [Fact]
    public void DoubleJournal_RestoresExistingTensorBitIdentically()
    {
        using var journal = new StreamingTensorJournal<double>(maxResidentBytes: 64);
        long[] sourceBits =
        {
            0L, unchecked((long)0x8000000000000000), 1L,
            0x7ff0000000000000, unchecked((long)0xfff0000000000000),
            0x7ff8000000001234,
        };
        var source = new Tensor<double>(
            sourceBits.Select(BitConverter.Int64BitsToDouble).ToArray(), new[] { 2, 3 });
        var record = journal.Append(source);
        var destination = new Tensor<double>(new[] { 2, 3 });
        int versionBeforeRestore = destination.Version;

        journal.Restore(record, destination);

        Assert.Equal(sourceBits, destination.ToArray().Select(BitConverter.DoubleToInt64Bits));
        Assert.Equal(new[] { 2, 3 }, record.Shape);
        Assert.Equal(versionBeforeRestore + 1, destination.Version);
    }

    [Fact]
    public void Int64Journal_RoundTripsSparseIndicesWithoutNumericConversion()
    {
        using var journal = new StreamingTensorJournal<long>(maxResidentBytes: 16);
        long[] expected = { 0, -1, int.MaxValue, (long)int.MaxValue + 1, long.MaxValue };
        var source = new Tensor<long>(expected, new[] { expected.Length });

        Tensor<long> replay = journal.Read(journal.Append(source));

        Assert.Equal(expected, replay.ToArray());
    }

    [Fact]
    public void Replay_UsesOneBoundedReusableBufferForOversizedRecord()
    {
        using var journal = new StreamingTensorJournal<float>(maxResidentBytes: 32);
        int[] expectedBits = Enumerable.Range(0, 37)
            .Select(i => SingleBits(i + 0.25f))
            .ToArray();
        var source = new Tensor<float>(
            expectedBits.Select(SingleFromBits).ToArray(), new[] { 37 });
        var actualBits = new int[source.Length];
        Tensor<float>? firstBuffer = null;
        int callbackCount = 0;

        var record = journal.Append(source);
        journal.Replay(record, (offset, buffer, count) =>
        {
            firstBuffer ??= buffer;
            Assert.Same(firstBuffer, buffer);
            Assert.InRange(count, 1, 4);
            for (int i = 0; i < count; i++)
                actualBits[offset + i] = SingleBits(buffer[i]);
            callbackCount++;
        });

        Assert.True(callbackCount > 1);
        Assert.Equal(expectedBits, actualBits);
        Assert.InRange(journal.GetReport().ResidentBytesPeak, 0, 32);
    }

    [Fact]
    public void Dispose_ReleasesResidentPayloadRootedByCallerHeldRecord()
    {
        var journal = new StreamingTensorJournal<float>(maxResidentBytes: 256);
        var record = journal.Append(new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 }));
        Assert.NotNull(record.Segments[0].ResidentData);

        journal.Dispose();

        Assert.Null(record.Segments[0].ResidentData);
    }

    [Fact]
    public void ContiguousOffsetViewRoundTripsAndNonContiguousViewIsRejected()
    {
        using var journal = new StreamingTensorJournal<float>(64);
        var source = new Tensor<float>(new float[] { 10, 20, 30, 40, 50, 60 }, new[] { 2, 3 });
        Tensor<float> contiguousOffsetView = source.Slice(1);
        Assert.True(contiguousOffsetView.IsContiguous);
        Assert.Equal(new float[] { 40, 50, 60 }, journal.Read(journal.Append(contiguousOffsetView)).ToArray());

        Tensor<float> nonContiguous = source.Transpose(new[] { 1, 0 });
        Assert.False(nonContiguous.IsContiguous);
        Assert.Throws<NotSupportedException>(() => journal.Append(nonContiguous));
    }

    [Fact]
    public void Restore_RejectsWrongShapeAndNonContiguousDestinationBeforeMutation()
    {
        using var journal = new StreamingTensorJournal<float>(64);
        var record = journal.Append(new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 }));
        var wrongShape = new Tensor<float>(new[] { 4 });
        var nonContiguous = new Tensor<float>(new[] { 2, 2 }).Transpose(new[] { 1, 0 });

        Assert.Throws<ArgumentException>(() => journal.Restore(record, wrongShape));
        Assert.Throws<NotSupportedException>(() => journal.Restore(record, nonContiguous));
        Assert.Equal(new float[4], wrongShape.ToArray());
    }

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void Restore_InvalidatesPrimedPhysicalGpuValueCache(bool activation, bool inference)
    {
        using var gpu = new DirectGpuTensorEngine();
        RequireGpuIfRequested(gpu);
        bool previousStrict = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var seed = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
            var destination = activation ? gpu.TensorMultiplyScalar(seed, 1.0f) : seed;
            if (!activation) gpu.RegisterResidentParamBuffer(destination);
            Assert.Equal(new float[] { 1, 2, 3, 4 }, destination.ToArray());
            Assert.True(gpu.IsDeviceResidentArray(destination.GetReadOnlyDataArray()));
            int autogradVersion = destination.Version;

            using var journal = new StreamingTensorJournal<float>(16);
            var record = journal.Append(new Tensor<float>(new float[] { 10, 20, 30, 40 }, new[] { 4 }));
            if (inference)
            {
                using (new InferenceModeScope<float>()) journal.Restore(record, destination);
                Assert.Equal(autogradVersion, destination.Version);
            }
            else journal.Restore(record, destination);

            Assert.Equal(new float[] { 20, 40, 60, 80 }, gpu.TensorMultiplyScalar(destination, 2.0f).ToArray());
        }
        finally { DirectGpuTensorEngine.ThrowOnGpuKernelFallback = previousStrict; }
    }

    [SkippableFact]
    public void ResidentCache_MutationThroughSharedViewInvalidatesSourceSnapshot()
    {
        using var gpu = new DirectGpuTensorEngine();
        RequireGpuIfRequested(gpu);
        bool previousStrict = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
        try
        {
            var source = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 4 });
            gpu.RegisterResidentParamBuffer(source);
            Tensor<float> view = source.Reshape(new[] { 2, 2 });
            view[0] = 10f;
            Assert.Equal(new float[] { 20, 4, 6, 8 }, gpu.TensorMultiplyScalar(source, 2f).ToArray());
        }
        finally { DirectGpuTensorEngine.ThrowOnGpuKernelFallback = previousStrict; }
    }

    private static void RequireGpuIfRequested(DirectGpuTensorEngine gpu)
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "A physical GPU was required, but no GPU backend initialized.");
        Skip.IfNot(gpu.IsGpuAvailable, "No physical GPU backend initialized.");
    }

    [Fact]
    public void RecordOwnershipAndRemoval_AreEnforced()
    {
        using var first = new StreamingTensorJournal<float>(maxResidentBytes: 64);
        using var second = new StreamingTensorJournal<float>(maxResidentBytes: 64);
        var record = first.Append(new Tensor<float>(new[] { 2 }));

        Assert.Throws<ArgumentException>(() => second.Read(record));
        first.Remove(record);
        Assert.Equal(0, first.Count);
        Assert.Throws<InvalidOperationException>(() => first.Read(record));
    }

    [Fact]
    public void UnsupportedElementTypeAndInvalidBudget_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new StreamingTensorJournal<float>(0));
        Assert.Throws<NotSupportedException>(() => new StreamingTensorJournal<decimal>(64));
    }

    [Fact]
    public void Append_IsReadOnlyForCopyOnWriteTensor()
    {
        var source = new Tensor<float>(new[] { 4 });
        var clone = (Tensor<float>)source.CloneShared();
        Assert.True(source.IsCowShared);
        Assert.True(clone.IsCowShared);

        using var journal = new StreamingTensorJournal<float>(32);
        var record = journal.Append(clone);

        Assert.True(source.IsCowShared);
        Assert.True(clone.IsCowShared);
        Assert.Equal(clone.ToArray(), journal.Read(record).ToArray());
    }

    [Fact]
    public void Remove_ReclaimsAndReusesBackingFileRanges()
    {
        using var journal = new StreamingTensorJournal<float>(4);
        var tensor = new Tensor<float>(Enumerable.Range(0, 64).Select(i => (float)i).ToArray(), new[] { 64 });

        for (int iteration = 0; iteration < 8; iteration++)
        {
            var record = journal.Append(tensor);
            Assert.Equal(tensor.Length * sizeof(float), journal.GetReport().BackingStoreBytes);
            journal.Remove(record);
            Assert.Equal(0, journal.GetReport().BackingStoreBytes);
        }
    }

    [Fact]
    public void Remove_ReusesInteriorFileRangeWithoutCorruptingNeighborRecords()
    {
        using var journal = new StreamingTensorJournal<float>(4);
        var first = new Tensor<float>(Enumerable.Range(0, 8).Select(i => i + 0.25f).ToArray(), new[] { 8 });
        var middle = new Tensor<float>(Enumerable.Range(0, 8).Select(i => i + 100.5f).ToArray(), new[] { 8 });
        var last = new Tensor<float>(Enumerable.Range(0, 8).Select(i => i + 200.75f).ToArray(), new[] { 8 });
        var replacement = new Tensor<float>(Enumerable.Range(0, 8).Select(i => i - 50.125f).ToArray(), new[] { 8 });

        var firstRecord = journal.Append(first);
        var middleRecord = journal.Append(middle);
        var lastRecord = journal.Append(last);
        long fullLength = journal.GetReport().BackingStoreBytes;
        journal.Remove(middleRecord);

        var replacementRecord = journal.Append(replacement);

        Assert.Equal(fullLength, journal.GetReport().BackingStoreBytes);
        Assert.Equal(first.ToArray(), journal.Read(firstRecord).ToArray());
        Assert.Equal(replacement.ToArray(), journal.Read(replacementRecord).ToArray());
        Assert.Equal(last.ToArray(), journal.Read(lastRecord).ToArray());

        journal.Remove(lastRecord);
        journal.Remove(replacementRecord);
        Assert.Equal(first.Length * sizeof(float), journal.GetReport().BackingStoreBytes);
    }

    [Fact]
    public void Replay_RejectsReentrantAndConcurrentLifecycleOperationsWithoutDeadlock()
    {
        using var journal = new StreamingTensorJournal<float>(16);
        var source = new Tensor<float>(Enumerable.Range(0, 20).Select(i => (float)i).ToArray(), new[] { 20 });
        var record = journal.Append(source);
        int callbacks = 0;

        journal.Replay(record, (_, _, _) =>
        {
            callbacks++;
            Assert.Throws<InvalidOperationException>(() => journal.Remove(record));
            Assert.Throws<InvalidOperationException>(() => journal.Dispose());
            Assert.Throws<InvalidOperationException>(() => journal.Replay(record, (_, _, _) => { }));

            Task<Exception?> concurrentRead = Task.Run(() => Record.Exception(() => journal.Read(record)));
            Assert.True(concurrentRead.Wait(TimeSpan.FromSeconds(5)),
                "A concurrent journal operation blocked behind the replay callback.");
            Assert.IsType<InvalidOperationException>(concurrentRead.Result);
        });

        Assert.True(callbacks > 1);
        Assert.Equal(source.ToArray(), journal.Read(record).ToArray());
    }

    [Fact]
    public void Dispose_RemovesPrivateBackingDirectory()
    {
        string root = Path.Combine(Path.GetTempPath(), "aidotnet-journal-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var journal = new StreamingTensorJournal<float>(4, root);
            journal.Append(new Tensor<float>(Enumerable.Range(0, 64).Select(i => (float)i).ToArray(), new[] { 64 }));
            Assert.Single(Directory.GetDirectories(root));

            journal.Dispose();

            Assert.Empty(Directory.GetDirectories(root));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ConcurrentRecordLifecycles_AreSerializedWithoutCorruption()
    {
        using var journal = new StreamingTensorJournal<double>(64);
        var failures = new ConcurrentQueue<Exception>();

        Parallel.For(0, 32, i =>
        {
            try
            {
                var source = new Tensor<double>(new[] { 16 });
                for (int j = 0; j < source.Length; j++) source[j] = i * 1000.0 + j;
                var record = journal.Append(source);
                Assert.Equal(source.ToArray(), journal.Read(record).ToArray());
                journal.Remove(record);
            }
            catch (Exception exception)
            {
                failures.Enqueue(exception);
            }
        });

        Assert.Empty(failures);
        Assert.Equal(0, journal.Count);
        Assert.InRange(journal.GetReport().ResidentBytesPeak, 0, 64);
    }

    [Fact]
    public void ScalarAndZeroLengthShapes_RoundTripWithoutSpecialCases()
    {
        using var journal = new StreamingTensorJournal<float>(8);
        var scalar = new Tensor<float>(Array.Empty<int>());
        scalar[0] = SingleFromBits(unchecked((int)0x80000000));
        var empty = new Tensor<float>(new[] { 0, 3 });

        var scalarReplay = journal.Read(journal.Append(scalar));
        var emptyReplay = journal.Read(journal.Append(empty));

        Assert.Equal(SingleBits(scalar[0]), SingleBits(scalarReplay[0]));
        Assert.Empty(emptyReplay.ToArray());
        Assert.Equal(new[] { 0, 3 }, emptyReplay._shape);
    }

    private static float SingleFromBits(int bits) => BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);

    private static int SingleBits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    [Fact]
    public void PropertiesAndRecordsRejectUseAfterDisposal()
    {
        var journal = new StreamingTensorJournal<float>(16);
        var record = journal.Append(new Tensor<float>(new[] { 2 }));
        journal.Dispose();

        Assert.Throws<ObjectDisposedException>(() => journal.Read(record));
        Assert.Throws<ObjectDisposedException>(() => journal.Remove(record));
        Assert.Throws<ObjectDisposedException>(() => journal.GetReport());
        Assert.Throws<ObjectDisposedException>(() => _ = journal.Count);
        Assert.Throws<ObjectDisposedException>(() => _ = journal.ResidentBytes);
    }
}
