using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Interop;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Interop;

/// <summary>
/// DLPack exchange: CPU round trips (contiguous, strided, legacy layout, empty, scalar), ownership (the producer's
/// deleter runs exactly once on dispose, never under a live span on finalization, and after queued kernels for a
/// finalized CUDA import; an export outlives its source; a repeated deleter call is not a double free), version and
/// dtype refusal, the read-only flag, and a CUDA round trip on a CUDA machine.
/// </summary>
[Collection("DLPack")]
public sealed class DLPackTests
{
    // ── A minimal C producer, built in unmanaged memory, with a counting deleter ─────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void Deleter(IntPtr self);

    private static readonly ConcurrentDictionary<IntPtr, int> s_deleterCalls = new();
    private static readonly Deleter s_countingDeleter = self => s_deleterCalls.AddOrUpdate(self, 1, (_, n) => n + 1);
    private static readonly IntPtr s_countingDeleterPtr = Marshal.GetFunctionPointerForDelegate(s_countingDeleter);

    private static int DeleterCalls(IntPtr self) => s_deleterCalls.TryGetValue(self, out int n) ? n : 0;

    /// <summary>A producer-owned float tensor: one block holding the header, shape, strides and data.</summary>
    private sealed unsafe class FakeProducer : IDisposable
    {
        public IntPtr Block { get; }
        public float* Data { get; }

        public FakeProducer(float[] values, long[] shape, long[]? strides, uint major = 1, uint minor = 3,
            ulong flags = 0, bool versioned = true, byte code = 2, byte bits = 32, int deviceType = 1, int deviceId = 0)
        {
            int header = versioned ? sizeof(DLManagedTensorVersionedLayout) : sizeof(DLManagedTensorLayout);
            Block = Marshal.AllocHGlobal(header + 16 * shape.Length + 4 * values.Length + 64);
            var shapePtr = (long*)((byte*)Block + header);
            var stridesPtr = shapePtr + shape.Length;
            Data = (float*)(stridesPtr + shape.Length);
            for (int i = 0; i < shape.Length; i++) shapePtr[i] = shape[i];
            if (strides is not null) for (int i = 0; i < strides.Length; i++) stridesPtr[i] = strides[i];
            for (int i = 0; i < values.Length; i++) Data[i] = values[i];
            var tensor = new DLTensorLayout
            {
                Data = (IntPtr)Data,
                DeviceType = deviceType,
                DeviceId = deviceId,
                NDim = shape.Length,
                Code = code,
                Bits = bits,
                Lanes = 1,
                Shape = shape.Length == 0 ? IntPtr.Zero : (IntPtr)shapePtr,
                Strides = strides is null ? IntPtr.Zero : (IntPtr)stridesPtr,
                ByteOffset = 0,
            };
            if (versioned)
                *(DLManagedTensorVersionedLayout*)Block = new DLManagedTensorVersionedLayout
                {
                    Major = major, Minor = minor, ManagerCtx = IntPtr.Zero, Deleter = s_countingDeleterPtr, Flags = flags, Tensor = tensor,
                };
            else
                *(DLManagedTensorLayout*)Block = new DLManagedTensorLayout
                {
                    Tensor = tensor, ManagerCtx = IntPtr.Zero, Deleter = s_countingDeleterPtr,
                };
        }

        public int Calls => DeleterCalls(Block);

        public void Dispose()
        {
            s_deleterCalls.TryRemove(Block, out _);
            Marshal.FreeHGlobal(Block);
        }
    }

    // The dlpack.h layouts, written out independently of the library's own declarations.
    [StructLayout(LayoutKind.Sequential)]
    private struct DLTensorLayout
    {
        public IntPtr Data;
        public int DeviceType;
        public int DeviceId;
        public int NDim;
        public byte Code;
        public byte Bits;
        public ushort Lanes;
        public IntPtr Shape;
        public IntPtr Strides;
        public ulong ByteOffset;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DLManagedTensorLayout
    {
        public DLTensorLayout Tensor;
        public IntPtr ManagerCtx;
        public IntPtr Deleter;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DLManagedTensorVersionedLayout
    {
        public uint Major;
        public uint Minor;
        public IntPtr ManagerCtx;
        public IntPtr Deleter;
        public ulong Flags;
        public DLTensorLayout Tensor;
    }

    private static unsafe DLManagedTensorVersionedLayout ReadVersioned(IntPtr p) => *(DLManagedTensorVersionedLayout*)p;

    // A raw unmanaged call, as a C consumer makes it (a delegate cast fails on .NET Framework when the pointer came
    // from a managed delegate of another type).
    private static unsafe void CallDeleter(IntPtr deleter, IntPtr self)
        => ((delegate* unmanaged[Cdecl]<IntPtr, void>)deleter)(self);

    private static Tensor<float> Iota(params int[] shape)
    {
        int n = shape.Aggregate(1, (a, b) => a * b);
        return new Tensor<float>(Enumerable.Range(0, n).Select(i => (float)i).ToArray(), shape);
    }

    // ── CPU round trips ──────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Cpu_RoundTrip_IsZeroCopy_AndReleasesTheExportOnDispose()
    {
        int liveBefore = DLPack.LiveExportCount;
        var source = Iota(3, 4);
        IntPtr capsule = source.ToDLPack();
        Assert.Equal(liveBefore + 1, DLPack.LiveExportCount);

        var header = ReadVersioned(capsule);
        Assert.Equal(1u, header.Major);
        Assert.Equal(1, header.Tensor.DeviceType);
        Assert.Equal(2, header.Tensor.NDim);
        Assert.Equal((byte)2, header.Tensor.Code);
        Assert.Equal((byte)32, header.Tensor.Bits);
        Assert.NotEqual(IntPtr.Zero, header.Tensor.Strides);   // mandatory since DLPack 1.2

        var imported = Tensor<float>.FromDLPack(capsule);
        Assert.Equal(new[] { 3, 4 }, imported.Shape.ToArray());
        Assert.Equal(source.ToArray(), imported.ToArray());

        // Zero-copy both ways: a write on either side is seen by the other.
        imported[1, 2] = -7f;
        Assert.Equal(-7f, source[1, 2]);
        source[2, 3] = 99f;
        Assert.Equal(99f, imported[2, 3]);

        Assert.Equal(liveBefore + 1, DLPack.LiveExportCount);
        imported.Dispose();   // the import's storage releases → our own export's deleter runs
        Assert.Equal(liveBefore, DLPack.LiveExportCount);
    }

    [Fact]
    public unsafe void Cpu_StridedView_ExportsItsStrides_AndRoundTrips()
    {
        var source = Iota(2, 3, 4);
        var transposed = source.Transpose(new[] { 2, 0, 1 });   // [4, 2, 3], a storage-sharing view
        Assert.False(transposed.IsContiguous);

        IntPtr capsule = transposed.ToDLPack();
        var header = ReadVersioned(capsule);
        var strides = (long*)header.Tensor.Strides;
        Assert.Equal(new long[] { 1, 12, 4 }, new[] { strides[0], strides[1], strides[2] });

        using var imported = Tensor<float>.FromDLPack(capsule);
        Assert.Equal(new[] { 4, 2, 3 }, imported.Shape.ToArray());
        Assert.Equal(transposed.ToArray(), imported.ToArray());
        Assert.Equal(source[1, 2, 3], imported[3, 1, 2]);
    }

    [Fact]
    public void Cpu_OffsetView_ExportsThePointerToItsFirstElement()
    {
        var source = Iota(4, 5);
        var row2 = source.Slice(0, 2, 3);   // rows [2, 3): a view at storage offset 10
        using var imported = Tensor<float>.FromDLPack(row2.ToDLPack());
        Assert.Equal(row2.ToArray(), imported.ToArray());
        Assert.Equal(10f, imported.ToArray()[0]);
    }

    [Fact]
    public void Cpu_LegacyLayout_RoundTrips()
    {
        int liveBefore = DLPack.LiveExportCount;
        var source = Iota(2, 5);
        var imported = Tensor<float>.FromDLPackLegacy(source.ToDLPackLegacy());
        Assert.Equal(source.ToArray(), imported.ToArray());
        imported.Dispose();
        Assert.Equal(liveBefore, DLPack.LiveExportCount);
    }

    [Fact]
    public void Cpu_IntegerAndDoubleElementTypes_RoundTrip()
    {
        var ints = new Tensor<int>(new[] { 1, -2, 3, int.MaxValue }, new[] { 4 });
        using (var back = Tensor<int>.FromDLPack(ints.ToDLPack())) Assert.Equal(ints.ToArray(), back.ToArray());
        var longs = new Tensor<long>(new[] { long.MinValue, 0L, 5L }, new[] { 3 });
        using (var back = Tensor<long>.FromDLPack(longs.ToDLPack())) Assert.Equal(longs.ToArray(), back.ToArray());
        var doubles = new Tensor<double>(new[] { 0.5, -1.25, 3e300 }, new[] { 3 });
        using (var back = Tensor<double>.FromDLPack(doubles.ToDLPack())) Assert.Equal(doubles.ToArray(), back.ToArray());
    }

    [Fact]
    public void Cpu_EmptyTensor_ExportsANullDataPointer_AndRoundTrips()
    {
        int liveBefore = DLPack.LiveExportCount;
        var empty = new Tensor<float>(new[] { 0, 3 });
        IntPtr capsule = empty.ToDLPack();
        Assert.Equal(IntPtr.Zero, ReadVersioned(capsule).Tensor.Data);
        var imported = Tensor<float>.FromDLPack(capsule);
        Assert.Equal(new[] { 0, 3 }, imported.Shape.ToArray());
        // Nothing to alias: the import released the export at once.
        Assert.Equal(liveBefore, DLPack.LiveExportCount);
    }

    [Fact]
    public void Cpu_ImportOfForeignStridedMemory_ReadsThroughTheStrides()
    {
        // A [2, 3] column-major producer tensor: element (i, j) at i + 2j.
        var values = new float[] { 0, 3, 1, 4, 2, 5 };
        using var producer = new FakeProducer(values, new long[] { 2, 3 }, new long[] { 1, 2 });
        var t = Tensor<float>.FromDLPack(producer.Block);
        Assert.Equal(new float[] { 0, 1, 2, 3, 4, 5 }, t.ToArray());
        t.Dispose();
        Assert.Equal(1, producer.Calls);
    }

    [Fact]
    public void Cpu_ImportOfNegativeStrides_IsAZeroCopyReversedView()
    {
        var values = new float[] { 10, 20, 30, 40 };
        using var producer = new FakeProducer(values, new long[] { 4 }, new long[] { -1 });
        // A reversed view: data points at the LAST element, stride -1 walks back to the first.
        unsafe
        {
            var header = (DLManagedTensorVersionedLayout*)producer.Block;
            header->Tensor.Data = (IntPtr)(producer.Data + 3);
        }
        using var t = Tensor<float>.FromDLPack(producer.Block);
        Assert.Equal(new float[] { 40, 30, 20, 10 }, t.ToArray());
        unsafe { producer.Data[0] = -1f; }
        Assert.Equal(-1f, t[3]);
    }

    // ── Ownership ───────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Import_RunsTheProducersDeleterExactlyOnce_WhenTheLastViewIsDisposed()
    {
        using var producer = new FakeProducer(new float[] { 1, 2, 3, 4, 5, 6 }, new long[] { 2, 3 }, new long[] { 3, 1 });
        var t = Tensor<float>.FromDLPack(producer.Block);
        var view = t.Reshape(3, 2);   // shares the storage
        Assert.Equal(0, producer.Calls);

        t.Dispose();
        Assert.Equal(0, producer.Calls);   // the view still uses the memory
        Assert.Equal(6f, view[2, 1]);

        view.Dispose();
        Assert.Equal(1, producer.Calls);

        t.Dispose();
        view.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        Assert.Equal(1, producer.Calls);   // repeated disposal and finalization never call it again
    }

    [Fact]
    public void Import_ThatIsNeverDisposed_KeepsTheMemory_ForASpanThatOutlivesTheTensor()
    {
        // A span over imported host memory is a raw pointer the GC does not track, so it outlives the tensor's last
        // reference. Finalization must not run the deleter under it: the import is leaked and counted instead.
        using var producer = new FakeProducer(new float[] { 1, 2, 3 }, new long[] { 3 }, new long[] { 1 });
        long leakedBefore = DLPack.LeakedHostImports;
        ReadOnlySpan<float> span = SpanOfADroppedImport(producer.Block);
        for (int i = 0; i < 5; i++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }

        Assert.True(DLPack.LeakedHostImports > leakedBefore, "the dropped import was never finalized");
        Assert.Equal(0, producer.Calls);
        Assert.Equal(new float[] { 1, 2, 3 }, span.ToArray());
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static ReadOnlySpan<float> SpanOfADroppedImport(IntPtr block)
    {
        var t = Tensor<float>.FromDLPack(block);
        Assert.Equal(2f, t[1]);
        return t.AsSpan();
    }

    [Fact]
    public unsafe void Export_OutlivesItsDisposedSource()
    {
        var source = Iota(64);
        IntPtr capsule = source.ToDLPack();
        source.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        // Allocate tensors that would reuse a recycled array, then read the exported memory directly.
        for (int i = 0; i < 8; i++) new Tensor<float>(new[] { 64 }).Fill(-5f);
        var header = ReadVersioned(capsule);
        var data = (float*)header.Tensor.Data;
        for (int i = 0; i < 64; i++) Assert.Equal(i, data[i]);
        CallDeleter(header.Deleter, capsule);
    }

    [Fact]
    public void Export_DeleterCalledTwice_DoesNotDoubleFree()
    {
        int liveBefore = DLPack.LiveExportCount;
        long unmatchedBefore = DLPack.UnmatchedDeleterCalls;
        IntPtr capsule = Iota(4).ToDLPack();
        IntPtr deleter = ReadVersioned(capsule).Deleter;

        CallDeleter(deleter, capsule);
        Assert.Equal(liveBefore, DLPack.LiveExportCount);
        CallDeleter(deleter, capsule);   // a contract violation: must be recognized, not free the block again
        Assert.Equal(unmatchedBefore + 1, DLPack.UnmatchedDeleterCalls);
        Assert.Equal(liveBefore, DLPack.LiveExportCount);
    }

    // ── Refusals ────────────────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Import_OfAnotherMajorVersion_CallsTheDeleter_AndThrows()
    {
        using var producer = new FakeProducer(new float[] { 1 }, new long[] { 1 }, new long[] { 1 }, major: 2);
        Assert.Throws<NotSupportedException>(() => Tensor<float>.FromDLPack(producer.Block));
        Assert.Equal(1, producer.Calls);
    }

    [Fact]
    public void Import_AsTheWrongElementType_ReleasesTheCapsule_AndThrows()
    {
        int liveBefore = DLPack.LiveExportCount;
        IntPtr capsule = Iota(3).ToDLPack();
        Assert.Throws<ArgumentException>(() => Tensor<double>.FromDLPack(capsule));
        Assert.Equal(liveBefore, DLPack.LiveExportCount);   // the refused import still ran the deleter
    }

    [Fact]
    public void Import_WithNullStridesAtVersion1_2_IsRefused_ButAcceptedBefore()
    {
        using (var strict = new FakeProducer(new float[] { 1, 2 }, new long[] { 2 }, strides: null, minor: 2))
        {
            Assert.Throws<ArgumentException>(() => Tensor<float>.FromDLPack(strict.Block));
            Assert.Equal(1, strict.Calls);
        }
        using var old = new FakeProducer(new float[] { 1, 2 }, new long[] { 2 }, strides: null, minor: 1);
        using var t = Tensor<float>.FromDLPack(old.Block);
        Assert.Equal(new float[] { 1, 2 }, t.ToArray());
    }

    [Fact]
    public void Import_ReadOnlyFlag_RefusesWrites()
    {
        using var producer = new FakeProducer(new float[] { 1, 2, 3 }, new long[] { 3 }, new long[] { 1 }, flags: 1);
        using var t = Tensor<float>.FromDLPack(producer.Block);
        Assert.Equal(2f, t[1]);
        Assert.Throws<InvalidOperationException>(() => t[1] = 5f);
        unsafe { Assert.Equal(2f, producer.Data[1]); }
    }

    [Fact]
    public void Import_OfALegacyCapsule_RunsItsDeleterOnce()
    {
        using var producer = new FakeProducer(new float[] { 7, 8 }, new long[] { 2 }, strides: null, versioned: false);
        var t = Tensor<float>.FromDLPackLegacy(producer.Block);
        Assert.Equal(new float[] { 7, 8 }, t.ToArray());
        t.Dispose();
        Assert.Equal(1, producer.Calls);
    }

    [Fact]
    public void Stream_FromHandle_RejectsTheAmbiguousZero_AndMapsTheReservedValues()
    {
        Assert.Throws<ArgumentException>(() => DLPackStream.FromHandle(IntPtr.Zero));
        Assert.Equal(DLPackStream.LegacyDefault, DLPackStream.FromHandle(new IntPtr(1)));
        Assert.Equal(DLPackStream.PerThreadDefault, DLPackStream.FromHandle(new IntPtr(2)));
        Assert.Equal(DLPackStream.NoSync, DLPackStream.FromHandle(new IntPtr(-1)));
        Assert.True(DLPackStream.FromHandle(new IntPtr(-1)).IsNoSync);
        Assert.Throws<ArgumentException>(() => DLPackStream.FromHandle(new IntPtr(-2)));
        Assert.Equal(DLPackStream.LegacyDefault, DLPackStream.Default);   // stream=None means the legacy default
        Assert.True(DLPackStream.NoSync.IsNoSync);
    }

    [Fact]
    public void Cpu_UnsignedAndBoolElementTypes_RoundTrip()
    {
        var bytes = new Tensor<byte>(new byte[] { 0, 7, 255 }, new[] { 3 });
        using (var back = Tensor<byte>.FromDLPack(bytes.ToDLPack())) Assert.Equal(bytes.ToArray(), back.ToArray());
        var ushorts = new Tensor<ushort>(new ushort[] { 0, 9, ushort.MaxValue }, new[] { 3 });
        using (var back = Tensor<ushort>.FromDLPack(ushorts.ToDLPack())) Assert.Equal(ushorts.ToArray(), back.ToArray());
        var uints = new Tensor<uint>(new[] { 0u, 11u, uint.MaxValue }, new[] { 3 });
        using (var back = Tensor<uint>.FromDLPack(uints.ToDLPack())) Assert.Equal(uints.ToArray(), back.ToArray());
        var ulongs = new Tensor<ulong>(new[] { 0ul, 13ul, ulong.MaxValue }, new[] { 3 });
        using (var back = Tensor<ulong>.FromDLPack(ulongs.ToDLPack())) Assert.Equal(ulongs.ToArray(), back.ToArray());
        var bools = new Tensor<bool>(new[] { true, false, true }, new[] { 3 });
        using (var back = Tensor<bool>.FromDLPack(bools.ToDLPack())) Assert.Equal(bools.ToArray(), back.ToArray());
    }

    [Fact]
    public void Export_OfAnElementTypeWithNoDLPackCode_Throws()
    {
        var t = new Tensor<decimal>(new[] { 1m, 2m }, new[] { 2 });
        Assert.Throws<NotSupportedException>(() => t.ToDLPack());
        Assert.Throws<NotSupportedException>(() => Tensor<decimal>.FromDLPack(Iota(2).ToDLPack()));
    }

    [Fact]
    public void Export_OfAPackedLayout_Throws_UntilReorderedToNchw()
    {
        var engine = new CpuEngine();
        var packed = engine.ReorderToNchwc(Iota(1, 8, 2, 2), TensorLayout.Nchwc8);
        Assert.Throws<InvalidOperationException>(() => packed.ToDLPack());
        using var back = Tensor<float>.FromDLPack(engine.ReorderToNchw(packed).ToDLPack());
        Assert.Equal(Iota(1, 8, 2, 2).ToArray(), back.ToArray());
    }

    [Theory]
    [InlineData(4)]    // kDLOpenCL
    [InlineData(10)]   // kDLROCM: no HIP device exchange yet (#1621)
    public void Import_OfAnUnsupportedDevice_CallsTheDeleter_AndThrows(int deviceType)
    {
        using var producer = new FakeProducer(new float[] { 1 }, new long[] { 1 }, new long[] { 1 }, deviceType: deviceType);
        Assert.Throws<NotSupportedException>(() => Tensor<float>.FromDLPack(producer.Block));
        Assert.Equal(1, producer.Calls);
    }

    [Fact]
    public void Import_WithANegativeExtent_CallsTheDeleter_AndThrows()
    {
        using var producer = new FakeProducer(new float[] { 1 }, new long[] { -1 }, new long[] { 1 });
        Assert.Throws<ArgumentException>(() => Tensor<float>.FromDLPack(producer.Block));
        Assert.Equal(1, producer.Calls);
    }

    /// <summary>
    /// A kDLCUDA capsule the CUDA import cannot take is refused before any device memory is touched, on every machine:
    /// a non-float element type, a strided layout, the read-only flag, and a device no active CUDA backend runs on
    /// (ordinal 7). Each refusal still runs the producer's deleter.
    /// </summary>
    [Fact]
    public void CudaImport_RefusesWhatItCannotTake_AndCallsTheDeleter()
    {
        const int kDLCUDA = 2;
        using (var asDouble = new FakeProducer(new float[] { 1, 2 }, new long[] { 1 }, new long[] { 1 }, code: 2, bits: 64, deviceType: kDLCUDA))
        {
            Assert.Throws<NotSupportedException>(() => Tensor<double>.FromDLPack(asDouble.Block));
            Assert.Equal(1, asDouble.Calls);
        }
        using (var strided = new FakeProducer(new float[4], new long[] { 2, 2 }, new long[] { 1, 2 }, deviceType: kDLCUDA))
        {
            Assert.Throws<NotSupportedException>(() => Tensor<float>.FromDLPack(strided.Block));
            Assert.Equal(1, strided.Calls);
        }
        using (var readOnly = new FakeProducer(new float[2], new long[] { 2 }, new long[] { 1 }, flags: 1, deviceType: kDLCUDA))
        {
            Assert.Throws<NotSupportedException>(() => Tensor<float>.FromDLPack(readOnly.Block));
            Assert.Equal(1, readOnly.Calls);
        }
        using (var otherDevice = new FakeProducer(new float[2], new long[] { 2 }, new long[] { 1 }, deviceType: kDLCUDA, deviceId: 7))
        {
            Assert.Throws<NotSupportedException>(() => Tensor<float>.FromDLPack(otherDevice.Block));
            Assert.Equal(1, otherDevice.Calls);
        }
        Assert.Throws<NotSupportedException>(() => DLPack.GetCudaConsumerStream(7));
    }

    [Fact]
    public void Stream_EqualityAndNames_FollowTheCudaHandle()
    {
        var handle = DLPackStream.FromHandle(new IntPtr(0x30));
        Assert.Equal("0x30", handle.ToString());
        Assert.Equal("LegacyDefault", DLPackStream.LegacyDefault.ToString());
        Assert.Equal("PerThreadDefault", DLPackStream.PerThreadDefault.ToString());
        Assert.Equal("NoSync", DLPackStream.NoSync.ToString());
        Assert.True(handle == DLPackStream.FromHandle(new IntPtr(0x30)));
        Assert.True(handle != DLPackStream.PerThreadDefault);
        Assert.True(handle.Equals((object)DLPackStream.FromHandle(new IntPtr(0x30))));
        Assert.False(handle.Equals("0x30"));
        Assert.Equal(handle.GetHashCode(), DLPackStream.FromHandle(new IntPtr(0x30)).GetHashCode());
        // stream=None (Default) and the legacy default name the same CUDA stream.
        Assert.True(DLPackStream.Default == DLPackStream.LegacyDefault);
        Assert.False(DLPackStream.NoSync == DLPackStream.Default);
    }

    // ── CUDA ────────────────────────────────────────────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Cuda_RoundTrip_ExportsTheDevicePointer_AndReadsBackTheSameValues()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        var previous = AiDotNetEngine.Current;
        var gpu = new DirectGpuTensorEngine();
        try
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.PlacementBackend is CudaBackend, "The active GPU backend is not CUDA.");
            AiDotNetEngine.Current = gpu;
            int liveBefore = DLPack.LiveExportCount;

            var source = Iota(8, 16).Gpu();
            Assert.True(source.IsGpuResident);
            IntPtr capsule = source.ToDLPack();
            var header = ReadVersioned(capsule);
            Assert.Equal(2, header.Tensor.DeviceType);   // kDLCUDA: the device pointer, not a host copy
            Assert.Equal(((CudaBackend)gpu.PlacementBackend).DeviceOrdinal, header.Tensor.DeviceId);

            var imported = Tensor<float>.FromDLPack(capsule);
            Assert.True(imported.IsGpuResident);
            Assert.Equal(Iota(8, 16).ToArray(), imported.ToArray());

            imported.Dispose();
            Assert.Equal(liveBefore, DLPack.LiveExportCount);
            source.Dispose();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }

    [SkippableFact]
    public void Cuda_ImportThatIsNeverDisposed_IsReleasedByTheNextOperation()
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        var previous = AiDotNetEngine.Current;
        var gpu = new DirectGpuTensorEngine();
        try
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.PlacementBackend is CudaBackend, "The active GPU backend is not CUDA.");
            AiDotNetEngine.Current = gpu;
            var source = Iota(8, 16).Gpu();
            int liveBefore = DLPack.LiveExportCount;
            IntPtr capsule = source.ToDLPack();
            Assert.Equal(liveBefore + 1, DLPack.LiveExportCount);

            ImportUseAndDrop(capsule);
            for (int i = 0; i < 5; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
            // The finalizer only queues the release (it never calls the driver); the next GPU operation (here, or
            // one a concurrently running test issues) waits for the queued work and then runs the deleter, which is
            // our own export's.
            using var next = Iota(4).Gpu();
            using var sum = gpu.TensorAdd(next, next);
            Assert.Equal(new float[] { 0, 2, 4, 6 }, sum.ToArray());
            Assert.Equal(liveBefore, DLPack.LiveExportCount);
            Assert.Equal(Iota(8, 16).ToArray(), source.ToArray());   // the producer's tensor is untouched
            source.Dispose();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }

    [SkippableFact]
    public void Cuda_ExportedBuffer_SurvivesTheTensorReleasingIt_UntilTheDeleter()
    {
        // The capsule holds only the device address. Releasing the producer tensor's buffer (here as a TensorScope or a
        // dropped gradient would) must not free it or hand it to the next allocation while the consumer can still read it.
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        var previous = AiDotNetEngine.Current;
        var gpu = new DirectGpuTensorEngine();
        try
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.PlacementBackend is CudaBackend, "The active GPU backend is not CUDA.");
            AiDotNetEngine.Current = gpu;
            var expected = Iota(64, 64).ToArray();
            var source = Iota(64, 64).Gpu();
            IntPtr address = source.TryGetGpuBuffer()!.Handle;
            IntPtr capsule = source.ToDLPack(DLPackStream.LegacyDefault);

            gpu.ReleaseDeadDeviceStorage(source);
            Assert.Null(source.DataVector._deviceState?.Buffer);
            // Same-size allocations: without the hold the pool hands the released memory straight back.
            var fillers = Enumerable.Range(0, 8).Select(i => gpu.TensorAdd(Iota(64, 64).Gpu(), Iota(64, 64).Gpu())).ToList();
            Assert.DoesNotContain(fillers, f => f.TryGetGpuBuffer()!.Handle == address);

            var imported = Tensor<float>.FromDLPack(capsule);
            Assert.Equal(expected, imported.ToArray());
            imported.Dispose();   // the deleter: the hold ends and the parked release runs
            foreach (var f in fillers) f.Dispose();
            source.Dispose();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }

    [SkippableFact]
    public async System.Threading.Tasks.Task Cuda_ImportedTensor_RunsThroughGeneratedKernels_ElementwiseAndMatMul()
    {
        // An import wraps the producer's pointer in an ExternalDeviceBuffer, not a CudaGpuBuffer: every device path,
        // the generated-kernel one included, must take it.
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        var previous = AiDotNetEngine.Current;
        var gpu = new DirectGpuTensorEngine();
        try
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.PlacementBackend is CudaBackend, "The active GPU backend is not CUDA.");
            AiDotNetEngine.Current = gpu;
            var backend = (CudaBackend)gpu.PlacementBackend!;
            // Centered on zero so the ReLU kernel has both halves to get right.
            var values = Iota(4, 8).ToArray().Select(v => v - 16f).ToArray();
            using var source = new Tensor<float>(values, new[] { 4, 8 }).Gpu();
            using var imported = Tensor<float>.FromDLPack(source.ToDLPack());

            var graph = AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenLowering.LowerUnaryPointwise<float>(
                AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenOpKind.ReLU, new[] { values.Length });
            using var output = backend.AllocateBuffer(values.Length);
            var result = await AiDotNet.Tensors.Engines.Compilation.Codegen.NativeGpuCodegenExecutorExtensions.EmitAndExecuteCodegenKernelAsync(
                backend, graph, AiDotNet.Tensors.Engines.Compilation.Codegen.Ir.CodegenElementType.Float32,
                new[] { imported.TryGetGpuBuffer()! }, new[] { output }, values.Length);
            Skip.If(result.Declined, "The CUDA code generator declined the graph: " + result.DeclineReason);
            Assert.Equal(values.Select(v => Math.Max(v, 0f)).ToArray(), backend.DownloadBuffer(output));

            using var sum = gpu.TensorAdd(imported, imported);
            Assert.Equal(values.Select(v => 2 * v).ToArray(), sum.ToArray());
            using var weights = new Tensor<float>(Enumerable.Repeat(1f, 8 * 2).ToArray(), new[] { 8, 2 });
            using var product = gpu.TensorMatMul(imported, weights);
            for (int r = 0; r < 4; r++)
            {
                float rowSum = values.Skip(8 * r).Take(8).Sum();
                Assert.Equal(rowSum, product.ToArray()[2 * r], 3);
                Assert.Equal(rowSum, product.ToArray()[2 * r + 1], 3);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ImportUseAndDrop(IntPtr capsule)
    {
        var imported = Tensor<float>.FromDLPack(capsule);
        Assert.True(imported.IsGpuResident);
        Assert.Equal(Iota(8, 16).ToArray(), imported.ToArray());
    }
}
