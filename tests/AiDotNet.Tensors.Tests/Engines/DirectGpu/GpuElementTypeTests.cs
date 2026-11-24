using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public class GpuElementTypeTests
{
    [Theory]
    [InlineData(GpuElementType.Float64, 8)]
    [InlineData(GpuElementType.Int64, 8)]
    [InlineData(GpuElementType.Float32, 4)]
    [InlineData(GpuElementType.Int32, 4)]
    [InlineData(GpuElementType.Float16, 2)]
    [InlineData(GpuElementType.BFloat16, 2)]
    [InlineData(GpuElementType.Int16, 2)]
    [InlineData(GpuElementType.Int8, 1)]
    [InlineData(GpuElementType.UInt8, 1)]
    [InlineData(GpuElementType.Bool, 1)]
    public void SizeInBytes_IsTheDeviceWidthOfTheType(GpuElementType type, int bytes)
        => Assert.Equal(bytes, GpuElementTypes.SizeInBytes(type));

    [Fact]
    public void SizeInBytes_OfEveryDefinedTypeButRawBytes_IsDefined()
    {
        // A type added to the enum without a size would only fail at its first allocation.
        foreach (GpuElementType type in Enum.GetValues(typeof(GpuElementType)))
        {
            if (type == GpuElementType.Unspecified)
                Assert.Throws<ArgumentOutOfRangeException>(() => GpuElementTypes.SizeInBytes(type));
            else
                Assert.InRange(GpuElementTypes.SizeInBytes(type), 1, 8);
        }
    }

    [Fact]
    public void TryFromClrType_MapsEachDeviceRepresentableType()
    {
        AssertMaps<float>(GpuElementType.Float32);
        AssertMaps<double>(GpuElementType.Float64);
        AssertMaps<Half>(GpuElementType.Float16);
        AssertMaps<BFloat16>(GpuElementType.BFloat16);
        AssertMaps<sbyte>(GpuElementType.Int8);
        AssertMaps<byte>(GpuElementType.UInt8);
        AssertMaps<short>(GpuElementType.Int16);
        AssertMaps<int>(GpuElementType.Int32);
        AssertMaps<long>(GpuElementType.Int64);
        AssertMaps<bool>(GpuElementType.Bool);
    }

    [Fact]
    public void TryFromClrType_RefusesATypeWithNoDeviceRepresentation()
    {
        Assert.False(GpuElementTypes.TryFromClrType<decimal>(out var type));
        Assert.Equal(GpuElementType.Unspecified, type);
        Assert.False(GpuElementTypes.TryFromClrType(typeof(string), out _));
    }

    [Fact]
    public void Classification_SeparatesFloatingPointIntegerAndBool()
    {
        foreach (GpuElementType type in Enum.GetValues(typeof(GpuElementType)))
        {
            bool floating = GpuElementTypes.IsFloatingPoint(type), integer = GpuElementTypes.IsInteger(type);
            Assert.False(floating && integer);
            if (type is GpuElementType.Bool or GpuElementType.Unspecified) Assert.False(floating || integer);
            else Assert.True(floating || integer);
        }
    }

    [SkippableFact]
    public void DeviceBuffers_ReportTheirElementType()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        using var floats = backend.AllocateBuffer(8);
        using var bytes = backend.AllocateByteBuffer(8);
        Assert.Equal(GpuElementType.Float32, floats.ElementType);
        Assert.Equal(GpuElementType.Unspecified, bytes.ElementType);
    }

    [SkippableFact]
    public void IntBuffers_ReportInt32_AndARecycledAllocationIsAFloatBufferAgain()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        // An odd size nothing else in the suite pools, so the float rents below can only be served by these buffers.
        const int Size = 7919;
        using (var sized = backend.AllocateIntBuffer(Size))
            Assert.Equal(GpuElementType.Int32, sized.ElementType);
        using (var uploaded = backend.AllocateIntBuffer(new int[Size]))
            Assert.Equal(GpuElementType.Int32, uploaded.ElementType);
        // Int and float buffers share one pool: whichever allocation comes back, it must not still say Int32.
        for (int i = 0; i < 4; i++)
        {
            using var floats = backend.AllocateBuffer(Size);
            Assert.Equal(GpuElementType.Float32, floats.ElementType);
        }
    }

    [SkippableFact]
    public void Tag_RefusesAnElementTypeOfAnotherWidth()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        using var floats = backend.AllocateBuffer(16);
        Assert.Throws<InvalidOperationException>(() => GpuBufferTyping.Tag(floats, GpuElementType.Float64));
        Assert.Equal(GpuElementType.Float32, floats.ElementType);
    }

    [SkippableFact]
    public void TypedBuffers_RoundTripEveryElementTypeTheBackendHolds()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");

        // Every type has storage on CUDA, HIP and OpenCL (double where the device has it); a backend that reports a type
        // must round-trip it bit for bit, and one that does not must refuse the allocation.
        RoundTrip(backend, GpuElementType.Float32, new[] { 1.5f, -2.25f, float.MaxValue, float.Epsilon, 0f });
        RoundTrip(backend, GpuElementType.Float64, new[] { 1.5, -2.25, double.MaxValue, double.Epsilon, 1e-300 });
        RoundTrip(backend, GpuElementType.Int32, new[] { 1, -2, int.MaxValue, int.MinValue, 0 });
        RoundTrip(backend, GpuElementType.Int64, new[] { 1L, -2L, long.MaxValue, long.MinValue, 1L << 40 });
        RoundTrip(backend, GpuElementType.Int16, new short[] { 1, -2, short.MaxValue, short.MinValue, 0 });
        RoundTrip(backend, GpuElementType.Int8, new sbyte[] { 1, -2, sbyte.MaxValue, sbyte.MinValue, 0 });
        RoundTrip(backend, GpuElementType.UInt8, new byte[] { 1, 2, byte.MaxValue, 0, 128 });
        RoundTrip(backend, GpuElementType.Bool, new[] { true, false, true, true, false });
#if NET6_0_OR_GREATER
        RoundTrip(backend, GpuElementType.Float16, new[] { (Half)1.5f, (Half)(-2.25f), Half.MaxValue, Half.Epsilon, (Half)0f });
#else
        // Without System.Half the Half shim is float-backed: it does not have the two-byte device layout, so a typed
        // transfer of it is refused rather than reinterpreted.
        if (backend.SupportsElementType(GpuElementType.Float16))
        {
            using var halves = backend.AllocateBuffer(5, GpuElementType.Float16);
            Assert.Throws<NotSupportedException>(() => backend.UploadBuffer<Half>(halves, new Half[5]));
            Assert.Throws<NotSupportedException>(() => backend.DownloadBuffer<Half>(halves, new Half[5]));
        }
        else
        {
            Assert.Throws<NotSupportedException>(() => backend.AllocateBuffer(5, GpuElementType.Float16));
        }
#endif
        RoundTrip(backend, GpuElementType.BFloat16, new[] { (BFloat16)1.5f, (BFloat16)(-2.25f), (BFloat16)3f, (BFloat16)0f, (BFloat16)100f });
    }

    // A view is its parent's memory: it reports the parent's elements, so a typed transfer through it is accepted.
    [SkippableFact]
    public void View_OfAnIntBuffer_IsAnIntBuffer()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is not IGpuBufferViews, "This backend has no buffer views.");
        var views = (IGpuBufferViews)backend!;
        var values = new int[4096];
        for (int i = 0; i < values.Length; i++) values[i] = int.MaxValue - i;
        using var parent = backend.AllocateIntBuffer(values);
        Assert.Equal(GpuElementType.Int32, parent.ElementType);
        using var view = views.TryCreateView(parent, 0, 1024);
        Skip.If(view is null, "The backend declined this view.");
        Assert.Equal(GpuElementType.Int32, view.ElementType);
        var back = new int[1024];
        backend.DownloadBuffer<int>(view, back);
        for (int i = 0; i < back.Length; i++) Assert.Equal(values[i], back[i]);
    }

    // The word-addressed backends count a view's offset in 4-byte words: a view of an 8-byte parent would address the
    // wrong range, so it is declined (or, where views are element-addressed, covers the parent's elements).
    [SkippableFact]
    public void View_OfADoubleBuffer_AddressesDoubles_OrIsDeclined()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is not IGpuBufferViews, "This backend has no buffer views.");
        Skip.If(!backend!.SupportsElementType(GpuElementType.Float64), "No double storage on this device.");
        var views = (IGpuBufferViews)backend;
        var values = new double[2048];
        for (int i = 0; i < values.Length; i++) values[i] = 1.0 / (i + 3);
        using var parent = backend.AllocateBuffer(values.Length, GpuElementType.Float64);
        backend.UploadBuffer<double>(parent, values);
        using var view = views.TryCreateView(parent, 1024, 512);
        if (view is null) return;   // declined: the contract's answer for a parent it cannot address
        Assert.Equal(GpuElementType.Float64, view.ElementType);
        Assert.Equal(512L * sizeof(double), view.SizeInBytes);
        var back = new double[512];
        backend.DownloadBuffer<double>(view, back);
        for (int i = 0; i < back.Length; i++) Assert.Equal(values[1024 + i], back[i]);
    }

    [SkippableFact]
    public void ByteRead_OfAPrefix_ReturnsThoseBytes()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        var values = new float[1 << 16];
        for (int i = 0; i < values.Length; i++) values[i] = i * 0.5f - 7f;
        using var buffer = backend.AllocateBuffer(values);
        foreach (int count in new[] { 0, 1, 4, 6, 4096 + 3 })
        {
            var bytes = backend.DownloadByteBuffer(buffer, count);
            Assert.Equal(count, bytes.Length);
            Assert.True(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()).Slice(0, count)
                .SequenceEqual(bytes), $"The first {count} bytes differ.");
        }
    }

    [SkippableFact]
    public void TypedTransfer_RefusesAHostTypeThatIsNotTheBuffersElementType()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        using var floats = backend.AllocateBuffer(4, GpuElementType.Float32);
        Assert.Throws<ArgumentException>(() => backend.UploadBuffer<int>(floats, new int[4]));
        Assert.Throws<ArgumentException>(() => backend.DownloadBuffer<double>(floats, new double[4]));
        Assert.Throws<ArgumentException>(() => backend.UploadBuffer<float>(floats, new float[5]));
        Assert.Throws<NotSupportedException>(() => backend.AllocateBuffer(4, GpuElementType.Unspecified));
        Assert.False(backend.SupportsElementType(GpuElementType.Unspecified));
        Assert.True(backend.SupportsElementType(GpuElementType.Float32));
    }

    // The float download moves Size * 4 bytes as 32-bit words: of a double buffer that is half its bytes read as
    // unrelated floats, of a byte-wide buffer a read past the allocation. Both are refused by name.
    [SkippableFact]
    public void FloatDownload_RefusesABufferThatIsNotFourByteWords()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        foreach (var type in new[] { GpuElementType.Float64, GpuElementType.Int64, GpuElementType.Float16, GpuElementType.Int8, GpuElementType.Bool })
        {
            if (!backend.SupportsElementType(type)) continue;
            using var typed = backend.AllocateBuffer(6, type);
            var refused = Assert.Throws<InvalidOperationException>(() => backend.DownloadBuffer(typed, new float[6]));
            Assert.Contains(type.ToString(), refused.Message);
            Assert.Throws<InvalidOperationException>(() => backend.DownloadBuffer(typed));
        }
        using var floats = backend.AllocateBuffer(new[] { 1f, 2f, 3f });
        Assert.Equal(new[] { 1f, 2f, 3f }, backend.DownloadBuffer(floats));
        using var ints = backend.AllocateBuffer(3, GpuElementType.Int32);
        backend.UploadBuffer<int>(ints, new[] { 7, -8, 9 });
        var words = new int[3];
        backend.DownloadBuffer<int>(ints, words);
        Assert.Equal(new[] { 7, -8, 9 }, words);
    }

    // A device copy counts elements of the buffers' type: of doubles it moves eight bytes each, at element offsets.
    [SkippableFact]
    public void DeviceCopy_OfATypedBuffer_MovesWholeElements()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        Skip.IfNot(backend.SupportsElementType(GpuElementType.Float64), "The backend holds no Float64 buffers.");
        var values = new[] { 1.5, -2.25, 1e300, double.Epsilon, 5.0, -6.0 };
        using var source = backend.AllocateBuffer(values.Length, GpuElementType.Float64);
        backend.UploadBuffer<double>(source, values);

        using var whole = backend.AllocateBuffer(values.Length, GpuElementType.Float64);
        backend.Copy(source, whole, values.Length);
        var read = new double[values.Length];
        backend.DownloadBuffer<double>(whole, read);
        Assert.Equal(values, read);

        using var buffered = backend.AllocateBuffer(values.Length, GpuElementType.Float64);
        backend.CopyBuffer(source, buffered, values.Length);
        backend.DownloadBuffer<double>(buffered, read);
        Assert.Equal(values, read);

        using var part = backend.AllocateBuffer(values.Length, GpuElementType.Float64);
        backend.UploadBuffer<double>(part, new double[values.Length]);
        backend.Copy(source, 2, part, 1, 3);
        backend.DownloadBuffer<double>(part, read);
        Assert.Equal(new[] { 0.0, 1e300, double.Epsilon, 5.0, 0.0, 0.0 }, read);

        // Elements of different widths: the count would mean different things on each side.
        using var floats = backend.AllocateBuffer(values.Length, GpuElementType.Float32);
        Assert.Throws<InvalidOperationException>(() => backend.Copy(source, floats, values.Length));
        Assert.Throws<InvalidOperationException>(() => backend.Copy(floats, 0, source, 0, 1));
        Assert.Throws<InvalidOperationException>(() => backend.CopyBuffer(source, floats, values.Length));
    }

#if NETCOREAPP
    // The table on a backend that holds only floats (the mock): every family for Float32, nothing else, and a wrapper
    // answers for what it wraps.
    [Fact]
    public void CapabilityTable_OfAFloatOnlyBackend_IsFloat32Alone()
    {
        IDirectGpuBackend backend = MockDirectGpuBackend.Create(new MockBackendState());
        var wrapped = new AiDotNet.Tensors.Engines.Gpu.DelegatingGpuBackend(backend);
        foreach (var candidate in new[] { backend, wrapped })
        {
            var rows = candidate.DescribeOperations();
            Assert.Equal(40, rows.Count);
            Assert.Equal(rows.Count, rows.Select(row => (row.Family, row.ElementType)).Distinct().Count());
            Assert.All(rows, row => Assert.Equal(row.ElementType == GpuElementType.Float32, row.IsNative));
            Assert.False(candidate.SupportsOperation(GpuOperationFamily.Storage, GpuElementType.Unspecified));
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.SupportsOperation((GpuOperationFamily)99, GpuElementType.Float32));
        Assert.Equal("Reduction<Float64>: CPU", new GpuOperationCapability(GpuOperationFamily.Reduction, GpuElementType.Float64, false).ToString());
        // Emulation qualifies a route the device has; a row the device cannot run is never "emulated".
        var emulated = new GpuOperationCapability(GpuOperationFamily.Reduction, GpuElementType.Float64, true, isEmulated: true);
        Assert.True(emulated.IsEmulated);
        Assert.Equal("Reduction<Float64>: emulated (on request)", emulated.ToString());
        Assert.NotEqual(emulated, new GpuOperationCapability(GpuOperationFamily.Reduction, GpuElementType.Float64, true));
        Assert.False(new GpuOperationCapability(GpuOperationFamily.Reduction, GpuElementType.Float64, false, isEmulated: true).IsEmulated);
    }
#endif

    // On a device, the table is what the backend's own capability members say, row for row.
    [SkippableFact]
    public void CapabilityTable_AgreesWithTheBackendsCapabilityMembers()
    {
        var gpu = AiDotNetEngine.Current as DirectGpuTensorEngine;
        Skip.If(gpu is null, "Not a DirectGpu engine.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No GPU backend on this host.");
        foreach (var row in backend.DescribeOperations())
        {
            bool held = backend.SupportsElementType(row.ElementType);
            bool expected = row.Family == GpuOperationFamily.Storage ? held
                : row.ElementType == GpuElementType.Float32 ? held
                : !held ? false
                // Integer and bool rows are what the backend's integer interfaces say for that type.
                : GpuIntegerLayout.IsIntegerOrBool(row.ElementType)
                    ? (row.Family == GpuOperationFamily.Elementwise ? backend is IGpuIntegerElementwise integers && integers.SupportsIntegerElementwise(row.ElementType)
                        : row.Family == GpuOperationFamily.Reduction && backend is IGpuIntegerReductions sums && sums.SupportsIntegerReductions(row.ElementType))
                : GpuNarrowFloatLayout.IsNarrowFloat(row.ElementType)
                    ? (row.Family == GpuOperationFamily.Elementwise
                        ? backend is IGpuNarrowFloatElementwise narrow && narrow.SupportsNarrowFloatBinary(GpuNarrowFloatBinaryOperation.Add, row.ElementType)
                        : row.Family == GpuOperationFamily.MatrixMultiply && backend is IGpuNarrowFloatMatrixMultiply multiply
                            && multiply.SupportsNarrowFloatMatrixMultiply(row.ElementType))
                : row.ElementType != GpuElementType.Float64 ? false
                : row.Family == GpuOperationFamily.Elementwise ? backend is IGpuFloat64Elementwise { SupportsFloat64Elementwise: true }
                : row.Family == GpuOperationFamily.Reduction ? backend is IGpuFloat64Reductions { SupportsFloat64Reductions: true }
                : backend is IGpuFloat64MatrixMultiply { SupportsFloat64MatrixMultiply: true };
            Assert.True(expected == row.IsNative, $"{row} disagrees with the backend ({backend.BackendName}).");
        }
        Assert.True(backend.SupportsOperation(GpuOperationFamily.MatrixMultiply, GpuElementType.Float32));
        // No 16-bit float type has a reduction behind this table.
        Assert.False(backend.SupportsOperation(GpuOperationFamily.Reduction, GpuElementType.BFloat16));
    }

    private static void RoundTrip<T>(IDirectGpuBackend backend, GpuElementType type, T[] values) where T : unmanaged
    {
        if (!backend.SupportsElementType(type))
        {
            Assert.Throws<NotSupportedException>(() => backend.AllocateBuffer(values.Length, type));
            return;
        }
        using var buffer = backend.AllocateBuffer(values.Length, type);
        Assert.Equal(type, buffer.ElementType);
        Assert.Equal(values.Length, buffer.Size);
        Assert.True(buffer.SizeInBytes >= (long)values.Length * GpuElementTypes.SizeInBytes(type));
        backend.UploadBuffer<T>(buffer, values);
        var back = new T[values.Length];
        backend.DownloadBuffer<T>(buffer, back);
        // Bitwise: a transfer must not round, reorder or reinterpret.
        Assert.True(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan())
            .SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.AsBytes(back.AsSpan())), $"{type} did not round-trip.");
    }

    private static void AssertMaps<T>(GpuElementType expected)
    {
        Assert.True(GpuElementTypes.TryFromClrType<T>(out var type));
        Assert.Equal(expected, type);
    }
}
