namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The element type a device buffer holds. Every <see cref="IGpuBuffer"/> reports one, so an operation can check the
/// type of the memory it was handed instead of assuming 32-bit float.
/// </summary>
public enum GpuElementType
{
    /// <summary>
    /// Raw bytes with no element interpretation of their own: the operation that reads the buffer decides (a byte
    /// workspace, a packed bit mask, an encoded payload).
    /// </summary>
    Unspecified = 0,

    /// <summary>IEEE 754 binary32 (<see cref="float"/>).</summary>
    Float32,

    /// <summary>IEEE 754 binary64 (<see cref="double"/>).</summary>
    Float64,

    /// <summary>IEEE 754 binary16 (<c>System.Half</c>).</summary>
    Float16,

    /// <summary>bfloat16: the top 16 bits of a binary32.</summary>
    BFloat16,

    /// <summary>Signed 8-bit integer (<see cref="sbyte"/>).</summary>
    Int8,

    /// <summary>Unsigned 8-bit integer (<see cref="byte"/>).</summary>
    UInt8,

    /// <summary>Signed 16-bit integer (<see cref="short"/>).</summary>
    Int16,

    /// <summary>Signed 32-bit integer (<see cref="int"/>).</summary>
    Int32,

    /// <summary>Signed 64-bit integer (<see cref="long"/>).</summary>
    Int64,

    /// <summary>Boolean, one byte per element (0 or 1).</summary>
    Bool,
}

/// <summary>Size, classification and CLR mapping of <see cref="GpuElementType"/>.</summary>
public static class GpuElementTypes
{
    /// <summary>Bytes one element of <paramref name="type"/> occupies on the device.</summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="type"/> is <see cref="GpuElementType.Unspecified"/> (raw bytes have no element size) or not a
    /// defined value.
    /// </exception>
    public static int SizeInBytes(GpuElementType type) => type switch
    {
        GpuElementType.Float64 or GpuElementType.Int64 => 8,
        GpuElementType.Float32 or GpuElementType.Int32 => 4,
        GpuElementType.Float16 or GpuElementType.BFloat16 or GpuElementType.Int16 => 2,
        GpuElementType.Int8 or GpuElementType.UInt8 or GpuElementType.Bool => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(type), type, "The element type has no element size."),
    };

    /// <summary>
    /// Whether a buffer of <paramref name="type"/> is addressed in 4-byte words: the float and int types, and an
    /// untyped (<see cref="GpuElementType.Unspecified"/>) buffer, which the float paths have always treated as words.
    /// A view whose offsets are counted in floats is only valid over such a parent.
    /// </summary>
    internal static bool IsWordAddressed(GpuElementType type)
        => type is GpuElementType.Unspecified or GpuElementType.Float32 or GpuElementType.Int32;

    /// <summary>Whether <paramref name="type"/> is a floating-point type.</summary>
    public static bool IsFloatingPoint(GpuElementType type)
        => type is GpuElementType.Float32 or GpuElementType.Float64 or GpuElementType.Float16 or GpuElementType.BFloat16;

    /// <summary>Whether <paramref name="type"/> is an integer type (<see cref="GpuElementType.Bool"/> is not).</summary>
    public static bool IsInteger(GpuElementType type)
        => type is GpuElementType.Int8 or GpuElementType.UInt8 or GpuElementType.Int16 or GpuElementType.Int32
            or GpuElementType.Int64;

    /// <summary>
    /// The element type a tensor of <typeparamref name="T"/> holds on a device, or false when <typeparamref name="T"/>
    /// has no device representation (a complex, decimal or user-defined numeric type).
    /// </summary>
    public static bool TryFromClrType<T>(out GpuElementType type) => TryFromClrType(typeof(T), out type);

    /// <summary>
    /// The element type a tensor of <paramref name="clrType"/> holds on a device, or false when it has no device
    /// representation.
    /// </summary>
    public static bool TryFromClrType(Type clrType, out GpuElementType type)
    {
        if (clrType is null) throw new ArgumentNullException(nameof(clrType));
        if (clrType == typeof(float)) type = GpuElementType.Float32;
        else if (clrType == typeof(double)) type = GpuElementType.Float64;
        else if (clrType == typeof(int)) type = GpuElementType.Int32;
        else if (clrType == typeof(long)) type = GpuElementType.Int64;
        else if (clrType == typeof(short)) type = GpuElementType.Int16;
        else if (clrType == typeof(sbyte)) type = GpuElementType.Int8;
        else if (clrType == typeof(byte)) type = GpuElementType.UInt8;
        else if (clrType == typeof(bool)) type = GpuElementType.Bool;
        else if (clrType == typeof(Half)) type = GpuElementType.Float16;
        else if (clrType == typeof(NumericOperations.BFloat16)) type = GpuElementType.BFloat16;
        else
        {
            type = GpuElementType.Unspecified;
            return false;
        }
        return true;
    }
}

/// <summary>A buffer whose allocation is shared between element types of one width, so its type is set per use.</summary>
internal interface IRetypableGpuBuffer
{
    /// <summary>Declares what the buffer's elements are from now on.</summary>
    void SetElementType(GpuElementType type);
}

internal static class GpuBufferTyping
{
    /// <summary>
    /// Declares <paramref name="buffer"/> to hold <paramref name="type"/> and returns it. The type must have the width
    /// the buffer was allocated for: a tag never changes how many bytes an element occupies.
    /// </summary>
    public static IGpuBuffer Tag(IGpuBuffer buffer, GpuElementType type)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (buffer.ElementType == type) return buffer;
        if (buffer is not IRetypableGpuBuffer retypable)
            throw new InvalidOperationException(
                $"A {buffer.GetType().Name} holds {buffer.ElementType} elements and cannot be declared {type}.");
        if (GpuElementTypes.SizeInBytes(type) != GpuElementTypes.SizeInBytes(buffer.ElementType))
            throw new InvalidOperationException(
                $"A buffer of {buffer.ElementType} elements cannot be declared {type}: the element widths differ.");
        retypable.SetElementType(type);
        return buffer;
    }

    /// <summary>
    /// Declares <paramref name="buffer"/> to hold raw bytes (<see cref="GpuElementType.Unspecified"/>) and returns it:
    /// for a backend that serves byte buffers from allocations of another element type.
    /// </summary>
    public static IGpuBuffer TagRaw(IGpuBuffer buffer)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (buffer.ElementType == GpuElementType.Unspecified) return buffer;
        if (buffer is not IRetypableGpuBuffer retypable)
            throw new InvalidOperationException(
                $"A {buffer.GetType().Name} holds {buffer.ElementType} elements and cannot be declared raw bytes.");
        retypable.SetElementType(GpuElementType.Unspecified);
        return buffer;
    }
}

/// <summary>
/// The precondition of every float transfer: it moves <c>Size * 4</c> bytes and reads them as 32-bit words.
/// </summary>
internal static class GpuFloatLayout
{
    /// <summary>
    /// Refuses a buffer whose elements are not 4-byte words. A float read of a <see cref="GpuElementType.Float64"/>
    /// buffer would return half of its bytes as unrelated floats, and one of a narrower type (or of raw bytes) would
    /// read past the allocation; the typed transfers (<see cref="GpuTypedBufferExtensions"/>) are the way to read those.
    /// </summary>
    public static void Require(IGpuBuffer buffer, string operation)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        var type = buffer.ElementType;
        if (type is GpuElementType.Float32 or GpuElementType.Int32) return;
        throw new InvalidOperationException(type == GpuElementType.Unspecified
            ? $"{operation} reads 32-bit words, but the buffer holds {buffer.SizeInBytes} raw bytes; read it with DownloadByteBuffer."
            : $"{operation} reads 32-bit words, but the buffer holds {buffer.Size} {type} elements; read it with the typed DownloadBuffer<T>.");
    }
}

/// <summary>The unit a device-to-device copy counts in.</summary>
internal static class GpuCopyLayout
{
    /// <summary>
    /// Bytes per element of a copy from <paramref name="source"/> to <paramref name="destination"/>: the copy's
    /// offsets and length count elements of the buffers' type, so a <see cref="GpuElementType.Float64"/> copy moves
    /// eight bytes an element (it moved four, half the data). Raw bytes (<see cref="GpuElementType.Unspecified"/>) keep
    /// the 4-byte unit their callers have always counted in. Buffers whose elements differ in width are refused: the
    /// count would mean different things on each side.
    /// </summary>
    public static int ElementBytes(IGpuBuffer source, IGpuBuffer destination, string operation)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        int from = Width(source.ElementType), to = Width(destination.ElementType);
        if (from != to)
            throw new InvalidOperationException(
                $"{operation} cannot copy {source.ElementType} elements ({from} bytes) into a {destination.ElementType} buffer ({to} bytes).");
        return from;
    }

    /// <summary>
    /// <see cref="ElementBytes"/>, after checking that <paramref name="count"/> elements from each offset lie inside
    /// both buffers: inside the elements a typed buffer declares, and inside the allocation of any buffer. A copy is a
    /// raw device memcpy or an unchecked kernel, which would otherwise read or write past either.
    /// </summary>
    public static int CheckedElementBytes(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset,
        int count, string operation)
    {
        int width = ElementBytes(source, destination, operation);
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), count, $"{operation}: the element count must be non-negative.");
        if (sourceOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(sourceOffset), sourceOffset, $"{operation}: the source offset must be non-negative.");
        if (destinationOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(destinationOffset), destinationOffset, $"{operation}: the destination offset must be non-negative.");
        // A typed buffer may sit in a larger allocation (a pooled or rounded one): its declared elements are the range.
        if (source.ElementType != GpuElementType.Unspecified && (long)sourceOffset + count > source.Size)
            throw new ArgumentOutOfRangeException(nameof(count), count,
                $"{operation}: {count} elements from offset {sourceOffset} run past the source's {source.Size} elements.");
        if (destination.ElementType != GpuElementType.Unspecified && (long)destinationOffset + count > destination.Size)
            throw new ArgumentOutOfRangeException(nameof(count), count,
                $"{operation}: {count} elements from offset {destinationOffset} run past the destination's {destination.Size} elements.");
        if (((long)sourceOffset + count) * width > source.SizeInBytes)
            throw new ArgumentOutOfRangeException(nameof(count), count,
                $"{operation}: {count} elements of {width} bytes from offset {sourceOffset} run past the source's {source.SizeInBytes} bytes.");
        if (((long)destinationOffset + count) * width > destination.SizeInBytes)
            throw new ArgumentOutOfRangeException(nameof(count), count,
                $"{operation}: {count} elements of {width} bytes from offset {destinationOffset} run past the destination's {destination.SizeInBytes} bytes.");
        return width;
    }

    private static int Width(GpuElementType type)
        => type == GpuElementType.Unspecified ? sizeof(float) : GpuElementTypes.SizeInBytes(type);

    /// <summary>Whether two ranges of <paramref name="bytes"/> bytes, at the given byte positions of one allocation, share a byte.</summary>
    public static bool Overlaps(long sourceByte, long destinationByte, long bytes)
        => bytes > 0 && sourceByte < destinationByte + bytes && destinationByte < sourceByte + bytes;

    /// <summary>
    /// A copy whose ranges overlap inside one allocation, done through a scratch buffer: a device copy of overlapping
    /// ranges is refused (OpenCL) or undefined (CUDA, HIP, Vulkan), and the destination must get what the source held
    /// before the copy.
    /// </summary>
    public static void ThroughScratch(IDirectGpuBackend backend, IGpuBuffer source, int sourceOffset, IGpuBuffer destination,
        int destinationOffset, int count)
    {
        using var scratch = source.ElementType == GpuElementType.Unspecified
            ? backend.AllocateByteBuffer(checked(count * sizeof(float)))
            : backend.AllocateBuffer(count, source.ElementType);
        backend.Copy(source, sourceOffset, scratch, 0, count);
        backend.Copy(scratch, 0, destination, destinationOffset, count);
    }
}

/// <summary>A byte allocation that can be declared a buffer of typed elements.</summary>
internal interface IElementTypedByteBuffer
{
    /// <summary>Declares the allocation <paramref name="count"/> elements of <paramref name="type"/>.</summary>
    void DeclareElements(GpuElementType type, int count);
}

/// <summary>The one implementation of <see cref="IGpuDevice.AllocateBuffer(int, GpuElementType)"/>.</summary>
internal static class GpuTypedAllocation
{
    public static IGpuBuffer Allocate(IDirectGpuBackend backend, int size, GpuElementType type)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), size, "Buffer size must be positive.");
        if (!backend.SupportsElementType(type))
            throw new NotSupportedException($"{backend.BackendName} cannot hold {type} buffers on {backend.DeviceName}.");
        // The 4-byte types keep their pooled allocation paths.
        if (type == GpuElementType.Float32) return backend.AllocateBuffer(size);
        if (type == GpuElementType.Int32) return backend.AllocateIntBuffer(size);
        long bytes = (long)size * GpuElementTypes.SizeInBytes(type);
        if (bytes > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(size), size, $"{size} {type} elements exceed a single allocation.");
        var buffer = backend.AllocateByteBuffer((int)bytes);
        if (buffer is not IElementTypedByteBuffer typed)
        {
            buffer.Dispose();
            throw new NotSupportedException($"{backend.BackendName} reports {type} storage but its byte buffers carry no element type.");
        }
        typed.DeclareElements(type, size);
        return buffer;
    }
}

/// <summary>Typed host transfers for device buffers of any element type.</summary>
public static class GpuTypedBufferExtensions
{
    /// <summary>Copies <paramref name="data"/> into the start of <paramref name="buffer"/>.</summary>
    /// <exception cref="ArgumentException">
    /// <typeparamref name="T"/> is not the buffer's element type, or the data is longer than the buffer.
    /// </exception>
    public static void UploadBuffer<T>(this IDirectGpuBackend backend, IGpuBuffer buffer, ReadOnlySpan<T> data) where T : unmanaged
    {
        Validate<T>(backend, buffer, data.Length);
        if (data.Length == 0) return;
        if (buffer.ElementType == GpuElementType.Float32)
            backend.UploadBufferInPlace(System.Runtime.InteropServices.MemoryMarshal.Cast<T, float>(data).ToArray(), buffer);
        else if (buffer.ElementType == GpuElementType.Int32)
            backend.UploadIntBufferInPlace(System.Runtime.InteropServices.MemoryMarshal.Cast<T, int>(data).ToArray(), buffer);
        else
            backend.UploadByteBuffer(buffer, System.Runtime.InteropServices.MemoryMarshal.AsBytes(data).ToArray());
    }

    /// <summary>Copies the first <paramref name="destination"/>.Length elements of <paramref name="buffer"/> to the host.</summary>
    /// <exception cref="ArgumentException">
    /// <typeparamref name="T"/> is not the buffer's element type, or the destination is longer than the buffer.
    /// </exception>
    public static void DownloadBuffer<T>(this IDirectGpuBackend backend, IGpuBuffer buffer, Span<T> destination) where T : unmanaged
    {
        Validate<T>(backend, buffer, destination.Length);
        if (destination.Length == 0) return;
        if (buffer.ElementType is GpuElementType.Float32 or GpuElementType.Int32)
        {
            // The 4-byte types live in the float allocation paths; their bits come back through the float download.
            var floats = backend.DownloadBuffer(buffer);
            System.Runtime.InteropServices.MemoryMarshal.Cast<float, T>(floats.AsSpan()).Slice(0, destination.Length).CopyTo(destination);
            return;
        }
        int bytes = destination.Length * GpuElementTypes.SizeInBytes(buffer.ElementType);
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(backend.DownloadByteBuffer(buffer, bytes).AsSpan())
            .Slice(0, destination.Length).CopyTo(destination);
    }

    private static void Validate<T>(IDirectGpuBackend backend, IGpuBuffer buffer, int length) where T : unmanaged
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (!GpuElementTypes.TryFromClrType<T>(out var hostType) || hostType != buffer.ElementType)
            throw new ArgumentException($"A buffer of {buffer.ElementType} elements cannot be transferred as {typeof(T).Name}.", nameof(buffer));
        // The transfer reinterprets host memory, so the host type must have the device layout (the Half shim on
        // frameworks without System.Half is wider than two bytes).
        if (System.Runtime.CompilerServices.Unsafe.SizeOf<T>() != GpuElementTypes.SizeInBytes(hostType))
            throw new NotSupportedException($"{typeof(T).Name} does not have the {hostType} device layout on this framework.");
        if (length > buffer.Size)
            throw new ArgumentException($"{length} elements exceed the buffer's {buffer.Size}.", nameof(buffer));
    }
}
