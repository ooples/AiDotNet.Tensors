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
