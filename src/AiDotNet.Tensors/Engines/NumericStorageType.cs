namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The element type of a raw numeric buffer: what <see cref="TensorDataEngineExtensions.TensorFromBytes{T}"/> decodes
/// (the <c>dtype</c> of <c>torch.frombuffer</c> / <c>numpy.frombuffer</c>).
/// </summary>
/// <remarks>
/// The values are stable: the device kernels receive them as integers. This is a storage format, not the codegen IR's
/// element type (<see cref="Compilation.Codegen.Ir.CodegenElementType"/>), which has no 16-bit or unsigned 32-bit
/// integers and carries packed sub-byte formats a byte buffer of whole elements cannot hold.
/// </remarks>
public enum NumericStorageType
{
    /// <summary>IEEE-754 binary16 (2 bytes).</summary>
    Float16 = 0,

    /// <summary>IEEE-754 binary32 (4 bytes).</summary>
    Float32 = 1,

    /// <summary>IEEE-754 binary64 (8 bytes).</summary>
    Float64 = 2,

    /// <summary>Signed 8-bit integer.</summary>
    Int8 = 3,

    /// <summary>Unsigned 8-bit integer.</summary>
    UInt8 = 4,

    /// <summary>Signed 16-bit integer.</summary>
    Int16 = 5,

    /// <summary>Unsigned 16-bit integer.</summary>
    UInt16 = 6,

    /// <summary>Signed 32-bit integer.</summary>
    Int32 = 7,

    /// <summary>Unsigned 32-bit integer.</summary>
    UInt32 = 8,

    /// <summary>Signed 64-bit integer.</summary>
    Int64 = 9,
}

/// <summary>Helpers for <see cref="NumericStorageType"/>.</summary>
public static class NumericStorageTypeExtensions
{
    /// <summary>The size of one element in bytes.</summary>
    /// <param name="type">The storage type.</param>
    /// <returns>1, 2, 4 or 8.</returns>
    public static int GetByteWidth(this NumericStorageType type) => type switch
    {
        NumericStorageType.Int8 or NumericStorageType.UInt8 => 1,
        NumericStorageType.Float16 or NumericStorageType.Int16 or NumericStorageType.UInt16 => 2,
        NumericStorageType.Float32 or NumericStorageType.Int32 or NumericStorageType.UInt32 => 4,
        NumericStorageType.Float64 or NumericStorageType.Int64 => 8,
        _ => throw new System.ArgumentOutOfRangeException(nameof(type), type, "Unknown numeric storage type."),
    };
}
