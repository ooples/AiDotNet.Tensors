namespace AiDotNet.Tensors.Engines.Compilation.Serialization;

/// <summary>
/// Wire-format constants for the compiled-plan binary serialization format.
/// All multi-byte values are little-endian. The format is deterministic —
/// serialize → deserialize → serialize produces byte-identical output.
/// </summary>
internal static class PlanFormatConstants
{
    /// <summary>Magic bytes: "ATNS" (AiDotNet Tensors Serialized), LE uint32.</summary>
    internal const uint Magic = 0x534E5441; // 'A','T','N','S' in LE

    /// <summary>
    /// Format version. Bumped on breaking layout changes. The reader rejects
    /// files with a higher version (forward-incompat) and files with version 0
    /// (corruption). Files with the same version are guaranteed readable.
    /// </summary>
    internal const ushort CurrentFormatVersion = 1;

    /// <summary>
    /// Tensor-codec version. Semantically distinct from the format version:
    /// same binary layout but different compiler/optimization output. A plan
    /// compiled under codec V2 might produce different intermediate shapes
    /// than codec V1 — the loader checks this and returns null (forces
    /// recompile) rather than silently mis-replaying.
    /// </summary>
    internal const int TensorCodecVersion = 1;

    // ── Plan type ───────────────────────────────────────────────────────────
    internal const byte PlanTypeInference = 0;
    internal const byte PlanTypeTraining  = 1;

    // ── Element type codes ──────────────────────────────────────────────────
    internal const byte ElementTypeFloat   = 0;
    internal const byte ElementTypeDouble  = 1;
    internal const byte ElementTypeInt32   = 2;
    internal const byte ElementTypeInt64   = 3;
    internal const byte ElementTypeFloat16 = 4;

    /// <summary>
    /// Returns the element type code for a given CLR type, or throws if
    /// the type isn't supported by the serialization format.
    /// </summary>
    internal static byte GetElementTypeCode<T>()
    {
        if (typeof(T) == typeof(float))  return ElementTypeFloat;
        if (typeof(T) == typeof(double)) return ElementTypeDouble;
        if (typeof(T) == typeof(int))    return ElementTypeInt32;
        if (typeof(T) == typeof(long))   return ElementTypeInt64;
        throw new NotSupportedException(
            $"Plan serialization does not support element type {typeof(T).FullName}. " +
            "Supported types: float, double, int, long.");
    }

    /// <summary>
    /// Returns the byte size of one element for the given type code.
    /// </summary>
    internal static int ElementSize(byte typeCode) => typeCode switch
    {
        ElementTypeFloat   => 4,
        ElementTypeDouble  => 8,
        ElementTypeInt32   => 4,
        ElementTypeInt64   => 8,
        ElementTypeFloat16 => 2,
        _ => throw new NotSupportedException($"Unknown element type code: {typeCode}"),
    };

    // ── SavedState type tags ────────────────────────────────────────────────
    // Used by SavedStateSerializer to disambiguate object[] entries.
    internal const byte TagNull       = 0x00;
    internal const byte TagInt32      = 0x01;
    internal const byte TagInt32Array = 0x02;
    internal const byte TagDouble     = 0x03;
    internal const byte TagTensorRef  = 0x04; // references a tensor table ID
    internal const byte TagBool       = 0x05;
    internal const byte TagFloat      = 0x06;
    internal const byte TagString     = 0x07;
    internal const byte TagByteArray  = 0x08;
    internal const byte TagEnum       = 0x09; // assembly-qualified type name + int value

    // ── Tensor table flags ──────────────────────────────────────────────────
    internal const byte TensorFlagWeight       = 0x01; // model parameter
    internal const byte TensorFlagLeafInput    = 0x02; // external input
    internal const byte TensorFlagIntermediate = 0x04; // pre-allocated buffer
    internal const byte TensorFlagHasData      = 0x08; // data section follows (weights have data; intermediates don't)
}
