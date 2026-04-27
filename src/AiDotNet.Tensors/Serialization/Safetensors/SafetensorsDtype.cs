// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.Serialization.Safetensors;

/// <summary>
/// Safetensors element-type tag — matches the strings the upstream
/// safetensors format uses in its JSON header (<c>F32</c>, <c>F64</c>,
/// <c>BF16</c>, <c>F16</c>, <c>I64</c>, <c>I32</c>, <c>I16</c>,
/// <c>I8</c>, <c>U8</c>, <c>BOOL</c>) plus AiDotNet.Tensors-specific
/// extensions for the sub-byte / FP8 dtypes that don't have an
/// upstream-blessed string yet.
/// </summary>
/// <remarks>
/// The extension tags are namespace-prefixed (<c>AIDN_*</c>) so a
/// safetensors file written with them can still be opened by upstream
/// readers — they'll just refuse to decode the unknown dtype, rather
/// than mis-interpret it as a known one.
/// </remarks>
public enum SafetensorsDtype
{
    F32,
    F64,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    BOOL,

    /// <summary>FP8 E4M3 — OCP spec, 1 sign / 4 exp / 3 mantissa.</summary>
    F8_E4M3,
    /// <summary>FP8 E5M2 — OCP spec, 1 sign / 5 exp / 2 mantissa.</summary>
    F8_E5M2,

    /// <summary>NF4 (normal-distribution 4-bit, QLoRA-style).</summary>
    AIDN_NF4,
    /// <summary>FP4 (E2M1).</summary>
    AIDN_FP4,
    /// <summary>Signed 4-bit integer, two values packed per byte.</summary>
    AIDN_INT4,
    /// <summary>Signed 3-bit integer, two values per byte (3-bit + 1 slack).</summary>
    AIDN_INT3,
    /// <summary>Signed 2-bit integer, four values per byte.</summary>
    AIDN_INT2,
    /// <summary>1-bit (BitNet convention: 0→-1, 1→+1).</summary>
    AIDN_INT1,
}

/// <summary>Conversion between safetensors dtype tags and their on-wire string form.</summary>
public static class SafetensorsDtypeExtensions
{
    /// <summary>Renders a dtype as the string the safetensors header uses.</summary>
    public static string ToHeaderString(this SafetensorsDtype dtype) => dtype switch
    {
        SafetensorsDtype.F32 => "F32",
        SafetensorsDtype.F64 => "F64",
        SafetensorsDtype.F16 => "F16",
        SafetensorsDtype.BF16 => "BF16",
        SafetensorsDtype.I64 => "I64",
        SafetensorsDtype.I32 => "I32",
        SafetensorsDtype.I16 => "I16",
        SafetensorsDtype.I8 => "I8",
        SafetensorsDtype.U8 => "U8",
        SafetensorsDtype.BOOL => "BOOL",
        SafetensorsDtype.F8_E4M3 => "F8_E4M3",
        SafetensorsDtype.F8_E5M2 => "F8_E5M2",
        SafetensorsDtype.AIDN_NF4 => "AIDN_NF4",
        SafetensorsDtype.AIDN_FP4 => "AIDN_FP4",
        SafetensorsDtype.AIDN_INT4 => "AIDN_INT4",
        SafetensorsDtype.AIDN_INT3 => "AIDN_INT3",
        SafetensorsDtype.AIDN_INT2 => "AIDN_INT2",
        SafetensorsDtype.AIDN_INT1 => "AIDN_INT1",
        _ => throw new InvalidOperationException($"Unknown safetensors dtype: {dtype}"),
    };

    /// <summary>
    /// Parses a header dtype string to the corresponding enum value.
    /// Throws <see cref="NotSupportedException"/> on an unknown tag —
    /// callers that want to pass through unknown dtypes should catch
    /// and skip the affected tensor rather than abort the whole load.
    /// </summary>
    public static SafetensorsDtype ParseHeaderString(string s) => s switch
    {
        "F32" => SafetensorsDtype.F32,
        "F64" => SafetensorsDtype.F64,
        "F16" => SafetensorsDtype.F16,
        "BF16" => SafetensorsDtype.BF16,
        "I64" => SafetensorsDtype.I64,
        "I32" => SafetensorsDtype.I32,
        "I16" => SafetensorsDtype.I16,
        "I8" => SafetensorsDtype.I8,
        "U8" => SafetensorsDtype.U8,
        "BOOL" => SafetensorsDtype.BOOL,
        "F8_E4M3" => SafetensorsDtype.F8_E4M3,
        "F8_E5M2" => SafetensorsDtype.F8_E5M2,
        "AIDN_NF4" => SafetensorsDtype.AIDN_NF4,
        "AIDN_FP4" => SafetensorsDtype.AIDN_FP4,
        "AIDN_INT4" => SafetensorsDtype.AIDN_INT4,
        "AIDN_INT3" => SafetensorsDtype.AIDN_INT3,
        "AIDN_INT2" => SafetensorsDtype.AIDN_INT2,
        "AIDN_INT1" => SafetensorsDtype.AIDN_INT1,
        _ => throw new NotSupportedException($"Unknown safetensors dtype tag: {s}"),
    };

    /// <summary>
    /// Element size in bytes. For sub-byte types this is the natural
    /// byte size of one packed unit (1 byte holding 2/4/8 elements).
    /// Callers that need element count vs. byte count should multiply
    /// by the per-byte packing factor for sub-byte types.
    /// </summary>
    public static int ElementByteSize(this SafetensorsDtype dtype) => dtype switch
    {
        SafetensorsDtype.F32 => 4,
        SafetensorsDtype.F64 => 8,
        SafetensorsDtype.F16 => 2,
        SafetensorsDtype.BF16 => 2,
        SafetensorsDtype.I64 => 8,
        SafetensorsDtype.I32 => 4,
        SafetensorsDtype.I16 => 2,
        SafetensorsDtype.I8 => 1,
        SafetensorsDtype.U8 => 1,
        SafetensorsDtype.BOOL => 1,
        SafetensorsDtype.F8_E4M3 => 1,
        SafetensorsDtype.F8_E5M2 => 1,
        SafetensorsDtype.AIDN_NF4 => 1,    // packed 2 per byte
        SafetensorsDtype.AIDN_FP4 => 1,    // packed 2 per byte
        SafetensorsDtype.AIDN_INT4 => 1,   // packed 2 per byte
        SafetensorsDtype.AIDN_INT3 => 1,   // packed 2 per byte (with slack bit)
        SafetensorsDtype.AIDN_INT2 => 1,   // packed 4 per byte
        SafetensorsDtype.AIDN_INT1 => 1,   // packed 8 per byte
        _ => throw new InvalidOperationException($"Unknown safetensors dtype: {dtype}"),
    };
}
