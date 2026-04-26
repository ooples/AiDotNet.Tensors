// Copyright (c) AiDotNet. All rights reserved.
// Codegen IR element type taxonomy — shared by every emitter (AVX-512
// C#, Triton, HIP, MSL, WGSL, GLSL). See Engines/Compilation/Codegen/
// Ir/ for the rest of the IR definitions and the lowering pass that
// produces a CodegenGraph from the existing LazyTensorScope /
// CompiledInferencePlan step list.

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// The element type of a tensor in the codegen IR. Each emitter
/// maps this to its target language's concrete scalar type (e.g.
/// Triton <c>tl.float32</c>, HIP <c>float</c>, C# <c>Vector512&lt;float&gt;</c>
/// lane element, WGSL <c>f32</c>).
/// </summary>
/// <remarks>
/// <para><b>Why an enum over <see cref="System.Type"/>:</b></para>
/// <para>
/// Codegen has to serialize IR across the wire (autotune cache
/// persistence, remote kernel lookup, cross-process build servers).
/// <see cref="System.Type"/> tokens aren't portable — the enum is
/// fixed-width, content-stable across .NET versions, and gives the
/// C# compiler exhaustive-switch coverage for emitter case
/// analysis.</para>
/// <para><b>Sub-byte values (NF4, FP4, BitNet) are first-class</b> —
/// they're part of our moat over PyTorch, which ships only int8/fp8.
/// Emitters that don't support a given sub-byte format throw
/// <see cref="System.NotSupportedException"/> at lowering time, not
/// at runtime; Phase D's guard system filters out unsupported
/// variants before a plan is cached.</para>
/// </remarks>
public enum CodegenElementType
{
    /// <summary>IEEE-754 binary32. The default target for CPU + GPU fast paths.</summary>
    Float32,
    /// <summary>IEEE-754 binary64. Needed for scientific / high-precision training.</summary>
    Float64,
    /// <summary>IEEE-754 binary16. Used in mixed-precision training on modern GPUs.</summary>
    Float16,
    /// <summary>Brain-float16 (bfloat16) — matches the float32 exponent range. Ampere+.</summary>
    BFloat16,
    /// <summary>OCP FP8 E4M3 (4 exponent, 3 mantissa) — training-mode signed-zero variant.</summary>
    FP8_E4M3,
    /// <summary>OCP FP8 E5M2 (5 exponent, 2 mantissa) — inference-friendly wider range.</summary>
    FP8_E5M2,
    /// <summary>Signed 32-bit integer.</summary>
    Int32,
    /// <summary>Signed 64-bit integer — long indices, reductions that can overflow int32.</summary>
    Int64,
    /// <summary>Signed 8-bit integer — post-training quantization target.</summary>
    Int8,
    /// <summary>Unsigned 8-bit integer — non-negative quantization (ReLU activations).</summary>
    UInt8,
    /// <summary>NormalFloat-4 — rotation-preserving 4-bit float used by QLoRA / bitsandbytes.</summary>
    NF4,
    /// <summary>IEEE-like FP4 — Micikevicius et al. "FP4 variant of OCP spec".</summary>
    FP4,
    /// <summary>3-bit (typically int3 packed 8 values per 3 bytes) — GPTQ extreme quantization.</summary>
    Int3,
    /// <summary>2-bit (typically int2 packed 4 values per byte) — GPTQ ultra-compression.</summary>
    Int2,
    /// <summary>1-bit (binary / XNOR-popcount).</summary>
    Int1,
    /// <summary>Boolean. Guarded by element-wise ops that accept only boolean inputs.</summary>
    Bool,
}

/// <summary>
/// Helpers for <see cref="CodegenElementType"/> that belong with the
/// enum but can't live on the enum itself.
/// </summary>
public static class CodegenElementTypeExtensions
{
    /// <summary>
    /// Returns the nominal byte width for codegen layout planning.
    /// Sub-byte types report their packed width in fractional bits;
    /// this helper returns the byte count rounded up (e.g. Int4 → 1).
    /// Callers that need exact bit-level layout (packing, stride
    /// computation) must consult <see cref="GetBitWidth"/>.
    /// </summary>
    public static int GetByteWidth(this CodegenElementType t) => t switch
    {
        CodegenElementType.Float64 => 8,
        CodegenElementType.Int64 => 8,
        CodegenElementType.Float32 => 4,
        CodegenElementType.Int32 => 4,
        CodegenElementType.Float16 => 2,
        CodegenElementType.BFloat16 => 2,
        CodegenElementType.Int8 => 1,
        CodegenElementType.UInt8 => 1,
        CodegenElementType.FP8_E4M3 => 1,
        CodegenElementType.FP8_E5M2 => 1,
        CodegenElementType.Bool => 1,
        // Sub-byte rounds up to 1 — emitters must consult GetBitWidth for packing.
        CodegenElementType.NF4 => 1,
        CodegenElementType.FP4 => 1,
        CodegenElementType.Int3 => 1,
        CodegenElementType.Int2 => 1,
        CodegenElementType.Int1 => 1,
        _ => throw new System.ArgumentOutOfRangeException(nameof(t), t, "Unknown element type."),
    };

    /// <summary>
    /// Returns the exact bit width — required for sub-byte packing
    /// math. For byte-multiple types returns 8 × <see cref="GetByteWidth"/>.
    /// </summary>
    public static int GetBitWidth(this CodegenElementType t) => t switch
    {
        CodegenElementType.Int1 => 1,
        CodegenElementType.Int2 => 2,
        CodegenElementType.Int3 => 3,
        CodegenElementType.NF4 => 4,
        CodegenElementType.FP4 => 4,
        _ => 8 * GetByteWidth(t),
    };

    /// <summary>
    /// Returns true for floating-point formats — emitters use this to
    /// gate transcendental function emission (<c>exp</c>, <c>log</c>,
    /// trig) that only makes sense on floats.
    /// </summary>
    public static bool IsFloatingPoint(this CodegenElementType t)
        => t == CodegenElementType.Float32
        || t == CodegenElementType.Float64
        || t == CodegenElementType.Float16
        || t == CodegenElementType.BFloat16
        || t == CodegenElementType.FP8_E4M3
        || t == CodegenElementType.FP8_E5M2
        || t == CodegenElementType.NF4
        || t == CodegenElementType.FP4;

    /// <summary>
    /// Returns true when the bit layout is narrower than one byte —
    /// the packed-storage paths only kick in for these types.
    /// </summary>
    public static bool IsSubByte(this CodegenElementType t)
        => t == CodegenElementType.Int1
        || t == CodegenElementType.Int2
        || t == CodegenElementType.Int3
        || t == CodegenElementType.NF4
        || t == CodegenElementType.FP4;
}
