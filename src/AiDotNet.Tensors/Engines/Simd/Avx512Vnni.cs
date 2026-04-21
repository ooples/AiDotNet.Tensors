using System;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// AVX-512 VNNI int8 dot-product helper. Target: Intel Cascade Lake and
/// later (Ice Lake, Sapphire Rapids) + AMD Zen 5+. VNNI's <c>vpdpbusd</c>
/// instruction does (signed int32 += uint8 × int8 × 4 per lane × 16
/// lanes) — 64 int8 MACs per instruction. That's 8× the throughput of
/// scalar int8 MACs over a future QLinearMatMul path that keeps int8
/// through the GEMM instead of dequantising.
///
/// <para><b>Scope (B5).</b> This file ships:</para>
/// <list type="bullet">
/// <item>The <see cref="CanUse"/> feature gate (wired to
/// <see cref="CpuFeatures.HasAVX512VNNI"/> / <see cref="CpuFeatures.HasAMX"/>).</item>
/// <item>The scalar reference <see cref="DotInt8"/> — the exact contract
/// the VNNI kernel must match.</item>
/// <item>A SIMD body sketched out but compiled out; the concrete VNNI
/// intrinsic signatures land with the real int8 GEMM integration. The
/// BCL exposes <see cref="System.Runtime.Intrinsics.X86.Avx512Vbmi"/> +
/// <c>AvxVnni</c> surface on .NET 9+ but the exact entry point
/// (<c>MultiplyWideningAndAdd</c> vs <c>VectorDotProduct</c>) keeps
/// changing across previews — the final wiring happens in the same PR
/// that refactors QLinearMatMul to preserve int8 operands.</item>
/// </list>
///
/// <para>AMX support is gated by <see cref="CpuFeatures.HasAMX"/> but not
/// wired — the .NET BCL doesn't yet expose <c>_tile_*</c> intrinsics. When
/// it does, the AMX path slots in behind the same gate.</para>
/// </summary>
internal static class Avx512Vnni
{
    /// <summary>
    /// True iff the current CPU + runtime supports VNNI int8 dot product
    /// AND the SIMD body is compiled in (future). Today this returns false
    /// even on VNNI-capable silicon — consumers fall through to the scalar
    /// reference. Flip when the SIMD body lands.
    /// </summary>
    public static bool CanUse => false;

    /// <summary>
    /// int32 dot product of <paramref name="a"/> (unsigned int8) and
    /// <paramref name="b"/> (signed int8) vectors of length
    /// <paramref name="length"/>. The contract a future VNNI SIMD kernel
    /// must preserve bit-exact.
    /// </summary>
    public static int DotInt8(ReadOnlySpan<byte> a, ReadOnlySpan<sbyte> b, int length)
    {
        if (a.Length < length || b.Length < length)
            throw new ArgumentException("a/b must hold at least `length` bytes.");
        int acc = 0;
        for (int i = 0; i < length; i++)
            acc += (int)a[i] * (int)b[i];
        return acc;
    }
}
