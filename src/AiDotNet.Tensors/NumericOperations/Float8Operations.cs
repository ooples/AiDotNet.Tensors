namespace AiDotNet.Tensors.NumericOperations;

/// <summary>
/// <see cref="Interfaces.INumericOperations{T}"/> adapter for
/// <see cref="Float8E4M3"/>. Every op upcasts to <c>float</c>, routes
/// through the float SIMD kernel library, and re-quantizes — same
/// pattern NVIDIA's FP8 GEMM uses when elementwise ops punctuate a
/// mixed-precision pipeline.
///
/// <para>E4M3-specific overrides: Inf saturates to <see cref="Float8E4M3.MaxFinite"/>
/// on encode (no Inf encoding exists in E4M3), so
/// <see cref="NumericOperationsViaFloat{T}.IsInfinity"/> is always false
/// and reductions over mixed ±values can't silently poison downstream
/// ops with NaN from +Inf + −Inf.</para>
/// </summary>
public sealed class Float8E4M3Operations : NumericOperationsViaFloat<Float8E4M3>
{
    protected override float ToFloatImpl(Float8E4M3 value) => value.ToFloat();
    protected override Float8E4M3 FromFloatImpl(float value) => Float8E4M3.FromFloat(value);

    public override Float8E4M3 Zero => Float8E4M3.Zero;
    public override Float8E4M3 One => Float8E4M3.FromFloat(1f);
    public override Float8E4M3 MinValue => Float8E4M3.MinFinite;
    public override Float8E4M3 MaxValue => Float8E4M3.MaxFinite;
    public override int PrecisionBits => 3; // 3 mantissa bits

    public override bool IsNaN(Float8E4M3 value) => value.IsNaN;
    public override bool IsInfinity(Float8E4M3 value) => false; // E4M3 has no Inf encoding.
}

/// <summary>
/// <see cref="Interfaces.INumericOperations{T}"/> adapter for
/// <see cref="Float8E5M2"/>. IEEE-style Inf and NaN encodings unlike
/// E4M3, so the predicates delegate to the struct's own flags.
/// </summary>
public sealed class Float8E5M2Operations : NumericOperationsViaFloat<Float8E5M2>
{
    protected override float ToFloatImpl(Float8E5M2 value) => value.ToFloat();
    protected override Float8E5M2 FromFloatImpl(float value) => Float8E5M2.FromFloat(value);

    public override Float8E5M2 Zero => Float8E5M2.Zero;
    public override Float8E5M2 One => Float8E5M2.FromFloat(1f);
    public override Float8E5M2 MinValue => Float8E5M2.MinFinite;
    public override Float8E5M2 MaxValue => Float8E5M2.MaxFinite;
    public override int PrecisionBits => 2; // 2 mantissa bits

    public override bool IsNaN(Float8E5M2 value) => value.IsNaN;
    public override bool IsInfinity(Float8E5M2 value) => value.IsInfinity;
}
