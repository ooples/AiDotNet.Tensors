using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// The elementwise activation applied by the pointwise/activation kernels (issue #839).
/// One parameterized kernel serves every activation; only the emitted transform differs.
/// </summary>
internal enum DirectPtxActivation
{
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
    GeluTanh,
    Silu
}

/// <summary>
/// Shared forward-activation emitter for the pointwise PTX kernels. Applies the activation
/// in place on register <paramref name="v"/> using scratch %f10/%f11 and <c>tanh.approx.f32</c>
/// (so tanh-based activations carry ~1e-3 approximation error, disclosed on the release gate).
/// Every immediate appears only as the last operand, matching the net471-safe convention.
/// </summary>
internal static class PtxActivationEmit
{
    private const string Half = "0f3F000000";       // 0.5
    private const string One = "0f3F800000";        // 1.0
    private const string LeakySlope = "0f3C23D70A"; // 0.01
    private const string GeluC1 = "0f3D372713";     // 0.044715
    private const string GeluC0 = "0f3F4C422A";     // sqrt(2/pi)

    internal static void Emit(StringBuilder ptx, DirectPtxActivation activation, string v)
    {
        switch (activation)
        {
            case DirectPtxActivation.Relu:
                ptx.AppendLine($"    max.f32 {v}, {v}, 0f00000000;");
                break;
            case DirectPtxActivation.LeakyRelu:
                ptx.AppendLine($"    mul.rn.f32 %f10, {v}, {LeakySlope};");
                ptx.AppendLine($"    max.f32 {v}, {v}, %f10;");
                break;
            case DirectPtxActivation.Tanh:
                ptx.AppendLine($"    tanh.approx.f32 {v}, {v};");
                break;
            case DirectPtxActivation.Sigmoid:
                EmitSigmoid(ptx, v, v);
                break;
            case DirectPtxActivation.Silu:
                EmitSigmoid(ptx, v, "%f11");   // s = sigmoid(v) -> %f11
                ptx.AppendLine($"    mul.rn.f32 {v}, {v}, %f11;");
                break;
            case DirectPtxActivation.GeluTanh:
                // 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
                ptx.AppendLine($"    mul.rn.f32 %f10, {v}, {v};");
                ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {v};");
                ptx.AppendLine($"    fma.rn.f32 %f10, %f10, {GeluC1}, {v};");
                ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {GeluC0};");
                ptx.AppendLine("    tanh.approx.f32 %f10, %f10;");
                ptx.AppendLine($"    add.rn.f32 %f10, %f10, {One};");
                ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {v};");
                ptx.AppendLine($"    mul.rn.f32 {v}, %f10, {Half};");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(activation));
        }
    }

    // sigmoid(x) = 0.5*tanh(0.5x) + 0.5, written into <paramref name="dst"/>.
    private static void EmitSigmoid(StringBuilder ptx, string x, string dst)
    {
        ptx.AppendLine($"    mul.rn.f32 %f10, {x}, {Half};");
        ptx.AppendLine("    tanh.approx.f32 %f10, %f10;");
        ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {Half};");
        ptx.AppendLine($"    add.rn.f32 {dst}, %f10, {Half};");
    }
}
