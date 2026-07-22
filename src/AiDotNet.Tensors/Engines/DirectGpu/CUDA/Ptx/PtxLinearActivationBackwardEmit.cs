using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Shared activation-gradient emitter for the dense-linear backward PTX kernels. Given
/// the forward preactivation <paramref name="z"/> (= X @ transpose(W) + bias, before
/// the activation) and the upstream gradient <paramref name="dy"/>, it writes
/// <c>out = dy * activation'(z)</c> into <paramref name="outReg"/>. This is the sole
/// activation-specific part of the FusedLinear*Backward family — dInput, dWeight and
/// dBias are computed from the resulting dZ by activation-agnostic GEMMs and a column
/// reduction, exactly mirroring how <see cref="PtxLinearActivationEmit"/> parameterizes
/// the forward.
///
/// Uses scratch registers %f10..%f13 and predicate %p3, and <c>tanh.approx.f32</c>
/// (so tanh/sigmoid/swish/gelu derivatives carry ~1e-3 approximation error, disclosed
/// on the release gate). Every immediate appears only as the last operand, matching the
/// net471-safe convention of the forward emitter.
/// </summary>
internal static class PtxLinearActivationBackwardEmit
{
    // Hex float constants (little-endian IEEE-754), shared with the forward emitter.
    private const string Half = "0f3F000000";       // 0.5
    private const string One = "0f3F800000";        // 1.0
    private const string NegOne = "0fBF800000";     // -1.0
    private const string Zero = "0f00000000";       // 0.0
    private const string LeakySlope = "0f3C23D70A"; // 0.01
    private const string GeluC1 = "0f3D372713";     // 0.044715
    private const string GeluC0 = "0f3F4C422A";     // sqrt(2/pi) = 0.7978845608
    private const string Three = "0f40400000";      // 3.0

    internal static void Emit(
        StringBuilder ptx, DirectPtxLinearActivation activation, string z, string dy, string outReg)
    {
        switch (activation)
        {
            case DirectPtxLinearActivation.None:
                // derivative 1
                ptx.AppendLine($"    mov.f32 {outReg}, {dy};");
                break;

            case DirectPtxLinearActivation.Relu:
                ptx.AppendLine($"    setp.gt.f32 %p3, {z}, {Zero};");
                ptx.AppendLine($"    selp.f32 %f10, {One}, {Zero}, %p3;");
                ptx.AppendLine($"    mul.rn.f32 {outReg}, {dy}, %f10;");
                break;

            case DirectPtxLinearActivation.LeakyRelu:
                ptx.AppendLine($"    setp.ge.f32 %p3, {z}, {Zero};");
                ptx.AppendLine($"    selp.f32 %f10, {One}, {LeakySlope}, %p3;");
                ptx.AppendLine($"    mul.rn.f32 {outReg}, {dy}, %f10;");
                break;

            case DirectPtxLinearActivation.Tanh:
                // 1 - tanh(z)^2
                ptx.AppendLine($"    tanh.approx.f32 %f10, {z};");
                ptx.AppendLine("    mul.rn.f32 %f11, %f10, %f10;");
                ptx.AppendLine($"    mul.rn.f32 %f11, %f11, {NegOne};");
                ptx.AppendLine($"    add.rn.f32 %f11, %f11, {One};");
                ptx.AppendLine($"    mul.rn.f32 {outReg}, {dy}, %f11;");
                break;

            case DirectPtxLinearActivation.Sigmoid:
                // s = 0.5*tanh(0.5z)+0.5 ; s*(1-s)
                EmitSigmoid(ptx, z, "%f10");
                ptx.AppendLine($"    mul.rn.f32 %f11, %f10, {NegOne};");
                ptx.AppendLine($"    add.rn.f32 %f11, %f11, {One};");     // 1 - s
                ptx.AppendLine("    mul.rn.f32 %f10, %f10, %f11;");        // s(1-s)
                ptx.AppendLine($"    mul.rn.f32 {outReg}, {dy}, %f10;");
                break;

            case DirectPtxLinearActivation.Swish:
                // s = sigmoid(z) ; d = s*(1 + z*(1-s))
                EmitSigmoid(ptx, z, "%f10");
                ptx.AppendLine($"    mul.rn.f32 %f11, %f10, {NegOne};");
                ptx.AppendLine($"    add.rn.f32 %f11, %f11, {One};");     // 1 - s
                ptx.AppendLine($"    mul.rn.f32 %f11, %f11, {z};");        // z(1-s)
                ptx.AppendLine($"    add.rn.f32 %f11, %f11, {One};");     // 1 + z(1-s)
                ptx.AppendLine("    mul.rn.f32 %f10, %f10, %f11;");        // s(1+z(1-s))
                ptx.AppendLine($"    mul.rn.f32 {outReg}, {dy}, %f10;");
                break;

            case DirectPtxLinearActivation.GeluTanh:
                EmitGeluDerivative(ptx, z);                                 // d -> %f10
                ptx.AppendLine($"    mul.rn.f32 {outReg}, {dy}, %f10;");
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(activation));
        }
    }

    /// <summary>
    /// Emits <c>out = dy * activation'(y)</c> where the derivative is expressed in terms of
    /// the post-activation output <paramref name="y"/> (what the backend saves for the
    /// smooth squashing activations): sigmoid' = y(1-y), tanh' = 1 - y^2. Only Sigmoid and
    /// Tanh have an exact output-form derivative; other activations must use <see cref="Emit"/>
    /// with the preactivation.
    /// </summary>
    internal static void EmitFromOutput(
        StringBuilder ptx, DirectPtxLinearActivation activation, string y, string dy, string outReg)
    {
        switch (activation)
        {
            case DirectPtxLinearActivation.Sigmoid:
                // y * (1 - y) = y - y^2
                ptx.AppendLine($"    mul.rn.f32 %f10, {y}, {y};");       // y^2
                ptx.AppendLine($"    sub.rn.f32 %f10, {y}, %f10;");       // y - y^2
                ptx.AppendLine($"    mul.rn.f32 {outReg}, {dy}, %f10;");
                break;
            case DirectPtxLinearActivation.Tanh:
                // 1 - y^2
                ptx.AppendLine($"    mul.rn.f32 %f10, {y}, {y};");       // y^2
                ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {NegOne};");
                ptx.AppendLine($"    add.rn.f32 %f10, %f10, {One};");     // 1 - y^2
                ptx.AppendLine($"    mul.rn.f32 {outReg}, {dy}, %f10;");
                break;
            default:
                throw new ArgumentOutOfRangeException(
                    nameof(activation), activation,
                    "Only Sigmoid and Tanh have an exact post-activation-form derivative.");
        }
    }

    private static void EmitSigmoid(StringBuilder ptx, string z, string sReg)
    {
        ptx.AppendLine($"    mul.rn.f32 {sReg}, {z}, {Half};");
        ptx.AppendLine($"    tanh.approx.f32 {sReg}, {sReg};");
        ptx.AppendLine($"    mul.rn.f32 {sReg}, {sReg}, {Half};");
        ptx.AppendLine($"    add.rn.f32 {sReg}, {sReg}, {Half};");
    }

    // gelu'(z) = 0.5(1+t) + 0.5*z*(1-t^2)*u'
    //   u  = c0*(z + c1 z^3),  t = tanh(u),  u' = c0*(1 + 3 c1 z^2). Result -> %f10.
    private static void EmitGeluDerivative(StringBuilder ptx, string z)
    {
        ptx.AppendLine($"    mul.rn.f32 %f10, {z}, {z};");            // z^2
        ptx.AppendLine($"    mul.rn.f32 %f11, %f10, {z};");           // z^3
        ptx.AppendLine($"    fma.rn.f32 %f11, %f11, {GeluC1}, {z};"); // z + c1 z^3
        ptx.AppendLine($"    mul.rn.f32 %f11, %f11, {GeluC0};");      // u
        ptx.AppendLine("    tanh.approx.f32 %f11, %f11;");            // t
        // u' -> %f12
        ptx.AppendLine($"    mul.rn.f32 %f12, %f10, {GeluC1};");      // c1 z^2
        ptx.AppendLine($"    mul.rn.f32 %f12, %f12, {Three};");       // 3 c1 z^2
        ptx.AppendLine($"    add.rn.f32 %f12, %f12, {One};");         // 1 + 3 c1 z^2
        ptx.AppendLine($"    mul.rn.f32 %f12, %f12, {GeluC0};");      // u'
        // left = 0.5(1+t) -> %f13
        ptx.AppendLine($"    add.rn.f32 %f13, %f11, {One};");
        ptx.AppendLine($"    mul.rn.f32 %f13, %f13, {Half};");
        // sech2 = 1 - t^2 -> %f10
        ptx.AppendLine("    mul.rn.f32 %f10, %f11, %f11;");           // t^2
        ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {NegOne};");
        ptx.AppendLine($"    add.rn.f32 %f10, %f10, {One};");         // 1 - t^2
        // right = 0.5 * z * sech2 * u'
        ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {z};");
        ptx.AppendLine("    mul.rn.f32 %f10, %f10, %f12;");
        ptx.AppendLine($"    mul.rn.f32 %f10, %f10, {Half};");
        // d = left + right -> %f10
        ptx.AppendLine("    add.rn.f32 %f10, %f10, %f13;");
    }
}
