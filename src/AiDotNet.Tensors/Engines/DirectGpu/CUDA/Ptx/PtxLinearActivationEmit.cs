using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Shared fused-epilogue activation emitter for the dense-linear PTX tiles
/// (<see cref="PtxFusedLinearTiledKernel"/> and <see cref="PtxFusedGemmBiasKernel"/>).
/// Applies the activation in place on register <paramref name="v"/> after the bias
/// add and before the final global store, using scratch register %f25. Uses
/// <c>tanh.approx.f32</c>, so tanh-based activations carry ~1e-3 approximation
/// error (disclosed on the release gate).
/// </summary>
internal static class PtxLinearActivationEmit
{
    internal static void Emit(StringBuilder ptx, DirectPtxLinearActivation activation, string v)
    {
        switch (activation)
        {
            case DirectPtxLinearActivation.None:
                break;
            case DirectPtxLinearActivation.Relu:
                ptx.AppendLine($"    max.f32 {v}, {v}, 0f00000000;");
                break;
            case DirectPtxLinearActivation.LeakyRelu:
                ptx.AppendLine($"    mul.rn.f32 %f25, {v}, 0f3C23D70A;"); // 0.01x
                ptx.AppendLine($"    max.f32 {v}, {v}, %f25;");
                break;
            case DirectPtxLinearActivation.Tanh:
                ptx.AppendLine($"    tanh.approx.f32 {v}, {v};");
                break;
            case DirectPtxLinearActivation.Sigmoid:
                // 1/(1+e^-x) = 0.5*tanh(0.5x) + 0.5 (mathematically exact).
                ptx.AppendLine($"    mul.rn.f32 %f25, {v}, 0f3F000000;");
                ptx.AppendLine("    tanh.approx.f32 %f25, %f25;");
                ptx.AppendLine("    mul.rn.f32 %f25, %f25, 0f3F000000;");
                ptx.AppendLine($"    add.rn.f32 {v}, %f25, 0f3F000000;");
                break;
            case DirectPtxLinearActivation.Swish:
                // x * sigmoid(x)
                ptx.AppendLine($"    mul.rn.f32 %f25, {v}, 0f3F000000;");
                ptx.AppendLine("    tanh.approx.f32 %f25, %f25;");
                ptx.AppendLine("    mul.rn.f32 %f25, %f25, 0f3F000000;");
                ptx.AppendLine("    add.rn.f32 %f25, %f25, 0f3F000000;");
                ptx.AppendLine($"    mul.rn.f32 {v}, {v}, %f25;");
                break;
            case DirectPtxLinearActivation.GeluTanh:
                // 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
                ptx.AppendLine($"    mul.rn.f32 %f25, {v}, {v};");
                ptx.AppendLine($"    mul.rn.f32 %f25, %f25, {v};");
                ptx.AppendLine($"    fma.rn.f32 %f25, %f25, 0f3D372713, {v};");
                ptx.AppendLine("    mul.rn.f32 %f25, %f25, 0f3F4C422A;");
                ptx.AppendLine("    tanh.approx.f32 %f25, %f25;");
                ptx.AppendLine("    add.rn.f32 %f25, %f25, 0f3F800000;");
                ptx.AppendLine($"    mul.rn.f32 %f25, %f25, {v};");
                ptx.AppendLine($"    mul.rn.f32 {v}, %f25, 0f3F000000;");
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(activation));
        }
    }
}
