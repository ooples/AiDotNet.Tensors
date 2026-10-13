// Copyright (c) AiDotNet. All rights reserved.
// Generated row / column kernel variants for Metal, served through the tuned-kernel registry (#1116).

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Metal instances of the registry's row-wise and column-wise variants, emitted from <see cref="TunedRowKernelTemplate"/>
/// (the same bodies OpenCL runs), plus the reference kernels the split backward slots need.
/// </summary>
/// <remarks>
/// Metal's established LayerNorm and RMSNorm backward kernels compute the input gradient and the parameter gradients
/// together on ONE thread, so neither half can stand alone as a slot reference. <c>ref_*</c> below split them: the
/// input gradient takes a thread per row and each parameter gradient a thread per column, with the arithmetic of the
/// serial kernels unchanged and in the same order (Metal's default fast-math may still contract a multiply-add
/// differently, so agreement is to rounding, not guaranteed to the bit).
/// </remarks>
internal static class MetalTunedRowKernels
{
    internal const string LayerNormBackwardInputReference = "ref_layer_norm_backward_input_rows";
    internal const string LayerNormGradParamsReference = "ref_layer_norm_grad_params_columns";
    internal const string RmsNormBackwardInputReference = "ref_rms_norm_backward_input_rows";
    internal const string RmsNormGradGammaReference = "ref_rms_norm_grad_gamma_columns";

    internal static string GetSource() => TunedRowKernelTemplate.GetSource(TunedKernelDialect.Metal) + ReferenceSource;

    private const string ReferenceSource = @"
kernel void ref_layer_norm_backward_input_rows(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device const float* gamma [[buffer(2)]],
    device const float* savedMean [[buffer(3)]], device const float* savedInverse [[buffer(4)]], device float* gradInput [[buffer(5)]],
    constant uint& rows [[buffer(6)]], constant uint& width [[buffer(7)]], uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    uint offset = row * width; float mean = savedMean[row], inverse = savedInverse[row];
    float sum1 = 0.0f, sum2 = 0.0f;
    for (uint i = 0; i < width; ++i) { uint index = offset + i; float normalized = (input[index] - mean) * inverse; sum1 += gradOutput[index] * gamma[i]; sum2 += gradOutput[index] * gamma[i] * normalized; }
    for (uint i = 0; i < width; ++i) { uint index = offset + i; float normalized = (input[index] - mean) * inverse; gradInput[index] = inverse * (gradOutput[index] * gamma[i] - sum1 / float(width) - normalized * sum2 / float(width)); }
}

kernel void ref_layer_norm_grad_params_columns(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]],
    device const float* savedMean [[buffer(2)]], device const float* savedInverse [[buffer(3)]],
    device float* gradGamma [[buffer(4)]], device float* gradBeta [[buffer(5)]],
    constant uint& rows [[buffer(6)]], constant uint& width [[buffer(7)]], uint col [[thread_position_in_grid]])
{
    if (col >= width) return;
    float gg = 0.0f, gb = 0.0f;
    for (uint row = 0; row < rows; ++row) { uint index = row * width + col; float normalized = (input[index] - savedMean[row]) * savedInverse[row]; gg += gradOutput[index] * normalized; gb += gradOutput[index]; }
    gradGamma[col] = gg; gradBeta[col] = gb;
}

kernel void ref_rms_norm_backward_input_rows(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device const float* gamma [[buffer(2)]],
    device const float* savedRms [[buffer(3)]], device float* gradInput [[buffer(4)]],
    constant uint& rows [[buffer(5)]], constant uint& width [[buffer(6)]], uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    uint offset = row * width; float rms = savedRms[row], inverse = 1.0f / rms;
    float sum = 0.0f; for (uint i = 0; i < width; ++i) { uint index = offset + i; sum += gradOutput[index] * gamma[i] * input[index]; }
    for (uint i = 0; i < width; ++i) { uint index = offset + i; float normalizedGradient = gradOutput[index] * gamma[i]; gradInput[index] = inverse * (normalizedGradient - input[index] * sum / (rms * rms * float(width))); }
}

kernel void ref_rms_norm_grad_gamma_columns(
    device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device const float* savedRms [[buffer(2)]],
    device float* gradGamma [[buffer(3)]], constant uint& rows [[buffer(4)]], constant uint& width [[buffer(5)]],
    uint col [[thread_position_in_grid]])
{
    if (col >= width) return;
    float acc = 0.0f;
    for (uint row = 0; row < rows; ++row) { uint index = row * width + col; float inverse = 1.0f / savedRms[row]; acc += gradOutput[index] * input[index] * inverse; }
    gradGamma[col] = acc;
}
";
}
