// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The fused decode-linear routes (#974): a linear projection, bias and tanh-GELU in one device pass, for the
/// decode-token shapes. Weights are canonical output-major <c>[outFeatures, inFeatures]</c> (row n holds output n's
/// weights); activations are contiguous. Every operand stays on the device.
/// </summary>
/// <remarks>
/// CUDA serves these with its direct-PTX specializations and cuBLAS; every other backend runs the shared
/// <c>dl_p_</c> kernels through <see cref="PortableFusedDecodeLinear"/>. All of them compute
/// <c>0.5 x (1 + tanh(0.79788456 (x + 0.044715 x^3)))</c>.
/// </remarks>
internal interface IFusedDecodeLinearBackend
{
    /// <summary>FP32: <c>output[N] = GELU(input[K] . weight[N, K]^T + bias[N])</c>.</summary>
    void FusedLinearGELUTransposedM1(IGpuBuffer input, IGpuBuffer outputMajorWeight, IGpuBuffer bias, IGpuBuffer output,
        int inFeatures, int outFeatures);

    /// <summary>
    /// FP16 operands, FP32 accumulation and output: <c>outputFloat[N] = GELU(inputHalf[K] . weightHalf[N, K]^T +
    /// biasFloat[N])</c>.
    /// </summary>
    void FusedLinearGELUFp16TransposedM1(IGpuBuffer inputHalf, IGpuBuffer outputMajorWeightHalf, IGpuBuffer biasFloat,
        IGpuBuffer outputFloat, int inFeatures, int outFeatures);

    /// <summary>
    /// The sixteen-row form of <see cref="FusedLinearGELUFp16TransposedM1"/>: <c>outputFloat[16, N] =
    /// GELU(inputHalf[16, K] . weightHalf[N, K]^T + biasFloat[N])</c>, input row-major.
    /// </summary>
    void FusedLinearGELUFp16TransposedM16(IGpuBuffer inputHalf, IGpuBuffer outputMajorWeightHalf, IGpuBuffer biasFloat,
        IGpuBuffer outputFloat, int inFeatures, int outFeatures);

    /// <summary>
    /// Symmetric W8A8 (zero-points exactly zero): <c>outputFloat[N] = GELU(int32(inputInt8[K] . weightsInt8[N, K]^T) *
    /// activationScale[0] * weightScales[N] + biasFloat[N])</c>. The products accumulate exactly in int32.
    /// </summary>
    void FusedLinearGELUW8A8TransposedM1(IGpuBuffer inputInt8, IGpuBuffer outputMajorWeightsInt8, IGpuBuffer activationScaleFloat,
        IGpuBuffer weightScalesFloat, IGpuBuffer biasFloat, IGpuBuffer outputFloat, int inputFeatures, int outputFeatures);

    /// <summary>
    /// The W8A8 epilogue alone, for an int32 accumulator computed elsewhere: <c>outputFloat[N] =
    /// GELU(accumulatorInt32[N] * activationScale[0] * weightScales[N] + biasFloat[N])</c>.
    /// </summary>
    void W8A8DequantBiasGelu(IGpuBuffer accumulatorInt32, IGpuBuffer activationScaleFloat, IGpuBuffer weightScalesFloat,
        IGpuBuffer biasFloat, IGpuBuffer outputFloat, int outputFeatures);
}
