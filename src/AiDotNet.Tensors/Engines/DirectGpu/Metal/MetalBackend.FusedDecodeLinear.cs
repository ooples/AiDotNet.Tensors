// Copyright (c) AiDotNet. All rights reserved.
// Metal IFusedDecodeLinearBackend (#974): the portable decode-linear routes over the shared dl_p_ kernels.
using P = AiDotNet.Tensors.Engines.DirectGpu.PortableFusedDecodeLinear;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IFusedDecodeLinearBackend
{
    void IFusedDecodeLinearBackend.FusedLinearGELUTransposedM1(IGpuBuffer input, IGpuBuffer outputMajorWeight, IGpuBuffer bias,
        IGpuBuffer output, int inFeatures, int outFeatures)
        => P.Linear(LaunchBatch, input, outputMajorWeight, bias, output, 1, inFeatures, outFeatures);

    void IFusedDecodeLinearBackend.FusedLinearGELUFp16TransposedM1(IGpuBuffer inputHalf, IGpuBuffer outputMajorWeightHalf,
        IGpuBuffer biasFloat, IGpuBuffer outputFloat, int inFeatures, int outFeatures)
        => P.LinearFp16(this, LaunchBatch, inputHalf, outputMajorWeightHalf, biasFloat, outputFloat, 1, inFeatures, outFeatures);

    void IFusedDecodeLinearBackend.FusedLinearGELUFp16TransposedM16(IGpuBuffer inputHalf, IGpuBuffer outputMajorWeightHalf,
        IGpuBuffer biasFloat, IGpuBuffer outputFloat, int inFeatures, int outFeatures)
        => P.LinearFp16Rows16(this, LaunchBatch, inputHalf, outputMajorWeightHalf, biasFloat, outputFloat, inFeatures, outFeatures);

    void IFusedDecodeLinearBackend.FusedLinearGELUW8A8TransposedM1(IGpuBuffer inputInt8, IGpuBuffer outputMajorWeightsInt8,
        IGpuBuffer activationScaleFloat, IGpuBuffer weightScalesFloat, IGpuBuffer biasFloat, IGpuBuffer outputFloat,
        int inputFeatures, int outputFeatures)
        => P.LinearW8A8(this, LaunchBatch, inputInt8, outputMajorWeightsInt8, activationScaleFloat, weightScalesFloat, biasFloat,
            outputFloat, inputFeatures, outputFeatures);

    void IFusedDecodeLinearBackend.W8A8DequantBiasGelu(IGpuBuffer accumulatorInt32, IGpuBuffer activationScaleFloat,
        IGpuBuffer weightScalesFloat, IGpuBuffer biasFloat, IGpuBuffer outputFloat, int outputFeatures)
        => P.W8A8Epilogue(LaunchBatch, accumulatorInt32, activationScaleFloat, weightScalesFloat, biasFloat, outputFloat, outputFeatures);
}
