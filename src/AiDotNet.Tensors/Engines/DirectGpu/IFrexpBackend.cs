// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Optional native FP32 decomposition into mantissa and exponent.</summary>
public interface IFrexpBackend
{
    /// <summary>
    /// Computes <c>input = mantissa * 2^exponent</c>. Exponents are stored as exactly
    /// representable float integers, matching the DirectGPU integer-tensor convention.
    /// </summary>
    void Frexp(
        IGpuBuffer input,
        IGpuBuffer mantissa,
        IGpuBuffer exponent,
        int length);
}
