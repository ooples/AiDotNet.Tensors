// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// A backend whose <see cref="IGpuMixedPrecisionConvBackend"/> does the 16-bit convolution math natively (cuDNN tensor
/// cores on CUDA), so autocast routes convolutions there (#1549). Internal, beside the shipped public interface: the
/// portable implementations only store 16-bit around an FP32 convolution, which autocast must not take since it is no
/// faster, and adding a member to the public interface would break its outside implementers.
/// </summary>
internal interface INativeReducedPrecisionConv
{
    /// <summary>True when the 16-bit convolutions run in native reduced-precision math on this device now.</summary>
    bool NativeReducedPrecisionMath { get; }
}
