// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The FP16-native kernels live in CudaBackend.cs and CudaBackend.HalfPrecision.cs.
public sealed partial class CudaBackend : IFp16ActivationBackend
{
    /// <remarks>Compression converts on the compute stream and frees the FP32 copy stream-ordered.</remarks>
    bool IFp16ActivationBackend.SupportsFp16ActivationStorage => IsAvailable;
}
