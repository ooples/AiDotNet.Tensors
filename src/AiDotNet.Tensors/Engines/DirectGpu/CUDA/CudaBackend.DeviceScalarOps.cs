// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// ScaleByDeviceScalar and MultiTensorSumOfSquares live in CudaBackend.cs.
public sealed partial class CudaBackend : IDeviceScalarOps
{
    bool IDeviceScalarOps.SupportsMultiTensorSumOfSquares => IsAvailable;
}
