// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// CUDA IFusedDecodeLinearBackend (#974): the routes are this backend's own public methods (direct PTX where a
// specialization is admitted, cuBLAS or the resident NVRTC kernels otherwise); the declaration only names the
// contract the other backends now share.
public sealed partial class CudaBackend : IFusedDecodeLinearBackend
{
}
