// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// Each of these kernels reads an element (and its row's statistics) before writing the same element, so an output equal
// to the input gives the out-of-place values; FusedLinearCrossEntropy_index_chunked_in_place_row_transforms_match
// checks it bit for bit at the chunk shapes the fused cross-entropy uses.
public sealed partial class CudaBackend : IInPlaceRowTransformsBackend
{
}
