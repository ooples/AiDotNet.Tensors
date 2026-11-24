// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend whose <see cref="IDirectGpuBackend.BiasAdd"/>, <see cref="IDirectGpuBackend.Exp"/>,
/// <see cref="IDirectGpuBackend.Softmax"/> and <see cref="IDirectGpuBackend.BroadcastMultiplyLastAxis"/> accept their
/// output buffer equal to their input buffer and then give exactly the values the out-of-place call writes. The chunked
/// fused cross-entropy then transforms its <c>[chunk, vocab]</c> logits in place and holds one block per direction
/// instead of two. A backend that binds each buffer once per dispatch (WebGPU forbids a writable alias) does not
/// implement it.
/// </summary>
[BackendSpecificRoute("An aliasing guarantee verified bit for bit on CUDA; every backend runs the same transforms out of place.",
    "BiasAdd", "Exp", "Softmax", "BroadcastMultiplyLastAxis")]
internal interface IInPlaceRowTransformsBackend
{
}
