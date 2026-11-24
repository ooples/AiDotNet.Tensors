// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend that transforms a <c>[rows, cols]</c> block in place: <see cref="IDirectGpuBackend.BiasAdd"/>,
/// <see cref="IDirectGpuBackend.Exp"/>, <see cref="IDirectGpuBackend.Softmax"/> and
/// <see cref="IDirectGpuBackend.BroadcastMultiplyLastAxis"/>, with exactly the values their reference kernels write out
/// of place (an out-of-place call may take another route, within its rounding). The chunked fused cross-entropy then transforms
/// its <c>[chunk, vocab]</c> logits in place and holds one block per direction instead of two. The in-place entry
/// points run kernels written for an aliased output; the out-of-place calls keep their own routes. A backend that binds
/// each buffer once per dispatch (WebGPU forbids a writable alias) does not implement it.
/// </summary>
[BackendSpecificRoute("In-place forms verified bit for bit on CUDA; every backend runs the same transforms out of place.",
    "BiasAdd", "Exp", "Softmax", "BroadcastMultiplyLastAxis")]
internal interface IInPlaceRowTransformsBackend
{
    /// <summary><c>data[r, c] += bias[c]</c>.</summary>
    void BiasAddInPlace(IGpuBuffer data, IGpuBuffer bias, int rows, int cols);

    /// <summary><c>data[i] = exp(data[i])</c>.</summary>
    void ExpInPlace(IGpuBuffer data, int size);

    /// <summary>Each row of <paramref name="data"/> replaced by its softmax.</summary>
    void SoftmaxInPlace(IGpuBuffer data, int rows, int cols);

    /// <summary><c>data[o, i] *= factors[i]</c>.</summary>
    void BroadcastMultiplyLastAxisInPlace(IGpuBuffer data, IGpuBuffer factors, int outerSize, int innerSize);
}
