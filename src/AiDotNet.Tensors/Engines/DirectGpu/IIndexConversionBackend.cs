// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The index helpers the any-rank gather and scatter paths use (#1700). Internal, beside <see cref="IDirectGpuBackend"/>
/// rather than on it: that interface has shipped, and a member added to it breaks every implementation outside this
/// package (package validation, CP0006). Every backend implements this one, as does <c>DelegatingGpuBackend</c>, so a
/// recording or deferred backend records the launches like any other.
/// </summary>
internal interface IIndexConversionBackend
{
    /// <summary>
    /// Raw int32 values (the storage of device-resident index tensors such as argmax indices) to float:
    /// output[i] = (float)raw[i]. The numeric float form the float-indexed kernels (take-along-dim) read.
    /// </summary>
    void Int32ToFloat(IGpuBuffer raw, IGpuBuffer output, int n);

    /// <summary>
    /// Each write's position along the scatter axis for [outer, srcDim, inner] writes, as a float:
    /// output[g] = (g / inner) % srcDim, computed in integers on the device (no host upload).
    /// </summary>
    void ScatterPositions(IGpuBuffer output, int srcDim, int inner, int n);
}
