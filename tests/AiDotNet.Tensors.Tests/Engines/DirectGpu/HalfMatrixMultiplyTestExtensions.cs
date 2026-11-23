// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The half matrix multiply of a backend seen through its fused-kernel interface: it now lives on the typed
/// <see cref="IGpuNarrowFloatMatrixMultiply"/>, which these reach.
/// </summary>
internal static class HalfMatrixMultiplyTestExtensions
{
    public static bool HasHgemm(this IGpuHalfFusedKernels kernels) => GpuNarrowFloatLayout.HasHalfMatrixMultiply(kernels);

    public static void Hgemm(this IGpuHalfFusedKernels kernels, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
        => ((IGpuNarrowFloatMatrixMultiply)kernels).MatrixMultiplyNarrowFloat(GpuElementType.Float16, a, b, c, m, n, k);
}
