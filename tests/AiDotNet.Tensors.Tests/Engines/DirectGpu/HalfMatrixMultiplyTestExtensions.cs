// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The half matrix multiply of a backend seen through its fused-kernel interface: the multiply itself lives on the
/// typed <see cref="IGpuNarrowFloatMatrixMultiply"/>, which <c>Hgemm</c> reaches.
/// </summary>
internal static class HalfMatrixMultiplyTestExtensions
{
    // The legacy half GEMM kernels, which the conversions and the mixed-precision multiply these tests call also need.
    // A backend with only the typed Float16 multiply does not have them.
    public static bool HasHgemm(this IGpuHalfFusedKernels kernels) => kernels.SupportsHgemm;

    public static void Hgemm(this IGpuHalfFusedKernels kernels, IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
        => ((IGpuNarrowFloatMatrixMultiply)kernels).MatrixMultiplyNarrowFloat(GpuElementType.Float16, a, b, c, m, n, k);
}
