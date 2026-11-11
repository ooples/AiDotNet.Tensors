using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // The jagged-rows grouped GEMM on the device through the backend's IGroupedGemmBackend, which every backend
    // implements: CUDA and HIP run one vendor-BLAS GEMM per group on buffer views, OpenCL, Vulkan, Metal and WebGPU one
    // launch of GroupedGemmKernelTemplate over every group. Nothing is gathered or padded to the largest group. Records
    // the same tape node as the CPU op, whose backward calls back into these overrides.
    internal override Tensor<T> GroupedRowsMatMul<T>(Tensor<T> a, Tensor<T> b, int[] ends, bool transposeB, Tensor<T>? bias = null)
    {
        using var __gpuOp = EnterGpuOp();
        var (g, m, k, n) = GroupedRowsShape(a, b, ends, transposeB);
        if (bias is not null) ValidateGroupBias(bias, g, n);
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || backend is not IGroupedGemmBackend grouped || m == 0 || n == 0 || k == 0)
            return base.GroupedRowsMatMul(a, b, ends, transposeB, bias);

        using var aB = GetOrAllocateBuffer(backend, a);
        using var bB = GetOrAllocateBuffer(backend, b);
        using var biasB = bias is null ? (OwnedBuffer?)null : GetOrAllocateBuffer(backend, bias);
        using var yB = AllocateFullyWrittenOutputBuffer(backend, m * n);
        if (!grouped.TryGroupedRowsGemm(aB.Buffer, bB.Buffer, yB.Buffer, ends, m, n, k, transposeB, biasB?.Buffer, bias?.Rank == 2))
        {
            GpuLaunchProbe.OnFallback("GroupedRowsMatMul", null);
            return base.GroupedRowsMatMul(a, b, ends, transposeB, bias);
        }
        var result = DeferTensorResult<T>(backend, yB.Buffer, m * n, new[] { m, n });
        yB.RelinquishOwnership();
        RecordGroupedRows(result, a, b, bias, ends, transposeB);
        return result;
    }

    internal override Tensor<T> GroupedRowsWeightGrad<T>(Tensor<T> a, Tensor<T> c, int[] ends)
    {
        using var __gpuOp = EnterGpuOp();
        int g = ends.Length, m = a._shape[0], k = a._shape[1], n = c._shape[1];
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IGroupedGemmBackend grouped
            || g == 0 || k == 0 || n == 0)
            return base.GroupedRowsWeightGrad(a, c, ends);

        using var aB = GetOrAllocateBuffer(backend, a);
        using var cB = GetOrAllocateBuffer(backend, c);
        using var wB = AllocateFullyWrittenOutputBuffer(backend, g * k * n);
        if (!grouped.TryGroupedRowsWeightGrad(aB.Buffer, cB.Buffer, wB.Buffer, ends, m, k, n))
        {
            GpuLaunchProbe.OnFallback("GroupedRowsWeightGrad", null);
            return base.GroupedRowsWeightGrad(a, c, ends);
        }
        var result = DeferTensorResult<T>(backend, wB.Buffer, g * k * n, new[] { g, k, n });
        wB.RelinquishOwnership();
        return result;
    }
}
