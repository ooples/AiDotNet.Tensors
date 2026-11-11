using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // Native where a view costs nothing at any offset (CUDA, HIP): expert row blocks start at arbitrary offsets, which
    // Vulkan/Metal/WebGPU/OpenCL views refuse off their alignment, so those keep the device-kernel composition.
    internal override bool UsesNativeGroupedRows =>
        !TryGetBackend(out var backend) || backend is IGpuBufferViews { ViewsAreFreeAtAnyOffset: true };

    // The jagged-rows grouped GEMM on the device: one GEMM per non-empty group at its own row count, on views of the
    // resident buffers (IGpuBufferViews, every backend), so nothing is gathered and nothing is padded to the largest
    // group. Records the same tape node as the CPU op, whose backward calls back into these overrides.
    internal override Tensor<T> GroupedRowsMatMul<T>(Tensor<T> a, Tensor<T> b, int[] ends, bool transposeB)
    {
        using var __gpuOp = EnterGpuOp();
        var (g, m, k, n) = GroupedRowsShape(a, b, ends, transposeB);
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || backend is not IGpuBufferViews views || m == 0 || n == 0 || k == 0)
            return base.GroupedRowsMatMul(a, b, ends, transposeB);

        try
        {
            using var aB = GetOrAllocateBuffer(backend, a);
            using var bB = GetOrAllocateBuffer(backend, b);
            using var yB = AllocateFullyWrittenOutputBuffer(backend, m * n);
            int start = 0;
            for (int i = 0; i < g; i++)
            {
                int rows = ends[i] - start;
                if (rows > 0)
                {
                    using var av = View(views, aB.Buffer, start * k, rows * k);
                    using var bv = View(views, bB.Buffer, i * k * n, k * n);
                    using var yv = View(views, yB.Buffer, start * n, rows * n);
                    if (transposeB) backend.MatMulTransposed(av, bv, yv, rows, n, k);
                    else backend.Gemm(av, bv, yv, rows, n, k);
                }
                start = ends[i];
            }
            if (start < m)
            {
                using var tail = View(views, yB.Buffer, start * n, (m - start) * n);
                backend.Fill(tail, 0f, (m - start) * n);
            }
            var result = DeferTensorResult<T>(backend, yB.Buffer, m * n, new[] { m, n });
            yB.RelinquishOwnership();
            Autodiff.DifferentiableOps.RecordBinary("GroupedMatMul", result, a, b, GroupedRowsMatMulBackward<T>,
                new object[] { ends, transposeB });
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("GroupedRowsMatMul", capabilityGap);
            return base.GroupedRowsMatMul(a, b, ends, transposeB);
        }
    }

    // w[g] = a_g^T c_g. The backend contract has no transposed-A GEMM, so each group's a and c blocks are transposed
    // into scratch ([K, rows] and [N, rows], contiguous per group) and multiplied as A^T (C^T)^T with MatMulTransposed.
    internal override Tensor<T> GroupedRowsWeightGrad<T>(Tensor<T> a, Tensor<T> c, int[] ends)
    {
        using var __gpuOp = EnterGpuOp();
        int g = ends.Length, m = a._shape[0], k = a._shape[1], n = c._shape[1];
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IGpuBufferViews views
            || m == 0 || n == 0 || k == 0)
            return base.GroupedRowsWeightGrad(a, c, ends);

        try
        {
            using var aB = GetOrAllocateBuffer(backend, a);
            using var cB = GetOrAllocateBuffer(backend, c);
            using var wB = AllocateFullyWrittenOutputBuffer(backend, g * k * n);
            using var aT = AllocateFullyWrittenOutputBuffer(backend, m * k);
            using var cT = AllocateFullyWrittenOutputBuffer(backend, m * n);
            int start = 0;
            for (int i = 0; i < g; i++)
            {
                int rows = ends[i] - start;
                using var wv = View(views, wB.Buffer, i * k * n, k * n);
                if (rows > 0)
                {
                    using var av = View(views, aB.Buffer, start * k, rows * k);
                    using var cv = View(views, cB.Buffer, start * n, rows * n);
                    using var atv = View(views, aT.Buffer, start * k, rows * k);
                    using var ctv = View(views, cT.Buffer, start * n, rows * n);
                    backend.Transpose(av, atv, rows, k);
                    backend.Transpose(cv, ctv, rows, n);
                    backend.MatMulTransposed(atv, ctv, wv, k, n, rows);
                }
                else backend.Fill(wv, 0f, k * n);
                start = ends[i];
            }
            var result = DeferTensorResult<T>(backend, wB.Buffer, g * k * n, new[] { g, k, n });
            wB.RelinquishOwnership();
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("GroupedRowsWeightGrad", capabilityGap);
            return base.GroupedRowsWeightGrad(a, c, ends);
        }
    }

    private static IGpuBuffer View(IGpuBufferViews views, IGpuBuffer parent, int offset, int count) =>
        views.TryCreateView(parent, offset, count)
        ?? throw new NotSupportedException($"The backend refused a buffer view at offset {offset} ({count} elements).");
}
