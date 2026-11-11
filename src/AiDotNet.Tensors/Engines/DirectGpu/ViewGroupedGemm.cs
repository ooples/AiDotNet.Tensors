// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="IGroupedGemmBackend"/> on CUDA and HIP: one vendor-BLAS GEMM per non-empty group at its exact row count, on
/// views of the caller's buffers (<see cref="IGpuBufferViews"/>, free at any offset on these two backends), so nothing
/// is gathered or padded. The weight gradient transposes each group's blocks into scratch (the backend contract has no
/// transposed-A GEMM) and multiplies them with <see cref="IDirectGpuBackend.MatMulTransposed"/>.
/// </summary>
internal static class ViewGroupedGemm
{
    private static IGpuBuffer View(IGpuBufferViews views, IGpuBuffer parent, int offset, int count) =>
        views.TryCreateView(parent, offset, count)
        ?? throw new InvalidOperationException(
            $"{views.GetType().Name} refused a buffer view at offset {offset} ({count} elements) although it reports views free at any offset.");

    private static bool Valid(int[] ends, int m)
    {
        if (ends.Length == 0) return false;
        int prev = 0;
        foreach (int e in ends)
        {
            if (e < prev) return false;
            prev = e;
        }
        return prev <= m;
    }

    internal static bool TryForward(IDirectGpuBackend backend, IGpuBuffer a, IGpuBuffer b, IGpuBuffer y, int[] ends,
        int m, int n, int k, bool transposeB, IGpuBuffer? bias, bool biasPerGroup)
    {
        if (backend is not IGpuBufferViews { ViewsAreFreeAtAnyOffset: true } views
            || m <= 0 || n <= 0 || k <= 0 || !Valid(ends, m))
            return false;
        int start = 0;
        for (int g = 0; g < ends.Length; g++)
        {
            int rows = ends[g] - start;
            if (rows > 0)
            {
                using var av = View(views, a, start * k, rows * k);
                using var bv = View(views, b, g * k * n, k * n);
                using var yv = View(views, y, start * n, rows * n);
                if (transposeB) backend.MatMulTransposed(av, bv, yv, rows, n, k);
                else backend.Gemm(av, bv, yv, rows, n, k);
                if (bias is not null)
                {
                    if (biasPerGroup)
                    {
                        using var biasView = View(views, bias, g * n, n);
                        backend.BiasAdd(yv, biasView, yv, rows, n);
                    }
                    else backend.BiasAdd(yv, bias, yv, rows, n);
                }
            }
            start = ends[g];
        }
        if (start < m)
        {
            using var tail = View(views, y, start * n, (m - start) * n);
            backend.Fill(tail, 0f, (m - start) * n);
        }
        return true;
    }

    internal static bool TryWeightGrad(IDirectGpuBackend backend, IGpuBuffer a, IGpuBuffer c, IGpuBuffer w, int[] ends,
        int m, int k, int n)
    {
        if (backend is not IGpuBufferViews { ViewsAreFreeAtAnyOffset: true } views
            || k <= 0 || n <= 0 || m < 0 || !Valid(ends, m))
            return false;
        int used = ends[ends.Length - 1];
        // Scratch for the transposed blocks of the rows that belong to a group (rows past the last offset take no part).
        using var aT = backend.AllocateBuffer(Math.Max(1, used * k));
        using var cT = backend.AllocateBuffer(Math.Max(1, used * n));
        int start = 0;
        for (int g = 0; g < ends.Length; g++)
        {
            int rows = ends[g] - start;
            using var wv = View(views, w, g * k * n, k * n);
            if (rows > 0)
            {
                using var av = View(views, a, start * k, rows * k);
                using var cv = View(views, c, start * n, rows * n);
                using var atv = View(views, aT, start * k, rows * k);
                using var ctv = View(views, cT, start * n, rows * n);
                backend.Transpose(av, atv, rows, k);
                backend.Transpose(cv, ctv, rows, n);
                backend.MatMulTransposed(atv, ctv, wv, k, n, rows);
            }
            else backend.Fill(wv, 0f, k * n);
            start = ends[g];
        }
        return true;
    }
}
