using System;
using System.Buffers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // Groups own disjoint output/dX/dW regions. Parallelize that ownership, leaving each
    // group's image/panel reduction order and GEMM options unchanged. A joined worker reuses
    // its scratch across groups; grouped panels/partials together request at most 16 MiB after pool rounding.
    // Existing waveform panel kernels already parallelize columns and keep their own dispatch.
    private static int GroupedPanelWorkers(in GroupedConv2DGeometry g, int panelLength, int partialLength = 0, bool sum = false)
    {
        if (g.Groups <= 1 || CpuParallelSettings.IsInParallelRegion) return 1;
        const long budget = 4_194_304;
        long perWorker = GroupedPoolLength(panelLength) + GroupedPoolLength(partialLength) * (sum ? 2 : 1);
        return Math.Max(1, (int)Math.Min(Math.Min(g.Groups, CpuParallelSettings.MaxDegreeOfParallelism), budget / Math.Max(1, perWorker)));
    }

    private static long GroupedPoolLength(int length)
    {
        if (length == 0) return 0;
        long size = 16;
        while (size < length) size <<= 1;
        return size;
    }

    // A caller can override the process-wide GEMM policy. Dedicated pool threads do not
    // inherit thread-local state, so restore both the effective policy and each worker's prior override.
    private readonly struct GroupedGemmPolicyScope : IDisposable
    {
        private readonly bool? _previous;
        internal GroupedGemmPolicyScope(bool deterministic)
        {
            _previous = BlasProvider.GetThreadLocalDeterministicMode();
            BlasProvider.SetThreadLocalDeterministicMode(deterministic);
        }
        public void Dispose() => BlasProvider.SetThreadLocalDeterministicMode(_previous);
    }

    private static bool TryGroupedParallelForward(float[] source, int inputOffset, float[] weights, int kernelOffset,
        float[] output, int outputOffset, in GroupedConv2DGeometry geometry)
    {
        var g = geometry;
        if (g.InHeight == 1 && g.KernelH == 1 && g.OutHeight == 1 && g.PadH == 0 && g.StrideH == 1
            && g.KernelW > 1 && (long)g.InChannelsPerGroup * g.KernelW * g.OutWidth >= 1_048_576) return false;
        int k = checked(g.InChannelsPerGroup * g.KernelH * g.KernelW);
        int spatial = checked(g.OutHeight * g.OutWidth), columns = Math.Min(256, spatial);
        int panelLength = checked(k * columns), workers = GroupedPanelWorkers(g, panelLength);
        if (workers <= 1) return false;
        bool deterministic = BlasProvider.IsDeterministicMode;
        CpuParallelSettings.ParallelForOrSerial(0, workers, (long)g.Batch * g.OutChannels * k * spatial, worker =>
        {
            using var policy = new GroupedGemmPolicyScope(deterministic);
            var panel = CpuStorageAllocationDiagnostics.RentWorkspace<float>(panelLength, "workspace.grouped_conv_forward_columns");
            try
            {
                for (int group = worker; group < g.Groups; group += workers)
                    for (int image = 0; image < g.Batch; image++)
                    {
                        int x = checked(inputOffset + (image * g.InChannels + group * g.InChannelsPerGroup) * g.InHeight * g.InWidth);
                        int w = checked(kernelOffset + group * g.OutChannelsPerGroup * k);
                        int y = checked(outputOffset + (image * g.OutChannels + group * g.OutChannelsPerGroup) * spatial);
                        for (int from = 0; from < spatial; from += columns)
                        {
                            int count = Math.Min(columns, spatial - from);
                            FillGroupedColumnPanel(source, x, panel, from, count, g);
                            BlasProvider.GemmEx(g.OutChannelsPerGroup, count, k, weights, w, k, false,
                                panel, 0, count, false, output, y + from, spatial);
                        }
                    }
            }
            finally { ArrayPool<float>.Shared.Return(panel); }
        }, deterministicSafe: true);
        return true;
    }

    private static bool TryGroupedParallelInputGradient(Tensor<float> destination, float[] weights, int kernelOffset,
        float[] gradient, int gradientOffset, in GroupedConv2DGeometry geometry, bool waveform)
    {
        if (waveform) return false;
        var g = geometry;
        int k = checked(g.InChannelsPerGroup * g.KernelH * g.KernelW);
        int spatial = checked(g.OutHeight * g.OutWidth), columns = Math.Min(256, spatial);
        int panelLength = checked(k * columns), workers = GroupedPanelWorkers(g, panelLength);
        if (workers <= 1) return false;
        var target = destination.GetCpuBackingForContiguousWrite(out int targetOffset);
        if (target is null) return false;
        bool deterministic = BlasProvider.IsDeterministicMode;
        CpuParallelSettings.ParallelForOrSerial(0, workers, (long)g.Batch * g.OutChannels * k * spatial, worker =>
        {
            using var policy = new GroupedGemmPolicyScope(deterministic);
            var panel = CpuStorageAllocationDiagnostics.RentWorkspace<float>(panelLength, "workspace.grouped_conv_dx_columns");
            try
            {
                for (int group = worker; group < g.Groups; group += workers)
                    for (int image = 0; image < g.Batch; image++)
                    {
                        int w = checked(kernelOffset + group * g.OutChannelsPerGroup * k);
                        int dy = checked(gradientOffset + (image * g.OutChannels + group * g.OutChannelsPerGroup) * spatial);
                        int x = checked(targetOffset + (image * g.InChannels + group * g.InChannelsPerGroup) * g.InHeight * g.InWidth);
                        for (int from = 0; from < spatial; from += columns)
                        {
                            int count = Math.Min(columns, spatial - from);
                            BlasProvider.GemmEx(k, count, g.OutChannelsPerGroup, weights, w, k, true,
                                gradient, dy + from, spatial, false, panel, 0, count);
                            AddGroupedColumnGradient(target.AsSpan(), x, panel, from, count, g);
                        }
                    }
            }
            finally { ArrayPool<float>.Shared.Return(panel); }
        }, deterministicSafe: true);
        return true;
    }

    private static bool TryGroupedParallelKernelGradient(Tensor<float> destination, float[] source, int inputOffset,
        float[] gradient, int gradientOffset, in GroupedConv2DGeometry geometry, bool accumulate)
    {
        var g = geometry;
        int k = checked(g.InChannelsPerGroup * g.KernelH * g.KernelW);
        int spatial = checked(g.OutHeight * g.OutWidth), columns = Math.Min(256, spatial);
        int weights = checked(g.OutChannelsPerGroup * k), panelLength = checked(k * columns);
        int workers = GroupedPanelWorkers(g, panelLength, weights, accumulate);
        if (workers <= 1) return false;
        var target = destination.GetCpuBackingForContiguousWrite(out int targetOffset);
        if (target is null) return false;
        bool deterministic = BlasProvider.IsDeterministicMode;
        CpuParallelSettings.ParallelForOrSerial(0, workers, (long)g.Batch * g.OutChannels * k * spatial, worker =>
        {
            using var policy = new GroupedGemmPolicyScope(deterministic);
            var panel = CpuStorageAllocationDiagnostics.RentWorkspace<float>(panelLength, "workspace.grouped_conv_columns");
            float[]? partial = null, sum = null;
            try
            {
                partial = CpuStorageAllocationDiagnostics.RentWorkspace<float>(weights, "workspace.grouped_conv_dw_partial");
                if (accumulate) sum = CpuStorageAllocationDiagnostics.RentWorkspace<float>(weights, "workspace.grouped_conv_dw_sum");
                for (int group = worker; group < g.Groups; group += workers)
                {
                    var total = sum is null ? target.AsSpan(targetOffset + group * weights, weights) : sum.AsSpan(0, weights);
                    total.Clear();
                    for (int image = 0; image < g.Batch; image++)
                        for (int from = 0; from < spatial; from += columns)
                        {
                            int count = Math.Min(columns, spatial - from);
                            int x = checked(inputOffset + (image * g.InChannels + group * g.InChannelsPerGroup) * g.InHeight * g.InWidth);
                            int dy = checked(gradientOffset + (image * g.OutChannels + group * g.OutChannelsPerGroup) * spatial + from);
                            FillGroupedColumnPanel(source, x, panel, from, count, g);
                            BlasProvider.GemmEx(g.OutChannelsPerGroup, k, count, gradient, dy, spatial, false,
                                panel, 0, count, true, partial, 0, k);
                            AddGroupedFloat(total, partial.AsSpan(0, weights));
                        }
                    if (sum is not null) AddGroupedFloat(target.AsSpan(targetOffset + group * weights, weights), total);
                }
            }
            finally
            {
                if (sum is not null) ArrayPool<float>.Shared.Return(sum);
                if (partial is not null) ArrayPool<float>.Shared.Return(partial);
                ArrayPool<float>.Shared.Return(panel);
            }
        }, deterministicSafe: true);
        return true;
    }
}
