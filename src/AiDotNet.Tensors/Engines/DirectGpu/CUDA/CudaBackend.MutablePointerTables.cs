using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    internal long MutableAdamTableRefreshes { get; private set; }
    internal long MutableGradientTableRefreshes { get; private set; }

    private static bool SameTableGeometry(MultiTensorTable table, IReadOnlyList<int> sizes, int lists)
    {
        if (table.Sizes.Length != sizes.Count || table.Handles.Length != lists * sizes.Count) return false;
        for (int i = 0; i < sizes.Count; i++) if (table.Sizes[i] != sizes[i]) return false;
        return true;
    }

    private MultiTensorTable GetMutableAdamTable(long owner, IReadOnlyList<IGpuBuffer> parameters,
        IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<IGpuBuffer> firstMoments,
        IReadOnlyList<IGpuBuffer> secondMoments, IReadOnlyList<int> sizes)
    {
        for (int i = 0; i < _retainedAdamTables.Count; i++)
        {
            var table = _retainedAdamTables[i];
            if (table.MutableOwner != owner || !SameTableGeometry(table, sizes, 4)) continue;
            if (RefreshMutableTable(table, parameters, gradients, firstMoments, secondMoments)) MutableAdamTableRefreshes++;
            _retainedAdamTables.RemoveAt(i);
            _retainedAdamTables.Add(table);
            return _adamTable = table;
        }
        var built = BuildAdamTable(parameters, gradients, firstMoments, secondMoments, sizes, retainAddresses: true);
        built.MutableOwner = owner;
        RetainMultiTensorTable(_retainedAdamTables, built); // same eight-entry limit, including immutable tables
        return _adamTable = built;
    }

    private MultiTensorTable GetMutableGradientTable(long owner, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<int> sizes)
    {
        for (int i = 0; i < _retainedGradientTables.Count; i++)
        {
            var table = _retainedGradientTables[i];
            if (table.MutableOwner != owner || !SameTableGeometry(table, sizes, 1)) continue;
            if (RefreshMutableTable(table, gradients, null, null, null)) MutableGradientTableRefreshes++;
            _retainedGradientTables.RemoveAt(i);
            _retainedGradientTables.Add(table);
            return _gradientTable = table;
        }
        var built = BuildGradientTable(gradients, sizes, retainAddresses: true);
        built.MutableOwner = owner;
        RetainMultiTensorTable(_retainedGradientTables, built);
        return _gradientTable = built;
    }

    private unsafe bool RefreshMutableTable(MultiTensorTable table, IReadOnlyList<IGpuBuffer> first,
        IReadOnlyList<IGpuBuffer>? second, IReadOnlyList<IGpuBuffer>? third, IReadOnlyList<IGpuBuffer>? fourth)
    {
        int count = first.Count;
        bool changed = table.PointerUploadPending;
        for (int i = 0; i < table.Handles.Length; i++)
        {
            IntPtr address = i < count ? first[i].Handle : i < 2 * count ? second![i - count].Handle
                : i < 3 * count ? third![i - 2 * count].Handle : fourth![i - 3 * count].Handle;
            if (table.Handles[i] == address) continue;
            table.PointerUploadPending = true;
            table.Handles[i] = address;
            WriteAddress(table.StagingAddresses, i, address);
            changed = true;
        }
        if (!changed) return false;
        // Enqueue behind previous readers on this same stream. Keep the managed staging pinned until the copy
        // completes; the wait is part of ordinary execution and is never hidden from timing. No device allocation.
        using var context = PushContext();
        fixed (byte* source = table.StagingAddresses)
        {
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuMemcpyHtoDAsync(table.Pointers!.Handle,
                (IntPtr)source, (ulong)table.StagingAddresses.Length, _stream), "cuMemcpyHtoDAsync(mutable table)");
            CuBlasNative.CheckCudaResult(CudaNativeBindings.cuStreamSynchronize(_stream), "cuStreamSynchronize(mutable table)");
        }
        // A failed copy or stream wait must force a full upload on the next call, even for the same handles.
        table.PointerUploadPending = false;
        return true;
    }
}
