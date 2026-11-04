using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// The global-norm clip over every gradient in one launch per operation, the HIP port of CudaBackend's
// IMultiTensorKernels: a device table of buffer addresses plus a (tensor, start) entry per chunk (256 elements for
// the finiteness reduction, 4096 for the deterministic norm record).
public sealed partial class HipBackend : IMultiTensorKernels
{
    private const int MultiTensorChunk = 256;
    private const int MultiTensorReductionMaxBlocks = 1024;

    // The gradient set repeats every step, so the table is built once and reused while the buffers and sizes match.
    private sealed class MultiTensorTable : IDisposable
    {
        public IntPtr[] Handles = Array.Empty<IntPtr>();
        public int[] Sizes = Array.Empty<int>();
        public int TotalChunks;
        public IGpuBuffer? Pointers;
        public IGpuBuffer? SizesBuffer;
        public IGpuBuffer? ChunkTensor;
        public IGpuBuffer? ChunkStart;

        public bool Matches(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes)
        {
            if (tensors.Count != Handles.Length) return false;
            for (int t = 0; t < Handles.Length; t++)
                if (tensors[t].Handle != Handles[t] || sizes[t] != Sizes[t]) return false;
            return true;
        }

        public void Dispose()
        {
            Pointers?.Dispose();
            SizesBuffer?.Dispose();
            ChunkTensor?.Dispose();
            ChunkStart?.Dispose();
        }
    }

    private MultiTensorTable? _multiTensorTable;
    // The deterministic norm's table: 4096-element chunks (Gpu.DeterministicGradientNorm.Chunk), part of its definition.
    private MultiTensorTable? _normTable;
    private IGpuBuffer? _normPartials;

    private void DisposeMultiTensorTable()
    {
        _multiTensorTable?.Dispose();
        _multiTensorTable = null;
        _normTable?.Dispose();
        _normTable = null;
        _normPartials?.Dispose();
        _normPartials = null;
    }

    // Int and address tables travel in float buffers: the kernel reinterprets the bytes.
    private IGpuBuffer UploadInts(int[] values)
    {
        var bits = new float[values.Length];
        Buffer.BlockCopy(values, 0, bits, 0, values.Length * sizeof(int));
        return AllocateBuffer(bits);
    }

    private MultiTensorTable GetMultiTensorTable(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes)
    {
        if (_multiTensorTable is { } cached && cached.Matches(tensors, sizes)) return cached;
        _multiTensorTable?.Dispose();
        _multiTensorTable = null;
        return _multiTensorTable = BuildMultiTensorTable(tensors, sizes, MultiTensorChunk);
    }

    private MultiTensorTable GetNormTable(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes)
    {
        if (_normTable is { } cached && cached.Matches(tensors, sizes)) return cached;
        _normTable?.Dispose();
        _normTable = null;
        return _normTable = BuildMultiTensorTable(tensors, sizes, Gpu.DeterministicGradientNorm.Chunk);
    }

    private MultiTensorTable BuildMultiTensorTable(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, int chunkSize)
    {

        int n = tensors.Count;
        var handles = new IntPtr[n];
        var addresses = new ulong[n];
        var sizeArray = new int[n];
        int totalChunks = 0;
        for (int t = 0; t < n; t++)
        {
            handles[t] = tensors[t].Handle;
            addresses[t] = unchecked((ulong)handles[t].ToInt64());
            sizeArray[t] = sizes[t];
            totalChunks = checked(totalChunks + (sizes[t] + chunkSize - 1) / chunkSize);
        }
        var chunkTensor = new int[totalChunks];
        var chunkStart = new int[totalChunks];
        int c = 0;
        for (int t = 0; t < n; t++)
            for (int start = 0; start < sizeArray[t]; start += chunkSize) { chunkTensor[c] = t; chunkStart[c] = start; c++; }

        var addressBits = new float[n * 2];
        Buffer.BlockCopy(addresses, 0, addressBits, 0, n * sizeof(ulong));
        var table = new MultiTensorTable { Handles = handles, Sizes = sizeArray, TotalChunks = totalChunks };
        try
        {
            table.Pointers = AllocateBuffer(addressBits);
            table.SizesBuffer = UploadInts(sizeArray);
            table.ChunkTensor = UploadInts(chunkTensor);
            table.ChunkStart = UploadInts(chunkStart);
        }
        catch
        {
            table.Dispose();
            throw;
        }
        return table;
    }

    private IntPtr GetMultiTensorKernel(string name)
    {
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new GpuCapabilityUnavailableException($"HIP kernel not found: {name}");
        return kernel;
    }

    public unsafe void MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        MultiTensorArgs.Validate(tensors, sizes);
        Fill(sumOfSquares, 0f, 2);
        if (tensors.Count == 0) return;
        var table = GetMultiTensorTable(tensors, sizes);
        var kernel = GetMultiTensorKernel("multi_tensor_sum_squares");
        IntPtr pH = TableHandle(table.Pointers), sH = TableHandle(table.SizesBuffer);
        IntPtr ctH = TableHandle(table.ChunkTensor), csH = TableHandle(table.ChunkStart);
        IntPtr outH = sumOfSquares.Handle;
        int totalChunks = table.TotalChunks;
        void** args = stackalloc void*[6];
        args[0] = &pH; args[1] = &sH; args[2] = &ctH; args[3] = &csH; args[4] = &totalChunks; args[5] = &outH;
        LaunchKernel(kernel, (uint)Math.Min(totalChunks, MultiTensorReductionMaxBlocks), MultiTensorChunk, args);
    }

    // The deterministic norm record (Gpu.DeterministicGradientNorm); the public MultiTensorSumOfSquares above is the
    // finiteness reduction IDeviceScalarOps uses.
    unsafe void IMultiTensorKernels.MultiTensorSumOfSquares(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer record)
    {
        MultiTensorArgs.ValidateRecordBuffer(record);
        MultiTensorArgs.Validate(tensors, sizes);
        var chunkKernel = GetMultiTensorKernel("dgn_chunk_partials");
        var finish = GetMultiTensorKernel("dgn_finish");
        int total = 0;
        for (int t = 0; t < sizes.Count; t++) total += Gpu.DeterministicGradientNorm.ChunkCount(sizes[t]);
        int words = Math.Max(1, total) * Gpu.DeterministicGradientNorm.PartialWords;
        if (_normPartials is not { } partials || partials.Size < words)
        {
            _normPartials?.Dispose();
            partials = _normPartials = AllocateBuffer(Math.Max(words, 64 * Gpu.DeterministicGradientNorm.PartialWords));
        }
        IntPtr partH = partials.Handle, recH = record.Handle;
        if (total > 0)
        {
            var table = GetNormTable(tensors, sizes);
            IntPtr pH = TableHandle(table.Pointers), sH = TableHandle(table.SizesBuffer);
            IntPtr ctH = TableHandle(table.ChunkTensor), csH = TableHandle(table.ChunkStart);
            void** a = stackalloc void*[6];
            a[0] = &pH; a[1] = &sH; a[2] = &ctH; a[3] = &csH; a[4] = &total; a[5] = &partH;
            LaunchKernel(chunkKernel, (uint)total, DeterministicNormKernels.GroupSize, a);
        }
        int headroom = Gpu.DeterministicGradientNorm.Headroom(total);
        void** f = stackalloc void*[4];
        f[0] = &partH; f[1] = &total; f[2] = &headroom; f[3] = &recH;
        LaunchKernel(finish, 1, DeterministicNormKernels.GroupSize, f);
    }

    public unsafe void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
        => ((IMultiTensorKernels)this).ClipScaleFromRecord(sumOfSquares, maxNorm, scale, Gpu.NonFiniteGradientPolicy.LeaveUnscaled);

    unsafe void IMultiTensorKernels.ClipScaleFromRecord(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale,
        Gpu.NonFiniteGradientPolicy policy)
    {
        MultiTensorArgs.ValidateRecordBuffer(sumOfSquares);
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        var kernel = GetMultiTensorKernel("dgn_clip_scale");
        IntPtr sH = sumOfSquares.Handle, cH = scale.Handle;
        uint bits = Gpu.DeterministicGradientNorm.SingleToUInt32Bits(maxNorm), pol = (uint)policy;
        void** args = stackalloc void*[4];
        args[0] = &sH; args[1] = &bits; args[2] = &pol; args[3] = &cH;
        LaunchKernel(kernel, 1, 1, args);
    }

    public unsafe void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        MultiTensorArgs.Validate(tensors, sizes);
        if (tensors.Count == 0) return;
        // dgn_mul: the IEEE product with subnormals kept, so the result matches every other backend.
        var table = GetNormTable(tensors, sizes);
        var kernel = GetMultiTensorKernel("dgn_multi_tensor_scale");
        IntPtr pH = TableHandle(table.Pointers), sH = TableHandle(table.SizesBuffer);
        IntPtr ctH = TableHandle(table.ChunkTensor), csH = TableHandle(table.ChunkStart);
        IntPtr scH = scale.Handle;
        int totalChunks = table.TotalChunks;
        void** args = stackalloc void*[6];
        args[0] = &pH; args[1] = &sH; args[2] = &ctH; args[3] = &csH; args[4] = &totalChunks; args[5] = &scH;
        LaunchKernel(kernel, (uint)Math.Min(totalChunks, 4096), DeterministicNormKernels.GroupSize, args);
    }

    private static IntPtr TableHandle(IGpuBuffer? buffer) =>
        buffer?.Handle ?? throw new InvalidOperationException("The multi-tensor table was not uploaded.");

    /// <inheritdoc />
    double IMultiTensorKernels.ReadSumOfSquares(IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        return MultiTensorArgs.DoubleFromSlots(DownloadBuffer(sumOfSquares));
    }
}