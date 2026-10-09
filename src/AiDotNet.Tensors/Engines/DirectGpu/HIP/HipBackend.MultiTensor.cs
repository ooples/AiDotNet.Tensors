using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// The global-norm clip over every gradient in one launch per operation, the HIP port of CudaBackend's
// IMultiTensorKernels: a device table of buffer addresses plus a (tensor, start) entry per 256-element chunk.
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

    private void DisposeMultiTensorTable()
    {
        _multiTensorTable?.Dispose();
        _multiTensorTable = null;
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
            totalChunks = checked(totalChunks + (sizes[t] + MultiTensorChunk - 1) / MultiTensorChunk);
        }
        var chunkTensor = new int[totalChunks];
        var chunkStart = new int[totalChunks];
        int c = 0;
        for (int t = 0; t < n; t++)
            for (int start = 0; start < sizeArray[t]; start += MultiTensorChunk) { chunkTensor[c] = t; chunkStart[c] = start; c++; }

        var addressBits = new float[n * 2];
        Buffer.BlockCopy(addresses, 0, addressBits, 0, n * sizeof(ulong));
        DisposeMultiTensorTable();
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
        _multiTensorTable = table;
        return table;
    }

    private IntPtr GetMultiTensorKernel(string name)
    {
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {name}");
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

    public unsafe void ClipScaleFromSumOfSquares(IGpuBuffer sumOfSquares, float maxNorm, IGpuBuffer scale)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        var kernel = GetMultiTensorKernel("clip_scale_from_sum_squares");
        IntPtr sH = sumOfSquares.Handle, cH = scale.Handle;
        void** args = stackalloc void*[3];
        args[0] = &sH; args[1] = &maxNorm; args[2] = &cH;
        LaunchKernel(kernel, 1, 1, args);
    }

    public unsafe void MultiTensorScaleByDeviceScalar(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes, IGpuBuffer scale)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        MultiTensorArgs.Validate(tensors, sizes);
        if (tensors.Count == 0) return;
        var table = GetMultiTensorTable(tensors, sizes);
        var kernel = GetMultiTensorKernel("multi_tensor_scale_by_device_scalar");
        IntPtr pH = TableHandle(table.Pointers), sH = TableHandle(table.SizesBuffer);
        IntPtr ctH = TableHandle(table.ChunkTensor), csH = TableHandle(table.ChunkStart);
        IntPtr scH = scale.Handle;
        void** args = stackalloc void*[5];
        args[0] = &pH; args[1] = &sH; args[2] = &ctH; args[3] = &csH; args[4] = &scH;
        LaunchKernel(kernel, (uint)table.TotalChunks, MultiTensorChunk, args);
    }

    private static IntPtr TableHandle(IGpuBuffer? buffer) =>
        buffer?.Handle ?? throw new InvalidOperationException("The multi-tensor table was not uploaded.");
}