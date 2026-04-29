// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Distributed.Native;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// NCCL-backed <see cref="IProcessGroup"/>. Bridges NVIDIA's collective-
/// communication library to the standard <see cref="IProcessGroup"/>
/// interface so DDP / FSDP / DTensor / pipeline / RPC implementations
/// don't change between in-process / TCP / NCCL backends.
///
/// <para>NCCL is the GPU-side counterpart of MPI / Gloo: it implements
/// every collective directly on top of NVLink / NVSwitch / InfiniBand,
/// hugely outperforming TCP for cross-machine GPU training. NCCL
/// requires a CUDA context per rank and its collectives operate on
/// CUDA device pointers — the wrapper allocates a temporary device
/// buffer per call (<c>cuMemAlloc</c>), uploads the host tensor, runs
/// the NCCL collective, downloads back, and frees. PyTorch's
/// <c>ProcessGroupNCCL</c> takes the same approach for tensors that
/// don't already live on-device.</para>
///
/// <para><b>Bring-up:</b> rank 0 calls <see cref="GetUniqueId"/> and
/// broadcasts the resulting blob to every other rank via an external
/// channel (file rendezvous, TCP rendezvous, etc.). Each rank then
/// passes the blob to the constructor with its <c>rank</c> /
/// <c>worldSize</c> and the wrapper completes
/// <c>ncclCommInitRank</c>.</para>
/// </summary>
public sealed class NcclProcessGroup : IProcessGroup
{
    /// <inheritdoc/>
    public int WorldSize { get; }
    /// <inheritdoc/>
    public int Rank { get; }
    /// <inheritdoc/>
    public string Backend => "nccl";

    private IntPtr _comm;
    private readonly object _lock = new();

    /// <summary>True when libnccl is loadable on this host.</summary>
    public static bool IsAvailable => NcclNative.IsAvailable;

    /// <summary>Returns the 128-byte NCCL unique-id blob produced by
    /// rank 0. Rank 0 broadcasts this to every other rank via an
    /// external channel; each rank passes it to the constructor.</summary>
    public static byte[] GetUniqueId()
    {
        if (!NcclNative.IsAvailable)
            throw new NotSupportedException(
                "libnccl is not loadable on this host. NCCL ships only for Linux + NVIDIA GPU hosts; " +
                "use the TCP / in-process backend on other platforms.");
        var status = NcclNative.ncclGetUniqueId(out var id);
        if (status != NcclNative.Result.Success)
            throw new InvalidOperationException($"ncclGetUniqueId returned {status}.");
        var blob = new byte[128];
        unsafe { Marshal.Copy((IntPtr)id.Data, blob, 0, 128); }
        return blob;
    }

    /// <summary>Constructs a NCCL process group on this rank. Caller is
    /// responsible for broadcasting the unique-id blob from rank 0 to
    /// every rank before calling.</summary>
    public NcclProcessGroup(int rank, int worldSize, byte[] uniqueIdBlob)
    {
        if (!NcclNative.IsAvailable)
            throw new NotSupportedException(
                "libnccl is not loadable on this host. NCCL ships only for Linux + NVIDIA GPU hosts.");
        if (uniqueIdBlob is null || uniqueIdBlob.Length != 128)
            throw new ArgumentException("NCCL unique id must be a 128-byte blob.", nameof(uniqueIdBlob));
        Rank = rank;
        WorldSize = worldSize;

        var id = default(NcclNative.UniqueId);
        unsafe { Marshal.Copy(uniqueIdBlob, 0, (IntPtr)id.Data, 128); }
        var status = NcclNative.ncclCommInitRank(out _comm, worldSize, id, rank);
        if (status != NcclNative.Result.Success)
            throw new InvalidOperationException($"ncclCommInitRank(rank={rank}, world={worldSize}) returned {status}.");
    }

    private static NcclNative.RedOp Map(ReduceOp op) => op switch
    {
        ReduceOp.Sum => NcclNative.RedOp.Sum,
        ReduceOp.Avg => NcclNative.RedOp.Avg,
        ReduceOp.Min => NcclNative.RedOp.Min,
        ReduceOp.Max => NcclNative.RedOp.Max,
        ReduceOp.Product => NcclNative.RedOp.Prod,
        _ => throw new NotSupportedException($"NCCL doesn't support reduce-op {op} (bitwise ops are integer-only and use a different surface).")
    };

    private static NcclNative.DataType MapDtype<T>()
    {
        if (typeof(T) == typeof(float)) return NcclNative.DataType.Float32;
        if (typeof(T) == typeof(double)) return NcclNative.DataType.Float64;
        if (typeof(T) == typeof(int)) return NcclNative.DataType.Int32;
        if (typeof(T) == typeof(long)) return NcclNative.DataType.Int64;
        if (typeof(T) == typeof(byte)) return NcclNative.DataType.Uint8;
        throw new NotSupportedException($"NCCL doesn't have a datatype mapping for {typeof(T).Name}.");
    }

    private static int ElemSize<T>() => Marshal.SizeOf<T>();

    /// <summary>Allocates a temp device buffer, uploads the host tensor,
    /// runs the supplied collective lambda with the device pointer, and
    /// downloads the result back to the host tensor. Frees the buffer
    /// before returning.</summary>
    private static void RunOnDevice<T>(Tensor<T> tensor, Func<IntPtr, NcclNative.Result> kernel)
    {
        ulong bytes = (ulong)tensor.Length * (ulong)ElemSize<T>();
        IntPtr dev = IntPtr.Zero;
        try
        {
            if (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemAlloc(out dev, bytes) != CudaResult.Success)
                throw new InvalidOperationException("cuMemAlloc failed.");
            var hostHandle = GCHandle.Alloc(GetBackingArray(tensor), GCHandleType.Pinned);
            try
            {
                if (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemcpyHtoD(dev, hostHandle.AddrOfPinnedObject(), bytes) != CudaResult.Success)
                    throw new InvalidOperationException("cuMemcpyHtoD failed.");
            }
            finally { hostHandle.Free(); }

            var status = kernel(dev);
            if (status != NcclNative.Result.Success)
                throw new InvalidOperationException($"NCCL collective returned {status}.");

            hostHandle = GCHandle.Alloc(GetBackingArray(tensor), GCHandleType.Pinned);
            try
            {
                if (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemcpyDtoH(hostHandle.AddrOfPinnedObject(), dev, bytes) != CudaResult.Success)
                    throw new InvalidOperationException("cuMemcpyDtoH failed.");
            }
            finally { hostHandle.Free(); }
        }
        finally
        {
            if (dev != IntPtr.Zero) AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemFree(dev);
        }
    }

    private static T[] GetBackingArray<T>(Tensor<T> t)
    {
        // Tensor<T> exposes its backing via AsSpan / AsWritableSpan; the wrapper here
        // is permitted to copy through a managed array, which keeps the pin scope tight.
        var arr = new T[t.Length];
        t.AsSpan().CopyTo(arr);
        return arr;
    }

    /// <inheritdoc/>
    public void AllReduce<T>(Tensor<T> tensor, ReduceOp op = ReduceOp.Sum)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        var dtype = MapDtype<T>();
        var ncclOp = Map(op);
        // NCCL allows in-place: same buffer for sendBuff and recvBuff.
        RunOnDevice(tensor, dev =>
            NcclNative.ncclAllReduce(dev, dev, (ulong)tensor.Length, dtype, ncclOp, _comm, IntPtr.Zero));
    }

    /// <inheritdoc/>
    public void Broadcast<T>(Tensor<T> tensor, int root)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        var dtype = MapDtype<T>();
        RunOnDevice(tensor, dev =>
            NcclNative.ncclBroadcast(dev, dev, (ulong)tensor.Length, dtype, root, _comm, IntPtr.Zero));
    }

    /// <inheritdoc/>
    public void Reduce<T>(Tensor<T> tensor, int root, ReduceOp op = ReduceOp.Sum)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        var dtype = MapDtype<T>();
        var ncclOp = Map(op);
        RunOnDevice(tensor, dev =>
            NcclNative.ncclReduce(dev, dev, (ulong)tensor.Length, dtype, ncclOp, root, _comm, IntPtr.Zero));
    }

    /// <inheritdoc/>
    public void AllGather<T>(Tensor<T> input, IList<Tensor<T>> output)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null || output.Count != WorldSize)
            throw new ArgumentException($"output must have length {WorldSize}.", nameof(output));
        var dtype = MapDtype<T>();
        ulong sendBytes = (ulong)input.Length * (ulong)ElemSize<T>();
        ulong recvBytes = sendBytes * (ulong)WorldSize;

        IntPtr sendDev = IntPtr.Zero, recvDev = IntPtr.Zero;
        try
        {
            if (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemAlloc(out sendDev, sendBytes) != CudaResult.Success)
                throw new InvalidOperationException("cuMemAlloc(send) failed.");
            if (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemAlloc(out recvDev, recvBytes) != CudaResult.Success)
                throw new InvalidOperationException("cuMemAlloc(recv) failed.");

            var sendArr = GetBackingArray(input);
            var sendHandle = GCHandle.Alloc(sendArr, GCHandleType.Pinned);
            try
            {
                AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemcpyHtoD(sendDev, sendHandle.AddrOfPinnedObject(), sendBytes);
            }
            finally { sendHandle.Free(); }

            var status = NcclNative.ncclAllGather(sendDev, recvDev, (ulong)input.Length, dtype, _comm, IntPtr.Zero);
            if (status != NcclNative.Result.Success)
                throw new InvalidOperationException($"ncclAllGather returned {status}.");

            // Slice recvDev into per-rank chunks back to the host.
            for (int r = 0; r < WorldSize; r++)
            {
                var chunk = new T[input.Length];
                var ch = GCHandle.Alloc(chunk, GCHandleType.Pinned);
                try
                {
                    var src = (IntPtr)((long)recvDev + r * (long)sendBytes);
                    AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemcpyDtoH(ch.AddrOfPinnedObject(), src, sendBytes);
                }
                finally { ch.Free(); }
                chunk.AsSpan().CopyTo(output[r].AsWritableSpan());
            }
        }
        finally
        {
            if (sendDev != IntPtr.Zero) AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemFree(sendDev);
            if (recvDev != IntPtr.Zero) AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemFree(recvDev);
        }
    }

    /// <inheritdoc/>
    public void ReduceScatter<T>(IList<Tensor<T>> input, Tensor<T> output, ReduceOp op = ReduceOp.Sum)
    {
        if (input is null || input.Count != WorldSize) throw new ArgumentException("input must have length = WorldSize.", nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        var dtype = MapDtype<T>();
        var ncclOp = Map(op);

        // Concatenate per-rank inputs into one buffer for ncclReduceScatter.
        ulong recvBytes = (ulong)output.Length * (ulong)ElemSize<T>();
        ulong sendBytes = recvBytes * (ulong)WorldSize;
        IntPtr sendDev = IntPtr.Zero, recvDev = IntPtr.Zero;
        try
        {
            if (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemAlloc(out sendDev, sendBytes) != CudaResult.Success) throw new InvalidOperationException("cuMemAlloc(send) failed.");
            if (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemAlloc(out recvDev, recvBytes) != CudaResult.Success) throw new InvalidOperationException("cuMemAlloc(recv) failed.");

            for (int r = 0; r < WorldSize; r++)
            {
                var arr = GetBackingArray(input[r]);
                var h = GCHandle.Alloc(arr, GCHandleType.Pinned);
                try
                {
                    var dst = (IntPtr)((long)sendDev + r * (long)recvBytes);
                    AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemcpyHtoD(dst, h.AddrOfPinnedObject(), recvBytes);
                }
                finally { h.Free(); }
            }

            var status = NcclNative.ncclReduceScatter(sendDev, recvDev, (ulong)output.Length, dtype, ncclOp, _comm, IntPtr.Zero);
            if (status != NcclNative.Result.Success) throw new InvalidOperationException($"ncclReduceScatter returned {status}.");

            var outArr = new T[output.Length];
            var oh = GCHandle.Alloc(outArr, GCHandleType.Pinned);
            try { AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemcpyDtoH(oh.AddrOfPinnedObject(), recvDev, recvBytes); }
            finally { oh.Free(); }
            outArr.AsSpan().CopyTo(output.AsWritableSpan());
        }
        finally
        {
            if (sendDev != IntPtr.Zero) AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemFree(sendDev);
            if (recvDev != IntPtr.Zero) AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaNativeBindings.cuMemFree(recvDev);
        }
    }

    /// <inheritdoc/>
    public void Send<T>(Tensor<T> tensor, int dst, int tag = 0)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        var dtype = MapDtype<T>();
        // NCCL Send/Recv are tag-less; tag is documented for caller-pairing
        // with the matching Recv but doesn't reach the wire. Standard
        // PyTorch ProcessGroupNCCL behaviour.
        _ = tag;
        RunOnDevice(tensor, dev =>
            NcclNative.ncclSend(dev, (ulong)tensor.Length, dtype, dst, _comm, IntPtr.Zero));
    }

    /// <inheritdoc/>
    public void Recv<T>(Tensor<T> tensor, int src, int tag = 0)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        var dtype = MapDtype<T>();
        _ = tag;
        RunOnDevice(tensor, dev =>
            NcclNative.ncclRecv(dev, (ulong)tensor.Length, dtype, src, _comm, IntPtr.Zero));
    }

    /// <inheritdoc/>
    public void Barrier()
    {
        // NCCL doesn't have a dedicated Barrier — emulate via a one-element AllReduce.
        var t = new Tensor<float>(new[] { 1 });
        AllReduce(t, ReduceOp.Sum);
    }

    /// <inheritdoc/>
    public Tensor<T> RecvDiscoverShape<T>(int src, int tag = 0) =>
        throw new NotSupportedException("NCCL doesn't support shape-discovering recv. Use RecvDiscoverShape on the in-process / TCP backend.");

    /// <inheritdoc/>
    public void Gather<T>(Tensor<T> input, IList<Tensor<T>>? output, int root) =>
        throw new NotSupportedException("NCCL Gather is composable as AllGather + drop on non-root; use AllGather instead.");

    /// <inheritdoc/>
    public void Scatter<T>(IList<Tensor<T>>? input, Tensor<T> output, int root) =>
        throw new NotSupportedException("NCCL Scatter has a separate API; emulate via Broadcast + slice for now.");

    /// <inheritdoc/>
    public IProcessGroup? NewGroup(IReadOnlyList<int> ranks) =>
        throw new NotSupportedException("NCCL sub-group construction (ncclCommSplit) is a follow-up; create separate NcclProcessGroup instances with disjoint world_size meanwhile.");

    /// <inheritdoc/>
    public IProcessGroup SplitGroup(int color, int? key = null) =>
        throw new NotSupportedException("NCCL sub-group construction is a follow-up — see NewGroup.");

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            if (_comm != IntPtr.Zero)
            {
                NcclNative.ncclCommDestroy(_comm);
                _comm = IntPtr.Zero;
            }
        }
    }
}
