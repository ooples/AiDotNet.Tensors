// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Distributed.Native;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// MPI-backed <see cref="IProcessGroup"/>. Bridges Open MPI / MPICH to
/// the standard <see cref="IProcessGroup"/> contract. Used for cross-
/// machine CPU-tensor training on supercomputer-class systems where
/// MPI is the canonical transport.
///
/// <para><b>Bring-up:</b> the constructor calls <c>MPI_Init</c> on first
/// instantiation (process-wide; subsequent constructors increment a
/// refcount), then queries <c>MPI_COMM_WORLD</c>'s rank/size. Every
/// rank must construct the group with the same <c>worldSize</c> as the
/// one <c>mpirun</c> launched. The <see cref="Rank"/> is read from
/// <c>MPI_Comm_rank</c> rather than passed by the caller.</para>
///
/// <para><b>Predefined-type constants.</b> Open MPI exports its
/// <c>MPI_FLOAT</c> etc. as Fortran-named globals (e.g. on Linux
/// <c>ompi_mpi_float</c>); MPICH exports them as small integer
/// constants. We resolve via <see cref="NativeLibrary.GetExport"/> at
/// construction with both naming conventions so the wrapper handles
/// either implementation.</para>
/// </summary>
public sealed class MpiProcessGroup : IProcessGroup
{
    /// <inheritdoc/>
    public int WorldSize { get; }
    /// <inheritdoc/>
    public int Rank { get; }
    /// <inheritdoc/>
    public string Backend => "mpi";

    private readonly IntPtr _commWorld;
    private readonly IntPtr _floatType;
    private readonly IntPtr _doubleType;
    private readonly IntPtr _intType;
    private readonly IntPtr _opSum;
    private readonly IntPtr _opMin;
    private readonly IntPtr _opMax;
    private readonly IntPtr _opProd;

    private static int _initRefcount;
    private static readonly object _initLock = new();
    private bool _disposed;

    /// <summary>True when libmpi (or msmpi) is loadable.</summary>
    public static bool IsAvailable => MpiNative.IsAvailable;

    /// <summary>Constructs an MPI process group bound to MPI_COMM_WORLD.</summary>
    public MpiProcessGroup()
    {
        if (!MpiNative.IsAvailable)
            throw new NotSupportedException("MPI is not loadable on this host. Install Open MPI / MPICH / MS-MPI and ensure libmpi is on the dynamic loader path.");

        lock (_initLock)
        {
            if (_initRefcount == 0)
            {
                int argc = 0; IntPtr argv = IntPtr.Zero;
                int rc = MpiNative.MPI_Init(ref argc, ref argv);
                if (rc != MpiNative.MPI_SUCCESS)
                    throw new InvalidOperationException($"MPI_Init returned {rc}.");
            }
            _initRefcount++;
        }

#if NET5_0_OR_GREATER
        // Resolve MPI_COMM_WORLD and the predefined op / type constants by symbol.
        // - Open MPI exports them as Fortran-named globals (ompi_mpi_*).
        // - MPICH defines them as small-integer macros (#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)),
        //   so they aren't visible as symbols at all. For MPICH we fall back to MPICH's documented
        //   integer-handle constants and pass them as IntPtrs.
        if (!NativeLibrary.TryLoad(GetMpiLibName(), typeof(MpiNative).Assembly, null, out var lib))
            throw new InvalidOperationException("Failed to resolve MPI library handle for symbol lookup.");
        try
        {
            // Open MPI symbol path (named globals).
            if (NativeLibrary.TryGetExport(lib, "ompi_mpi_comm_world", out _))
            {
                _commWorld = NativeLibrary.GetExport(lib, "ompi_mpi_comm_world");
                _floatType = NativeLibrary.GetExport(lib, "ompi_mpi_float");
                _doubleType = NativeLibrary.GetExport(lib, "ompi_mpi_double");
                _intType = NativeLibrary.GetExport(lib, "ompi_mpi_int");
                _opSum = NativeLibrary.GetExport(lib, "ompi_mpi_op_sum");
                _opMin = NativeLibrary.GetExport(lib, "ompi_mpi_op_min");
                _opMax = NativeLibrary.GetExport(lib, "ompi_mpi_op_max");
                _opProd = NativeLibrary.GetExport(lib, "ompi_mpi_op_prod");
            }
            else
            {
                // MPICH / MS-MPI integer-handle constants (mpi.h #defines reproduced here).
                _commWorld = (IntPtr)0x44000000;
                _floatType = (IntPtr)0x4c00040a;
                _doubleType = (IntPtr)0x4c00080b;
                _intType = (IntPtr)0x4c000405;
                _opSum = (IntPtr)0x58000003;
                _opMin = (IntPtr)0x58000002;
                _opMax = (IntPtr)0x58000001;
                _opProd = (IntPtr)0x58000004;
            }
        }
        finally
        {
            NativeLibrary.Free(lib);
        }
#else
        throw new NotSupportedException("MpiProcessGroup requires .NET 5+ for NativeLibrary.GetExport.");
#endif

        if (MpiNative.MPI_Comm_size(_commWorld, out int worldSize) != MpiNative.MPI_SUCCESS)
            throw new InvalidOperationException("MPI_Comm_size failed.");
        if (MpiNative.MPI_Comm_rank(_commWorld, out int rank) != MpiNative.MPI_SUCCESS)
            throw new InvalidOperationException("MPI_Comm_rank failed.");
        WorldSize = worldSize;
        Rank = rank;
    }

#if NET5_0_OR_GREATER
    private static IntPtr ResolveSymbol(IntPtr lib, params string[] names)
    {
        foreach (var n in names)
        {
            if (NativeLibrary.TryGetExport(lib, n, out var addr))
                return addr;
        }
        throw new InvalidOperationException($"Could not resolve any of the MPI symbol names: {string.Join(", ", names)}.");
    }
#endif

    private static string GetMpiLibName()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return "libmpi.so.40";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return "libmpi.dylib";
        return "msmpi.dll";
    }

    private IntPtr Type<T>()
    {
        if (typeof(T) == typeof(float)) return _floatType;
        if (typeof(T) == typeof(double)) return _doubleType;
        if (typeof(T) == typeof(int)) return _intType;
        throw new NotSupportedException($"MPI binding does not yet expose a predefined type for {typeof(T).Name}.");
    }

    private IntPtr Op(ReduceOp op) => op switch
    {
        ReduceOp.Sum => _opSum,
        ReduceOp.Avg => _opSum,
        ReduceOp.Min => _opMin,
        ReduceOp.Max => _opMax,
        ReduceOp.Product => _opProd,
        _ => throw new NotSupportedException($"MPI binding does not yet expose an op for {op}."),
    };

    private static int ElemSize<T>() => Marshal.SizeOf<T>();

    /// <inheritdoc/>
    public void AllReduce<T>(Tensor<T> tensor, ReduceOp op = ReduceOp.Sum)
    {
        var arr = new T[tensor.Length];
        tensor.AsSpan().CopyTo(arr);
        var send = GCHandle.Alloc(arr, GCHandleType.Pinned);
        var recv = GCHandle.Alloc(new T[tensor.Length], GCHandleType.Pinned);
        try
        {
            int rc = MpiNative.MPI_Allreduce(send.AddrOfPinnedObject(), recv.AddrOfPinnedObject(),
                tensor.Length, Type<T>(), Op(op), _commWorld);
            if (rc != MpiNative.MPI_SUCCESS) throw new InvalidOperationException($"MPI_Allreduce returned {rc}.");
            var result = (T[])recv.Target!;
            if (op == ReduceOp.Avg)
            {
                var ops = MathHelper.GetNumericOperations<T>();
                var div = ops.FromDouble(1.0 / WorldSize);
                for (int i = 0; i < result.Length; i++) result[i] = ops.Multiply(result[i], div);
            }
            result.AsSpan().CopyTo(tensor.AsWritableSpan());
        }
        finally { send.Free(); recv.Free(); }
    }

    /// <inheritdoc/>
    public void Broadcast<T>(Tensor<T> tensor, int root)
    {
        var arr = new T[tensor.Length];
        if (Rank == root) tensor.AsSpan().CopyTo(arr);
        var h = GCHandle.Alloc(arr, GCHandleType.Pinned);
        try
        {
            int rc = MpiNative.MPI_Bcast(h.AddrOfPinnedObject(), tensor.Length, Type<T>(), root, _commWorld);
            if (rc != MpiNative.MPI_SUCCESS) throw new InvalidOperationException($"MPI_Bcast returned {rc}.");
            arr.AsSpan().CopyTo(tensor.AsWritableSpan());
        }
        finally { h.Free(); }
    }

    /// <inheritdoc/>
    public void Reduce<T>(Tensor<T> tensor, int root, ReduceOp op = ReduceOp.Sum)
    {
        var arr = new T[tensor.Length];
        tensor.AsSpan().CopyTo(arr);
        var send = GCHandle.Alloc(arr, GCHandleType.Pinned);
        var recv = GCHandle.Alloc(new T[tensor.Length], GCHandleType.Pinned);
        try
        {
            int rc = MpiNative.MPI_Reduce(send.AddrOfPinnedObject(), recv.AddrOfPinnedObject(),
                tensor.Length, Type<T>(), Op(op), root, _commWorld);
            if (rc != MpiNative.MPI_SUCCESS) throw new InvalidOperationException($"MPI_Reduce returned {rc}.");
            if (Rank == root)
            {
                var result = (T[])recv.Target!;
                if (op == ReduceOp.Avg)
                {
                    var ops = MathHelper.GetNumericOperations<T>();
                    var div = ops.FromDouble(1.0 / WorldSize);
                    for (int i = 0; i < result.Length; i++) result[i] = ops.Multiply(result[i], div);
                }
                result.AsSpan().CopyTo(tensor.AsWritableSpan());
            }
        }
        finally { send.Free(); recv.Free(); }
    }

    /// <inheritdoc/>
    public void AllGather<T>(Tensor<T> input, IList<Tensor<T>> output)
    {
        if (output.Count != WorldSize) throw new ArgumentException($"output must have length {WorldSize}.", nameof(output));
        var sendArr = new T[input.Length];
        input.AsSpan().CopyTo(sendArr);
        var recvArr = new T[input.Length * WorldSize];
        var send = GCHandle.Alloc(sendArr, GCHandleType.Pinned);
        var recv = GCHandle.Alloc(recvArr, GCHandleType.Pinned);
        try
        {
            int rc = MpiNative.MPI_Allgather(send.AddrOfPinnedObject(), input.Length, Type<T>(),
                recv.AddrOfPinnedObject(), input.Length, Type<T>(), _commWorld);
            if (rc != MpiNative.MPI_SUCCESS) throw new InvalidOperationException($"MPI_Allgather returned {rc}.");
            for (int r = 0; r < WorldSize; r++)
                recvArr.AsSpan(r * input.Length, input.Length).CopyTo(output[r].AsWritableSpan());
        }
        finally { send.Free(); recv.Free(); }
    }

    /// <inheritdoc/>
    public void ReduceScatter<T>(IList<Tensor<T>> input, Tensor<T> output, ReduceOp op = ReduceOp.Sum)
    {
        if (input.Count != WorldSize) throw new ArgumentException($"input must have length {WorldSize}.", nameof(input));
        var combined = new T[output.Length * WorldSize];
        for (int r = 0; r < WorldSize; r++)
            input[r].AsSpan().CopyTo(combined.AsSpan(r * output.Length, output.Length));
        var recv = new T[output.Length];
        var sh = GCHandle.Alloc(combined, GCHandleType.Pinned);
        var rh = GCHandle.Alloc(recv, GCHandleType.Pinned);
        try
        {
            int rc = MpiNative.MPI_Reduce_scatter_block(sh.AddrOfPinnedObject(), rh.AddrOfPinnedObject(),
                output.Length, Type<T>(), Op(op), _commWorld);
            if (rc != MpiNative.MPI_SUCCESS) throw new InvalidOperationException($"MPI_Reduce_scatter_block returned {rc}.");
            if (op == ReduceOp.Avg)
            {
                var ops = MathHelper.GetNumericOperations<T>();
                var div = ops.FromDouble(1.0 / WorldSize);
                for (int i = 0; i < recv.Length; i++) recv[i] = ops.Multiply(recv[i], div);
            }
            recv.AsSpan().CopyTo(output.AsWritableSpan());
        }
        finally { sh.Free(); rh.Free(); }
    }

    /// <inheritdoc/>
    public void Send<T>(Tensor<T> tensor, int dst, int tag = 0)
    {
        var arr = new T[tensor.Length];
        tensor.AsSpan().CopyTo(arr);
        var h = GCHandle.Alloc(arr, GCHandleType.Pinned);
        try
        {
            int rc = MpiNative.MPI_Send(h.AddrOfPinnedObject(), tensor.Length, Type<T>(), dst, tag, _commWorld);
            if (rc != MpiNative.MPI_SUCCESS) throw new InvalidOperationException($"MPI_Send returned {rc}.");
        }
        finally { h.Free(); }
    }

    /// <inheritdoc/>
    public void Recv<T>(Tensor<T> tensor, int src, int tag = 0)
    {
        var arr = new T[tensor.Length];
        var h = GCHandle.Alloc(arr, GCHandleType.Pinned);
        try
        {
            int rc = MpiNative.MPI_Recv(h.AddrOfPinnedObject(), tensor.Length, Type<T>(), src, tag, _commWorld, IntPtr.Zero);
            if (rc != MpiNative.MPI_SUCCESS) throw new InvalidOperationException($"MPI_Recv returned {rc}.");
            arr.AsSpan().CopyTo(tensor.AsWritableSpan());
        }
        finally { h.Free(); }
    }

    /// <inheritdoc/>
    public void Barrier()
    {
        int rc = MpiNative.MPI_Barrier(_commWorld);
        if (rc != MpiNative.MPI_SUCCESS) throw new InvalidOperationException($"MPI_Barrier returned {rc}.");
    }

    /// <inheritdoc/>
    public Tensor<T> RecvDiscoverShape<T>(int src, int tag = 0) =>
        throw new NotSupportedException("MPI doesn't have a single-call shape-discovering recv. Use Probe + Get_count + Recv pattern, or InProcessGroup.");

    /// <inheritdoc/>
    public void Gather<T>(Tensor<T> input, IList<Tensor<T>>? output, int root) =>
        throw new NotSupportedException("MpiProcessGroup.Gather not yet wired (uses MPI_Gather). Use AllGather + drop on non-root meanwhile.");

    /// <inheritdoc/>
    public void Scatter<T>(IList<Tensor<T>>? input, Tensor<T> output, int root) =>
        throw new NotSupportedException("MpiProcessGroup.Scatter not yet wired (uses MPI_Scatter). Use Broadcast + slice meanwhile.");

    /// <inheritdoc/>
    public IProcessGroup? NewGroup(IReadOnlyList<int> ranks) =>
        throw new NotSupportedException("MPI sub-group construction (MPI_Comm_create) is a follow-up.");

    /// <inheritdoc/>
    public IProcessGroup SplitGroup(int color, int? key = null) =>
        throw new NotSupportedException("MPI sub-group construction (MPI_Comm_split) is a follow-up.");

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_initLock)
        {
            if (_disposed) return; // idempotent — guard against unbalanced MPI_Finalize.
            _disposed = true;
            _initRefcount--;
            if (_initRefcount == 0)
            {
                MpiNative.MPI_Finalize();
            }
        }
    }
}
#endif
