// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.Distributed.Native;

/// <summary>
/// P/Invoke bindings into <c>libmpi</c> (Open MPI / MPICH). MPI is the
/// canonical cross-machine collective-communication library on
/// supercomputer-class systems. Used by the
/// <see cref="MpiProcessGroup"/> wrapper to provide an
/// <see cref="IProcessGroup"/> backend.
///
/// <para>SONAME is <c>libmpi.so.40</c> on most modern Open MPI releases
/// and <c>libmpi.so.12</c> on MPICH. The <see cref="IsAvailable"/>
/// probe attempts both. On Windows the binding looks for
/// <c>msmpi.dll</c> from Microsoft MPI.</para>
///
/// <para>The MPI C API uses opaque <c>MPI_Comm</c> / <c>MPI_Op</c> /
/// <c>MPI_Datatype</c> handles. Different MPI implementations represent
/// these differently (Open MPI uses pointers, MPICH uses small ints),
/// so the wrapper uses <c>IntPtr</c> for portability and looks up the
/// pre-defined constants (<c>MPI_COMM_WORLD</c>, <c>MPI_FLOAT</c>, etc.)
/// at init time via <c>MPI_Init</c>.</para>
/// </summary>
internal static class MpiNative
{
    private const string Lib = "mpi";

#if NET5_0_OR_GREATER
    static MpiNative()
    {
        NativeLibrary.SetDllImportResolver(typeof(MpiNative).Assembly, ResolveLib);
    }

    private static IntPtr ResolveLib(string name, System.Reflection.Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            // Try Open MPI first, then MPICH.
            if (NativeLibrary.TryLoad("libmpi.so.40", asm, path, out var h)) return h;
            if (NativeLibrary.TryLoad("libmpi.so.12", asm, path, out h)) return h;
            if (NativeLibrary.TryLoad("libmpi.so", asm, path, out h)) return h;
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            if (NativeLibrary.TryLoad("libmpi.dylib", asm, path, out var h)) return h;
        }
        else
        {
            if (NativeLibrary.TryLoad("msmpi.dll", asm, path, out var h)) return h;
        }
        return IntPtr.Zero;
    }
#endif

    public const int MPI_SUCCESS = 0;

    [DllImport(Lib, EntryPoint = "MPI_Init")]
    public static extern int MPI_Init(ref int argc, ref IntPtr argv);

    [DllImport(Lib, EntryPoint = "MPI_Finalize")]
    public static extern int MPI_Finalize();

    [DllImport(Lib, EntryPoint = "MPI_Comm_size")]
    public static extern int MPI_Comm_size(IntPtr comm, out int size);

    [DllImport(Lib, EntryPoint = "MPI_Comm_rank")]
    public static extern int MPI_Comm_rank(IntPtr comm, out int rank);

    [DllImport(Lib, EntryPoint = "MPI_Allreduce")]
    public static extern int MPI_Allreduce(IntPtr sendBuf, IntPtr recvBuf, int count, IntPtr dataType, IntPtr op, IntPtr comm);

    [DllImport(Lib, EntryPoint = "MPI_Bcast")]
    public static extern int MPI_Bcast(IntPtr buffer, int count, IntPtr dataType, int root, IntPtr comm);

    [DllImport(Lib, EntryPoint = "MPI_Reduce")]
    public static extern int MPI_Reduce(IntPtr sendBuf, IntPtr recvBuf, int count, IntPtr dataType, IntPtr op, int root, IntPtr comm);

    [DllImport(Lib, EntryPoint = "MPI_Allgather")]
    public static extern int MPI_Allgather(IntPtr sendBuf, int sendCount, IntPtr sendType,
        IntPtr recvBuf, int recvCount, IntPtr recvType, IntPtr comm);

    [DllImport(Lib, EntryPoint = "MPI_Reduce_scatter_block")]
    public static extern int MPI_Reduce_scatter_block(IntPtr sendBuf, IntPtr recvBuf, int recvCount,
        IntPtr dataType, IntPtr op, IntPtr comm);

    [DllImport(Lib, EntryPoint = "MPI_Send")]
    public static extern int MPI_Send(IntPtr buf, int count, IntPtr dataType, int dest, int tag, IntPtr comm);

    [DllImport(Lib, EntryPoint = "MPI_Recv")]
    public static extern int MPI_Recv(IntPtr buf, int count, IntPtr dataType, int src, int tag, IntPtr comm, IntPtr status);

    [DllImport(Lib, EntryPoint = "MPI_Barrier")]
    public static extern int MPI_Barrier(IntPtr comm);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            // We can't safely call MPI_Init here (it's process-wide and
            // ungrateful about being called from a probe). Instead, we
            // resolve the entry-point symbol via NativeLibrary and accept
            // that as confirmation. On older runtimes the DllImport JIT
            // will throw DllNotFoundException at first invocation and we
            // catch it.
#if NET5_0_OR_GREATER
            if (!NativeLibrary.TryLoad(Lib, typeof(MpiNative).Assembly, null, out var handle)) return false;
            try { return NativeLibrary.GetExport(handle, "MPI_Init") != IntPtr.Zero; }
            finally { NativeLibrary.Free(handle); }
#else
            return false;
#endif
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
