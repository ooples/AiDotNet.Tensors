// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.Distributed.Native;

/// <summary>
/// P/Invoke bindings into <c>libnccl</c>. NCCL is NVIDIA's collective-
/// communication library — the GPU-side counterpart of MPI / Gloo. It
/// implements all-reduce / broadcast / reduce / all-gather / reduce-
/// scatter / send / recv on top of NVLink and InfiniBand transports.
///
/// <para>Standard usage:
/// <c>ncclGetUniqueId(&amp;id)</c> → broadcast id to every rank →
/// <c>ncclCommInitRank(&amp;comm, world, id, rank)</c> → call
/// <c>ncclAllReduce</c> etc. → <c>ncclCommDestroy(comm)</c>. The
/// <see cref="NcclProcessGroup"/> wrapper handles this lifecycle so
/// callers stay on the <see cref="IProcessGroup"/> abstraction.</para>
///
/// <para>SONAME is <c>libnccl.so.2</c> on Linux. NCCL doesn't ship for
/// Windows / macOS — <see cref="IsAvailable"/> is false there.</para>
/// </summary>
internal static class NcclNative
{
    private const string Lib = "nccl";

#if NET5_0_OR_GREATER
    static NcclNative()
    {
        NativeLibrary.SetDllImportResolver(typeof(NcclNative).Assembly, ResolveLib);
    }

    private static IntPtr ResolveLib(string name, System.Reflection.Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (NativeLibrary.TryLoad("libnccl.so.2", asm, path, out var h)) return h;
            if (NativeLibrary.TryLoad("libnccl.so", asm, path, out h)) return h;
        }
        return IntPtr.Zero;
    }
#endif

    public enum Result
    {
        Success = 0,
        UnhandledCudaError = 1,
        SystemError = 2,
        InternalError = 3,
        InvalidArgument = 4,
        InvalidUsage = 5,
        RemoteError = 6,
        InProgress = 7,
        NumResults = 8,
    }

    public enum DataType
    {
        Int8 = 0, Char = 0,
        Uint8 = 1,
        Int32 = 2, Int = 2,
        Uint32 = 3,
        Int64 = 4,
        Uint64 = 5,
        Float16 = 6, Half = 6,
        Float32 = 7, Float = 7,
        Float64 = 8, Double = 8,
        Bfloat16 = 9,
    }

    public enum RedOp
    {
        Sum = 0,
        Prod = 1,
        Max = 2,
        Min = 3,
        Avg = 4,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct UniqueId
    {
        // ncclUniqueId is 128 bytes — opaque blob.
        public unsafe fixed byte Data[128];
    }

    [DllImport(Lib)] public static extern Result ncclGetUniqueId(out UniqueId uniqueId);
    [DllImport(Lib)] public static extern Result ncclCommInitRank(out IntPtr comm, int nranks, UniqueId commId, int rank);
    [DllImport(Lib)] public static extern Result ncclCommDestroy(IntPtr comm);
    [DllImport(Lib)] public static extern Result ncclCommAbort(IntPtr comm);
    [DllImport(Lib)] public static extern Result ncclCommCount(IntPtr comm, out int count);
    [DllImport(Lib)] public static extern Result ncclCommUserRank(IntPtr comm, out int rank);

    [DllImport(Lib)]
    public static extern Result ncclAllReduce(IntPtr sendBuff, IntPtr recvBuff, ulong count,
        DataType dataType, RedOp op, IntPtr comm, IntPtr stream);

    [DllImport(Lib)]
    public static extern Result ncclBroadcast(IntPtr sendBuff, IntPtr recvBuff, ulong count,
        DataType dataType, int root, IntPtr comm, IntPtr stream);

    [DllImport(Lib)]
    public static extern Result ncclReduce(IntPtr sendBuff, IntPtr recvBuff, ulong count,
        DataType dataType, RedOp op, int root, IntPtr comm, IntPtr stream);

    [DllImport(Lib)]
    public static extern Result ncclAllGather(IntPtr sendBuff, IntPtr recvBuff, ulong sendCount,
        DataType dataType, IntPtr comm, IntPtr stream);

    [DllImport(Lib)]
    public static extern Result ncclReduceScatter(IntPtr sendBuff, IntPtr recvBuff, ulong recvCount,
        DataType dataType, RedOp op, IntPtr comm, IntPtr stream);

    [DllImport(Lib)]
    public static extern Result ncclSend(IntPtr sendBuff, ulong count,
        DataType dataType, int peer, IntPtr comm, IntPtr stream);

    [DllImport(Lib)]
    public static extern Result ncclRecv(IntPtr recvBuff, ulong count,
        DataType dataType, int peer, IntPtr comm, IntPtr stream);

    [DllImport(Lib)] public static extern Result ncclGroupStart();
    [DllImport(Lib)] public static extern Result ncclGroupEnd();

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
            // ncclGetUniqueId is callable without a CUDA context, so this is
            // a clean availability check. If libnccl is missing, the call
            // throws DllNotFoundException at JIT site.
            return ncclGetUniqueId(out _) == Result.Success;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }
}
