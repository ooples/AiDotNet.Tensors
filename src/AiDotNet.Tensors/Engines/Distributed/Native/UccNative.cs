// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.Distributed.Native;

/// <summary>
/// P/Invoke bindings into <c>libucc</c> (Unified Collective Communication).
/// UCC is the OpenUCX collective-communication library, used by recent
/// PyTorch / Megatron-LM builds as an alternative to NCCL on certain
/// transports. The <see cref="UccProcessGroup"/> wrapper bridges UCC to
/// the <see cref="IProcessGroup"/> abstraction.
///
/// <para>SONAME: <c>libucc.so.1</c>. UCC ships only on Linux as part of
/// the OpenUCX suite — <see cref="IsAvailable"/> is false on Windows /
/// macOS.</para>
/// </summary>
internal static class UccNative
{
    private const string Lib = "ucc";

#if NET5_0_OR_GREATER
    static UccNative()
    {
        NativeLibrary.SetDllImportResolver(typeof(UccNative).Assembly, ResolveLib);
    }

    private static IntPtr ResolveLib(string name, System.Reflection.Assembly asm, DllImportSearchPath? path)
    {
        if (name != Lib) return IntPtr.Zero;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            if (NativeLibrary.TryLoad("libucc.so.1", asm, path, out var h)) return h;
            if (NativeLibrary.TryLoad("libucc.so", asm, path, out h)) return h;
        }
        return IntPtr.Zero;
    }
#endif

    public enum Status
    {
        Ok = 0,
        InProgress = 1,
        OperationInitialized = 2,
        Error = -1,
        ErrNotImplemented = -3,
        ErrInvalidParam = -7,
        ErrUnsupported = -22,
    }

    public enum CollOpType
    {
        AllReduce = 0,
        AllGather = 1,
        AllGatherV = 2,
        AllToAll = 3,
        AllToAllV = 4,
        Barrier = 5,
        Bcast = 6,
        Fanin = 7,
        Fanout = 8,
        Gather = 9,
        GatherV = 10,
        Reduce = 11,
        ReduceScatter = 12,
        ReduceScatterV = 13,
        Scatter = 14,
        ScatterV = 15,
    }

    public enum DataType
    {
        Predefined_Int8 = 0,
        Predefined_Int32 = 2,
        Predefined_Int64 = 4,
        Predefined_Float32 = 8,
        Predefined_Float64 = 9,
        Predefined_BFloat16 = 10,
    }

    public enum ReductionOp
    {
        Sum = 0,
        Prod = 1,
        Max = 2,
        Min = 3,
        Avg = 4,
        BAnd = 5,
        BOr = 6,
        BXor = 7,
    }

    [DllImport(Lib, EntryPoint = "ucc_init")]
    public static extern Status ucc_init(IntPtr libParams, IntPtr libConfig, out IntPtr libHandle);

    [DllImport(Lib, EntryPoint = "ucc_finalize")]
    public static extern Status ucc_finalize(IntPtr libHandle);

    [DllImport(Lib, EntryPoint = "ucc_context_create")]
    public static extern Status ucc_context_create(IntPtr libHandle, IntPtr params_, IntPtr config, out IntPtr context);

    [DllImport(Lib, EntryPoint = "ucc_context_destroy")]
    public static extern Status ucc_context_destroy(IntPtr context);

    [DllImport(Lib, EntryPoint = "ucc_team_create_post")]
    public static extern Status ucc_team_create_post(IntPtr[] contexts, uint numContexts, IntPtr params_, out IntPtr team);

    [DllImport(Lib, EntryPoint = "ucc_team_destroy")]
    public static extern Status ucc_team_destroy(IntPtr team);

    [DllImport(Lib, EntryPoint = "ucc_collective_init")]
    public static extern Status ucc_collective_init(IntPtr collArgs, out IntPtr request, IntPtr team);

    [DllImport(Lib, EntryPoint = "ucc_collective_post")]
    public static extern Status ucc_collective_post(IntPtr request);

    [DllImport(Lib, EntryPoint = "ucc_collective_test")]
    public static extern Status ucc_collective_test(IntPtr request);

    [DllImport(Lib, EntryPoint = "ucc_collective_finalize")]
    public static extern Status ucc_collective_finalize(IntPtr request);

    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try
        {
#if NET5_0_OR_GREATER
            if (!NativeLibrary.TryLoad(Lib, typeof(UccNative).Assembly, null, out var handle)) return false;
            try { return NativeLibrary.GetExport(handle, "ucc_init") != IntPtr.Zero; }
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
