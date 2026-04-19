using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// P/Invoke bindings for the NVIDIA Collective Communications Library
/// (NCCL) — the foundation for multi-GPU distributed training (data /
/// tensor / pipeline parallel). This is the documented <c>nccl.h</c>
/// surface shipped with NCCL 2.x; the bindings are gated on the
/// <c>libnccl.so.2</c> / <c>nccl.dll</c> library being loadable at
/// runtime. <see cref="NcclComm"/> wraps these into an idiomatic
/// managed surface.
///
/// <para><b>Availability probe:</b> call
/// <see cref="NcclComm.IsAvailable"/> to check without crashing — a
/// missing library returns false rather than throwing.</para>
/// </summary>
public static class NcclNative
{
    private const string NcclLibrary = "nccl";

    [DllImport(NcclLibrary, EntryPoint = "ncclGetVersion")]
    public static extern NcclResult ncclGetVersion(out int version);

    [DllImport(NcclLibrary, EntryPoint = "ncclGetUniqueId")]
    public static extern NcclResult ncclGetUniqueId(out NcclUniqueId uniqueId);

    [DllImport(NcclLibrary, EntryPoint = "ncclCommInitRank")]
    public static extern NcclResult ncclCommInitRank(
        out IntPtr comm, int nranks, NcclUniqueId commId, int rank);

    [DllImport(NcclLibrary, EntryPoint = "ncclCommInitAll")]
    public static extern NcclResult ncclCommInitAll(
        [Out] IntPtr[] comms, int ndev, [In] int[] devlist);

    [DllImport(NcclLibrary, EntryPoint = "ncclCommDestroy")]
    public static extern NcclResult ncclCommDestroy(IntPtr comm);

    [DllImport(NcclLibrary, EntryPoint = "ncclCommUserRank")]
    public static extern NcclResult ncclCommUserRank(IntPtr comm, out int rank);

    [DllImport(NcclLibrary, EntryPoint = "ncclCommCount")]
    public static extern NcclResult ncclCommCount(IntPtr comm, out int count);

    [DllImport(NcclLibrary, EntryPoint = "ncclAllReduce")]
    public static extern NcclResult ncclAllReduce(
        IntPtr sendbuff, IntPtr recvbuff, ulong count,
        NcclDataType datatype, NcclRedOp op, IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, EntryPoint = "ncclAllGather")]
    public static extern NcclResult ncclAllGather(
        IntPtr sendbuff, IntPtr recvbuff, ulong sendcount,
        NcclDataType datatype, IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, EntryPoint = "ncclReduceScatter")]
    public static extern NcclResult ncclReduceScatter(
        IntPtr sendbuff, IntPtr recvbuff, ulong recvcount,
        NcclDataType datatype, NcclRedOp op, IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, EntryPoint = "ncclBroadcast")]
    public static extern NcclResult ncclBroadcast(
        IntPtr sendbuff, IntPtr recvbuff, ulong count,
        NcclDataType datatype, int root, IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, EntryPoint = "ncclReduce")]
    public static extern NcclResult ncclReduce(
        IntPtr sendbuff, IntPtr recvbuff, ulong count,
        NcclDataType datatype, NcclRedOp op, int root, IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, EntryPoint = "ncclSend")]
    public static extern NcclResult ncclSend(
        IntPtr sendbuff, ulong count, NcclDataType datatype,
        int peer, IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, EntryPoint = "ncclRecv")]
    public static extern NcclResult ncclRecv(
        IntPtr recvbuff, ulong count, NcclDataType datatype,
        int peer, IntPtr comm, IntPtr stream);

    [DllImport(NcclLibrary, EntryPoint = "ncclGroupStart")]
    public static extern NcclResult ncclGroupStart();

    [DllImport(NcclLibrary, EntryPoint = "ncclGroupEnd")]
    public static extern NcclResult ncclGroupEnd();

    [DllImport(NcclLibrary, EntryPoint = "ncclGetErrorString", CharSet = CharSet.Ansi)]
    public static extern IntPtr ncclGetErrorStringPtr(NcclResult result);

    public static string GetErrorString(NcclResult result)
    {
        var ptr = ncclGetErrorStringPtr(result);
        return ptr == IntPtr.Zero ? $"NCCL error {result}" : Marshal.PtrToStringAnsi(ptr) ?? result.ToString();
    }

    internal static void Check(NcclResult result, string op)
    {
        if (result != NcclResult.Success)
            throw new InvalidOperationException($"NCCL {op} failed: {GetErrorString(result)}");
    }
}

/// <summary>NCCL return codes.</summary>
public enum NcclResult
{
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    RemoteError = 6,
    InProgress = 7,
}

/// <summary>Element types supported by NCCL collectives.</summary>
public enum NcclDataType
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
    BFloat16 = 9,
}

/// <summary>Reduction ops.</summary>
public enum NcclRedOp
{
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

/// <summary>128-byte opaque unique-id blob NCCL uses for rendezvous.</summary>
[StructLayout(LayoutKind.Sequential, Size = 128)]
public struct NcclUniqueId
{
    // 128 bytes of opaque data — NCCL populates via ncclGetUniqueId
    // and consumes via ncclCommInitRank. We treat it as opaque.
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
    public byte[] Internal;
}
