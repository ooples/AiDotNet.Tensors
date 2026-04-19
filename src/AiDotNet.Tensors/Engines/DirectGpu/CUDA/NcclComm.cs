namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Managed wrapper around an NCCL communicator
/// (<c>ncclComm_t</c>). Owns a single rank's view of the collective
/// group. Disposing releases the native communicator.
///
/// <para><b>Typical lifecycle (rank 0):</b></para>
/// <code>
/// var id = NcclComm.CreateUniqueId();
/// // broadcast id bytes to every other rank via your control plane
/// using var comm = NcclComm.InitRank(nRanks: 4, rank: 0, id);
/// comm.AllReduce(devPtr, count, NcclDataType.Float32, NcclRedOp.Sum, streamHandle);
/// </code>
///
/// <para>For single-process multi-GPU (the simplest case),
/// <see cref="InitAll"/> creates all <c>nRanks</c> comms in one
/// call — no id broadcast required.</para>
/// </summary>
public sealed class NcclComm : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    /// <summary>Native ncclComm_t. Exposed for interop with other
    /// NCCL-aware libraries.</summary>
    public IntPtr Handle => _handle;

    /// <summary>Rank of this communicator within its group (0..nRanks-1).</summary>
    public int Rank { get; }

    /// <summary>Total rank count in the group.</summary>
    public int NRanks { get; }

    private NcclComm(IntPtr handle, int rank, int nRanks)
    {
        _handle = handle;
        Rank = rank;
        NRanks = nRanks;
    }

    /// <summary>Check whether libnccl is loadable.</summary>
    public static bool IsAvailable
    {
        get
        {
            try
            {
                NcclNative.ncclGetVersion(out _);
                return true;
            }
            catch { return false; }
        }
    }

    /// <summary>Library version, e.g. 21803 for NCCL 2.18.3.</summary>
    public static int GetVersion()
    {
        NcclNative.Check(NcclNative.ncclGetVersion(out int v), "GetVersion");
        return v;
    }

    /// <summary>Mint a fresh 128-byte unique id. Rank 0 calls this
    /// then distributes the bytes to every other rank via whatever
    /// control plane the app uses (MPI, shared filesystem, RPC).</summary>
    public static NcclUniqueId CreateUniqueId()
    {
        NcclNative.Check(NcclNative.ncclGetUniqueId(out var id), "GetUniqueId");
        return id;
    }

    /// <summary>Initialize one rank of a multi-process NCCL group.</summary>
    public static NcclComm InitRank(int nRanks, int rank, NcclUniqueId id)
    {
        if (nRanks <= 0) throw new ArgumentOutOfRangeException(nameof(nRanks));
        if (rank < 0 || rank >= nRanks) throw new ArgumentOutOfRangeException(nameof(rank));
        NcclNative.Check(NcclNative.ncclCommInitRank(out IntPtr h, nRanks, id, rank), "CommInitRank");
        return new NcclComm(h, rank, nRanks);
    }

    /// <summary>Single-process multi-GPU — allocate all comms at once.
    /// Returns an array of <paramref name="deviceIds"/>.Length comms,
    /// one per device. Caller disposes each.</summary>
    public static NcclComm[] InitAll(int[] deviceIds)
    {
        if (deviceIds is null) throw new ArgumentNullException(nameof(deviceIds));
        if (deviceIds.Length == 0) return Array.Empty<NcclComm>();
        var handles = new IntPtr[deviceIds.Length];
        NcclNative.Check(
            NcclNative.ncclCommInitAll(handles, deviceIds.Length, deviceIds),
            "CommInitAll");
        var result = new NcclComm[deviceIds.Length];
        for (int i = 0; i < result.Length; i++)
            result[i] = new NcclComm(handles[i], i, deviceIds.Length);
        return result;
    }

    // ──────────── collectives ────────────

    public void AllReduce(IntPtr sendBuf, IntPtr recvBuf, ulong count,
        NcclDataType dtype, NcclRedOp op, IntPtr stream)
        => NcclNative.Check(
            NcclNative.ncclAllReduce(sendBuf, recvBuf, count, dtype, op, _handle, stream),
            "AllReduce");

    public void AllGather(IntPtr sendBuf, IntPtr recvBuf, ulong sendCount,
        NcclDataType dtype, IntPtr stream)
        => NcclNative.Check(
            NcclNative.ncclAllGather(sendBuf, recvBuf, sendCount, dtype, _handle, stream),
            "AllGather");

    public void ReduceScatter(IntPtr sendBuf, IntPtr recvBuf, ulong recvCount,
        NcclDataType dtype, NcclRedOp op, IntPtr stream)
        => NcclNative.Check(
            NcclNative.ncclReduceScatter(sendBuf, recvBuf, recvCount, dtype, op, _handle, stream),
            "ReduceScatter");

    public void Broadcast(IntPtr sendBuf, IntPtr recvBuf, ulong count,
        NcclDataType dtype, int root, IntPtr stream)
        => NcclNative.Check(
            NcclNative.ncclBroadcast(sendBuf, recvBuf, count, dtype, root, _handle, stream),
            "Broadcast");

    public void Reduce(IntPtr sendBuf, IntPtr recvBuf, ulong count,
        NcclDataType dtype, NcclRedOp op, int root, IntPtr stream)
        => NcclNative.Check(
            NcclNative.ncclReduce(sendBuf, recvBuf, count, dtype, op, root, _handle, stream),
            "Reduce");

    /// <summary>Point-to-point send within a group.</summary>
    public void Send(IntPtr buf, ulong count, NcclDataType dtype, int peer, IntPtr stream)
        => NcclNative.Check(
            NcclNative.ncclSend(buf, count, dtype, peer, _handle, stream),
            "Send");

    /// <summary>Point-to-point recv within a group.</summary>
    public void Recv(IntPtr buf, ulong count, NcclDataType dtype, int peer, IntPtr stream)
        => NcclNative.Check(
            NcclNative.ncclRecv(buf, count, dtype, peer, _handle, stream),
            "Recv");

    /// <summary>Open an NCCL group — every collective issued between
    /// this call and <see cref="GroupEnd"/> is batched into a single
    /// fused submission. Essential for P2P send/recv (the host-blocks-
    /// on-send deadlock is avoided by fusing with recv).</summary>
    public static void GroupStart() => NcclNative.Check(NcclNative.ncclGroupStart(), "GroupStart");
    public static void GroupEnd()   => NcclNative.Check(NcclNative.ncclGroupEnd(),   "GroupEnd");

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_handle != IntPtr.Zero)
        {
            // Best-effort: NCCL errors on destroy are unrecoverable, log
            // and move on rather than throwing from finalizer/Dispose.
            try { NcclNative.ncclCommDestroy(_handle); } catch { }
            _handle = IntPtr.Zero;
        }
    }
}
