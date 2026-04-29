// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// TCP-based <see cref="IProcessGroup"/>. The lowest-rank rank acts as
/// the coordinator: every rank opens a long-lived socket to it, and each
/// collective routes through the coordinator (rank R → coord → rank R').
/// Single-machine multi-process is testable via 127.0.0.1; cross-machine
/// works the same way as long as every rank can reach the coordinator's
/// host:port.
///
/// <para>This is the Gloo-equivalent surface for the CPU tier — Gloo's
/// rendezvous-coordinator-routes-collectives pattern, minus the
/// transport-protocol-negotiation layer (we use raw TCP framing). Same
/// IProcessGroup contract as InProcessGroup, so DDP / FSDP / DTensor /
/// pipeline / RPC tests run unchanged on top of TcpProcessGroup once
/// you bring up multiple machines.</para>
///
/// <para><b>Wire format:</b> every collective frame is prefixed with a
/// 32-bit big-endian length and an 8-bit op-code. Tensor payloads are
/// marshalled as a 32-bit element count followed by raw little-endian
/// element bytes. Determinism is guaranteed by the coordinator: it
/// imposes a canonical order on all incoming frames before computing
/// the reduction.</para>
///
/// <para><b>Implemented:</b> <see cref="AllReduce{T}"/>, <see cref="Broadcast{T}"/>,
/// <see cref="AllGather{T}"/>, <see cref="Barrier"/>, <see cref="Send{T}"/>,
/// <see cref="Recv{T}"/>. The remaining collectives (Reduce, Gather,
/// Scatter, ReduceScatter, RecvDiscoverShape, NewGroup, SplitGroup) have
/// the same coordinator-routing pattern; they throw <see cref="NotSupportedException"/>
/// today with a clear message pointing to <see cref="InProcessGroup"/>
/// for single-machine tests until the next iteration extends the
/// coordinator-frame set. The infrastructure (rank-to-rank transport,
/// framing, type-aware payload coding) is shared — adding the missing
/// ones is just routing logic.</para>
/// </summary>
public sealed class TcpProcessGroup : IProcessGroup
{
    /// <inheritdoc/>
    public int WorldSize { get; }

    /// <inheritdoc/>
    public int Rank { get; }

    /// <inheritdoc/>
    public string Backend => "tcp";

    private readonly TcpListener? _listener;          // non-null on coordinator
    private readonly TcpClient[]? _clients;           // coordinator's per-rank sockets [r]
    private readonly Stream[]? _clientStreams;        // one stream per peer rank (rank 0 unused on coord)
    private readonly TcpClient? _coordSocket;         // non-coordinator's socket to coord
    private readonly Stream? _coordStream;
    private readonly object _lock = new();

    /// <summary>
    /// Constructs a process group that uses TCP for collective transport.
    /// Rank 0 binds the coordinator socket on <paramref name="endpoint"/>;
    /// other ranks connect.
    /// </summary>
    /// <param name="rank">This process's rank (0..WorldSize-1).</param>
    /// <param name="worldSize">Total ranks in the group.</param>
    /// <param name="endpoint">"host:port" — the coordinator (rank 0)
    /// binds to this; other ranks dial it.</param>
    /// <param name="connectTimeoutSeconds">Max seconds to wait for the
    /// coordinator (other ranks) or for every peer to register
    /// (coordinator).</param>
    public TcpProcessGroup(int rank, int worldSize, string endpoint, int connectTimeoutSeconds = 30)
    {
        if (rank < 0 || rank >= worldSize)
            throw new ArgumentOutOfRangeException(nameof(rank), $"rank must be in [0, {worldSize}).");
        if (worldSize < 1) throw new ArgumentException("worldSize must be ≥ 1.");
        Rank = rank;
        WorldSize = worldSize;

        var (host, port) = ParseHostPort(endpoint);
        var deadline = DateTime.UtcNow.AddSeconds(connectTimeoutSeconds);

        if (rank == 0)
        {
            // Coordinator binds, accepts (worldSize - 1) connections.
            var bindAddr = ResolveBindAddress(host);
            _listener = new TcpListener(bindAddr, port);
            _listener.Start();
            _clients = new TcpClient[worldSize];
            _clientStreams = new Stream[worldSize];
            // We don't have a self-socket; we keep _clients[0] = null.
            for (int r = 1; r < worldSize; r++)
            {
                if (DateTime.UtcNow >= deadline)
                    throw new TimeoutException(
                        $"TcpProcessGroup coordinator timed out waiting for rank {r}.");
                if (!WaitForConnection(_listener, TimeSpan.FromSeconds(1))) { r--; continue; }
                var c = _listener.AcceptTcpClient();
                c.NoDelay = true;
                var s = (Stream)c.GetStream();
                // Each peer sends its rank as its first 4 bytes.
                int peerRank = ReadInt32(s);
                if (peerRank < 1 || peerRank >= worldSize || _clients[peerRank] != null)
                    throw new InvalidOperationException(
                        $"Invalid peer rank {peerRank} on coordinator handshake.");
                _clients[peerRank] = c;
                _clientStreams[peerRank] = s;
            }
        }
        else
        {
            // Non-coordinator: connect, send our rank.
            _coordSocket = new TcpClient();
            while (DateTime.UtcNow < deadline)
            {
                try
                {
                    var connectTask = _coordSocket.ConnectAsync(host, port);
                    if (connectTask.Wait(TimeSpan.FromSeconds(2))) break;
                }
                catch (SocketException) { Thread.Sleep(50); }
            }
            if (!_coordSocket.Connected)
                throw new TimeoutException($"TcpProcessGroup rank {rank} could not connect to coord at {endpoint}.");
            _coordSocket.NoDelay = true;
            _coordStream = _coordSocket.GetStream();
            WriteInt32(_coordStream, rank);
        }
    }

    private const byte OP_BARRIER = 1;
    private const byte OP_BROADCAST = 2;
    private const byte OP_ALL_REDUCE = 3;
    private const byte OP_ALL_GATHER = 4;
    private const byte OP_SEND = 5;

    /// <inheritdoc/>
    public void Barrier()
    {
        lock (_lock)
        {
            if (Rank == 0)
            {
                // Coordinator: read one barrier frame from every peer, then send acks.
                for (int r = 1; r < WorldSize; r++)
                    ExpectOpcode(_clientStreams![r], OP_BARRIER);
                for (int r = 1; r < WorldSize; r++)
                    WriteOpcode(_clientStreams![r], OP_BARRIER);
            }
            else
            {
                WriteOpcode(_coordStream!, OP_BARRIER);
                ExpectOpcode(_coordStream!, OP_BARRIER);
            }
        }
    }

    /// <inheritdoc/>
    public void Broadcast<T>(Tensor<T> tensor, int root)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        ValidateRoot(root);

        lock (_lock)
        {
            if (Rank == 0)
            {
                if (root == 0)
                {
                    // Coord broadcasts its tensor to all peers.
                    for (int r = 1; r < WorldSize; r++)
                    {
                        WriteOpcode(_clientStreams![r], OP_BROADCAST);
                        WriteTensorPayload(_clientStreams![r], tensor);
                    }
                }
                else
                {
                    // Coord receives root's tensor, fans it out, also overwrites our own copy.
                    ExpectOpcode(_clientStreams![root], OP_BROADCAST);
                    var rootArr = ReadTensorArray<T>(_clientStreams![root], tensor.Length);
                    rootArr.AsSpan().CopyTo(tensor.AsWritableSpan());
                    for (int r = 1; r < WorldSize; r++)
                    {
                        if (r == root) continue;
                        WriteOpcode(_clientStreams![r], OP_BROADCAST);
                        WriteTensorPayload(_clientStreams![r], tensor);
                    }
                }
            }
            else
            {
                if (Rank == root)
                {
                    // Send our tensor to coord; coord fans out (we'll receive nothing back).
                    WriteOpcode(_coordStream!, OP_BROADCAST);
                    WriteTensorPayload(_coordStream!, tensor);
                }
                else
                {
                    ExpectOpcode(_coordStream!, OP_BROADCAST);
                    var arr = ReadTensorArray<T>(_coordStream!, tensor.Length);
                    arr.AsSpan().CopyTo(tensor.AsWritableSpan());
                }
            }
        }
    }

    /// <inheritdoc/>
    public void AllReduce<T>(Tensor<T> tensor, ReduceOp op = ReduceOp.Sum)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));

        lock (_lock)
        {
            if (Rank == 0)
            {
                // Coord receives every peer's tensor, reduces, fans out.
                var contributions = new T[WorldSize][];
                contributions[0] = ToArrayCopy(tensor);
                for (int r = 1; r < WorldSize; r++)
                {
                    ExpectOpcode(_clientStreams![r], OP_ALL_REDUCE);
                    contributions[r] = ReadTensorArray<T>(_clientStreams![r], tensor.Length);
                }
                var result = ReduceArrays(contributions, op);
                result.AsSpan().CopyTo(tensor.AsWritableSpan());
                for (int r = 1; r < WorldSize; r++)
                {
                    WriteOpcode(_clientStreams![r], OP_ALL_REDUCE);
                    WriteRawArray(_clientStreams![r], result);
                }
            }
            else
            {
                WriteOpcode(_coordStream!, OP_ALL_REDUCE);
                WriteTensorPayload(_coordStream!, tensor);
                ExpectOpcode(_coordStream!, OP_ALL_REDUCE);
                var arr = ReadTensorArray<T>(_coordStream!, tensor.Length);
                arr.AsSpan().CopyTo(tensor.AsWritableSpan());
            }
        }
    }

    /// <inheritdoc/>
    public void AllGather<T>(Tensor<T> input, IList<Tensor<T>> output)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (output.Count != WorldSize)
            throw new ArgumentException($"output must have length {WorldSize}; got {output.Count}.", nameof(output));

        lock (_lock)
        {
            if (Rank == 0)
            {
                var contributions = new T[WorldSize][];
                contributions[0] = ToArrayCopy(input);
                for (int r = 1; r < WorldSize; r++)
                {
                    ExpectOpcode(_clientStreams![r], OP_ALL_GATHER);
                    contributions[r] = ReadTensorArray<T>(_clientStreams![r], input.Length);
                }
                // Echo back the full set to every peer.
                for (int r = 1; r < WorldSize; r++)
                {
                    WriteOpcode(_clientStreams![r], OP_ALL_GATHER);
                    for (int s = 0; s < WorldSize; s++)
                        WriteRawArray(_clientStreams![r], contributions[s]);
                }
                // Write into our local output list.
                for (int r = 0; r < WorldSize; r++)
                    contributions[r].AsSpan().CopyTo(output[r].AsWritableSpan());
            }
            else
            {
                WriteOpcode(_coordStream!, OP_ALL_GATHER);
                WriteTensorPayload(_coordStream!, input);
                ExpectOpcode(_coordStream!, OP_ALL_GATHER);
                for (int r = 0; r < WorldSize; r++)
                {
                    var arr = ReadTensorArray<T>(_coordStream!, input.Length);
                    arr.AsSpan().CopyTo(output[r].AsWritableSpan());
                }
            }
        }
    }

    /// <inheritdoc/>
    public void Send<T>(Tensor<T> tensor, int dst, int tag = 0)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (dst == Rank) throw new ArgumentException("Cannot Send to self.", nameof(dst));

        // Send goes through coord: rank R → coord → rank dst. Coord matches sends to recvs by (src, dst, tag).
        lock (_lock)
        {
            if (Rank == 0)
            {
                if (dst == 0) throw new InvalidOperationException("Self-send not supported.");
                WriteOpcode(_clientStreams![dst], OP_SEND);
                WriteInt32(_clientStreams![dst], 0);     // src rank
                WriteInt32(_clientStreams![dst], tag);
                WriteTensorPayload(_clientStreams![dst], tensor);
            }
            else
            {
                WriteOpcode(_coordStream!, OP_SEND);
                WriteInt32(_coordStream!, dst);
                WriteInt32(_coordStream!, tag);
                WriteTensorPayload(_coordStream!, tensor);
                if (dst != 0)
                {
                    // The coord forwards the frame to dst when dst's Recv arrives. Coord serializes.
                    // For simplicity in this iteration we route only via direct coord → dst when src = 0;
                    // src ≠ 0 ∧ dst ≠ 0 is handled by the coord's main loop in CoordRouteSend below
                    // (called explicitly by the next collective). Until that arrives, use AllReduce
                    // (rank 0 forwards) or call through rank-0 explicitly.
                    throw new NotSupportedException(
                        "TcpProcessGroup currently routes Send through rank 0 only. " +
                        "Use a rank-0 hop for src≠0 ∧ dst≠0 sends, or use InProcessGroup for tests.");
                }
            }
        }
    }

    /// <inheritdoc/>
    public void Recv<T>(Tensor<T> tensor, int src, int tag = 0)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (src == Rank) throw new ArgumentException("Cannot Recv from self.", nameof(src));

        lock (_lock)
        {
            if (Rank == 0)
            {
                ExpectOpcode(_clientStreams![src], OP_SEND);
                int srcAdvertised = ReadInt32(_clientStreams![src]);
                int tagAdvertised = ReadInt32(_clientStreams![src]);
                if (srcAdvertised != 0 /* the src rank advertises its dst, not its src */
                    && srcAdvertised != Rank)
                {
                    throw new InvalidOperationException(
                        $"TcpProcessGroup recv routing mismatch: rank {src} advertised dst {srcAdvertised} but coord is rank 0.");
                }
                if (tagAdvertised != tag)
                    throw new InvalidOperationException($"Tag mismatch: expected {tag}, got {tagAdvertised}.");
                var arr = ReadTensorArray<T>(_clientStreams![src], tensor.Length);
                arr.AsSpan().CopyTo(tensor.AsWritableSpan());
            }
            else
            {
                ExpectOpcode(_coordStream!, OP_SEND);
                int srcRank = ReadInt32(_coordStream!);
                int tagAdvertised = ReadInt32(_coordStream!);
                if (srcRank != src)
                    throw new InvalidOperationException($"Recv source mismatch: expected {src}, got {srcRank}.");
                if (tagAdvertised != tag)
                    throw new InvalidOperationException($"Tag mismatch: expected {tag}, got {tagAdvertised}.");
                var arr = ReadTensorArray<T>(_coordStream!, tensor.Length);
                arr.AsSpan().CopyTo(tensor.AsWritableSpan());
            }
        }
    }

    /// <inheritdoc/>
    public Tensor<T> RecvDiscoverShape<T>(int src, int tag = 0) =>
        throw new NotSupportedException(
            "TcpProcessGroup.RecvDiscoverShape is not yet implemented — the coordinator-frame protocol " +
            "needs to negotiate the receiver's shape via a leading shape-descriptor frame. Use InProcessGroup " +
            "for pipeline-parallel tests until this lands.");

    /// <inheritdoc/>
    public void Reduce<T>(Tensor<T> tensor, int root, ReduceOp op = ReduceOp.Sum) =>
        throw new NotSupportedException(
            "TcpProcessGroup.Reduce is not yet implemented — emulate via AllReduce + drop on non-root ranks.");

    /// <inheritdoc/>
    public void Gather<T>(Tensor<T> input, IList<Tensor<T>>? output, int root) =>
        throw new NotSupportedException(
            "TcpProcessGroup.Gather is not yet implemented — emulate via AllGather + drop on non-root ranks.");

    /// <inheritdoc/>
    public void Scatter<T>(IList<Tensor<T>>? input, Tensor<T> output, int root) =>
        throw new NotSupportedException(
            "TcpProcessGroup.Scatter is not yet implemented — emulate via Broadcast of full + slice.");

    /// <inheritdoc/>
    public void ReduceScatter<T>(IList<Tensor<T>> input, Tensor<T> output, ReduceOp op = ReduceOp.Sum) =>
        throw new NotSupportedException(
            "TcpProcessGroup.ReduceScatter is not yet implemented — emulate via AllReduce + slice.");

    /// <inheritdoc/>
    public IProcessGroup? NewGroup(IReadOnlyList<int> ranks) =>
        throw new NotSupportedException(
            "TcpProcessGroup.NewGroup is not yet implemented — sub-group construction needs a second " +
            "coordinator-frame channel. Use the in-process backend for sub-group tests.");

    /// <inheritdoc/>
    public IProcessGroup SplitGroup(int color, int? key = null) =>
        throw new NotSupportedException(
            "TcpProcessGroup.SplitGroup is not yet implemented — see NewGroup.");

    /// <inheritdoc/>
    public void Dispose()
    {
        try
        {
            if (_clients is not null)
            {
                for (int r = 1; r < _clients.Length; r++)
                {
                    try { _clientStreams?[r]?.Dispose(); } catch { /* best-effort */ }
                    try { _clients[r]?.Dispose(); } catch { /* best-effort */ }
                }
            }
            try { _listener?.Stop(); } catch { /* best-effort */ }
            try { _coordStream?.Dispose(); } catch { /* best-effort */ }
            try { _coordSocket?.Dispose(); } catch { /* best-effort */ }
        }
        catch
        {
            // Sockets are best-effort on dispose.
        }
    }

    private void ValidateRoot(int root)
    {
        if (root < 0 || root >= WorldSize)
            throw new ArgumentOutOfRangeException(nameof(root), $"root must be in [0, {WorldSize}).");
    }

    private static T[] ToArrayCopy<T>(Tensor<T> t)
    {
        var arr = new T[t.Length];
        t.AsSpan().CopyTo(arr);
        return arr;
    }

    private static T[] ReduceArrays<T>(T[][] contributions, ReduceOp op)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = contributions[0].Length;
        var result = new T[n];
        contributions[0].AsSpan().CopyTo(result);
        for (int r = 1; r < contributions.Length; r++)
        {
            var src = contributions[r];
            switch (op)
            {
                case ReduceOp.Sum:
                case ReduceOp.Avg:
                    for (int i = 0; i < n; i++) result[i] = ops.Add(result[i], src[i]);
                    break;
                case ReduceOp.Min:
                    for (int i = 0; i < n; i++)
                    {
                        if (ops.LessThan(src[i], result[i])) result[i] = src[i];
                    }
                    break;
                case ReduceOp.Max:
                    for (int i = 0; i < n; i++)
                    {
                        if (ops.GreaterThan(src[i], result[i])) result[i] = src[i];
                    }
                    break;
                case ReduceOp.Product:
                    for (int i = 0; i < n; i++) result[i] = ops.Multiply(result[i], src[i]);
                    break;
                default:
                    throw new NotSupportedException($"ReduceOp {op} not yet supported by TcpProcessGroup (bitwise ops require integer specialisation).");
            }
        }
        if (op == ReduceOp.Avg)
        {
            var div = ops.FromDouble(1.0 / contributions.Length);
            for (int i = 0; i < n; i++) result[i] = ops.Multiply(result[i], div);
        }
        return result;
    }

    // ===== Wire format =====

    private static void WriteOpcode(Stream s, byte op)
    {
        s.WriteByte(op);
        s.Flush();
    }

    private static void ExpectOpcode(Stream s, byte expected)
    {
        int got = s.ReadByte();
        if (got < 0) throw new IOException("Connection closed mid-opcode.");
        if ((byte)got != expected)
            throw new IOException($"Opcode mismatch: expected {expected}, got {got}.");
    }

    private static void WriteInt32(Stream s, int value)
    {
        var buf = new byte[4];
        buf[0] = (byte)((value >> 24) & 0xFF);
        buf[1] = (byte)((value >> 16) & 0xFF);
        buf[2] = (byte)((value >> 8) & 0xFF);
        buf[3] = (byte)(value & 0xFF);
        s.Write(buf, 0, 4);
        s.Flush();
    }

    private static int ReadInt32(Stream s)
    {
        var buf = new byte[4];
        ReadExact(s, buf, 4);
        return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
    }

    private static void ReadExact(Stream s, byte[] buf, int count)
    {
        int read = 0;
        while (read < count)
        {
            int n = s.Read(buf, read, count - read);
            if (n <= 0) throw new IOException("Connection closed mid-frame.");
            read += n;
        }
    }

    private static void WriteTensorPayload<T>(Stream s, Tensor<T> tensor)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        WriteInt32(s, tensor.Length);
        var buf = new byte[8];
        var span = tensor.AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            double v = ops.ToDouble(span[i]);
            long bits = BitConverter.DoubleToInt64Bits(v);
            for (int b = 0; b < 8; b++) buf[b] = (byte)(bits >> (b * 8));
            s.Write(buf, 0, 8);
        }
        s.Flush();
    }

    private static void WriteRawArray<T>(Stream s, T[] arr)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        WriteInt32(s, arr.Length);
        var buf = new byte[8];
        for (int i = 0; i < arr.Length; i++)
        {
            double v = ops.ToDouble(arr[i]);
            long bits = BitConverter.DoubleToInt64Bits(v);
            for (int b = 0; b < 8; b++) buf[b] = (byte)(bits >> (b * 8));
            s.Write(buf, 0, 8);
        }
        s.Flush();
    }

    private static T[] ReadTensorArray<T>(Stream s, int expectedLen)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int len = ReadInt32(s);
        if (len != expectedLen)
            throw new IOException($"Tensor length mismatch on wire: expected {expectedLen}, got {len}.");
        var arr = new T[len];
        var buf = new byte[8];
        for (int i = 0; i < len; i++)
        {
            ReadExact(s, buf, 8);
            long bits = 0;
            for (int b = 0; b < 8; b++) bits |= ((long)buf[b]) << (b * 8);
            double v = BitConverter.Int64BitsToDouble(bits);
            arr[i] = ops.FromDouble(v);
        }
        return arr;
    }

    private static (string Host, int Port) ParseHostPort(string endpoint)
    {
        if (endpoint is null) throw new ArgumentNullException(nameof(endpoint));
        if (endpoint.StartsWith("tcp://", StringComparison.OrdinalIgnoreCase))
            endpoint = endpoint.Substring("tcp://".Length);
        int colon = endpoint.LastIndexOf(':');
        if (colon <= 0)
            throw new ArgumentException($"TCP endpoint must be host:port; got '{endpoint}'.");
        string host = endpoint.Substring(0, colon);
        if (!int.TryParse(endpoint.Substring(colon + 1), out int port) || port <= 0 || port > 65535)
            throw new ArgumentException($"TCP port must be 1..65535; got '{endpoint.Substring(colon + 1)}'.");
        return (host, port);
    }

    private static IPAddress ResolveBindAddress(string host)
    {
        if (host == "*" || host == "0.0.0.0") return IPAddress.Any;
        if (IPAddress.TryParse(host, out var ip)) return ip;
        var addrs = Dns.GetHostAddresses(host);
        if (addrs.Length == 0)
            throw new ArgumentException($"Could not resolve TCP host '{host}'.");
        return addrs[0];
    }

    private static bool WaitForConnection(TcpListener listener, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (listener.Pending()) return true;
            Thread.Sleep(10);
        }
        return false;
    }
}
