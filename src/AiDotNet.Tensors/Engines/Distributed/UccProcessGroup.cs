// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Distributed.Native;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// UCC-backed <see cref="IProcessGroup"/>. UCC (Unified Collective
/// Communication) is the OpenUCX collective library. PyTorch ships a
/// <c>ProcessGroupUCC</c> in recent builds; we expose the same surface
/// here so callers can flip between NCCL / MPI / UCC by changing one
/// line.
///
/// <para>UCC's collective lifecycle is more complex than MPI's: the
/// caller initialises a request via <c>ucc_collective_init</c>, posts
/// it via <c>ucc_collective_post</c>, polls via
/// <c>ucc_collective_test</c> until <c>OK</c>, and finally releases the
/// request via <c>ucc_collective_finalize</c>. The
/// <see cref="UccNative"/> bindings expose every entry point used by
/// PyTorch's UCC backend.</para>
///
/// <para><b>Bring-up</b> requires a UCC team handle that's typically
/// constructed via an out-of-band rendezvous (file / TCP / etcd). The
/// integration sketch lives in <see cref="UccNative"/>'s docs; the
/// constructor below throws with a clear pointer until the team-bring-
/// up code lands. Implementing the team-handshake is mechanical (see
/// PyTorch's <c>torch/csrc/distributed/c10d/ProcessGroupUCC.cpp</c>),
/// just lengthy.</para>
/// </summary>
public sealed class UccProcessGroup : IProcessGroup
{
    /// <inheritdoc/>
    public int WorldSize { get; }
    /// <inheritdoc/>
    public int Rank { get; }
    /// <inheritdoc/>
    public string Backend => "ucc";

    /// <summary>True when libucc is loadable.</summary>
    public static bool IsAvailable => UccNative.IsAvailable;

    /// <summary>Constructs a UCC process group. The team-bring-up
    /// (rendezvous → ucc_init → ucc_context_create → ucc_team_create_post →
    /// poll until ready) is a 200+ LOC follow-up; this constructor
    /// throws clearly so callers don't accidentally fall through to a
    /// non-functional collective. The <see cref="UccNative"/> bindings
    /// are live for users who want to wire the team handshake against
    /// their own rendezvous.</summary>
    public UccProcessGroup(int rank, int worldSize)
    {
        if (!UccNative.IsAvailable)
            throw new NotSupportedException("libucc is not loadable on this host. UCC ships only on Linux as part of the OpenUCX suite.");
        Rank = rank;
        WorldSize = worldSize;
        throw new NotSupportedException(
            "UccProcessGroup team bring-up not yet wired — see PyTorch's ProcessGroupUCC.cpp for the rendezvous → ucc_init → ucc_context_create → ucc_team_create_post sequence. " +
            "The native bindings (UccNative) are live; use NcclProcessGroup, MpiProcessGroup, or TcpProcessGroup until this lands.");
    }

    /// <inheritdoc/>
    public void AllReduce<T>(Tensor<T> tensor, ReduceOp op = ReduceOp.Sum) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void Reduce<T>(Tensor<T> tensor, int root, ReduceOp op = ReduceOp.Sum) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void Broadcast<T>(Tensor<T> tensor, int root) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void AllGather<T>(Tensor<T> input, IList<Tensor<T>> output) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void Gather<T>(Tensor<T> input, IList<Tensor<T>>? output, int root) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void Scatter<T>(IList<Tensor<T>>? input, Tensor<T> output, int root) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void ReduceScatter<T>(IList<Tensor<T>> input, Tensor<T> output, ReduceOp op = ReduceOp.Sum) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void Send<T>(Tensor<T> tensor, int dst, int tag = 0) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void Recv<T>(Tensor<T> tensor, int src, int tag = 0) => throw new NotSupportedException();
    /// <inheritdoc/>
    public Tensor<T> RecvDiscoverShape<T>(int src, int tag = 0) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void Barrier() => throw new NotSupportedException();
    /// <inheritdoc/>
    public IProcessGroup? NewGroup(IReadOnlyList<int> ranks) => throw new NotSupportedException();
    /// <inheritdoc/>
    public IProcessGroup SplitGroup(int color, int? key = null) => throw new NotSupportedException();
    /// <inheritdoc/>
    public void Dispose() { /* no-op */ }
}
