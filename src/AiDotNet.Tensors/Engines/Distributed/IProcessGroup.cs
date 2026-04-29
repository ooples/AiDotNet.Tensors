// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// Reduction operation applied to a collective (all_reduce, reduce, etc).
/// Mirrors <c>torch.distributed.ReduceOp</c>.
/// </summary>
public enum ReduceOp
{
    /// <summary>Sum across ranks.</summary>
    Sum,
    /// <summary>Mean across ranks (= Sum / WorldSize).</summary>
    Avg,
    /// <summary>Element-wise minimum across ranks.</summary>
    Min,
    /// <summary>Element-wise maximum across ranks.</summary>
    Max,
    /// <summary>Element-wise product across ranks.</summary>
    Product,
    /// <summary>Bitwise AND (integer types only).</summary>
    BAnd,
    /// <summary>Bitwise OR (integer types only).</summary>
    BOr,
    /// <summary>Bitwise XOR (integer types only).</summary>
    BXor,
}

/// <summary>
/// Process group — the abstraction that turns a set of cooperating processes
/// (or threads, in the in-process backend) into a collective-operation
/// surface. Mirrors <c>torch.distributed.ProcessGroup</c>.
///
/// <para>Backends implement this interface to provide the wire-level
/// transport (NCCL on GPU, Gloo over TCP on CPU, in-process for tests).
/// User code calls the collective primitives <see cref="AllReduce{T}"/>,
/// <see cref="Broadcast{T}"/>, etc. without caring about the underlying
/// transport.</para>
///
/// <para><b>Determinism:</b> all collectives are synchronous from the
/// caller's perspective — <see cref="AllReduce{T}"/> returns when every
/// rank has completed the reduction. Async overloads with explicit
/// <see cref="IDistributedHandle"/> wait handles are provided where the
/// pattern is common (DDP gradient overlap).</para>
/// </summary>
public interface IProcessGroup : IDisposable
{
    /// <summary>Total ranks in the group (0 ≤ rank &lt; WorldSize).</summary>
    int WorldSize { get; }

    /// <summary>This process's rank within the group.</summary>
    int Rank { get; }

    /// <summary>Backend label — "in-process", "tcp", "nccl", "gloo".
    /// Surfaced for diagnostics + dispatch decisions.</summary>
    string Backend { get; }

    // ────────────────────────────────────────────────────────────────────
    // Synchronous collective ops — every rank in the group must call the
    // same op with the same shape/dtype/op or behaviour is undefined.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>All-reduce: every rank ends up with the per-element
    /// reduction of all ranks' input tensors. <paramref name="tensor"/>
    /// is updated in place.</summary>
    void AllReduce<T>(Tensor<T> tensor, ReduceOp op = ReduceOp.Sum);

    /// <summary>Reduce: rank <paramref name="root"/> ends up with the
    /// reduction; other ranks' tensors are left undefined (PyTorch
    /// matches: only root has the result).</summary>
    void Reduce<T>(Tensor<T> tensor, int root, ReduceOp op = ReduceOp.Sum);

    /// <summary>Broadcast: rank <paramref name="root"/>'s tensor is copied
    /// to every other rank. Non-root ranks pass a tensor with the same
    /// shape/dtype as root; its contents are overwritten.</summary>
    void Broadcast<T>(Tensor<T> tensor, int root);

    /// <summary>All-gather: every rank ends up with a list of WorldSize
    /// tensors, one from each rank. <paramref name="output"/> must have
    /// length <see cref="WorldSize"/>; each entry is overwritten with the
    /// matching rank's input.</summary>
    void AllGather<T>(Tensor<T> input, IList<Tensor<T>> output);

    /// <summary>Gather: rank <paramref name="root"/> ends up with the
    /// list of WorldSize tensors. Non-root ranks may pass null/empty
    /// for <paramref name="output"/>.</summary>
    void Gather<T>(Tensor<T> input, IList<Tensor<T>>? output, int root);

    /// <summary>Scatter: rank <paramref name="root"/> distributes a list
    /// of WorldSize tensors; rank R receives <c>input[R]</c>. Non-root
    /// ranks pass null/empty <paramref name="input"/>.</summary>
    void Scatter<T>(IList<Tensor<T>>? input, Tensor<T> output, int root);

    /// <summary>Reduce-scatter: input is a list of WorldSize tensors per
    /// rank; the per-rank reduction across the list is scattered back so
    /// rank R ends up with the reduction of slice R.</summary>
    void ReduceScatter<T>(IList<Tensor<T>> input, Tensor<T> output, ReduceOp op = ReduceOp.Sum);

    /// <summary>Point-to-point send. Pairs with <see cref="Recv{T}"/> on
    /// rank <paramref name="dst"/>.</summary>
    void Send<T>(Tensor<T> tensor, int dst, int tag = 0);

    /// <summary>Point-to-point recv. Pairs with <see cref="Send{T}"/> on
    /// rank <paramref name="src"/>.</summary>
    void Recv<T>(Tensor<T> tensor, int src, int tag = 0);

    /// <summary>Synchronization barrier — returns when every rank has
    /// reached this call.</summary>
    void Barrier();

    // ────────────────────────────────────────────────────────────────────
    // Sub-group construction — split a large group into smaller ones
    // (e.g. tensor-parallel × data-parallel grids).
    // ────────────────────────────────────────────────────────────────────

    /// <summary>Creates a sub-group containing only the specified ranks.
    /// Returns null on ranks not in <paramref name="ranks"/>; the caller
    /// must check before using the returned group.</summary>
    IProcessGroup? NewGroup(IReadOnlyList<int> ranks);

    /// <summary>Splits the current group by a per-rank color: ranks with
    /// the same color end up in the same sub-group. Used for tensor-
    /// parallel × data-parallel grid decomposition.</summary>
    IProcessGroup SplitGroup(int color, int? key = null);
}

/// <summary>
/// Async-completion handle returned by non-blocking collective overloads.
/// Pattern: kick off the collective, do unrelated work, then call
/// <see cref="Wait"/> to block until completion. Used by DDP to overlap
/// gradient all-reduce with backward computation of subsequent layers.
/// </summary>
public interface IDistributedHandle : IDisposable
{
    /// <summary>Blocks until the underlying collective finishes.</summary>
    void Wait();

    /// <summary>True if the collective has already completed.</summary>
    bool IsCompleted { get; }
}
