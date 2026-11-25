using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend that serves allocations made between a compiled plan's steps from that plan's idle step memory, the way
/// PyTorch's caching allocator serves validation and eager inference from the blocks a training step freed.
/// </summary>
/// <remarks>
/// <para>
/// A captured step graph bakes in its scratch addresses, so the plan cannot free them between steps. Lending is the
/// equivalent: while the plan is between steps, the backend's ordinary allocations (outside every capture) may be
/// carved out of the buffers the plan lends. The plan reclaims them before its next step runs. A borrowed allocation
/// still alive then is moved to memory of its own (a stream-ordered device copy, and the buffer object keeps its
/// identity with the new address), so the next replay may overwrite the lent memory.
/// </para>
/// <para>
/// All of this is ordered on the backend's compute stream, where replays also run: memory lent after a replay is
/// first used by work queued after it, and a replay after reclaim runs after every borrower's earlier work.
/// </para>
/// </remarks>
internal interface IIdleDeviceMemoryLender
{
    /// <summary>
    /// Lends <paramref name="buffers"/> (owned by <paramref name="owner"/>, whose contents nothing reads before the
    /// owner next writes them) until <see cref="ReclaimIdle"/> with the same owner.
    /// </summary>
    void LendIdle(object owner, IReadOnlyCollection<IGpuBuffer> buffers);

    /// <summary>
    /// As <see cref="LendIdle"/>, but the buffers are only collected (<paramref name="collect"/>, on the allocating
    /// thread) when an allocation first needs memory the backend's pool cannot serve; null lends nothing. Finding a
    /// plan's idle buffers walks its whole step, so a plan whose steps nothing allocates between pays nothing.
    /// </summary>
    void LendIdleOnDemand(object owner, System.Func<IReadOnlyCollection<IGpuBuffer>?> collect);

    /// <summary>Takes back everything <paramref name="owner"/> lent, moving any allocation still using it.</summary>
    void ReclaimIdle(object owner);

    /// <summary>
    /// Moves every allocation still using lent memory to memory of its own, whoever lent it. A capture's preparation
    /// calls this before it reads device addresses into tables its graph keeps.
    /// </summary>
    void MoveLiveLoans();
}

/// <summary>
/// A lender whose captures must not bake in a lent address. A plan opens <see cref="SuspendLoansForCapture"/> before it
/// prepares a capture, not only around the capture itself, because the preparation records device addresses too: the
/// pre-pass that binds the step's buffers, the step I/O, the optimizer's and the loss scaler's pointer tables.
/// </summary>
internal interface IIdleLoanCaptureGuard
{
    /// <summary>
    /// Moves every live loan to memory of its own and serves no new loan until the result is disposed (nests with the
    /// capture's own suspension).
    /// </summary>
    System.IDisposable SuspendLoansForCapture();
}
