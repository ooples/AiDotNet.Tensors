// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// The tickets of the in-kernel split-K reductions: a split kernel's last block per output tile (the one whose atomic
// increment of that tile's ticket reaches the split count) sums every split's partial for the tile in split order and
// resets the ticket. That replaces a separate reduce launch per split convolution, which on a ResNet-18 step was ~24
// launches of ~2 us each, mostly launch latency.
public sealed partial class CudaBackend
{
    // One zeroed uint per output tile, shared by every split kernel: they all run on the backend's one compute stream,
    // one after another, and each leaves its tickets at zero.
    private IGpuBuffer? _splitKTickets;

    // 64K tiles: far more than any split convolution here launches (ResNet-18's largest grid is 72 x 4).
    private const int SplitKTicketCapacity = 1 << 16;

    /// <summary>Kill switch for A/B and tests: AIDOTNET_SPLITK_INKERNEL=0 keeps the separate reduce launches.</summary>
    internal static bool SplitKInKernel { get; set; } = Environment.GetEnvironmentVariable("AIDOTNET_SPLITK_INKERNEL") != "0";

    /// <summary>
    /// The zeroed tickets for a split kernel over <paramref name="tiles"/> output tiles, or null (the caller then launches
    /// its separate reduce): switched off, too many tiles, or not yet allocated during a capture, where allocating would
    /// abort it (the eager warmup steps allocate it first).
    /// </summary>
    private IGpuBuffer? TrySplitKTickets(long tiles)
    {
        if (!SplitKInKernel || tiles <= 0 || tiles > SplitKTicketCapacity) return null;
        if (_splitKTickets is { } existing) return existing;
        if (IsStreamCapturing()) return null;
        var tickets = AllocateBuffer(SplitKTicketCapacity);
        ((IDeviceZeroFill)this).ZeroBuffer(tickets, SplitKTicketCapacity);
        _splitKTickets = tickets;
        return tickets;
    }
}
