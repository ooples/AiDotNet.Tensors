using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace AiDotNet.Tensors.Engines.BlasManaged.Pool;

/// <summary>
/// Per-worker state for <see cref="StreamingWorkerPool"/>. Cache-line-aligned
/// (64 bytes on x64) to prevent false sharing between worker threads — each
/// worker's seq counter, action reference, and chunk range live on a private
/// cache line so updates don't bounce.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 64)]
internal struct WorkerSlot
{
    /// <summary>Dispatch generation counter. Incremented by caller on new work; worker reads to detect dispatch.</summary>
    [FieldOffset(0)] public long Seq;
    /// <summary>Body to execute for the worker's assigned chunk(s). Worker reads after observing Seq increment.</summary>
    [FieldOffset(8)] public Action<int>? Body;
    /// <summary>Inclusive chunk start.</summary>
    [FieldOffset(16)] public int ChunkStart;
    /// <summary>Exclusive chunk end.</summary>
    [FieldOffset(20)] public int ChunkEnd;
    /// <summary>0 = worker is hot (spinning); 1 = worker is parked on ParkEvent.</summary>
    [FieldOffset(24)] public int ParkPending;
    /// <summary>Park back-stop after spin loop exhausts.</summary>
    [FieldOffset(32)] public ManualResetEventSlim? ParkEvent;
}
