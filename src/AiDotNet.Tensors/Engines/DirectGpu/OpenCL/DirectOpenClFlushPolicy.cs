// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// When to call clFlush. An OpenCL driver may hold enqueued commands until a flush or a blocking call: on AMD's
    /// Windows driver a ResNet-50 forward sat in the queue while the host enqueued all of it, and ran only when the
    /// result was read, so host enqueue time and device time added up instead of overlapping (12.3 ms per Predict,
    /// half of it with the device idle). Flushing after every command costs host time instead (12.5 ms).
    /// </summary>
    /// <remarks>
    /// After the host waits on the device (the device is then idle), the next command is flushed at once so the device
    /// starts, then the flush interval doubles up to <see cref="MaxCommandsPerFlush"/>, by which time the device has
    /// enough queued to stay busy while the host enqueues the next batch. State is per thread, like the command queues
    /// (each host thread gets its own queue from <see cref="DirectOpenClContext.CommandQueue"/>).
    /// </remarks>
    internal static class DirectOpenClFlushPolicy
    {
        /// <summary>The longest interval between flushes, in commands. 8, 16 and 32 measured alike; 64 lost 1 ms.</summary>
        internal const int MaxCommandsPerFlush = 16;

        [ThreadStatic] private static int t_unflushed;
        [ThreadStatic] private static int t_interval;
        [ThreadStatic] private static int t_held;

        /// <summary>
        /// Counts one command submitted on this thread; true when the queue should be flushed now. The ramp advances
        /// only on <see cref="OnFlushed"/>, so a flush that failed is due again at the next command. Never true while
        /// <see cref="Hold"/> is in force.
        /// </summary>
        internal static bool OnSubmitted() => ++t_unflushed >= Math.Max(1, t_interval) && t_held == 0;

        /// <summary>
        /// Defers this thread's flushes until the matching <see cref="Release"/>, so commands enqueued in between
        /// reach the device in one flush the caller makes (a resident program's chain). Nests.
        /// </summary>
        internal static void Hold() => t_held++;

        /// <summary>Ends a <see cref="Hold"/>. A command enqueued while held is still counted: the next one after
        /// the release flushes on the usual schedule unless the caller flushed.</summary>
        internal static void Release()
        {
            if (t_held == 0) throw new InvalidOperationException("Release without a matching Hold.");
            t_held--;
        }

        /// <summary>The queue was flushed: the count restarts and the interval doubles, up to the cap.</summary>
        internal static void OnFlushed()
        {
            t_unflushed = 0;
            t_interval = Math.Min(Math.Max(1, t_interval) * 2, MaxCommandsPerFlush);
        }

        /// <summary>The host waited for the device on this thread: the device is idle, so restart the ramp.</summary>
        internal static void OnHostWaited()
        {
            t_unflushed = 0;
            t_interval = 1;
        }
    }
}
