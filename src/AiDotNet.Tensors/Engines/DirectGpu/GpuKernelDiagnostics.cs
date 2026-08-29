// Copyright (c) AiDotNet. All rights reserved.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu
{
    /// <summary>
    /// Launch-time diagnostics for GPU kernels — the library's own and any a caller adds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// WHAT THIS EXISTS TO FIX. A device launch is asynchronous and unchecked by construction: an
    /// out-of-range write in one kernel is not reported at that kernel, it is reported at whatever
    /// synchronises next, and a bad handle or arg is not reported at all — the driver faults and the
    /// process dies with no managed frame to blame. Both failure modes are routinely observed as a
    /// crash in code that had nothing to do with the cause, and in CI as a shard that dies having
    /// written no results at all.
    /// </para>
    /// <para>
    /// <see cref="GpuLaunchProbe"/> already answers "did this op reach the device". This answers the
    /// three questions that come after a crash: <i>was the launch itself well formed</i>, <i>which
    /// launch was it</i>, and <i>what ran just before it</i>.
    /// </para>
    /// <list type="bullet">
    /// <item><b>Validation</b> (always on, cached per kernel, an int compare per launch): a released
    /// handle, an argument count that disagrees with the kernel's own
    /// <c>CL_KERNEL_NUM_ARGS</c>, a work-group larger than the kernel admits, a local-memory request
    /// the device cannot satisfy, a global size that is not a multiple of the local size.</item>
    /// <item><b>The journal</b> (always on, allocation-free): a fixed ring of the most recent
    /// launches. When the process dies this is the only surviving evidence of what it was doing, and
    /// it can be written to disk from a crash handler or a test teardown.</item>
    /// <item><b>Deep checks</b> (opt-in, <c>AIDOTNET_GPU_KERNEL_DIAGNOSTICS=1</c>): buffer capacity
    /// against the range a launch will address, and assertions a kernel declares about its own
    /// launch shape — for instance that its work-group size is a power of two, which several tree
    /// reductions require and none previously stated.</item>
    /// <item><b>Synchronous mode</b> (opt-in, <c>AIDOTNET_GPU_SYNC_LAUNCHES=1</c>): finish after
    /// every launch so an asynchronous fault is attributed to the launch that caused it instead of
    /// the next unrelated one. Slow by design; this is the mode you turn on to find a crash.</item>
    /// </list>
    /// <para>
    /// The validation entry points are public so a caller extending the library with their own
    /// kernels gets the same checks, rather than the library validating only what it happens to ship.
    /// </para>
    /// </remarks>
    public static class GpuKernelDiagnostics
    {
        private const int JournalCapacity = 64;

        private static readonly LaunchRecord[] _journal = new LaunchRecord[JournalCapacity];
        private static long _sequence = -1;

        private static readonly bool _deepChecks =
            IsSet("AIDOTNET_GPU_KERNEL_DIAGNOSTICS");

        private static readonly bool _synchronousLaunches =
            IsSet("AIDOTNET_GPU_SYNC_LAUNCHES");

        private static bool IsSet(string name)
        {
            var value = Environment.GetEnvironmentVariable(name);
            return !string.IsNullOrEmpty(value)
                   && !string.Equals(value, "0", StringComparison.Ordinal)
                   && !string.Equals(value, "false", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Whether the opt-in deep checks are enabled.</summary>
        public static bool DeepChecksEnabled => _deepChecks;

        /// <summary>
        /// Whether every launch should be followed by a device synchronise so asynchronous faults
        /// are attributed to the launch that caused them.
        /// </summary>
        public static bool SynchronousLaunches => _synchronousLaunches;

        private struct LaunchRecord
        {
            public string? Kernel;
            public long GlobalSize;
            public long LocalSize;
            public int ArgCount;
            public int ManagedThreadId;
            public long Ticks;
        }

        /// <summary>Records a launch in the ring. Allocation-free on the hot path.</summary>
        public static void RecordLaunch(string kernelName, long globalSize, long localSize, int argCount)
        {
            long slot = Interlocked.Increment(ref _sequence);
            int index = (int)(((slot % JournalCapacity) + JournalCapacity) % JournalCapacity);

            _journal[index] = new LaunchRecord
            {
                Kernel = kernelName,
                GlobalSize = globalSize,
                LocalSize = localSize,
                ArgCount = argCount,
                ManagedThreadId = Environment.CurrentManagedThreadId,
                Ticks = DateTime.UtcNow.Ticks,
            };
        }

        /// <summary>The most recent launches, oldest first. Empty when nothing has launched.</summary>
        public static IReadOnlyList<string> RecentLaunches()
        {
            long last = Interlocked.Read(ref _sequence);
            if (last < 0) return Array.Empty<string>();

            int count = (int)Math.Min(last + 1, JournalCapacity);
            var lines = new List<string>(count);

            for (long slot = Math.Max(0, last - count + 1); slot <= last; slot++)
            {
                int index = (int)(((slot % JournalCapacity) + JournalCapacity) % JournalCapacity);
                var record = _journal[index];
                if (record.Kernel is null) continue;

                lines.Add(string.Format(
                    CultureInfo.InvariantCulture,
                    "#{0} {1} global={2} local={3} args={4} thread={5} at={6:O}",
                    slot, record.Kernel, record.GlobalSize, record.LocalSize, record.ArgCount,
                    record.ManagedThreadId, new DateTime(record.Ticks, DateTimeKind.Utc)));
            }

            return lines;
        }

        /// <summary>
        /// Writes the journal to <paramref name="path"/>. Safe to call from a crash handler or a
        /// test teardown; never throws.
        /// </summary>
        public static void DumpTo(string path)
        {
            try
            {
                var text = new StringBuilder();
                text.AppendLine("# GPU launch journal (most recent last)");
                foreach (var line in RecentLaunches()) text.AppendLine(line);
                File.WriteAllText(path, text.ToString());
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            catch (ArgumentException) { }
        }

        /// <summary>
        /// Validates a launch's shape before it reaches the driver.
        /// </summary>
        /// <param name="kernelName">Kernel name, for the message.</param>
        /// <param name="handleIsValid">False when the kernel handle has been released.</param>
        /// <param name="stagedArgCount">Arguments the caller has set.</param>
        /// <param name="declaredArgCount">The kernel's own argument count, or -1 if unknown.</param>
        /// <param name="globalSize">Total work items.</param>
        /// <param name="localSize">Work-group size.</param>
        /// <param name="maxWorkGroupSize">The kernel's maximum work-group size, or -1 if unknown.</param>
        /// <param name="requiresPowerOfTwoWorkGroup">
        /// Whether the kernel's algorithm requires a power-of-two work-group. Tree reductions do;
        /// giving one an odd work-group silently drops a lane, which is invisible to Min and Max
        /// because they are idempotent and shows up only in sums.
        /// </param>
        public static void ValidateLaunch(
            string kernelName,
            bool handleIsValid,
            int stagedArgCount,
            int declaredArgCount,
            long globalSize,
            long localSize,
            long maxWorkGroupSize = -1,
            bool requiresPowerOfTwoWorkGroup = false)
        {
            if (!handleIsValid)
            {
                throw new ObjectDisposedException(
                    kernelName,
                    $"GPU kernel '{kernelName}' was launched after its handle was released.");
            }

            if (localSize <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(localSize),
                    $"{kernelName}: work-group size must be positive (got {localSize}).");
            }

            if (globalSize % localSize != 0)
            {
                throw new ArgumentException(
                    $"{kernelName}: global size {globalSize} is not a multiple of work-group size "
                        + $"{localSize}. The driver rejects this, and rounding it up silently would "
                        + "run work items the kernel may not guard against.",
                    nameof(globalSize));
            }

            // DELIBERATELY BEHIND THE FLAG, not always on. OpenCL arguments persist on the kernel
            // object between launches, so a launcher may legitimately set a subset and rely on the
            // rest still holding the values it set earlier. Failing that by default would break
            // working callers -- including any the library does not ship -- to catch a fault that,
            // by construction, only matters while you are hunting one.
            if (_deepChecks && declaredArgCount >= 0 && stagedArgCount != declaredArgCount)
            {
                throw new ArgumentException(
                    $"{kernelName}: {stagedArgCount} argument(s) were set but the kernel declares "
                        + $"{declaredArgCount}. Launching with a mismatched set reads whatever the "
                        + "previous launch left in the unset slots. (If this launcher intentionally "
                        + "relies on argument persistence, it is not a defect — it is only reported "
                        + "under AIDOTNET_GPU_KERNEL_DIAGNOSTICS.)",
                    nameof(stagedArgCount));
            }

            if (maxWorkGroupSize >= 0 && localSize > maxWorkGroupSize)
            {
                throw new ArgumentException(
                    $"{kernelName}: work-group size {localSize} exceeds the {maxWorkGroupSize} this "
                        + "kernel admits on this device.",
                    nameof(localSize));
            }

            if (requiresPowerOfTwoWorkGroup && !IsPowerOfTwo(localSize))
            {
                throw new ArgumentException(
                    $"{kernelName}: work-group size {localSize} is not a power of two, which this "
                        + "kernel's reduction requires. A halving tree over a non-power-of-two group "
                        + "drops its top lane — invisible in min/max, wrong in sums.",
                    nameof(localSize));
            }
        }

        /// <summary>True for 1, 2, 4, 8, ... — see the reduction note on <see cref="ValidateLaunch"/>.</summary>
        public static bool IsPowerOfTwo(long value) => value > 0 && (value & (value - 1)) == 0;

        /// <summary>
        /// Checks that a buffer can hold every element a launch will address.
        /// </summary>
        /// <remarks>
        /// Deep check: callers pass the highest index the kernel will touch, which they know and the
        /// driver does not. An out-of-range device write corrupts whatever is next in device memory
        /// and is reported, if at all, by an unrelated later operation.
        /// </remarks>
        public static void ValidateCapacity(
            string kernelName, string bufferName, long bufferElements, long addressedElements)
        {
            if (!_deepChecks) return;

            if (addressedElements > bufferElements)
            {
                throw new ArgumentException(
                    $"{kernelName}: '{bufferName}' holds {bufferElements} element(s) but the launch "
                        + $"addresses {addressedElements}. The excess would be written past the end "
                        + "of device memory the buffer owns.",
                    bufferName);
            }
        }

        /// <summary>Formats the journal for inclusion in an exception or a test failure.</summary>
        public static string DescribeRecentLaunches(int maximum = 12)
        {
            var recent = RecentLaunches();
            if (recent.Count == 0) return "(no GPU launches recorded)";

            var text = new StringBuilder();
            int skip = Math.Max(0, recent.Count - maximum);
            for (int i = skip; i < recent.Count; i++) text.AppendLine("  " + recent[i]);
            return text.ToString();
        }
    }
}
