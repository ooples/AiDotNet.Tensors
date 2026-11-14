// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Pure P/Invoke OpenCL context - no managed GPU runtime dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// OpenCL context wrapper using pure P/Invoke. No managed GPU runtime dependency.
    ///
    /// <para><b>Threading model</b> (issue #414): the OpenCL context is shared across
    /// all host threads (one per device), but each host thread that touches
    /// <see cref="CommandQueue"/> or <see cref="ProfilingCommandQueue"/> gets its
    /// own per-thread <c>cl_command_queue</c> lazily created on first access. This
    /// eliminates the host-side enqueue serialization point that previously made
    /// <c>Parallel.ForEach(... IEngine work ...)</c> slower than serial.</para>
    ///
    /// <para>Per-thread queues share the same OpenCL context and device, so device
    /// buffers / programs / kernels are interchangeable across them — but each
    /// thread's enqueues no longer block on another thread's clEnqueue*. The GPU
    /// driver schedules across queues independently; on small consumer devices the
    /// hardware may still serialize kernel execution, but the host-side enqueue
    /// path becomes truly parallel.</para>
    /// </summary>
    internal sealed class DirectOpenClContext : IDisposable
    {
        /// <summary>Contexts this process created and has not yet disposed.</summary>
        private static int s_liveContexts;

        /// <summary>
        /// With <c>AIDOTNET_OPENCL_TRACE_CONTEXTS=1</c>, where each live context was created, so a failure to create one
        /// (the driver's host budget exhausted by contexts nobody disposed: each holds its compiled programs, ~100 MB)
        /// names the code that leaked them.
        /// </summary>
        private static readonly ConcurrentDictionary<IntPtr, string>? s_creationSites = CreateCreationSiteTrace();

        // With tracing on, the contexts still alive when the process exits (never disposed) are also written to
        // %TEMP%/aidotnet-opencl-live-contexts-<pid>.txt, so a leak is found even in a run where it caused no failure.
        private static ConcurrentDictionary<IntPtr, string>? CreateCreationSiteTrace()
        {
            if (Environment.GetEnvironmentVariable("AIDOTNET_OPENCL_TRACE_CONTEXTS") != "1") return null;
            var sites = new ConcurrentDictionary<IntPtr, string>();
            AppDomain.CurrentDomain.ProcessExit += (_, _) =>
            {
                try
                {
                    var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
                        $"aidotnet-opencl-live-contexts-{System.Diagnostics.Process.GetCurrentProcess().Id}.txt");
                    System.IO.File.WriteAllText(path, LiveContextReport());
                }
                catch (System.IO.IOException) { }
                catch (UnauthorizedAccessException) { }
            };
            return sites;
        }

        /// <summary>Contexts this process created and has not yet disposed.</summary>
        internal static int LiveContextCount => Volatile.Read(ref s_liveContexts);

        private static string LiveContextReport()
        {
            var report = $" ({LiveContextCount} OpenCL context(s) of this process are still alive";
            if (s_creationSites is null)
                return report + "; set AIDOTNET_OPENCL_TRACE_CONTEXTS=1 to record where each was created)";
            var sites = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var site in s_creationSites.Values) sites[site] = sites.TryGetValue(site, out int n) ? n + 1 : 1;
            var lines = new List<string>();
            foreach (var pair in sites) lines.Add($"{pair.Value} x {pair.Key}");
            lines.Sort(StringComparer.Ordinal);
            return report + "; created at:" + Environment.NewLine + string.Join(Environment.NewLine, lines) + ")";
        }

        private IntPtr _context;
        private OpenClNativeContextHandle? _native;
        private ThreadLocal<IntPtr>? _threadCommandQueue;
        private ThreadLocal<IntPtr>? _threadProfilingCommandQueue;
        private IntPtr _device;
        private IntPtr _platform;
        private bool _profilingSupported;
        private bool _disposed;
        private readonly object _retirementLock = new object();
        private readonly List<RetiredMemoryObject> _retiredMemoryObjects = new List<RetiredMemoryObject>();
        private readonly List<PendingHostTransfer> _pendingHostTransfers = new List<PendingHostTransfer>();
        private readonly ConcurrentDictionary<IntPtr, byte> _completedQueues = new ConcurrentDictionary<IntPtr, byte>();

        /// <summary>
        /// Every command queue not yet released, across all contexts. A buffer remembers the last queue it ran on and
        /// is retired behind a marker enqueued there; if that queue belonged to an engine since disposed, enqueueing on
        /// the released handle faults the process (0xC0000005, seen when a collected buffer from an earlier engine was
        /// freed by the next allocation). A retirement whose queue is not live releases at once, which OpenCL makes
        /// safe: a released memory object is deleted only after the commands using it complete.
        /// </summary>
        internal static readonly ConcurrentDictionary<IntPtr, byte> LiveQueues = new ConcurrentDictionary<IntPtr, byte>();
        // Per-context ordering complements OpenClNativeBindings' process-wide native creation
        // gate. It prevents duplicate first-touch work within this context, while the bindings
        // gate also protects different contexts, profiling probes, and explicit streams from an
        // AMD driver race. Neither lock is present on the steady-state enqueue path.
        private readonly object _queueCreateLock = new object();

        private readonly struct RetiredMemoryObject
        {
            internal readonly IntPtr MemoryObject;
            internal readonly IntPtr CompletionEvent;
            internal readonly long ByteSize;

            internal RetiredMemoryObject(IntPtr memoryObject, IntPtr completionEvent, long byteSize)
            {
                MemoryObject = memoryObject;
                CompletionEvent = completionEvent;
                ByteSize = byteSize;
            }
        }

        private readonly struct PendingHostTransfer
        {
            internal readonly IntPtr CompletionEvent;
            internal readonly GCHandle PinnedMemory;

            internal PendingHostTransfer(IntPtr completionEvent, GCHandle pinnedMemory)
            {
                CompletionEvent = completionEvent;
                PinnedMemory = pinnedMemory;
            }
        }

        public IntPtr Context => _context;

        /// <summary>
        /// Gets the OpenCL command queue for the <b>current host thread</b>. Each
        /// thread that touches this property gets its own lazily-created
        /// <c>cl_command_queue</c> on the shared <see cref="Context"/> + <see cref="Device"/>
        /// (issue #414). The returned <c>IntPtr</c> is valid only on the thread that
        /// fetched it; do not store and use it from another thread — fetch fresh
        /// per thread.
        /// </summary>
        public IntPtr CommandQueue
        {
            get
            {
                if (_disposed) throw new ObjectDisposedException(nameof(DirectOpenClContext));
                var tl = _threadCommandQueue
                    ?? throw new GpuCapabilityUnavailableException("DirectOpenClContext not initialized.");
                return tl.Value;
            }
        }

        /// <summary>
        /// Gets a profiling-enabled command queue for performance measurements.
        /// Uses CL_QUEUE_PROFILING_ENABLE to allow clGetEventProfilingInfo.
        /// Per-thread (issue #414): each host thread that touches this property gets
        /// its own lazily-created profiling queue, mirroring <see cref="CommandQueue"/>.
        /// Returns <see cref="IntPtr.Zero"/> if the device or driver refused to
        /// create a profiling queue at init time.
        /// </summary>
        public IntPtr ProfilingCommandQueue
        {
            get
            {
                if (_disposed) throw new ObjectDisposedException(nameof(DirectOpenClContext));
                if (!_profilingSupported) return IntPtr.Zero;
                var tl = _threadProfilingCommandQueue
                    ?? throw new GpuCapabilityUnavailableException("DirectOpenClContext not initialized.");
                return tl.Value;
            }
        }

        /// <summary>
        /// Gets whether profiling is enabled on this context.
        /// </summary>
        public bool IsProfilingEnabled => _profilingSupported;

        public IntPtr Device => _device;

        internal bool IsQueueKnownComplete(IntPtr queue)
            => queue != IntPtr.Zero && _completedQueues.ContainsKey(queue);

        internal void RegisterMemoryObject(IDirectOpenClMemoryObject memory)
        {
            ReapCompletedResources();
            DirectOpenClSubmission.Register(memory);
            _native?.Track(memory.NativeHandle, OpenClChildKind.MemObject);
        }

        /// <summary>
        /// Retains a pinned managed transfer buffer until its non-blocking OpenCL command completes. A
        /// <c>fixed</c> statement that ends immediately after <c>clEnqueueRead/WriteBuffer</c> is not a valid
        /// asynchronous lifetime: the GC may move that array while the device still owns its pointer.
        /// </summary>
        internal void TrackHostTransfer(IntPtr completionEvent, GCHandle pinnedMemory)
        {
            if (completionEvent == IntPtr.Zero)
            {
                if (pinnedMemory.IsAllocated) pinnedMemory.Free();
                throw new ArgumentException("An asynchronous host transfer requires a completion event.", nameof(completionEvent));
            }

            lock (_retirementLock)
                _pendingHostTransfers.Add(new PendingHostTransfer(completionEvent, pinnedMemory));
            ReapCompletedResources();
        }

        /// <summary>
        /// Detaches a native memory object from future submissions and retires it behind a marker on its
        /// last-use queue. This is deliberately asynchronous: pool overflow must not turn into a hidden
        /// <c>clFinish</c> on the allocation hot path.
        /// </summary>
        internal void RetireMemoryObject(
            IDirectOpenClMemoryObject memory,
            IntPtr memoryObject,
            long byteSize)
        {
            if (memoryObject == IntPtr.Zero) return;
            // Claim it: a buffer resurrected by a finalizer can be disposed after its context was reclaimed, which
            // already released it (see OpenClNativeContextHandle).
            if (!ClaimNative(memoryObject))
            {
                lock (DirectOpenClSubmission.Gate) DirectOpenClSubmission.Unregister(memoryObject, memory);
                return;
            }

            bool releaseImmediately = false;
            lock (DirectOpenClSubmission.Gate)
            {
                DirectOpenClSubmission.Unregister(memoryObject, memory);
                IntPtr lastQueue = memory.LastSubmissionQueue;
                memory.LastSubmissionQueue = IntPtr.Zero;

                if (_disposed || lastQueue == IntPtr.Zero || IsQueueKnownComplete(lastQueue) || !LiveQueues.ContainsKey(lastQueue))
                {
                    releaseImmediately = true;
                }
                else
                {
                    int err = OpenClNativeBindings.EnqueueMarkerWithWaitList(
                        lastQueue, 0, null, out IntPtr completionEvent);
                    if (err == OpenClNativeBindings.CL_SUCCESS && completionEvent != IntPtr.Zero)
                    {
                        // Start the marker without waiting. Reaping is performed opportunistically by later
                        // allocations/retirements and deterministically during context disposal.
                        OpenClNativeBindings.Flush(lastQueue);
                        lock (_retirementLock)
                            _retiredMemoryObjects.Add(
                                new RetiredMemoryObject(memoryObject, completionEvent, byteSize));
                    }
                    else
                    {
                        // Event creation failed, so there is no safe asynchronous ownership token. Finish only
                        // this exceptional reclamation path rather than risking a use-after-release.
                        OpenClNativeBindings.Finish(lastQueue);
                        releaseImmediately = true;
                    }
                }
            }

            if (releaseImmediately)
                ReleaseMemoryObject(memoryObject, byteSize);
            else
                ReapCompletedResources();
        }

        /// <summary>
        /// Waits for every retired memory object's completion marker and releases its allocation. An explicit cache
        /// release calls this: a buffer the pool drained while queued work still used it is retired behind a marker, and
        /// without the wait its memory stays reserved after the release reports it freed.
        /// </summary>
        internal void WaitAndReleaseRetiredMemory()
        {
            lock (_retirementLock)
            {
                if (_retiredMemoryObjects.Count == 0) return;
                var events = new IntPtr[_retiredMemoryObjects.Count];
                for (int i = 0; i < events.Length; i++) events[i] = _retiredMemoryObjects[i].CompletionEvent;
                OpenClNativeBindings.WaitForEvents((uint)events.Length, events);
                foreach (var retired in _retiredMemoryObjects)
                {
                    OpenClNativeBindings.ReleaseEvent(retired.CompletionEvent);
                    ReleaseMemoryObject(retired.MemoryObject, retired.ByteSize);
                }
                _retiredMemoryObjects.Clear();
            }
        }

        /// <summary>Reclaims event-fenced memory and host pins whose commands completed, without blocking.</summary>
        internal void ReapCompletedResources()
        {
            lock (_retirementLock)
            {
                for (int i = _retiredMemoryObjects.Count - 1; i >= 0; i--)
                {
                    var retired = _retiredMemoryObjects[i];
                    if (!IsEventComplete(retired.CompletionEvent)) continue;
                    _retiredMemoryObjects.RemoveAt(i);
                    OpenClNativeBindings.ReleaseEvent(retired.CompletionEvent);
                    ReleaseMemoryObject(retired.MemoryObject, retired.ByteSize);
                }

                for (int i = _pendingHostTransfers.Count - 1; i >= 0; i--)
                {
                    var transfer = _pendingHostTransfers[i];
                    if (!IsEventComplete(transfer.CompletionEvent)) continue;
                    _pendingHostTransfers.RemoveAt(i);
                    OpenClNativeBindings.ReleaseEvent(transfer.CompletionEvent);
                    if (transfer.PinnedMemory.IsAllocated) transfer.PinnedMemory.Free();
                }
            }
        }

        internal int PendingMemoryRetirementCount
        {
            get { lock (_retirementLock) return _retiredMemoryObjects.Count; }
        }

        private static bool IsEventComplete(IntPtr eventHandle)
        {
            IntPtr statusPtr = Marshal.AllocHGlobal(sizeof(int));
            try
            {
                int err = OpenClNativeBindings.GetEventInfo(
                    eventHandle,
                    OpenClNativeBindings.CL_EVENT_COMMAND_EXECUTION_STATUS,
                    (UIntPtr)sizeof(int),
                    statusPtr,
                    out _);
                if (err != OpenClNativeBindings.CL_SUCCESS) return false;
                // Negative values are terminal command errors and are just as safe to reclaim as success.
                return Marshal.ReadInt32(statusPtr) <= OpenClNativeBindings.CL_COMPLETE;
            }
            finally
            {
                Marshal.FreeHGlobal(statusPtr);
            }
        }

        private void ReleaseMemoryObject(IntPtr memoryObject, long byteSize)
        {
            // The caller claimed it (RetireMemoryObject), so it is ours to release. A sub-buffer's reference on its parent
            // goes only after the sub-buffer itself (see OpenClNativeContextHandle.RetainViewParent).
            IntPtr viewParent = _native?.TakeViewParent(memoryObject) ?? IntPtr.Zero;
            OpenClNativeBindings.ReleaseMemObject(memoryObject);
            if (viewParent != IntPtr.Zero) OpenClNativeBindings.ReleaseMemObject(viewParent);
            GpuKernelDiagnostics.RecordBufferReleased(byteSize);
        }

        internal void CompleteQueueForDisposal(IntPtr queue)
        {
            if (queue == IntPtr.Zero) return;
            lock (DirectOpenClSubmission.Gate)
            {
                int err = OpenClNativeBindings.Finish(queue);
                if (err != OpenClNativeBindings.CL_SUCCESS)
                    throw new InvalidOperationException($"Failed to finish OpenCL command queue during disposal: {err}");
                _completedQueues.TryAdd(queue, 0);
                LiveQueues.TryRemove(queue, out _);
            }
            ReapCompletedResources();
        }

        public string DeviceName { get; private set; } = string.Empty;
        public string DeviceVendor { get; private set; } = string.Empty;
        public string DeviceBoardName { get; private set; } = string.Empty;
        public string DriverVersion { get; private set; } = string.Empty;
        public string OpenClVersion { get; private set; } = string.Empty;
        public ulong DeviceType { get; private set; }
        public uint MaxComputeUnits { get; private set; }
        public ulong GlobalMemSize { get; private set; }
        public ulong LocalMemSize { get; private set; }

        /// <summary>
        /// Maximum bytes the device allows in a single buffer allocation
        /// (<c>CL_DEVICE_MAX_MEM_ALLOC_SIZE</c>). The OpenCL spec guarantees
        /// this is at least <c>max(GlobalMemSize / 4, 128 MiB)</c>, but the
        /// real cap on consumer GPUs is often substantially lower than the
        /// total VRAM. Issue #285: blindly calling clCreateBuffer with a
        /// size above this returns <c>CL_INVALID_BUFFER_SIZE (-61)</c>;
        /// this value lets callers validate ahead of time and fall back
        /// gracefully to CPU or chunked execution.
        /// </summary>
        public ulong MaxMemAllocSize { get; private set; }

        /// <summary>
        /// Byte alignment this device requires of a sub-buffer origin (<c>CL_DEVICE_MEM_BASE_ADDR_ALIGN</c>, which the
        /// device reports in bits). An origin off this alignment fails clCreateSubBuffer with
        /// <c>CL_MISALIGNED_SUB_BUFFER_OFFSET (-13)</c>, so a view caller can rule the offset out, and say what the
        /// device wanted, without paying a failed call. Zero when the query failed, which leaves the check to the driver.
        /// </summary>
        public uint SubBufferAlignmentBytes { get; private set; }

        /// <summary>
        /// GPU memory bandwidth in bytes/second (approximate).
        /// </summary>
        public ulong MemoryBandwidth { get; private set; }

        /// <summary>
        /// GPU clock frequency in MHz.
        /// </summary>
        public uint ClockFrequencyMHz { get; private set; }

        /// <summary>
        /// Maximum total work items per work group (e.g., 256, 1024).
        /// </summary>
        public ulong MaxWorkGroupSize { get; private set; }

        /// <summary>
        /// Maximum number of work item dimensions (typically 3).
        /// </summary>
        public uint MaxWorkItemDimensions { get; private set; }

        /// <summary>
        /// Maximum work items per dimension (e.g., [1024, 1024, 64] or [256, 256, 256]).
        /// </summary>
        public ulong[] MaxWorkItemSizes { get; private set; } = Array.Empty<ulong>();

        /// <summary>
        /// Device extensions string (for capability detection).
        /// </summary>
        public string Extensions { get; private set; } = string.Empty;

        /// <summary>
        /// Whether cl_khr_fp16 (half precision) is supported.
        /// </summary>
        public bool SupportsFp16 { get; private set; }

        /// <summary>
        /// Whether cl_khr_subgroups is supported.
        /// </summary>
        public bool SupportsSubgroups { get; private set; }

        /// <summary>
        /// Gets whether OpenCL is available on this system (any platform / any device type).
        /// Returning true here does not guarantee a real GPU is present — use
        /// <see cref="IsGpuAvailable"/> to gate GPU-backend creation.
        /// </summary>
        public static bool IsAvailable => OpenClNativeBindings.IsAvailable;

        /// <summary>
        /// Gets whether a real GPU device is exposed by any OpenCL platform on this system.
        /// False on CPU-only machines that happen to have an OpenCL ICD installed, and
        /// false when <c>AIDOTNET_DISABLE_OPENCL=1</c> is set. Use this (not
        /// <see cref="IsAvailable"/>) before constructing <c>OpenClBackend</c> on a
        /// CPU-only run, otherwise the ~591-kernel cache will compile against the CPU
        /// runtime and inflate process RSS by 2-3 GB for no benefit.
        /// </summary>
        public static bool IsGpuAvailable => OpenClNativeBindings.IsGpuAvailable;

        /// <summary>
        /// Gets the total number of OpenCL GPU devices across all platforms.
        /// </summary>
        public static int GetDeviceCount()
        {
            if (!IsAvailable) return 0;

            try
            {
                int err = OpenClNativeBindings.GetPlatformIDs(0, null, out uint numPlatforms);
                if (err != OpenClNativeBindings.CL_SUCCESS || numPlatforms == 0)
                    return 0;

                var platforms = new IntPtr[numPlatforms];
                err = OpenClNativeBindings.GetPlatformIDs(numPlatforms, platforms, out _);
                if (err != OpenClNativeBindings.CL_SUCCESS)
                    return 0;

                int totalDevices = 0;
                foreach (var platform in platforms)
                {
                    err = OpenClNativeBindings.GetDeviceIDs(platform, OpenClNativeBindings.CL_DEVICE_TYPE_GPU, 0, null, out uint numDevices);
                    if (err == OpenClNativeBindings.CL_SUCCESS)
                    {
                        totalDevices += (int)numDevices;
                    }
                }
                // Opt-in CI fallback: count CPU OpenCL devices (POCL) only when no GPU exists
                // and AIDOTNET_OPENCL_ALLOW_CPU=1, matching the selection fallback in Initialize.
                if (totalDevices == 0 && OpenClNativeBindings.AllowCpuOpenClDevice)
                {
                    foreach (var platform in platforms)
                    {
                        err = OpenClNativeBindings.GetDeviceIDs(platform, OpenClNativeBindings.CL_DEVICE_TYPE_CPU, 0, null, out uint numCpu);
                        if (err == OpenClNativeBindings.CL_SUCCESS)
                        {
                            totalDevices += (int)numCpu;
                        }
                    }
                }
                return totalDevices;
            }
            catch
            {
                return 0;
            }
        }

        public DirectOpenClContext() : this(0)
        {
        }

        public DirectOpenClContext(int deviceIndex)
        {
            Initialize(deviceIndex);
        }

        private void Initialize(int deviceIndex)
        {
            // Get platforms
            int err = OpenClNativeBindings.GetPlatformIDs(0, null, out uint numPlatforms);
            if (err != OpenClNativeBindings.CL_SUCCESS || numPlatforms == 0)
                throw new InvalidOperationException("No OpenCL platforms found");

            var platforms = new IntPtr[numPlatforms];
            err = OpenClNativeBindings.GetPlatformIDs(numPlatforms, platforms, out _);
            if (err != OpenClNativeBindings.CL_SUCCESS)
                throw new InvalidOperationException($"Failed to get OpenCL platforms: {err}");

            // Find devices of a given type across all platforms and select by global index.
            // GPU is always attempted first; the CPU pass below runs only under the opt-in
            // flag when no GPU DEVICE EXISTS AT ALL, so production selection is unchanged.
            // anyDevicesOfType reports whether the platform exposed >=1 device of the type, which
            // is distinct from "the requested index was in range" — an out-of-range index against
            // an existing GPU must NOT silently fall through to a CPU device.
            bool TrySelectDevice(ulong deviceType, out bool anyDevicesOfType)
            {
                anyDevicesOfType = false;
                int currentIndex = 0;
                foreach (var platform in platforms)
                {
                    int e = OpenClNativeBindings.GetDeviceIDs(platform, deviceType, 0, null, out uint numDevices);
                    if (e == OpenClNativeBindings.CL_SUCCESS && numDevices > 0)
                    {
                        anyDevicesOfType = true;
                        var devices = new IntPtr[numDevices];
                        e = OpenClNativeBindings.GetDeviceIDs(platform, deviceType, numDevices, devices, out _);
                        if (e == OpenClNativeBindings.CL_SUCCESS)
                        {
                            // Check if the requested device index is within this platform's devices
                            if (deviceIndex >= currentIndex && deviceIndex < currentIndex + (int)numDevices)
                            {
                                _platform = platform;
                                _device = devices[deviceIndex - currentIndex];
                                return true;
                            }
                            currentIndex += (int)numDevices;
                        }
                    }
                }
                return false;
            }

            bool gpuSelected = TrySelectDevice(OpenClNativeBindings.CL_DEVICE_TYPE_GPU, out bool anyGpu);
            if (!gpuSelected && !anyGpu && OpenClNativeBindings.AllowCpuOpenClDevice)
            {
                // No GPU device exists AT ALL, but AIDOTNET_OPENCL_ALLOW_CPU=1 — bind a CPU OpenCL
                // device (e.g. POCL) so the DirectGpu kernels EXECUTE for CPU-vs-GPU parity in CI.
                // Guarded by !anyGpu so an out-of-range index on a real GPU throws below instead of
                // silently binding a CPU device.
                TrySelectDevice(OpenClNativeBindings.CL_DEVICE_TYPE_CPU, out _);
            }

            if (_device == IntPtr.Zero)
                throw new InvalidOperationException(
                    OpenClNativeBindings.AllowCpuOpenClDevice
                        ? "No OpenCL GPU or CPU devices found"
                        : "No GPU devices found");

            // Get device info
            DeviceName = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_NAME);
            DeviceVendor = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_VENDOR);
            DeviceType = OpenClNativeBindings.GetDeviceInfoULong(_device, OpenClNativeBindings.CL_DEVICE_TYPE);
            DriverVersion = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_DRIVER_VERSION);
            OpenClVersion = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_VERSION);
            MaxComputeUnits = OpenClNativeBindings.GetDeviceInfoUInt(_device, OpenClNativeBindings.CL_DEVICE_MAX_COMPUTE_UNITS);
            GlobalMemSize = OpenClNativeBindings.GetDeviceInfoULong(_device, OpenClNativeBindings.CL_DEVICE_GLOBAL_MEM_SIZE);
            LocalMemSize = OpenClNativeBindings.GetDeviceInfoULong(_device, OpenClNativeBindings.CL_DEVICE_LOCAL_MEM_SIZE);
            // Per-allocation cap (issue #285) — query at context init so the
            // hot allocation path (DirectOpenClBuffer ctor) can guard against
            // CL_INVALID_BUFFER_SIZE (-61) without a syscall per allocation.
            MaxMemAllocSize = OpenClNativeBindings.GetDeviceInfoULong(_device, OpenClNativeBindings.CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            // Sub-buffer origin alignment (issue: refused buffer views). Reported in bits, so divide once here rather
            // than per view.
            SubBufferAlignmentBytes =
                OpenClNativeBindings.GetDeviceInfoUInt(_device, OpenClNativeBindings.CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8;

            // Get work group capabilities
            MaxWorkGroupSize = (ulong)OpenClNativeBindings.GetDeviceInfoSizeT(_device, OpenClNativeBindings.CL_DEVICE_MAX_WORK_GROUP_SIZE);
            MaxWorkItemDimensions = OpenClNativeBindings.GetDeviceInfoUInt(_device, OpenClNativeBindings.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);

            // Get max work item sizes per dimension
            if (MaxWorkItemDimensions > 0)
            {
                var sizes = OpenClNativeBindings.GetDeviceInfoSizeTArray(_device, OpenClNativeBindings.CL_DEVICE_MAX_WORK_ITEM_SIZES, (int)MaxWorkItemDimensions);
                MaxWorkItemSizes = new ulong[sizes.Length];
                for (int i = 0; i < sizes.Length; i++)
                {
                    MaxWorkItemSizes[i] = (ulong)sizes[i];
                }
            }

            // Get extensions for capability detection
            Extensions = OpenClNativeBindings.GetDeviceInfoString(_device, OpenClNativeBindings.CL_DEVICE_EXTENSIONS);
            SupportsFp16 = Extensions.Contains("cl_khr_fp16");
            SupportsSubgroups = Extensions.Contains("cl_khr_subgroups");
            if (Extensions.Contains("cl_amd_device_attribute_query", StringComparison.OrdinalIgnoreCase))
            {
                DeviceBoardName = OpenClNativeBindings.GetDeviceInfoString(
                    _device, OpenClNativeBindings.CL_DEVICE_BOARD_NAME_AMD);
            }

            // Get clock frequency for theoretical GFLOPS calculation
            ClockFrequencyMHz = OpenClNativeBindings.GetDeviceInfoUInt(_device, OpenClNativeBindings.CL_DEVICE_MAX_CLOCK_FREQUENCY);

            // Create context
            var deviceArray = new IntPtr[] { _device };
            // Contexts reclaimed without Dispose wait in the deferred-free queue; free them before asking the driver
            // for another, whose host budget they would otherwise still hold.
            DirectOpenClGpuBuffer.PendingFrees.Drain();
            _context = OpenClNativeBindings.CreateContext(IntPtr.Zero, 1, deviceArray, IntPtr.Zero, IntPtr.Zero, out err);
            if (err != OpenClNativeBindings.CL_SUCCESS || _context == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL context: {err}" + LiveContextReport());
            Interlocked.Increment(ref s_liveContexts);
            s_creationSites?.TryAdd(_context, CreationSite());
            // Owns the context and everything created in it, so a context that is collected without being disposed
            // still gives its native objects back (see OpenClNativeContextHandle). The callback is static: capturing
            // this context would root it from its own handle.
            _native = new OpenClNativeContextHandle(_context, OnNativeContextReleased);

            // Per-thread command queues (issue #414). The original design used a
            // single shared queue here, which made Parallel.ForEach(... IEngine
            // work ...) serialize on host-side clEnqueue* calls — observed
            // 26× slower than 1× serial. Now each host thread that touches
            // CommandQueue gets its own clCreateCommandQueue on the same context;
            // ThreadLocal.Values lets Dispose release every queue created across
            // every thread.
            _threadCommandQueue = new ThreadLocal<IntPtr>(
                () => CreateCommandQueueForCurrentThread(profilingEnabled: false),
                trackAllValues: true);

            // Probe profiling-queue support once on the ctor thread so callers can
            // ask IsProfilingEnabled without paying a per-thread create + roll-back.
            IntPtr probeProfilingQueue = OpenClNativeBindings.CreateCommandQueue(
                _context, _device, OpenClNativeBindings.CL_QUEUE_PROFILING_ENABLE, out err);
            if (err == OpenClNativeBindings.CL_SUCCESS && probeProfilingQueue != IntPtr.Zero)
            {
                // Probe succeeded — release the probe queue and lazy-create per-thread
                // profiling queues on demand via the ThreadLocal factory.
                OpenClNativeBindings.ReleaseCommandQueue(probeProfilingQueue);
                _profilingSupported = true;
                _threadProfilingCommandQueue = new ThreadLocal<IntPtr>(
                    () => CreateCommandQueueForCurrentThread(profilingEnabled: true),
                    trackAllValues: true);
            }
            else
            {
                _profilingSupported = false;
                _threadProfilingCommandQueue = null;
            }
        }

        private IntPtr CreateCommandQueueForCurrentThread(bool profilingEnabled)
        {
            // Disposal can race with a thread that's lazy-fetching CommandQueue;
            // returning IntPtr.Zero in that case lets callers fail fast via the
            // null-check ladder rather than calling a released cl_command_queue.
            if (_disposed) return IntPtr.Zero;
            ulong properties = profilingEnabled
                ? OpenClNativeBindings.CL_QUEUE_PROFILING_ENABLE
                : 0;
            // Serialize first-touch within this context. OpenClNativeBindings additionally
            // serializes the actual native call across every context in the process.
            IntPtr q;
            int err;
            lock (_queueCreateLock)
            {
                // Re-check disposal under the lock so a concurrent Dispose
                // that already released the context can't be raced into a
                // call that uses the freed context handle.
                if (_disposed) return IntPtr.Zero;
                q = OpenClNativeBindings.CreateCommandQueue(_context, _device, properties, out err);
                if (err == OpenClNativeBindings.CL_SUCCESS && q != IntPtr.Zero)
                {
                    LiveQueues.TryAdd(q, 0);
                    _native?.Track(q, OpenClChildKind.CommandQueue);
                }
            }
            if (err != OpenClNativeBindings.CL_SUCCESS || q == IntPtr.Zero)
            {
                throw new InvalidOperationException(
                    $"Failed to create per-thread OpenCL command queue (profiling={profilingEnabled}): err={err}");
            }
            return q;
        }

        /// <summary>
        /// Waits for all enqueued commands on the <b>current thread's</b> command
        /// queue to complete. After issue #414, each host thread has its own
        /// command queue, so callers that need a true cross-thread barrier should
        /// call <see cref="FinishAll"/> instead.
        /// </summary>
        public void Finish()
        {
            if (_disposed) return;
            var tl = _threadCommandQueue;
            if (tl is null) return;
            // Only finish the queue if this thread has already materialized one;
            // touching tl.Value would lazy-create a queue just to finish nothing.
            if (tl.IsValueCreated)
            {
                IntPtr q = tl.Value;
                if (q != IntPtr.Zero) OpenClNativeBindings.Finish(q);
            }
        }

        /// <summary>
        /// Cross-thread barrier: finishes every per-thread command queue created
        /// across the lifetime of this context. Use sparingly — typically only at
        /// the end of a Parallel.ForEach block before reading device buffers from
        /// the orchestrating thread.
        /// </summary>
        public void FinishAll()
        {
            if (_disposed) return;
            var tl = _threadCommandQueue;
            if (tl is null) return;
            foreach (IntPtr q in QueuesOf(tl))
            {
                if (q != IntPtr.Zero) OpenClNativeBindings.Finish(q);
            }
        }

        /// <summary>
        /// Waits for all commands on the current thread's profiling queue to complete.
        /// </summary>
        public void FinishProfiling()
        {
            if (_disposed) return;
            var tl = _threadProfilingCommandQueue;
            if (tl is null) return;
            if (tl.IsValueCreated)
            {
                IntPtr q = tl.Value;
                if (q != IntPtr.Zero) OpenClNativeBindings.Finish(q);
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            GC.SuppressFinalize(this);

            // Stop new submissions and finish every implicit queue under the same gate used to enqueue
            // kernels. This closes the check/enqueue/dispose race and makes context teardown a real lifetime
            // boundary instead of releasing queue handles while work is still outstanding.
            lock (DirectOpenClSubmission.Gate)
            {
                if (_disposed) return;
                _disposed = true;

                var regularToFinish = _threadCommandQueue;
                if (regularToFinish is not null)
                {
                    foreach (IntPtr q in QueuesOf(regularToFinish))
                    {
                        if (q == IntPtr.Zero) continue;
                        OpenClNativeBindings.Finish(q);
                        _completedQueues.TryAdd(q, 0);
                        LiveQueues.TryRemove(q, out _);
                    }
                }

                var profilingToFinish = _threadProfilingCommandQueue;
                if (profilingToFinish is not null)
                {
                    foreach (IntPtr q in QueuesOf(profilingToFinish))
                    {
                        if (q == IntPtr.Zero) continue;
                        OpenClNativeBindings.Finish(q);
                        _completedQueues.TryAdd(q, 0);
                        LiveQueues.TryRemove(q, out _);
                    }
                }
            }

            // Every retirement marker is now either complete or belongs to an explicit stream. Waiting for
            // those event handles covers the latter without needing to retain or enumerate stream wrappers.
            lock (_retirementLock)
            {
                int eventCount = _retiredMemoryObjects.Count + _pendingHostTransfers.Count;
                if (eventCount > 0)
                {
                    var events = new IntPtr[eventCount];
                    int eventIndex = 0;
                    for (int i = 0; i < _retiredMemoryObjects.Count; i++)
                        events[eventIndex++] = _retiredMemoryObjects[i].CompletionEvent;
                    for (int i = 0; i < _pendingHostTransfers.Count; i++)
                        events[eventIndex++] = _pendingHostTransfers[i].CompletionEvent;
                    OpenClNativeBindings.WaitForEvents((uint)events.Length, events);

                    for (int i = 0; i < _retiredMemoryObjects.Count; i++)
                    {
                        var retired = _retiredMemoryObjects[i];
                        OpenClNativeBindings.ReleaseEvent(retired.CompletionEvent);
                        ReleaseMemoryObject(retired.MemoryObject, retired.ByteSize);
                    }
                    _retiredMemoryObjects.Clear();

                    for (int i = 0; i < _pendingHostTransfers.Count; i++)
                    {
                        var transfer = _pendingHostTransfers[i];
                        OpenClNativeBindings.ReleaseEvent(transfer.CompletionEvent);
                        if (transfer.PinnedMemory.IsAllocated) transfer.PinnedMemory.Free();
                    }
                    _pendingHostTransfers.Clear();
                }
            }

            var profiling = _threadProfilingCommandQueue;
            _threadProfilingCommandQueue = null;
            if (profiling is not null)
            {
                foreach (IntPtr q in QueuesOf(profiling))
                {
                    if (q == IntPtr.Zero || !ClaimNative(q)) continue;
                    OpenClNativeBindings.ReleaseCommandQueue(q);
                }
                profiling.Dispose();
            }

            var regular = _threadCommandQueue;
            _threadCommandQueue = null;
            if (regular is not null)
            {
                foreach (IntPtr q in QueuesOf(regular))
                {
                    if (q == IntPtr.Zero || !ClaimNative(q)) continue;
                    OpenClNativeBindings.ReleaseCommandQueue(q);
                }
                regular.Dispose();
            }

            // Releases the context (and anything a wrapper never released); the count drops in its callback.
            _native?.ReleaseAll();
            _context = IntPtr.Zero;
        }

        // A context collected without Dispose (its engine or backend was dropped) still holds ~110 MB of driver host
        // memory. Its release is handed to the deferred-free queue the buffer finalizers use, so it runs on an ordinary
        // thread (the next allocation, or the next context creation, drains it) and never concurrently with one of its
        // own buffers being freed there; claiming (ClaimNative) keeps either order safe.
        ~DirectOpenClContext()
        {
            if (_disposed || _native is null) return;
            DirectOpenClGpuBuffer.PendingFrees.Enqueue(Dispose);
        }

        // The queues a per-thread ThreadLocal holds. A context reclaimed without Dispose is disposed from the deferred-free
        // queue after the GC has already finalized its ThreadLocals (Values then throws); its queues are still released,
        // as tracked children of the native handle, so none is lost.
        private static IList<IntPtr> QueuesOf(ThreadLocal<IntPtr> queues)
        {
            try
            {
                return queues.Values;
            }
            catch (ObjectDisposedException)
            {
                return Array.Empty<IntPtr>();
            }
        }

        // Runs when the native context is released, by Dispose (also when the finalizer queued it). Static on purpose.
        private static void OnNativeContextReleased(IntPtr context)
        {
            Interlocked.Decrement(ref s_liveContexts);
            s_creationSites?.TryRemove(context, out _);
        }

        /// <summary>Records a native object created in this context, released with it if its wrapper never is.</summary>
        internal void TrackNative(IntPtr child, OpenClChildKind kind) => _native?.Track(child, kind);

        /// <summary>Claims a native object for release by its wrapper: false when the context already released it.</summary>
        internal bool ClaimNative(IntPtr child) => _native is null || _native.TryClaim(child);

        /// <summary>Keeps a sub-buffer's parent alive until the sub-buffer is released: false when the retain failed.</summary>
        internal bool RetainViewParent(IntPtr view, IntPtr parent) => _native is null || _native.RetainViewParent(view, parent);

        // The first frames outside the OpenCL backend: the code that asked for this context.
        private static string CreationSite()
        {
            var frames = new System.Diagnostics.StackTrace(1, false).GetFrames();
            var parts = new List<string>();
            foreach (var frame in frames)
            {
                var method = frame?.GetMethod();
                var type = method?.DeclaringType;
                if (type is null) continue;
                string ns = type.Namespace ?? string.Empty;
                if (ns.StartsWith("AiDotNet.Tensors.Engines", StringComparison.Ordinal) && !ns.Contains("Tests")) continue;
                if (ns.StartsWith("System", StringComparison.Ordinal) || ns.StartsWith("Xunit", StringComparison.Ordinal)) continue;
                parts.Add($"{type.FullName}.{method!.Name}");
                if (parts.Count == 2) break;
            }
            return parts.Count == 0 ? "(inside the library)" : string.Join(" <- ", parts);
        }
    }
}
