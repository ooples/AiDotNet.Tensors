// Copyright (c) AiDotNet. All rights reserved.
// Pure P/Invoke OpenCL context - no managed GPU runtime dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
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
        private IntPtr _context;
        private ThreadLocal<IntPtr>? _threadCommandQueue;
        private ThreadLocal<IntPtr>? _threadProfilingCommandQueue;
        private IntPtr _device;
        private IntPtr _platform;
        private bool _profilingSupported;
        private bool _disposed;
        // Per-instance lock that serializes the ONE-TIME clCreateCommandQueue
        // call per worker thread. The OpenCL 1.2 spec § 5.1.1 lists
        // clCreateCommandQueue as thread-safe, but at least AMD's RDNA1 driver
        // (gfx1012:xnack-, Adrenalin 24.x) crashes the host process with an
        // access violation in amdocl64.dll when ≥ 4 threads call it
        // concurrently. The lock costs ~tens of microseconds at first-touch
        // per thread and zero on every subsequent kernel launch (the queue
        // handle is then cached in the thread's ThreadLocal slot). This
        // matches PyTorch's CUDA stream pool which serialises cudaStreamCreate
        // for the same reason on older NVIDIA drivers.
        private readonly object _queueCreateLock = new object();

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
                    ?? throw new InvalidOperationException("DirectOpenClContext not initialized.");
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
                    ?? throw new InvalidOperationException("DirectOpenClContext not initialized.");
                return tl.Value;
            }
        }

        /// <summary>
        /// Gets whether profiling is enabled on this context.
        /// </summary>
        public bool IsProfilingEnabled => _profilingSupported;

        public IntPtr Device => _device;

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

            // Find GPU devices across all platforms and select by index
            int currentIndex = 0;
            foreach (var platform in platforms)
            {
                err = OpenClNativeBindings.GetDeviceIDs(platform, OpenClNativeBindings.CL_DEVICE_TYPE_GPU, 0, null, out uint numDevices);
                if (err == OpenClNativeBindings.CL_SUCCESS && numDevices > 0)
                {
                    var devices = new IntPtr[numDevices];
                    err = OpenClNativeBindings.GetDeviceIDs(platform, OpenClNativeBindings.CL_DEVICE_TYPE_GPU, numDevices, devices, out _);
                    if (err == OpenClNativeBindings.CL_SUCCESS)
                    {
                        // Check if the requested device index is within this platform's devices
                        if (deviceIndex >= currentIndex && deviceIndex < currentIndex + (int)numDevices)
                        {
                            _platform = platform;
                            _device = devices[deviceIndex - currentIndex];
                            break;
                        }
                        currentIndex += (int)numDevices;
                    }
                }
            }

            if (_device == IntPtr.Zero)
                throw new InvalidOperationException("No GPU devices found");

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
            _context = OpenClNativeBindings.CreateContext(IntPtr.Zero, 1, deviceArray, IntPtr.Zero, IntPtr.Zero, out err);
            if (err != OpenClNativeBindings.CL_SUCCESS || _context == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL context: {err}");

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
            // SERIALISE the native clCreateCommandQueue call across host
            // threads — see _queueCreateLock field doc for the rationale (AMD
            // RDNA1 driver crashes amdocl64.dll under concurrent invocation
            // despite the OpenCL 1.2 spec listing this entry point as
            // thread-safe). Cost: ~tens of microseconds at first-touch per
            // worker thread; ZERO on subsequent kernel launches (the queue
            // handle is cached by the ThreadLocal slot and re-used directly).
            IntPtr q;
            int err;
            lock (_queueCreateLock)
            {
                // Re-check disposal under the lock so a concurrent Dispose
                // that already released the context can't be raced into a
                // call that uses the freed context handle.
                if (_disposed) return IntPtr.Zero;
                q = OpenClNativeBindings.CreateCommandQueue(_context, _device, properties, out err);
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
            foreach (IntPtr q in tl.Values)
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
            // Mark disposed FIRST so any concurrent lazy-init from a worker thread
            // sees the flag and returns IntPtr.Zero instead of creating a new queue
            // that would leak past disposal.
            _disposed = true;

            var profiling = _threadProfilingCommandQueue;
            _threadProfilingCommandQueue = null;
            if (profiling is not null)
            {
                foreach (IntPtr q in profiling.Values)
                {
                    if (q != IntPtr.Zero) OpenClNativeBindings.ReleaseCommandQueue(q);
                }
                profiling.Dispose();
            }

            var regular = _threadCommandQueue;
            _threadCommandQueue = null;
            if (regular is not null)
            {
                foreach (IntPtr q in regular.Values)
                {
                    if (q != IntPtr.Zero) OpenClNativeBindings.ReleaseCommandQueue(q);
                }
                regular.Dispose();
            }

            if (_context != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseContext(_context);
                _context = IntPtr.Zero;
            }
        }
    }
}
