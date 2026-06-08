using System.Threading;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Global configuration for the AiDotNet execution engine.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNetEngine provides a singleton pattern for managing the active execution engine.
/// By default, operations run on the CPU. Users can switch to GPU or other accelerators
/// by setting the Current property.
/// </para>
/// <para><b>For Beginners:</b> This is like a settings panel for your calculations.
///
/// Example usage:
/// <code>
/// // Default: Use CPU
/// var result = vector1.Add(vector2);  // Runs on CPU
///
/// // Switch to GPU
/// AiDotNetEngine.Current = new GpuEngine();
/// var result2 = vector1.Add(vector2);  // Now runs on GPU!
///
/// // Auto-detect best hardware
/// AiDotNetEngine.AutoDetectAndConfigureGpu();
/// </code>
/// </para>
/// </remarks>
public static class AiDotNetEngine
{
    private static IEngine _current;

    /// <summary>
    /// Static constructor initializes with CPU engine by default.
    /// </summary>
    static AiDotNetEngine()
    {
        _current = new CpuEngine();
    }

    /// <summary>
    /// Gets or sets the current execution engine.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Changing the engine affects all subsequent operations. The change is global
    /// and thread-safe. Uses lock-free Volatile.Read/Write for optimal performance
    /// on the hot read path.
    /// </para>
    /// <para><b>For Beginners:</b> This is like choosing between CPU and GPU mode.
    ///
    /// Common patterns:
    /// <code>
    /// // Use CPU (default, works for all types)
    /// AiDotNetEngine.Current = new CpuEngine();
    ///
    /// // Use GPU (faster for float, fallback to CPU for other types)
    /// AiDotNetEngine.Current = new GpuEngine();
    ///
    /// // Auto-detect (recommended)
    /// AiDotNetEngine.AutoDetectAndConfigureGpu();
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when attempting to set to null.</exception>
    public static IEngine Current
    {
        get
        {
            return Volatile.Read(ref _current);
        }
        set
        {
            if (value == null)
            {
                throw new ArgumentNullException(nameof(value), "Engine cannot be null");
            }

            Volatile.Write(ref _current, value);
        }
    }

    /// <summary>
    /// User-configurable log sink for engine-level events (GPU auto-detect status,
    /// engine resets, etc.). When non-null, replaces the default Console.WriteLine
    /// destination — wire to your own logger (Microsoft.Extensions.Logging, Serilog,
    /// xUnit ITestOutputHelper, structured-event bus, etc.). Defaults to null
    /// (Console output preserved for backwards-compat) but settable at any time;
    /// changes take effect on the next event emission.
    /// </summary>
    /// <example>
    /// <code>
    /// // Route engine events to ILogger:
    /// AiDotNetEngine.Logger = msg =&gt; logger.LogInformation(msg);
    ///
    /// // Suppress all engine output:
    /// AiDotNetEngine.Logger = _ =&gt; { };
    ///
    /// // Restore default Console output:
    /// AiDotNetEngine.Logger = null;
    /// </code>
    /// </example>
    public static Action<string>? Logger { get; set; }

    /// <summary>
    /// Emits an engine log line. Honors (in order) the explicit <paramref name="verbose"/>
    /// flag passed by the caller, the <c>AIDOTNET_QUIET</c> env var (forces silence), and
    /// the <see cref="Logger"/> callback (overrides Console destination).
    /// </summary>
    private static void EmitLog(string message, bool verbose)
    {
        if (!verbose) return;
        // AIDOTNET_QUIET forces silence regardless of caller's verbose flag.
        // Lets users globally silence engine chatter from a single env var
        // without threading the flag through every caller.
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_QUIET")))
            return;

        // Guard sink invocation — a user-supplied Logger callback that
        // throws (broken ILogger pipeline, disposed sink, malformed format
        // string) must NOT break engine startup or ResetToCpu. Same logic
        // for Console.WriteLine: a redirected/closed stdout (Windows
        // service contexts, build hosts piping logs through a fragile
        // wrapper) can throw IOException, and the engine has no business
        // failing because the host's log transport is misconfigured.
        var sink = Logger;
        if (sink is not null)
        {
            try { sink(message); return; }
            catch { /* sink errors are non-fatal — fall through to Console */ }
        }
        try { Console.WriteLine(message); }
        catch { /* stdout transport errors are non-fatal */ }
    }

    /// <summary>
    /// Automatically detects and configures GPU acceleration if available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Attempts to initialize DirectGpu acceleration. If successful, the <see cref="Current"/>
    /// engine is switched to <c>DirectGpuTensorEngine</c>. If GPU is not available or
    /// initialization fails, the engine remains on <c>CpuEngine</c>.
    /// </para>
    /// <para><b>For Beginners:</b> Call this once at application startup for automatic optimization.
    /// <code>
    /// // In your Program.cs or Main():
    /// AiDotNetEngine.AutoDetectAndConfigureGpu();
    ///
    /// // Now all operations will automatically use GPU if available!
    /// </code>
    /// Safe to call even without a GPU — falls back to CPU silently.</para>
    /// <para>
    /// Original parameterless overload. Preserved as a distinct method (not collapsed
    /// into a <c>verbose = true</c> default on the verbose-aware overload below) so
    /// previously-compiled consumers that took a binary reference to
    /// <c>AutoDetectAndConfigureGpu()</c> keep linking against the same metadata signature.
    /// Forwards to <see cref="AutoDetectAndConfigureGpu(bool)"/> with verbose enabled.</para>
    /// </remarks>
    /// <returns>True if GPU was successfully configured, false otherwise.</returns>
    public static bool AutoDetectAndConfigureGpu()
        => AutoDetectAndConfigureGpu(verbose: true);

    /// <summary>
    /// Verbose-aware overload of <see cref="AutoDetectAndConfigureGpu()"/>. Same GPU
    /// detection / fallback behavior; the <paramref name="verbose"/> flag gates whether
    /// the status banner is emitted through the configured log sink.
    /// </summary>
    /// <param name="verbose">When true, emits status via <see cref="Logger"/>
    /// if set, otherwise Console.WriteLine. Pass false for module-init or library-internal
    /// use where polluting stdout is undesirable (test runners, hosted services, anything
    /// parsing stdout). Override globally via the <c>AIDOTNET_QUIET</c> env var — when
    /// set to any non-empty value, suppresses output regardless of this parameter.
    /// Plug your own log sink via the <see cref="Logger"/> property.</param>
    /// <returns>True if GPU was successfully configured, false otherwise.</returns>
    public static bool AutoDetectAndConfigureGpu(bool verbose)
    {
        // Once the GPU circuit breaker has tripped (a sticky CUDA fault corrupted the context),
        // never re-adopt the GPU for the rest of the process — re-engaging would immediately fault
        // again on the dead context. Stay on CPU.
        if (_gpuCircuitBroken)
        {
            Current = new CpuEngine();
            return false;
        }

        try
        {
            var gpuEngine = new DirectGpuTensorEngine();
            bool supportsGpu = gpuEngine.SupportsGpu;

            if (supportsGpu && GpuPassesCorrectnessProbe(gpuEngine))
            {
                Current = gpuEngine;
                TryRegisterGpuOffloadAllocator(gpuEngine, verbose);
                EmitLog($"[AiDotNet] GPU acceleration enabled: {gpuEngine.Name}", verbose);
                return true;
            }

            // SupportsGpu only reports that a device exists and the engine
            // constructed — NOT that it computes correctly. Some drivers/devices
            // (and misconfigured OpenCL stacks) construct fine but then throw or
            // return garbage at runtime; adopting such a backend makes EVERY
            // downstream op crash or silently produce wrong results. The probe
            // above runs a tiny known-answer op and only adopts the GPU when it
            // returns the right values, else we dispose it and stay on CPU. This
            // is a self-validation gate, not a CPU force — a working GPU is still
            // adopted.
            gpuEngine.Dispose();
            EmitLog(supportsGpu
                ? "[AiDotNet] GPU detected but failed its correctness self-check, using CPU"
                : "[AiDotNet] GPU not available, using CPU", verbose);
            return false;
        }
        catch (Exception ex)
        {
            EmitLog($"[AiDotNet] Failed to initialize DirectGpu: {ex.Message}", verbose);
            EmitLog("[AiDotNet] Falling back to CPU", verbose);
            return false;
        }
    }

    /// <summary>
    /// Runs a tiny known-answer workload on <paramref name="gpuEngine"/> and verifies the
    /// result, so a device that is present but computes incorrectly (driver/runtime issues,
    /// misconfigured OpenCL stacks) is rejected before it becomes <see cref="Current"/>.
    /// </summary>
    /// <remarks>
    /// Validates the full compute + host-readback path — the same path that otherwise
    /// surfaces as a runtime crash or all-zero results downstream. A 2×2 matmul has a unique
    /// exact integer answer in FP32, so a correct backend matches it bit-for-bit; any throw,
    /// NaN, or wrong value fails the probe. Returns false (never throws) so module-init
    /// auto-detection always degrades gracefully to CPU.
    /// </remarks>
    private static bool GpuPassesCorrectnessProbe(DirectGpuTensorEngine gpuEngine)
    {
        try
        {
            // [[1,2],[3,4]] @ [[5,6],[7,8]] == [[19,22],[43,50]]
            var a = new LinearAlgebra.Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
            var b = new LinearAlgebra.Tensor<float>(new float[] { 5f, 6f, 7f, 8f }, new[] { 2, 2 });
            var result = ((IEngine)gpuEngine).TensorMatMul(a, b);

            if (result == null || result.Length != 4)
                return false;

            ReadOnlySpan<float> expected = stackalloc float[] { 19f, 22f, 43f, 50f };
            for (int i = 0; i < 4; i++)
            {
                float v = result.GetFlatIndexValue(i);
                if (float.IsNaN(v) || Math.Abs(v - expected[i]) > 1e-3f)
                    return false;
            }

            return true;
        }
        catch
        {
            // Any failure (kernel build, allocation, readback, dispatch) means the
            // device can't be trusted for real work — reject it and fall back to CPU.
            return false;
        }
    }

    /// <summary>
    /// Auto-registers the GPU offload allocator that matches the adopted backend
    /// so weights/optimizer-moments tagged <see cref="LinearAlgebra.WeightLifetime.GpuPinned"/>
    /// (via <c>TensorAllocator.RentPinnedOnGpu</c>) actually become GPU-resident.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Without this, <see cref="LinearAlgebra.WeightRegistry.OffloadAllocator"/> stays
    /// null after GPU adoption, <c>RentPinnedOnGpu</c> silently falls back to CPU-pinned
    /// memory, <c>Tensor.TryGetGpuBuffer()</c> returns null for those tensors, and the
    /// GPU-resident optimizer step (<c>GpuOptimizer.TryXStep</c> → <c>backend.XUpdate</c>)
    /// can never fire — every optimizer silently runs the CPU weight update. This was the
    /// missing keystone behind the GPU-resident-optimizer path never executing on GPU.
    /// </para>
    /// <para>
    /// Only CUDA ships an offload allocator that is also an
    /// <c>IGpuDevicePointerWrapper</c> (the contract <c>TryGetGpuBuffer</c> needs to wrap
    /// an externally-owned device pointer), so GPU-resident weights are CUDA-only today;
    /// other backends keep CPU-resident moments and the optimizer falls back to CPU there.
    /// Best-effort: any failure leaves the registry unconfigured and GPU compute still works.
    /// </para>
    /// </remarks>
    private static void TryRegisterGpuOffloadAllocator(DirectGpuTensorEngine gpuEngine, bool verbose)
    {
        try
        {
            // Respect an allocator a caller already configured — don't stomp it.
            if (LinearAlgebra.WeightRegistry.OffloadAllocator is not null)
                return;

            var directGpu = ((IEngine)gpuEngine).DirectGpu;
            var backendName = directGpu?.BackendName?.ToUpperInvariant();
            DirectGpu.IGpuOffloadAllocator? allocator = backendName switch
            {
                // Share the compute backend's CUDA context so pinned moment/weight
                // buffers live in the SAME context as the optimizer kernels that
                // read them — no cross-context access (which made the GPU-resident
                // optimizer's host-readback sync flaky).
                "CUDA" or "NVIDIA" => new DirectGpu.CUDA.CudaOffloadAllocator(
                    directGpu?.Backend is DirectGpu.CUDA.CudaBackend cudaBackend ? cudaBackend.CudaContextHandle : IntPtr.Zero),
                // Hip/Metal/OpenCl/Vulkan/WebGpu offload allocators exist but are
                // not IGpuDevicePointerWrapper yet, so they cannot back a
                // GPU-resident weight buffer. Leave the registry unconfigured for
                // them (CPU-resident moments, CPU optimizer fallback).
                _ => null,
            };

            if (allocator is null)
                return;

            if (!allocator.IsAvailable)
            {
                allocator.Dispose();
                return;
            }

            LinearAlgebra.WeightRegistry.Configure(LinearAlgebra.WeightRegistry.CurrentOptions, allocator);
            EmitLog($"[AiDotNet] GPU offload allocator registered ({backendName}) — weights/optimizer moments can be GPU-resident", verbose);
        }
        catch (Exception ex)
        {
            // Non-fatal: GPU compute works without GPU-resident weights; the
            // optimizer simply keeps its CPU fallback.
            EmitLog($"[AiDotNet] GPU offload allocator registration skipped: {ex.Message}", verbose);
        }
    }

    /// <summary>
    /// Resets the engine to the default CPU engine.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is useful for testing or when you explicitly want to disable GPU acceleration.
    /// </para>
    /// </remarks>
    public static void ResetToCpu()
    {
        Current = new CpuEngine();
        EmitLog("[AiDotNet] Reset to CPU engine", verbose: true);
    }

    private static volatile bool _gpuCircuitBroken;

    /// <summary>True once a sticky CUDA fault has forced a permanent CPU fallback for this process.</summary>
    public static bool GpuCircuitBroken => _gpuCircuitBroken;

    /// <summary>
    /// Trips the process-wide GPU circuit breaker after a sticky/context-corrupting CUDA fault
    /// (illegal address 700, context destroyed 709, launch failed 719). Switches the engine to CPU
    /// and blocks any further GPU adoption, so a single GPU fault degrades to CPU instead of
    /// cascading the same failure across every subsequent op. Safe to call from inside a CUDA error
    /// path — it never throws.
    /// </summary>
    public static void TripGpuCircuitBreaker(string operation, int cudaErrorCode)
    {
        if (_gpuCircuitBroken)
        {
            return;
        }

        _gpuCircuitBroken = true;
        try
        {
            if (_current is DirectGpuTensorEngine)
            {
                _current = new CpuEngine();
            }
        }
        catch
        {
            // The breaker must never throw — it runs from within a CUDA error handler.
        }

        EmitLog($"[AiDotNet] GPU circuit breaker tripped by '{operation}' (CUDA error {cudaErrorCode}); " +
            "falling back to CPU for the rest of the process.", verbose: true);
    }

    /// <summary>
    /// Begins a GPU execution context for the current engine, if supported.
    /// </summary>
    /// <param name="options">Optional execution options.</param>
    /// <returns>A GPU execution context, or null if GPU is not available.</returns>
    public static GpuExecutionContext? BeginGpuContext(GpuExecutionOptions? options = null)
    {
        if (Current is DirectGpuTensorEngine gpuEngine)
        {
            return gpuEngine.BeginGpuContext(options);
        }

        return null;
    }

    /// <summary>
    /// Executes an action within a GPU execution context for the current engine.
    /// </summary>
    /// <param name="action">The action to execute.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>True if executed on GPU, false if GPU is not available.</returns>
    public static bool WithGpuContext(Action<GpuExecutionContext> action, GpuExecutionOptions? options = null)
    {
        if (action == null) throw new ArgumentNullException(nameof(action));

        if (Current is DirectGpuTensorEngine gpuEngine)
        {
            return gpuEngine.WithGpuContext(action, options);
        }

        return false;
    }

    /// <summary>
    /// Executes a function within a GPU execution context for the current engine.
    /// </summary>
    /// <typeparam name="TResult">The result type.</typeparam>
    /// <param name="func">The function to execute.</param>
    /// <param name="fallback">Fallback function if GPU is not available.</param>
    /// <param name="options">Optional execution options.</param>
    /// <returns>The function result.</returns>
    public static TResult WithGpuContext<TResult>(Func<GpuExecutionContext, TResult> func, Func<TResult> fallback, GpuExecutionOptions? options = null)
    {
        if (func is null)
            throw new ArgumentNullException(nameof(func));
        if (fallback is null)
            throw new ArgumentNullException(nameof(fallback));

        if (Current is DirectGpuTensorEngine gpuEngine)
        {
            return gpuEngine.WithGpuContext(func, fallback, options);
        }

        return fallback();
    }

    /// <summary>
    /// Gets information about the current engine configuration.
    /// </summary>
    /// <returns>A string describing the current engine.</returns>
    public static string GetEngineInfo()
    {
        var engine = Current;
        return $"Engine: {engine.Name}, GPU Support: {engine.SupportsGpu}";
    }

    /// <summary>
    /// Gets whether deterministic reduction mode is currently enabled.
    /// When true, floating-point reductions (matmul, softmax, sums) produce
    /// bit-identical results across runs on the same hardware.
    /// </summary>
    public static bool DeterministicMode => BlasProvider.IsDeterministicMode;

    /// <summary>
    /// Enables or disables deterministic reduction mode for CPU tensor operations.
    /// </summary>
    /// <param name="deterministic">True to enable deterministic mode; false to use the default high-throughput mode.</param>
    /// <remarks>
    /// <para>
    /// Multi-threaded BLAS (MKL.NET) can produce slightly different floating-point results across
    /// runs of the same inputs because parallel reduction order is not stable — floating-point
    /// addition is non-associative, so any variation in accumulation order produces a slightly
    /// different final value. In long training runs, small per-step drift can compound into
    /// observable trajectory divergence (e.g. final val-loss differences between otherwise-identical
    /// seeded runs).
    /// </para>
    /// <para>
    /// Enabling deterministic mode:
    /// <list type="bullet">
    ///   <item>Has <c>BlasProvider.TryGemm</c> and <c>IsMklVerified</c> short-circuit at the top,
    ///         bypassing both MKL.NET managed GEMM and any native BLAS GEMM paths entirely.</item>
    ///   <item>Routes matmul through the blocked C# fallback (<c>MatrixMultiplyHelper.MultiplyBlocked</c>
    ///         for matrices, <c>CpuEngine.TensorMatMul2D</c>'s <c>Parallel.For</c> path for tensors),
    ///         which writes disjoint output rows per thread with a fixed inner accumulation order —
    ///         bit-exact regardless of thread count.</item>
    ///   <item>Leaves native BLAS thread-control functions alone. Deterministic mode does <b>not</b>
    ///         invoke <c>mkl_set_num_threads</c> or <c>mkl_set_dynamic</c>; those are only touched
    ///         when <c>AIDOTNET_BLAS_THREADS</c> is explicitly set via env var.</item>
    ///   <item>Leaves all other tensor reductions alone — <c>TensorSum</c>, <c>TensorSumOfSquares</c>,
    ///         <c>TensorMean</c>, and <c>TensorLogSoftmax</c> are already deterministic in this engine.</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>Performance trade-off:</b> deterministic mode skips MKL's multi-threaded GEMM, which can
    /// reduce matmul throughput for large matrices. Only enable when reproducibility is required
    /// (tests, scientific reproducibility, paper experiments). Safe to call at any time; takes
    /// effect on the next BLAS call.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Turn this on if you need two identical training runs to produce
    /// identical numerical results — useful for debugging, test assertions, and research
    /// reproducibility. Leave it off for production training where raw speed matters more than
    /// bit-for-bit reproducibility.
    /// </para>
    /// </remarks>
    public static void SetDeterministicMode(bool deterministic)
    {
        BlasProvider.SetDeterministicMode(deterministic);
    }
}
