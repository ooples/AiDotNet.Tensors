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
    /// Automatically detects and configures GPU acceleration if available.     
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method attempts to initialize DirectGpu acceleration. If successful, the Current
    /// engine is switched to DirectGpuTensorEngine. If GPU is not available or initialization fails,
    /// the engine remains on CpuEngine.
    /// </para>
    /// <para><b>For Beginners:</b> Call this once at application startup for automatic optimization.
    ///
    /// <code>
    /// // In your Program.cs or Main():
    /// AiDotNetEngine.AutoDetectAndConfigureGpu();
    ///
    /// // Now all operations will automatically use GPU if available!
    /// </code>
    ///
    /// This is safe to call even if you don't have a GPU - it will just stay on CPU mode.
    /// </para>
    /// </remarks>
    /// <returns>True if GPU was successfully configured, false otherwise.</returns>
    public static bool AutoDetectAndConfigureGpu()
    {
        try
        {
            var gpuEngine = new DirectGpuTensorEngine();

            if (gpuEngine.SupportsGpu)
            {
                Current = gpuEngine;
                Console.WriteLine($"[AiDotNet] GPU acceleration enabled: {gpuEngine.Name}");
                return true;
            }

            gpuEngine.Dispose();
            Console.WriteLine("[AiDotNet] GPU not available, using CPU");
            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AiDotNet] Failed to initialize DirectGpu: {ex.Message}");
            Console.WriteLine("[AiDotNet] Falling back to CPU");
            return false;
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
        Console.WriteLine("[AiDotNet] Reset to CPU engine");
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
