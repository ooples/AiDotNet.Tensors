using AiDotNet.Tensors.Engines.Optimization;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Single source of truth for "should this op go through cuDNN / cuBLAS
/// or through the generic CUDA kernel?" on the CUDA backend. Combines
/// runtime availability of the vendor library with the user-controlled
/// opt-out flag on <see cref="TensorCodecOptions.Current"/>.
///
/// <para>Consumers call <see cref="Scope"/> at the top of their op impl
/// (Conv2D, BatchNorm, MatMul) so <see cref="PerformanceProfiler"/>
/// records the dispatched path — e.g. <c>"Conv2D.cuDNN"</c> vs
/// <c>"Conv2D.generic"</c>. A test can then verify via
/// <c>PerformanceProfiler.Instance.GetStats("Conv2D.cuDNN")</c> that the
/// vendor path actually ran.</para>
///
/// <para>Dispatch status (verified against CudaBackend, AiDotNet#1159):
/// <list type="bullet">
/// <item><b>Conv2D</b> — routes to cuDNN (<c>cudnnConvolutionForward</c> on GPU
/// buffers via <c>CuDnnConvolution.Conv2DForwardGpu</c>) when
/// <see cref="UseCudnnForConv"/>; otherwise the generic Winograd/tiled/im2col
/// kernel.</item>
/// <item><b>MatMul / GEMM</b> — always routes to cuBLAS (<c>cublasSgemm</c>);
/// the scope is labelled <c>"MatMul.cuBLAS"</c> unconditionally.</item>
/// <item><b>BatchNorm</b> — still runs the hand-written CUDA kernel on every
/// path. The <see cref="UseCudnnForBatchNorm"/> helper and dispatch-label
/// scope exist, but the cuDNN BatchNorm call is NOT yet wired; it can be
/// plugged into <c>CudaBackend.BatchNorm</c> with a single
/// <c>if (UseCudnnForBatchNorm) ... else ...</c> check mirroring Conv2D
/// (remaining work tracked by AiDotNet#1159).</item>
/// </list></para>
/// </summary>
internal static class CudaDispatchPolicy
{
    /// <summary>
    /// Returns true when Conv2D should run through cuDNN: user hasn't
    /// set <see cref="TensorCodecOptions.UseCudnn"/> to false and the
    /// cuDNN library is loadable on this host.
    /// </summary>
    public static bool UseCudnnForConv
        => TensorCodecOptions.Current.UseCudnn && CuDnnConvolution.IsAvailable;

    /// <summary>
    /// Returns true when BatchNorm should run through cuDNN: user hasn't
    /// set <see cref="TensorCodecOptions.UseCudnnBatchNorm"/> to false
    /// and the cuDNN library is loadable on this host.
    /// </summary>
    public static bool UseCudnnForBatchNorm
        => TensorCodecOptions.Current.UseCudnnBatchNorm && CuDnnContext.IsAvailable;

    /// <summary>
    /// Returns true when MatMul / GEMM should run through cuBLAS: user
    /// hasn't set <see cref="TensorCodecOptions.UseCublas"/> to false and
    /// the cuBLAS handle is available. CudaBackend always has cuBLAS
    /// loaded as a prerequisite of initialisation, so the availability
    /// leg is effectively "are we running on CudaBackend at all".
    /// </summary>
    public static bool UseCublasForMatMul
        => TensorCodecOptions.Current.UseCublas;

    /// <summary>
    /// Returns true when fp32 cuBLAS GEMMs should use TF32 (TensorFloat-32)
    /// accumulation on Ampere+ (compute capability ≥ 8.0). TF32 rounds the
    /// 23-bit fp32 mantissa to 10 bits for the multiply, accumulates at full
    /// fp32, and runs on the Tensor Cores. The result is ~5× the throughput
    /// of strict fp32 with bit-comparable training-loss curves on every
    /// published benchmark (NVIDIA: BERT, ResNet-50, Mask-R-CNN, T5).
    /// </summary>
    /// <remarks>
    /// <para>Default: true. Opt out by:
    /// <list type="bullet">
    /// <item>setting <c>AIDOTNET_DISABLE_TF32</c> to any non-empty value
    /// before process startup (checked at <see cref="CudaBackend"/> init time), or</item>
    /// <item>flipping the static <see cref="AllowTF32"/> property at runtime —
    /// changes take effect on the next <see cref="CudaBackend"/> instance
    /// (current backends keep the math mode they were initialized with).</item>
    /// </list></para>
    /// <para>When to disable: numerical-reproducibility runs that need bit-exact
    /// fp32 arithmetic (regression tests pinned to a reference output), Volta or
    /// older hardware (the property short-circuits via compute-capability check
    /// in <see cref="CudaBackend"/>, so this is for forcing the legacy path on
    /// Ampere+ rather than relaxing it on older hardware).</para>
    /// </remarks>
    public static bool AllowTF32
    {
        get
        {
            if (_allowTF32Override.HasValue) return _allowTF32Override.Value;
            return string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_TF32"));
        }
        set => _allowTF32Override = value;
    }

    private static bool? _allowTF32Override;

    /// <summary>
    /// Push a <see cref="PerformanceProfiler"/> scope with a kernel-path
    /// label so downstream telemetry can distinguish vendor vs generic
    /// kernel runs.
    /// </summary>
    /// <param name="op">Op name — typically <c>"Conv2D"</c>, <c>"BatchNorm"</c>,
    /// or <c>"MatMul"</c>.</param>
    /// <param name="useVendor">True iff the vendor path (cuDNN/cuBLAS) ran.</param>
    /// <returns>A scope that records the op timing on Dispose.</returns>
    public static IDisposable Scope(string op, bool useVendor)
    {
        string label = useVendor ? $"{op}.cuDNN" : $"{op}.generic";
        // Special case: MatMul + useVendor == true is cuBLAS, not cuDNN.
        if (useVendor && op == "MatMul") label = "MatMul.cuBLAS";
        return PerformanceProfiler.Instance.Profile(label);
    }
}
