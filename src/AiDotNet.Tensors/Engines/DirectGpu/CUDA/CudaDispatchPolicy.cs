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
/// <para>The actual vendor dispatch (calling cudnnConvolutionForward on
/// GPU buffers) is tracked in a follow-up; today the CudaBackend Conv2D
/// / BatchNorm methods still run their hand-written kernels. The
/// dispatch-label scope + availability helpers are here so that path can
/// be plugged in later with a single <c>if (UseCudnn) ... else ...</c>
/// check without further wiring.</para>
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
