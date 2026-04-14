namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Intel oneDNN accelerator — <b>disabled in the supply-chain-independence build</b>.
///
/// <para>
/// Historically this class loaded the native <c>dnnl.dll</c> (provided by the optional
/// <c>AiDotNet.Native.OneDNN</c> NuGet package) at runtime and exposed TryConv2D /
/// TrySigmoid / TryReLU / TryAdd / TryMultiply / TrySoftmax hot paths that
/// <see cref="Engines.CpuEngine"/> preferentially used over the in-house kernels.
/// After <c>feat/finish-mkl-replacement</c> (Issue #131 completion + supply-chain
/// directive), every entry point returns <c>false</c> immediately so callers fall
/// through to their managed fallback:
/// <list type="bullet">
///   <item><description>Conv2D → FusedConv + SimdGemm (im2col+GEMM), Winograd for 3×3 stride=1, SIMD direct for other small kernels</description></item>
///   <item><description>Element-wise (sigmoid / ReLU / add / multiply / softmax) → SimdKernels (AVX2-vectorized)</description></item>
/// </list>
/// </para>
/// <para>
/// This stub replaces ~1930 lines of P/Invoke / primitive-cache / mem-desc plumbing
/// that became unreachable after the disable. See git history for the original
/// implementation if a user ever wants to opt back in by reverting this file.
/// </para>
/// </summary>
internal static class OneDnnProvider
{
    /// <summary>Always false — external oneDNN is disabled.</summary>
    internal static bool IsAvailable => false;

    internal static unsafe bool TryConv2D(
        float* input, int batch, int inChannels, int height, int width,
        float* kernel, int outChannels, int kernelH, int kernelW,
        float* output, int outHeight, int outWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => false;

    internal static unsafe bool TryReLU(float* data, int length) => false;

    internal static unsafe bool TrySigmoid(float* data, int length) => false;

    internal static unsafe bool TryAdd(float* src0, float* src1, float* dst, int length) => false;

    internal static unsafe bool TryMultiply(float* src0, float* src1, float* dst, int length) => false;

    internal static unsafe bool TrySoftmax(float* input, float* output, int outerSize, int axisSize) => false;
}
