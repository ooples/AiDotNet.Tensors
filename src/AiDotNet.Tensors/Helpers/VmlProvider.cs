namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Intel MKL VML (Vector Math Library) accelerator — <b>disabled in the
/// supply-chain-independence build</b>.
///
/// <para>
/// Historically this class dynamically loaded <c>mkl_rt.dll</c> at runtime and
/// exposed <c>vs*</c> / <c>vmd*</c> bindings for vectorized transcendentals
/// (Exp / Log / Tanh / Sin / Cos / Sqrt / Erf / Abs / Sigmoid). Consumers
/// would call <c>if (!VmlProvider.TryExp(...)) { SimdKernels.Exp(...); }</c>
/// and opportunistically hit the faster VML path when the MKL runtime was
/// present.
/// </para>
/// <para>
/// After <c>feat/finish-mkl-replacement</c>, every entry point returns
/// <c>false</c> immediately and no native library is loaded. The SimdKernels
/// fallback (Herumi exp, Pade sigmoid, Schraudolph fast-exp, vectorized
/// tanh/log/sin/cos/sqrt/erf/abs) is already AVX2-vectorized and was the
/// primary path in all benchmarks — disabling VML has negligible measurable
/// impact at the shapes we care about.
/// </para>
/// <para>
/// This stub replaces ~520 lines of unmanaged-function-pointer P/Invoke
/// plumbing that became unreachable after the disable. See git history for
/// the original implementation if a user ever wants to opt back in.
/// </para>
/// </summary>
internal static class VmlProvider
{
    /// <summary>Always false — external VML is disabled.</summary>
    public static bool IsAvailable => false;

    /// <summary>Always false — external VML is disabled.</summary>
    internal static bool IsInitialized => false;

    public static unsafe bool TryExp(float* input, float* output, int length) => false;
    public static unsafe bool TryExp(double* input, double* output, int length) => false;

    public static unsafe bool TryLn(float* input, float* output, int length) => false;
    public static unsafe bool TryLn(double* input, double* output, int length) => false;

    public static unsafe bool TryTanh(float* input, float* output, int length) => false;
    public static unsafe bool TryTanh(double* input, double* output, int length) => false;

    public static unsafe bool TrySigmoid(float* input, float* output, int length) => false;

    public static unsafe bool TrySqrt(float* input, float* output, int length) => false;

    public static unsafe bool TryAbs(float* input, float* output, int length) => false;

    public static unsafe bool TrySin(float* input, float* output, int length) => false;

    public static unsafe bool TryCos(float* input, float* output, int length) => false;

    public static unsafe bool TryErf(float* input, float* output, int length) => false;
}
