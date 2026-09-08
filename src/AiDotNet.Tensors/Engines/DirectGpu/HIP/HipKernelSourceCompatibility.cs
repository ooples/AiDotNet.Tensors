namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// Adapts CUDA-compatible kernel source to the subset accepted by hipRTC.
/// </summary>
internal static class HipKernelSourceCompatibility
{
    private const string MathHeader = "#include <math.h>";
    private const string FloatHeader = "#include <float.h>";

    private const string CompatibilityPreamble = @"
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#ifndef FLT_MAX
#define FLT_MAX __FLT_MAX__
#endif

#ifndef NULL
#define NULL nullptr
#endif

// CUDA reductions in shared source are written for 32-thread warps. AMD
// wavefronts can be wider, so explicitly retain the source algorithm's
// 32-lane subgroup boundary when mapping the CUDA intrinsic to HIP.
#ifndef __shfl_down_sync
#define __shfl_down_sync(activeMask, value, delta) __shfl_down(value, delta, 32)
#endif
";

    internal static string Prepare(string source)
    {
        if (source is null)
        {
            throw new ArgumentNullException(nameof(source));
        }

        string sourceWithoutUnsupportedHeader = RemoveHeader(source, MathHeader);
        sourceWithoutUnsupportedHeader = RemoveHeader(sourceWithoutUnsupportedHeader, FloatHeader);

        return CompatibilityPreamble + sourceWithoutUnsupportedHeader;
    }

    private static string RemoveHeader(string source, string header)
    {
        return source
            .Replace(header + "\r\n", string.Empty)
            .Replace(header + "\n", string.Empty)
            .Replace(header, string.Empty);
    }
}
