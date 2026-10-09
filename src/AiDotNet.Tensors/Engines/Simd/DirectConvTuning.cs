using System.Collections.Concurrent;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>The conv pass a <see cref="DirectConvShape"/> describes.</summary>
internal enum DirectConvPass
{
    Forward = 0,
    BackwardInput = 1,
    BackwardKernel = 2,
}

/// <summary>Which implementation runs a conv pass.</summary>
internal enum DirectConvRoute
{
    /// <summary>The engine's existing routes (im2col GEMM, Winograd, transposed conv, ...).</summary>
    Im2Col = 0,

    /// <summary><see cref="DirectConvAvx2"/>.</summary>
    Direct = 1,
}

/// <summary>
/// One float32 NCHW conv pass, keyed by everything that changes which implementation is fastest. The output size
/// follows from the rest.
/// </summary>
internal readonly record struct DirectConvShape(
    DirectConvPass Pass,
    int Batch,
    int InChannels,
    int OutChannels,
    int Height,
    int Width,
    int KernelHeight,
    int KernelWidth,
    int StrideH,
    int StrideW,
    int PadH,
    int PadW,
    int DilationH,
    int DilationW)
{
    public int OutputHeight => StrideH <= 0 ? 0 : (Height + 2 * PadH - DilationH * (KernelHeight - 1) - 1) / StrideH + 1;

    public int OutputWidth => StrideW <= 0 ? 0 : (Width + 2 * PadW - DilationW * (KernelWidth - 1) - 1) / StrideW + 1;

    /// <summary>The multiply-adds of the pass, times two.</summary>
    public double FloatingPointOperations =>
        2.0 * Batch * OutChannels * OutputHeight * OutputWidth * InChannels * KernelHeight * KernelWidth;
}

/// <summary>
/// The tuned choice for one <see cref="DirectConvShape"/>: the route, and for a direct forward or input-gradient pass
/// the number of tasks its output tiles are split into (0 = the kernel's default).
/// </summary>
internal readonly record struct DirectConvConfiguration(DirectConvRoute Route, int TargetTasks);

/// <summary>
/// Configurations activated per exact conv shape (by <see cref="DirectConvEvolutionAutotuner"/>). A shape with no
/// entry keeps <see cref="DirectConvAvx2.DefaultChoosesDirect"/>.
/// </summary>
internal static class DirectConvTuning
{
    private static readonly ConcurrentDictionary<DirectConvShape, DirectConvConfiguration> s_active = new();

    public static bool TryGet(in DirectConvShape shape, out DirectConvConfiguration configuration)
    {
        if (s_active.IsEmpty)
        {
            configuration = default;
            return false;
        }
        return s_active.TryGetValue(shape, out configuration);
    }

    public static void Activate(in DirectConvShape shape, DirectConvConfiguration configuration) => s_active[shape] = configuration;

    public static bool Deactivate(in DirectConvShape shape) => s_active.TryRemove(shape, out _);

    public static void Clear() => s_active.Clear();
}
