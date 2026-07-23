using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The pooling family's contribution to the GPU-free cubin pipeline: which
/// modules exist. Everything else lives in <see cref="DirectPtxCubinToolCore"/>,
/// which is byte-identical on every branch that carries it so the shared half
/// merges cleanly when the families land together.
/// </summary>
internal static class DirectPtxPoolingCubinTool
{
    internal const string Family = "pooling";

    /// <summary>
    /// Every module this family can admit. Each shape is filtered through the
    /// kernel's own admission check rather than repeating its list here, so a
    /// drift between the two cannot silently skip a shipped module or try to
    /// emit one the kernel rejects.
    /// </summary>
    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        var planeShapes = new[] { (256, 128), (2048, 64), (2048, 128), (8192, 128) };
        foreach ((int rows, int spatial) in planeShapes)
        {
            if (PtxFusedGlobalAvgPoolF32Kernel.IsSupportedShape(rows, spatial))
                yield return new DirectPtxModuleSource(
                    $"global-avgpool-f32-v1-r{rows}-s{spatial}",
                    PtxFusedGlobalAvgPoolF32Kernel.EntryPoint,
                    PtxFusedGlobalAvgPoolF32Kernel.EmitPtx(8, 6, rows, spatial));

            if (PtxFusedGlobalMaxPoolF32Kernel.IsSupportedShape(rows, spatial))
                yield return new DirectPtxModuleSource(
                    $"global-maxpool-f32-v1-r{rows}-s{spatial}",
                    PtxFusedGlobalMaxPoolF32Kernel.EntryPoint,
                    PtxFusedGlobalMaxPoolF32Kernel.EmitPtx(8, 6, rows, spatial));
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-pooling-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-pooling-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-pooling-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
