using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The gather family's contribution to the GPU-free cubin pipeline: which
/// modules exist. Everything else lives in <see cref="DirectPtxCubinToolCore"/>,
/// which is byte-identical on every branch that carries it so the shared half
/// merges cleanly when the families land together.
/// </summary>
internal static class DirectPtxGatherCubinTool
{
    internal const string Family = "gather";

    /// <summary>
    /// Every module this family can admit. Each shape is filtered through the
    /// kernel's own admission check rather than repeating its list here, so a
    /// drift between the two cannot silently skip a shipped module or try to
    /// emit one the kernel rejects.
    /// </summary>
    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        foreach ((int numIndices, int featureSize) in new[]
                 { (256, 128), (2048, 64), (2048, 128), (8192, 128) })
        {
            if (!PtxFusedGatherF32Kernel.IsSupportedShape(numIndices, featureSize)) continue;
            yield return new DirectPtxModuleSource(
                $"gather-f32-v1-n{numIndices}-f{featureSize}",
                PtxFusedGatherF32Kernel.EntryPoint,
                PtxFusedGatherF32Kernel.EmitPtx(8, 6, numIndices, featureSize));
        }

        // Index-select takes FLOAT indices with a truncating cast, a different
        // contract from the int-index gather, so it is a separate module set.
        foreach (int numIndices in new[] { 1024, 4096, 16_384, 65_536 })
        foreach (int innerSize in new[] { 32, 64, 128, 256, 512 })
        {
            if (!PtxFusedIndexSelectF32Kernel.IsSupportedShape(numIndices, innerSize)) continue;
            yield return new DirectPtxModuleSource(
                $"index-select-f32-v1-n{numIndices}-i{innerSize}",
                PtxFusedIndexSelectF32Kernel.EntryPoint,
                PtxFusedIndexSelectF32Kernel.EmitPtx(8, 6, numIndices, innerSize));
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-gather-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-gather-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-gather-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
