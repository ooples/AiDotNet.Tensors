using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The optimizer family's contribution to the GPU-free cubin pipeline: which
/// modules exist. Everything else lives in <see cref="DirectPtxCubinToolCore"/>,
/// which is byte-identical on every branch that carries it so the shared half
/// merges cleanly when the families land together.
/// </summary>
internal static class DirectPtxOptimizerCubinTool
{
    internal const string Family = "optimizer";

    /// <summary>
    /// Every module this family can admit. Each shape is filtered through the
    /// kernel's own admission check rather than repeating its list here, so a
    /// drift between the two cannot silently skip a shipped module or try to
    /// emit one the kernel rejects.
    /// </summary>
    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        // Both optimizers key their module on the shape plus whether the
        // weight-decay term is emitted; every other scalar is a launch
        // parameter, so the key space is finite and fully enumerable here.
        foreach (int size in new[] { 65_536, 262_144, 1_048_576, 4_194_304 })
        foreach (bool weightDecay in new[] { false, true })
        {
            if (PtxFusedSgdMomentumF32Kernel.IsSupportedShape(size))
                yield return new DirectPtxModuleSource(
                    $"sgd-momentum-f32-v1-n{size}-wd{(weightDecay ? 1 : 0)}",
                    PtxFusedSgdMomentumF32Kernel.EntryPoint,
                    PtxFusedSgdMomentumF32Kernel.EmitPtx(8, 6, size, weightDecay));

            if (PtxFusedAdamUpdateF32Kernel.IsSupportedShape(size))
                yield return new DirectPtxModuleSource(
                    $"adam-update-f32-v1-n{size}-wd{(weightDecay ? 1 : 0)}",
                    PtxFusedAdamUpdateF32Kernel.EntryPoint,
                    PtxFusedAdamUpdateF32Kernel.EmitPtx(8, 6, size, weightDecay));
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-optimizer-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-optimizer-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-optimizer-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
