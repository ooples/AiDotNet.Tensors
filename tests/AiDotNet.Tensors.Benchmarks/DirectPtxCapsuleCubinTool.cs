using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The capsule-squash family's contribution to the GPU-free cubin pipeline (#833): which modules exist. Everything
/// else lives in <see cref="DirectPtxCubinToolCore"/>.
/// </summary>
internal static class DirectPtxCapsuleCubinTool
{
    internal const string Family = "capsule";

    /// <summary>Forward and backward for every admitted (capsules, width) bucket.</summary>
    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        foreach ((int capsules, int width) in new[] { (1152, 8), (9216, 8), (1280, 16), (10240, 16) })
        {
            // Filter through the kernel's own admission check rather than trusting this list.
            if (!PtxFusedSquashF32Kernel.IsSupportedShape(capsules, width)) continue;
            foreach (SquashDirection direction in new[] { SquashDirection.Forward, SquashDirection.Backward })
                yield return new DirectPtxModuleSource(
                    $"squash-{(direction == SquashDirection.Forward ? "forward" : "backward")}-f32-v1-n{capsules}-d{width}",
                    PtxFusedSquashF32Kernel.EntryPointOf(direction),
                    PtxFusedSquashF32Kernel.EmitPtx(8, 6, direction, capsules, width),
                    PtxFusedSquashF32Kernel.DefaultBlockThreads);
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-capsule-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-capsule-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-capsule-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
