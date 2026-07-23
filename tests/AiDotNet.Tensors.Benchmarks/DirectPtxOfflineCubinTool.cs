using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The layout family's contribution to the GPU-free cubin pipeline: which
/// modules exist. Everything else - compiling, hashing, manifesting, verifying,
/// and disassembling - lives in <see cref="DirectPtxCubinToolCore"/>, which is
/// identical on every branch that carries it so the families merge cleanly.
/// </summary>
internal static class DirectPtxOfflineCubinTool
{
    internal const string Family = "layout";

    /// <summary>
    /// Every module the layout family can admit. The shape domains are closed,
    /// so this enumeration is complete by construction: if a kernel gains a
    /// shape, this list grows with it and the gate sees the new cubin.
    /// </summary>
    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        foreach (int size in new[] { 65_536, 262_144, 1_048_576, 4_194_304 })
        {
            yield return new DirectPtxModuleSource(
                $"cast-f32-to-f16-v1-n{size}",
                PtxFusedCastF32ToF16Kernel.EntryPoint,
                PtxFusedCastF32ToF16Kernel.EmitPtx(8, 6, size));
            yield return new DirectPtxModuleSource(
                $"cast-f16-to-f32-v1-n{size}",
                PtxFusedCastF16ToF32Kernel.EntryPoint,
                PtxFusedCastF16ToF32Kernel.EmitPtx(8, 6, size));
        }

        foreach (int rows in new[] { 512, 1024, 2048, 4096 })
        foreach (int columns in new[] { 512, 1024, 2048, 4096 })
        {
            yield return new DirectPtxModuleSource(
                $"transpose2d-f32-v1-r{rows}-c{columns}",
                PtxFusedTranspose2DF32Kernel.EntryPoint,
                PtxFusedTranspose2DF32Kernel.EmitPtx(8, 6, rows, columns));
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-layout-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-layout-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-layout-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
