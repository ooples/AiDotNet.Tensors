using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The loss family's contribution to the GPU-free cubin pipeline: which
/// modules exist. Everything else lives in <see cref="DirectPtxCubinToolCore"/>,
/// which is byte-identical on every branch that carries it so the shared half
/// merges cleanly when the families land together.
/// </summary>
internal static class DirectPtxLossCubinTool
{
    internal const string Family = "loss";

    /// <summary>
    /// Every module this family can admit. Each shape is filtered through the
    /// kernel's own admission check rather than repeating its list here, so a
    /// drift between the two cannot silently skip a shipped module or try to
    /// emit one the kernel rejects.
    /// </summary>
    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        foreach ((int rows, int columns) in new[]
                 { (256, 128), (2048, 64), (2048, 128), (8192, 128) })
        {
            if (!PtxFusedMseLossF32Kernel.IsSupportedShape(rows, columns)) continue;
            yield return new DirectPtxModuleSource(
                $"mse-loss-f32-v1-r{rows}-c{columns}",
                PtxFusedMseLossF32Kernel.EntryPoint,
                PtxFusedMseLossF32Kernel.EmitPtx(8, 6, rows, columns));
        }

        // The two gradient operators have different launch ABIs - MSE takes a
        // broadcast gradOutput scalar and a scale, MAE takes neither - so each
        // is its own module.
        foreach (int size in new[] { 65_536, 262_144, 1_048_576, 4_194_304 })
        {
            if (!PtxFusedLossBackwardF32Kernel.IsSupportedShape(size)) continue;
            foreach (DirectPtxLossBackwardOp op in new[]
                     {
                         DirectPtxLossBackwardOp.MeanSquaredError,
                         DirectPtxLossBackwardOp.MeanAbsoluteError,
                     })
            {
                yield return new DirectPtxModuleSource(
                    $"loss-backward-{op}-f32-v1-n{size}",
                    PtxFusedLossBackwardF32Kernel.EntryPointFor(op),
                    PtxFusedLossBackwardF32Kernel.EmitPtx(8, 6, op, size));
            }
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-loss-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-loss-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-loss-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
