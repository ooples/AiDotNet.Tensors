using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The spectral family's contribution to the GPU-free cubin pipeline: which
/// modules exist. Everything else lives in <see cref="DirectPtxCubinToolCore"/>,
/// which is byte-identical on every branch that carries it so the shared half
/// merges cleanly when the families land together.
/// </summary>
internal static class DirectPtxSpectralCubinTool
{
    internal const string Family = "spectral";

    /// <summary>
    /// Every module this family can admit. Each shape is filtered through the
    /// kernel's own admission check rather than repeating its list here, so a
    /// drift between the two cannot silently skip a shipped module or try to
    /// emit one the kernel rejects.
    /// </summary>
    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        foreach (int pairs in new[] { 65_536, 262_144, 1_048_576, 4_194_304 })
        {
            if (PtxFusedComplexMultiplyF32Kernel.IsSupportedShape(pairs))
                yield return new DirectPtxModuleSource(
                    $"complex-multiply-f32-v1-n{pairs}",
                    PtxFusedComplexMultiplyF32Kernel.EntryPoint,
                    PtxFusedComplexMultiplyF32Kernel.EmitPtx(8, 6, pairs));

            if (!PtxFusedComplexUnaryF32Kernel.IsSupportedShape(pairs)) continue;
            foreach (DirectPtxComplexUnaryOp op in new[]
                     {
                         DirectPtxComplexUnaryOp.Conjugate,
                         DirectPtxComplexUnaryOp.Magnitude,
                     })
            {
                yield return new DirectPtxModuleSource(
                    $"complex-{op}-f32-v1-n{pairs}",
                    PtxFusedComplexUnaryF32Kernel.EntryPointFor(op),
                    PtxFusedComplexUnaryF32Kernel.EmitPtx(8, 6, op, pairs));
            }
        }

        // The split-complex operators land on separate real/imaginary arrays
        // rather than the interleaved layout, so they are their own modules.
        foreach (int count in new[] { 65_536, 262_144, 1_048_576, 4_194_304 })
        {
            if (!PtxSplitComplexUnaryF32Kernel.IsSupportedShape(count)) continue;
            foreach (DirectPtxSplitComplexUnaryOp op in new[]
                     {
                         DirectPtxSplitComplexUnaryOp.Magnitude,
                         DirectPtxSplitComplexUnaryOp.MagnitudeSquared,
                     })
            {
                yield return new DirectPtxModuleSource(
                    $"split-complex-{op}-f32-v1-n{count}",
                    PtxSplitComplexUnaryF32Kernel.EntryPointFor(op),
                    PtxSplitComplexUnaryF32Kernel.EmitPtx(8, 6, op, count));
            }
        }

        // The split-complex binary operators and the split conjugate landed
        // after this gate was first written; enumerating them here keeps every
        // kernel in the family covered by machine-code evidence.
        foreach (int count in new[] { 65_536, 262_144, 1_048_576, 4_194_304 })
        {
            if (PtxSplitComplexBinaryF32Kernel.IsSupportedShape(count))
            {
                foreach (DirectPtxSplitComplexBinaryOp op in new[]
                         {
                             DirectPtxSplitComplexBinaryOp.Multiply,
                             DirectPtxSplitComplexBinaryOp.Add,
                         })
                {
                    yield return new DirectPtxModuleSource(
                        $"split-complex-{op}-f32-v1-n{count}",
                        PtxSplitComplexBinaryF32Kernel.EntryPointFor(op),
                        PtxSplitComplexBinaryF32Kernel.EmitPtx(8, 6, op, count));
                }
            }

            if (PtxSplitComplexConjugateF32Kernel.IsSupportedShape(count))
                yield return new DirectPtxModuleSource(
                    $"split-complex-conjugate-f32-v1-n{count}",
                    PtxSplitComplexConjugateF32Kernel.EntryPoint,
                    PtxSplitComplexConjugateF32Kernel.EmitPtx(8, 6, count));
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-spectral-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-spectral-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-spectral-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
