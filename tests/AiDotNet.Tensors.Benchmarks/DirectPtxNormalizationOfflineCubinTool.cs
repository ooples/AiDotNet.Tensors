using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The normalization2 family's contribution to the GPU-free cubin pipeline: which
/// modules exist. Everything else lives in <see cref="DirectPtxCubinToolCore"/>,
/// which is byte-identical on every branch that carries it so the shared half
/// merges cleanly when the families land together.
/// </summary>
internal static class DirectPtxNormalizationOfflineCubinTool
{
    internal const string Family = "normalization2";

    /// <summary>
    /// Every module this family can admit. Each shape is filtered through the
    /// kernel's own admission check rather than repeating its list here, so a
    /// drift between the two cannot silently skip a shipped module or try to
    /// emit one the kernel rejects.
    /// </summary>
    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        // Every row-normalization operation across the admitted row counts. The
        // epsilon these kernels once baked is a launch parameter now, so the
        // module set is finite and this enumeration is complete - previously a
        // caller with a non-default epsilon produced PTX no artifact matched.
        foreach (DirectPtxRowNormalizationOperation op in
                 (DirectPtxRowNormalizationOperation[])System.Enum.GetValues(
                     typeof(DirectPtxRowNormalizationOperation)))
        foreach (int rows in new[] { 256, 2_048, 8_192 })
        {
            if (!PtxRowNormalizationD64Kernel.IsSupportedRows(rows)) continue;
            string ptx;
            try
            {
                ptx = PtxRowNormalizationD64Kernel.EmitPtx(8, 6, op, rows);
            }
            catch (System.ArgumentException)
            {
                // Not every operation admits every row count; the kernel's own
                // validation is the authority on which pairs exist.
                continue;
            }
            yield return new DirectPtxModuleSource(
                $"row-normalization-{op}-v1-r{rows}",
                PtxRowNormalizationD64Kernel.GetEntryPoint(op),
                ptx);
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-normalization2-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-normalization2-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-normalization2-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
