using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// The reduction family's contribution to the GPU-free cubin pipeline: which
/// modules exist. Everything else lives in <see cref="DirectPtxCubinToolCore"/>.
/// </summary>
internal static class DirectPtxReductionCubinTool
{
    internal const string Family = "reduction";

    /// <summary>
    /// Every module the reduction family can admit. The row-sum and
    /// L2-normalize kernels share one closed (rows, columns) bucket set; the
    /// operator-parameterized kernel multiplies its own bucket set by four
    /// operators, each of which is a separate module.
    /// </summary>
    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        var warpRowShapes = new[] { (256, 128), (2048, 64), (2048, 128), (8192, 128) };
        foreach ((int rows, int columns) in warpRowShapes)
        {
            yield return new DirectPtxModuleSource(
                $"row-sum-f32-v1-r{rows}-c{columns}",
                PtxFusedRowReduceF32Kernel.EntryPoint,
                PtxFusedRowReduceF32Kernel.EmitPtx(8, 6, rows, columns));
            yield return new DirectPtxModuleSource(
                $"row-l2normalize-f32-v1-r{rows}-c{columns}",
                PtxFusedRowL2NormalizeF32Kernel.EntryPoint,
                PtxFusedRowL2NormalizeF32Kernel.EmitPtx(8, 6, rows, columns));
        }

        var operators = new[]
        {
            DirectPtxRowReduceOp.Mean,
            DirectPtxRowReduceOp.Max,
            DirectPtxRowReduceOp.Min,
            DirectPtxRowReduceOp.SumOfSquares,
        };
        foreach (int rows in new[] { 256, 512, 1024, 2048, 4096 })
        foreach (int columns in new[] { 128, 256, 512, 1024 })
        foreach (DirectPtxRowReduceOp op in operators)
        {
            yield return new DirectPtxModuleSource(
                $"row-reduce-{op}-f32-v1-r{rows}-c{columns}",
                PtxFusedRowReduceOpF32Kernel.EntryPointFor(op),
                PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, op, rows, columns));
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-reduction-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-reduction-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-reduction-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
