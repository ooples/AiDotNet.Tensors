using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Defines the complete SM86 online-attention release matrix for the host-only
/// cubin pipeline. The same rows are exercised by <c>DirectPtxWmmaTests</c>.
/// Caller-controlled epsilon is a launch parameter, so this finite matrix is
/// exhaustive rather than covering only one baked scalar value.
/// </summary>
internal static class DirectPtxAttentionOfflineCubinTool
{
    internal const string Family = "attention";
    private const float Scale = 0.125f;
    private const float Epsilon = 1e-5f;

    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        foreach ((int sequence, bool causal, bool epilogue, int warps) in
                 DirectPtxOnlineAttentionReleaseMatrix.BasicCases)
        {
            yield return new DirectPtxModuleSource(
                $"online-attention-d64-v3-basic-s{sequence}-c{Bool(causal)}-e{Bool(epilogue)}-w{warps}",
                PtxOnlineFusedAttention128x64Kernel.EntryPoint,
                PtxOnlineFusedAttention128x64Kernel.EmitPtx(
                    8, 6, causal, epilogue, Scale, Epsilon, sequence,
                    emitSoftmaxStats: true, warps),
                warps * 32);
        }

        foreach ((int batch, int queryHeads, int keyValueHeads, int querySequence,
                  int keyValueSequence, bool causal, int causalQueryOffset, bool epilogue) in
                 DirectPtxOnlineAttentionReleaseMatrix.FamilyCases)
        {
            int warps = System.Math.Min(
                8, querySequence / PtxOnlineFusedAttention128x64Kernel.QueryTileRows);
            yield return new DirectPtxModuleSource(
                $"online-attention-d64-v3-family-b{batch}-hq{queryHeads}-hkv{keyValueHeads}" +
                $"-sq{querySequence}-skv{keyValueSequence}-c{Bool(causal)}-o{causalQueryOffset}" +
                $"-e{Bool(epilogue)}-w{warps}",
                PtxOnlineFusedAttention128x64Kernel.EntryPoint,
                PtxOnlineFusedAttention128x64Kernel.EmitFamilyPtx(
                    8, 6, queryHeads, keyValueHeads, causal, epilogue,
                    Scale, Epsilon, querySequence, keyValueSequence,
                    emitSoftmaxStats: true, warps, causalQueryOffset),
                warps * 32);
        }
    }

    internal static int Generate(string[] args) => args.Length < 2
        ? Usage("--generate-direct-ptx-attention-offline-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[0], args[1]);

    internal static int Verify(string[] args) => args.Length < 2
        ? Usage("--verify-direct-ptx-attention-offline-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[0], args[1]);

    internal static int AuditSass(string[] args) => args.Length < 3
        ? Usage("--audit-direct-ptx-attention-offline-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[0], args[1], args[2]);

    private static int Bool(bool value) => value ? 1 : 0;

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
