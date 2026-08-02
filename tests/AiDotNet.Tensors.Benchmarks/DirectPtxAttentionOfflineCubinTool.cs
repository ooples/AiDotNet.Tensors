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

    private static readonly (int Sequence, bool Causal, bool Epilogue, int Warps)[] BasicCases =
    [
        (16, false, false, 1),
        (16, true, true, 1),
        (32, false, false, 1),
        (32, true, true, 2),
        (64, false, false, 2),
        (64, true, true, 4),
        (128, false, false, 4),
        (128, false, true, 8),
        (128, true, false, 4),
        (128, true, true, 8)
    ];

    private static readonly
        (int Batch, int QueryHeads, int KeyValueHeads, int QuerySequence,
         int KeyValueSequence, bool Causal, int CausalQueryOffset)[] FamilyCases =
    [
        (2, 4, 2, 32, 64, false, 0),
        (2, 8, 1, 32, 64, true, 0),
        (2, 8, 2, 32, 64, true, 32),
        (1, 4, 4, 128, 32, true, 0),
        (1, 4, 2, 128, 64, true, -64)
    ];

    internal static IEnumerable<DirectPtxModuleSource> EnumerateModules()
    {
        foreach ((int sequence, bool causal, bool epilogue, int warps) in BasicCases)
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
                  int keyValueSequence, bool causal, int causalQueryOffset) in FamilyCases)
        {
            int warps = System.Math.Min(8, querySequence / 16);
            yield return new DirectPtxModuleSource(
                $"online-attention-d64-v3-family-b{batch}-hq{queryHeads}-hkv{keyValueHeads}" +
                $"-sq{querySequence}-skv{keyValueSequence}-c{Bool(causal)}-o{causalQueryOffset}-w{warps}",
                PtxOnlineFusedAttention128x64Kernel.EntryPoint,
                PtxOnlineFusedAttention128x64Kernel.EmitFamilyPtx(
                    8, 6, queryHeads, keyValueHeads, causal, fuseLayerNormGelu: false,
                    Scale, Epsilon, querySequence, keyValueSequence,
                    emitSoftmaxStats: true, warps, causalQueryOffset),
                warps * 32);
        }
    }

    internal static int Generate(string[] args) => args.Length < 3
        ? Usage("--generate-direct-ptx-attention-offline-cubins <ptxas-path> <output-directory>")
        : DirectPtxCubinToolCore.Generate(Family, EnumerateModules(), args[1], args[2]);

    internal static int Verify(string[] args) => args.Length < 3
        ? Usage("--verify-direct-ptx-attention-offline-cubins <ptxas-path> <artifact-directory>")
        : DirectPtxCubinToolCore.Verify(Family, EnumerateModules(), args[1], args[2]);

    internal static int AuditSass(string[] args) => args.Length < 4
        ? Usage("--audit-direct-ptx-attention-offline-sass <nvdisasm-path> <artifact-directory> <output-directory>")
        : DirectPtxCubinToolCore.AuditSass(Family, args[1], args[2], args[3]);

    private static int Bool(bool value) => value ? 1 : 0;

    private static int Usage(string text)
    {
        System.Console.Error.WriteLine("usage: " + text);
        return 2;
    }
}
