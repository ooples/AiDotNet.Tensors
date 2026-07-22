#if NET5_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static partial class PtxVisionEmitter
{
    private static DirectPtxVisionDefinition EmitMeshgrid2D(
        DirectPtxVisionSpec spec,
        DirectPtxArchitectureFamily architecture,
        int ccMajor,
        int ccMinor)
    {
        int n0 = spec.D0, n1 = spec.D1;
        int outputIndex = spec.Flags & 1;
        bool xy = (spec.Flags & 2) != 0;
        if ((n0, n1) is not ((256, 256) or (1024, 256)) ||
            outputIndex is not (0 or 1))
            throw new NotSupportedException("Meshgrid2D specialization is not emitted.");

        int total = checked(n0 * n1);
        int sourceLength = outputIndex == 0 ? n0 : n1;
        var ptx = Begin(spec, ccMajor, ccMinor, "source", "output");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<2>;");
        LoadParameters(ptx, "source", "output");
        EmitGlobalIndex(ptx, total);
        if (!xy && outputIndex == 0)
            ptx.AppendLine($"    div.u32 %r3, %r2, {n1};");
        else if (xy && outputIndex == 1)
            ptx.AppendLine($"    div.u32 %r3, %r2, {n0};");
        else
            ptx.AppendLine($"    rem.u32 %r3, %r2, {sourceLength};");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2; ld.global.f32 %f0, [%rd3];");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4; add.u64 %rd5, %rd1, %rd4; st.global.f32 [%rd5], %f0;");
        DirectPtxExtent outputExtent = xy ? new(n1, n0) : new(n0, n1);
        return Definition(spec, architecture,
            $"n0-{n0}-n1-{n1}-output-{outputIndex}-xy-{xy}",
            [
                Tensor("source", DirectPtxPhysicalLayout.Vector,
                    new(sourceLength), DirectPtxTensorAccess.Read),
                Tensor("output", DirectPtxPhysicalLayout.RowMajor2D,
                    outputExtent, DirectPtxTensorAccess.Write)
            ],
            Semantics(("indexing", xy ? "xy" : "ij"),
                ("output-index", outputIndex.ToString()),
                ("broadcast", "direct register load/store; no materialized repeat")),
            Finish(ptx), total, maxRegisters: 12, minBlocksPerSm: 4);
    }
}
#endif
