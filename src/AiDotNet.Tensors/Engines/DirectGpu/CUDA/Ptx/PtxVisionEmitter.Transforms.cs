#if NET5_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static partial class PtxVisionEmitter
{
    private static DirectPtxVisionDefinition EmitCross3(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        int outer = spec.D0, inner = spec.D1;
        if ((outer, inner) is not ((256, 1) or (1024, 1) or (256, 64)))
            throw new NotSupportedException($"Cross3 [{outer},3,{inner}] is not emitted.");
        int vectors = checked(outer * inner);
        var ptx = Begin(spec, ccMajor, ccMinor, "a", "b", "output");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        LoadParameters(ptx, "a", "b", "output");
        EmitGlobalIndex(ptx, vectors);
        if (inner == 1)
            ptx.AppendLine("    mul.lo.u32 %r3, %r2, 3;");
        else
        {
            ptx.AppendLine($"    rem.u32 %r4, %r2, {inner};");
            ptx.AppendLine($"    div.u32 %r5, %r2, {inner};");
            ptx.AppendLine($"    mad.lo.u32 %r3, %r5, {3 * inner}, %r4;");
        }
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3; add.u64 %rd5, %rd1, %rd3; add.u64 %rd6, %rd2, %rd3;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd4]; ld.global.f32 %f1, [%rd4+" + (inner * 4) + "]; ld.global.f32 %f2, [%rd4+" + (inner * 8) + "];");
        ptx.AppendLine("    ld.global.f32 %f3, [%rd5]; ld.global.f32 %f4, [%rd5+" + (inner * 4) + "]; ld.global.f32 %f5, [%rd5+" + (inner * 8) + "];");
        ptx.AppendLine("    neg.f32 %f9, %f4; neg.f32 %f10, %f5; neg.f32 %f11, %f3;");
        ptx.AppendLine("    mul.rn.f32 %f6, %f1, %f5; fma.rn.f32 %f6, %f2, %f9, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f2, %f3; fma.rn.f32 %f7, %f0, %f10, %f7;");
        ptx.AppendLine("    mul.rn.f32 %f8, %f0, %f4; fma.rn.f32 %f8, %f1, %f11, %f8;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f6; st.global.f32 [%rd6+" + (inner * 4) + "], %f7; st.global.f32 [%rd6+" + (inner * 8) + "], %f8;");
        return Definition(spec, architecture, $"outer{outer}-inner{inner}",
            [
                Tensor("a", DirectPtxPhysicalLayout.RowMajor3D, new(outer, 3, inner), DirectPtxTensorAccess.Read),
                Tensor("b", DirectPtxPhysicalLayout.RowMajor3D, new(outer, 3, inner), DirectPtxTensorAccess.Read),
                Tensor("output", DirectPtxPhysicalLayout.RowMajor3D, new(outer, 3, inner), DirectPtxTensorAccess.Write)
            ], Semantics(("axis-size", "3"), ("inner-size", inner.ToString())),
            Finish(ptx), vectors, maxRegisters: 24, minBlocksPerSm: 4);
    }

    private static DirectPtxVisionDefinition EmitMasksToBoxes(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        int n = spec.D0, h = spec.D1, w = spec.D2;
        if ((n, h, w) is not ((256, 28, 28) or (64, 64, 64)))
            throw new NotSupportedException("MasksToBoxes shape is not emitted.");
        var ptx = Begin(spec, ccMajor, ccMinor, "masks", "boxes");
        ptx.AppendLine("    .reg .pred %p<8>; .reg .b32 %r<24>; .reg .b64 %rd<12>; .reg .f32 %f<8>;");
        LoadParameters(ptx, "masks", "boxes"); EmitGlobalIndex(ptx, n);
        ptx.AppendLine($"    mov.u32 %r3, {w}; mov.u32 %r4, {h}; mov.s32 %r5, -1; mov.s32 %r6, -1; mov.u32 %r7, 0;");
        ptx.AppendLine($"    mul.lo.u32 %r8, %r2, {h * w};");
        ptx.AppendLine("MASK_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r7, {h * w}; @%p1 bra MASK_DONE;");
        ptx.AppendLine("    add.u32 %r9, %r8, %r7; mul.wide.u32 %rd2, %r9, 4; add.u64 %rd3, %rd0, %rd2; ld.global.f32 %f0, [%rd3];");
        ptx.AppendLine("    setp.eq.f32 %p2, %f0, 0f00000000; @%p2 bra MASK_NEXT;");
        ptx.AppendLine($"    rem.u32 %r10, %r7, {w}; div.u32 %r11, %r7, {w};");
        ptx.AppendLine("    min.u32 %r3, %r3, %r10; min.u32 %r4, %r4, %r11; max.s32 %r5, %r5, %r10; max.s32 %r6, %r6, %r11;");
        ptx.AppendLine("MASK_NEXT: add.u32 %r7, %r7, 1; bra MASK_LOOP;");
        ptx.AppendLine("MASK_DONE:");
        ptx.AppendLine("    setp.lt.s32 %p3, %r5, 0; selp.u32 %r3, 0, %r3, %p3; selp.u32 %r4, 0, %r4, %p3; selp.u32 %r5, 0, %r5, %p3; selp.u32 %r6, 0, %r6, %p3;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f1, %r3; cvt.rn.f32.u32 %f2, %r4; cvt.rn.f32.u32 %f3, %r5; cvt.rn.f32.u32 %f4, %r6;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 16; add.u64 %rd5, %rd1, %rd4; st.global.v4.f32 [%rd5], {%f1,%f2,%f3,%f4};");
        return Definition(spec, architecture, $"n{n}-{h}x{w}",
            [
                Tensor("masks", DirectPtxPhysicalLayout.RowMajor3D, new(n, h, w), DirectPtxTensorAccess.Read),
                Tensor("boxes", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Write)
            ], Semantics(("empty-mask", "zero box"), ("predicate", "nonzero")),
            Finish(ptx), n, maxRegisters: 32, minBlocksPerSm: 2);
    }
}
#endif
