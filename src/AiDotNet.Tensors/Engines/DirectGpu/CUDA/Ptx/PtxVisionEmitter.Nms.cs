#if NET5_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static partial class PtxVisionEmitter
{
    private static DirectPtxVisionDefinition EmitNms(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        int length = spec.D0;
        RequireOneOf(length, nameof(length), 256, 1024);
        bool batched = (spec.Flags & 1) != 0;
        float threshold = BitConverter.Int32BitsToSingle(spec.ScalarBits);
        if (!float.IsFinite(threshold) || threshold < 0 || threshold > 1)
            throw new ArgumentOutOfRangeException(nameof(spec), "NMS threshold must be in [0,1].");
        var ptx = Begin(spec, ccMajor, ccMinor,
            "boxes", "scores", "class_ids", "suppressed", "output", "output_count");
        ptx.AppendLine("    .reg .pred %p<16>; .reg .b32 %r<32>; .reg .b64 %rd<24>; .reg .f32 %f<48>;");
        LoadParameters(ptx, "boxes", "scores", "class_ids", "suppressed", "output", "output_count");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x; mov.u32 %r1, %tid.x; or.b32 %r2, %r0, %r1; setp.ne.u32 %p0, %r2, 0; @%p0 bra DONE;");
        ptx.AppendLine("    mov.u32 %r3, 0; mov.u32 %r4, 0;"); // count, iteration
        ptx.AppendLine("NMS_ITERATION:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r4, {length}; @%p1 bra NMS_COMPLETE;");
        ptx.AppendLine("    mov.s32 %r5, -1; mov.f32 %f0, 0fFF7FFFFF; mov.u32 %r6, 0;");
        ptx.AppendLine("NMS_FIND:");
        ptx.AppendLine($"    setp.ge.u32 %p2, %r6, {length}; @%p2 bra NMS_FOUND;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r6, 4; add.u64 %rd7, %rd3, %rd6; ld.global.f32 %f1, [%rd7]; setp.ne.f32 %p3, %f1, 0f00000000; @%p3 bra NMS_FIND_NEXT;");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6; ld.global.f32 %f2, [%rd8]; setp.lt.s32 %p4, %r5, 0; @%p4 bra NMS_SELECT;");
        ptx.AppendLine("    testp.nan.f32 %p5, %f2; @%p5 bra NMS_FIND_NEXT; testp.nan.f32 %p6, %f0; @%p6 bra NMS_SELECT;");
        ptx.AppendLine("    setp.gt.f32 %p7, %f2, %f0; @%p7 bra NMS_SELECT; setp.ne.f32 %p8, %f2, %f0; @%p8 bra NMS_FIND_NEXT; setp.lt.u32 %p9, %r6, %r5; @!%p9 bra NMS_FIND_NEXT;");
        ptx.AppendLine("NMS_SELECT: mov.u32 %r5, %r6; mov.f32 %f0, %f2;");
        ptx.AppendLine("NMS_FIND_NEXT: add.u32 %r6, %r6, 1; bra NMS_FIND;");
        ptx.AppendLine("NMS_FOUND: setp.lt.s32 %p10, %r5, 0; @%p10 bra NMS_COMPLETE;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r5, 4; add.u64 %rd10, %rd3, %rd9; mov.f32 %f3, 0f3F800000; st.global.f32 [%rd10], %f3;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r3, 4; add.u64 %rd12, %rd4, %rd11; cvt.rn.f32.u32 %f4, %r5; st.global.f32 [%rd12], %f4; add.u32 %r3, %r3, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r5, 16; add.u64 %rd14, %rd0, %rd13; ld.global.v4.f32 {%f5,%f6,%f7,%f8}, [%rd14];");
        ptx.AppendLine("    sub.rn.f32 %f9, %f7, %f5; sub.rn.f32 %f10, %f8, %f6; setp.gt.f32 %p11, %f9, 0f00000000; setp.gt.f32 %p12, %f10, 0f00000000; and.pred %p11, %p11, %p12; mov.f32 %f11, 0f00000000; @%p11 mul.rn.f32 %f11, %f9, %f10;");
        if (batched)
            ptx.AppendLine("    add.u64 %rd15, %rd2, %rd9; ld.global.f32 %f12, [%rd15];");
        ptx.AppendLine("    mov.u32 %r7, 0;");
        ptx.AppendLine("NMS_SUPPRESS:");
        ptx.AppendLine($"    setp.ge.u32 %p2, %r7, {length}; @%p2 bra NMS_NEXT_ITERATION;");
        ptx.AppendLine("    mul.wide.u32 %rd16, %r7, 4; add.u64 %rd17, %rd3, %rd16; ld.global.f32 %f13, [%rd17]; setp.ne.f32 %p3, %f13, 0f00000000; @%p3 bra NMS_SUPPRESS_NEXT;");
        if (batched)
        {
            ptx.AppendLine("    add.u64 %rd18, %rd2, %rd16; ld.global.f32 %f14, [%rd18]; setp.ne.f32 %p4, %f14, %f12; @%p4 bra NMS_SUPPRESS_NEXT;");
        }
        ptx.AppendLine("    mul.wide.u32 %rd19, %r7, 16; add.u64 %rd20, %rd0, %rd19; ld.global.v4.f32 {%f15,%f16,%f17,%f18}, [%rd20];");
        ptx.AppendLine("    sub.rn.f32 %f19, %f17, %f15; sub.rn.f32 %f20, %f18, %f16; setp.gt.f32 %p5, %f19, 0f00000000; setp.gt.f32 %p6, %f20, 0f00000000; and.pred %p5, %p5, %p6; mov.f32 %f21, 0f00000000; @%p5 mul.rn.f32 %f21, %f19, %f20;");
        ptx.AppendLine("    min.f32 %f22, %f7, %f17; max.f32 %f23, %f5, %f15; sub.rn.f32 %f22, %f22, %f23; max.f32 %f22, %f22, 0f00000000;");
        ptx.AppendLine("    min.f32 %f23, %f8, %f18; max.f32 %f24, %f6, %f16; sub.rn.f32 %f23, %f23, %f24; max.f32 %f23, %f23, 0f00000000; mul.rn.f32 %f25, %f22, %f23;");
        ptx.AppendLine("    add.rn.f32 %f26, %f11, %f21; sub.rn.f32 %f26, %f26, %f25; setp.gt.f32 %p7, %f26, 0f00000000; @!%p7 bra NMS_SUPPRESS_NEXT; div.rn.f32 %f27, %f25, %f26;");
        ptx.AppendLine($"    setp.gt.f32 %p8, %f27, {F(threshold)}; @%p8 st.global.f32 [%rd17], %f3;");
        ptx.AppendLine("NMS_SUPPRESS_NEXT: add.u32 %r7, %r7, 1; bra NMS_SUPPRESS;");
        ptx.AppendLine("NMS_NEXT_ITERATION: add.u32 %r4, %r4, 1; bra NMS_ITERATION;");
        ptx.AppendLine("NMS_COMPLETE: cvt.rn.f32.u32 %f28, %r3; st.global.f32 [%rd5], %f28;");
        string code = Finish(ptx);
        return Definition(spec, architecture, $"n{length}-threshold-{threshold:R}-batched-{batched}",
            [
                Tensor("boxes", DirectPtxPhysicalLayout.BoxXyxy, new(length, 4), DirectPtxTensorAccess.Read),
                Tensor("scores", DirectPtxPhysicalLayout.Vector, new(length), DirectPtxTensorAccess.Read),
                Tensor("class-ids", DirectPtxPhysicalLayout.Vector,
                    new(batched ? length : 1), DirectPtxTensorAccess.Read),
                Tensor("suppressed", DirectPtxPhysicalLayout.Vector, new(length), DirectPtxTensorAccess.ReadWrite),
                Tensor("output", DirectPtxPhysicalLayout.Vector, new(length), DirectPtxTensorAccess.Write),
                Tensor("output-count", DirectPtxPhysicalLayout.Vector, new(1), DirectPtxTensorAccess.Write)
            ], Semantics(("stable-tie", "lower original index"), ("batched", batched.ToString()),
                ("threshold", threshold.ToString("R", System.Globalization.CultureInfo.InvariantCulture)),
                ("execution", "single deterministic controller thread")),
            code, 1, maxRegisters: 48, minBlocksPerSm: 1);
    }
}
#endif
