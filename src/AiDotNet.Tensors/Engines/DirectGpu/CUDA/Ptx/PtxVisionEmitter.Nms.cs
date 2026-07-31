using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static partial class PtxVisionEmitter
{
    private static DirectPtxVisionDefinition EmitNms(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        const uint BlockThreads = 256;
        const int SharedBytes = checked((int)(BlockThreads * 2 * sizeof(float)));
        int length = spec.D0;
        RequireOneOf(length, nameof(length), 256, 1024);
        bool batched = (spec.Flags & 1) != 0;
        float threshold = PtxCompat.Int32BitsToSingle(spec.ScalarBits);
        if (!PtxCompat.IsFinite(threshold) || threshold < 0 || threshold > 1)
            throw new ArgumentOutOfRangeException(nameof(spec), "NMS threshold must be in [0,1].");
        var ptx = Begin(spec, ccMajor, ccMinor,
            "boxes", "scores", "class_ids", "suppressed", "output", "output_count");
        ptx.AppendLine("    .reg .pred %p<16>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<32>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        ptx.AppendLine($"    .shared .align 4 .b32 nms_scores[{BlockThreads}];");
        ptx.AppendLine($"    .shared .align 4 .b32 nms_indices[{BlockThreads}];");
        LoadParameters(ptx, "boxes", "scores", "class_ids", "suppressed", "output", "output_count");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, 0;"); // kept count, identical in every thread
        ptx.AppendLine("    mov.u64 %rd6, nms_scores;");
        ptx.AppendLine("    mov.u64 %rd7, nms_indices;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd6, %rd8;");
        ptx.AppendLine("    add.u64 %rd10, %rd7, %rd8;");
        ptx.AppendLine("NMS_ITERATION:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r1, {length}; @%p0 bra NMS_COMPLETE;");
        EmitNmsBlockArgMax(ptx, length, BlockThreads);
        ptx.AppendLine("    ld.shared.s32 %r6, [%rd7];");
        ptx.AppendLine("    setp.lt.s32 %p0, %r6, 0; @%p0 bra NMS_COMPLETE;");
        ptx.AppendLine("    mov.f32 %f28, 0f3F800000;");
        ptx.AppendLine("    setp.ne.u32 %p1, %r0, 0; @%p1 bra NMS_SELECTED;");
        ptx.AppendLine("    mul.wide.u32 %rd16, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd17, %rd3, %rd16; st.global.f32 [%rd17], %f28;");
        ptx.AppendLine("    mul.wide.u32 %rd18, %r1, 4; add.u64 %rd19, %rd4, %rd18;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f29, %r6; st.global.f32 [%rd19], %f29;");
        ptx.AppendLine("NMS_SELECTED:");
        ptx.AppendLine("    bar.sync 0;");
        EmitNmsBlockSuppression(ptx, length, threshold, batched, BlockThreads);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    add.u32 %r1, %r1, 1; bra NMS_ITERATION;");
        ptx.AppendLine("NMS_COMPLETE:");
        ptx.AppendLine("    setp.ne.u32 %p0, %r0, 0; @%p0 bra DONE;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f29, %r1; st.global.f32 [%rd5], %f29;");
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
                ("execution", "deterministic cooperative block: parallel argmax and suppression")),
            code, 1, maxRegisters: 48, minBlocksPerSm: 1,
            blockThreads: BlockThreads, maxStaticSharedBytes: SharedBytes);
    }

    private static void EmitNmsBlockArgMax(
        StringBuilder ptx, int length, uint blockThreads)
    {
        ptx.AppendLine("    mov.s32 %r3, -1; mov.f32 %f0, 0fFF7FFFFF; mov.u32 %r2, %r0;");
        ptx.AppendLine("NMS_FIND_LOCAL:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r2, {length}; @%p1 bra NMS_FIND_LOCAL_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r2, 4; add.u64 %rd12, %rd3, %rd11;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd12]; setp.ne.f32 %p2, %f1, 0f00000000; @%p2 bra NMS_FIND_LOCAL_NEXT;");
        ptx.AppendLine("    add.u64 %rd13, %rd1, %rd11; ld.global.f32 %f2, [%rd13];");
        ptx.AppendLine("    setp.lt.s32 %p3, %r3, 0; @%p3 bra NMS_FIND_LOCAL_SELECT;");
        ptx.AppendLine("    testp.notanumber.f32 %p4, %f2; @%p4 bra NMS_FIND_LOCAL_NEXT;");
        ptx.AppendLine("    testp.notanumber.f32 %p5, %f0; @%p5 bra NMS_FIND_LOCAL_SELECT;");
        ptx.AppendLine("    setp.gt.f32 %p6, %f2, %f0; @%p6 bra NMS_FIND_LOCAL_SELECT;");
        ptx.AppendLine("    setp.ne.f32 %p7, %f2, %f0; @%p7 bra NMS_FIND_LOCAL_NEXT;");
        ptx.AppendLine("    setp.lt.u32 %p8, %r2, %r3; @!%p8 bra NMS_FIND_LOCAL_NEXT;");
        ptx.AppendLine("NMS_FIND_LOCAL_SELECT: mov.u32 %r3, %r2; mov.f32 %f0, %f2;");
        ptx.AppendLine($"NMS_FIND_LOCAL_NEXT: add.u32 %r2, %r2, {blockThreads}; bra NMS_FIND_LOCAL;");
        ptx.AppendLine("NMS_FIND_LOCAL_DONE:");
        ptx.AppendLine("    st.shared.f32 [%rd9], %f0; st.shared.u32 [%rd10], %r3; bar.sync 0;");

        for (uint offset = blockThreads / 2; offset > 0; offset >>= 1)
        {
            string choose = $"NMS_REDUCE_CHOOSE_{offset}";
            string store = $"NMS_REDUCE_STORE_{offset}";
            string sync = $"NMS_REDUCE_SYNC_{offset}";
            ptx.AppendLine($"    setp.ge.u32 %p0, %r0, {offset}; @%p0 bra {sync};");
            ptx.AppendLine($"    ld.shared.f32 %f3, [%rd9+{offset * 4}]; ld.shared.s32 %r4, [%rd10+{offset * 4}];");
            ptx.AppendLine($"    setp.lt.s32 %p1, %r4, 0; @%p1 bra {store};");
            ptx.AppendLine($"    setp.lt.s32 %p2, %r3, 0; @%p2 bra {choose};");
            ptx.AppendLine($"    testp.notanumber.f32 %p3, %f3; @%p3 bra {store};");
            ptx.AppendLine($"    testp.notanumber.f32 %p4, %f0; @%p4 bra {choose};");
            ptx.AppendLine($"    setp.gt.f32 %p5, %f3, %f0; @%p5 bra {choose};");
            ptx.AppendLine($"    setp.ne.f32 %p6, %f3, %f0; @%p6 bra {store};");
            ptx.AppendLine($"    setp.lt.u32 %p7, %r4, %r3; @%p7 bra {choose}; bra {store};");
            ptx.AppendLine($"{choose}: mov.u32 %r3, %r4; mov.f32 %f0, %f3;");
            ptx.AppendLine($"{store}: st.shared.f32 [%rd9], %f0; st.shared.u32 [%rd10], %r3;");
            ptx.AppendLine($"{sync}: bar.sync 0;");
        }
    }

    private static void EmitNmsBlockSuppression(
        StringBuilder ptx, int length, float threshold, bool batched,
        uint blockThreads)
    {
        ptx.AppendLine("    mul.wide.u32 %rd16, %r6, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd17, %r6, 16; add.u64 %rd18, %rd0, %rd17;");
        ptx.AppendLine("    ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd18];");
        ptx.AppendLine("    sub.rn.f32 %f8, %f6, %f4; sub.rn.f32 %f9, %f7, %f5;");
        ptx.AppendLine("    setp.gt.f32 %p1, %f8, 0f00000000; setp.gt.f32 %p2, %f9, 0f00000000; and.pred %p1, %p1, %p2;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000; @%p1 mul.rn.f32 %f10, %f8, %f9;");
        if (batched)
            ptx.AppendLine("    add.u64 %rd19, %rd2, %rd16; ld.global.f32 %f11, [%rd19];");
        ptx.AppendLine("    mov.u32 %r7, %r0;");
        ptx.AppendLine("NMS_SUPPRESS_LOCAL:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r7, {length}; @%p1 bra NMS_SUPPRESS_LOCAL_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd20, %r7, 4; add.u64 %rd21, %rd3, %rd20;");
        ptx.AppendLine("    ld.global.f32 %f12, [%rd21]; setp.ne.f32 %p2, %f12, 0f00000000; @%p2 bra NMS_SUPPRESS_LOCAL_NEXT;");
        if (batched)
        {
            ptx.AppendLine("    add.u64 %rd22, %rd2, %rd20; ld.global.f32 %f13, [%rd22];");
            ptx.AppendLine("    setp.ne.f32 %p3, %f13, %f11; @%p3 bra NMS_SUPPRESS_LOCAL_NEXT;");
        }
        ptx.AppendLine("    mul.wide.u32 %rd23, %r7, 16; add.u64 %rd24, %rd0, %rd23;");
        ptx.AppendLine("    ld.global.v4.f32 {%f14,%f15,%f16,%f17}, [%rd24];");
        ptx.AppendLine("    sub.rn.f32 %f18, %f16, %f14; sub.rn.f32 %f19, %f17, %f15;");
        ptx.AppendLine("    setp.gt.f32 %p4, %f18, 0f00000000; setp.gt.f32 %p5, %f19, 0f00000000; and.pred %p4, %p4, %p5;");
        ptx.AppendLine("    mov.f32 %f20, 0f00000000; @%p4 mul.rn.f32 %f20, %f18, %f19;");
        ptx.AppendLine("    min.f32 %f21, %f6, %f16; max.f32 %f22, %f4, %f14; sub.rn.f32 %f21, %f21, %f22; max.f32 %f21, %f21, 0f00000000;");
        ptx.AppendLine("    min.f32 %f22, %f7, %f17; max.f32 %f23, %f5, %f15; sub.rn.f32 %f22, %f22, %f23; max.f32 %f22, %f22, 0f00000000;");
        ptx.AppendLine("    mul.rn.f32 %f24, %f21, %f22; add.rn.f32 %f25, %f10, %f20; sub.rn.f32 %f25, %f25, %f24;");
        ptx.AppendLine("    setp.gt.f32 %p6, %f25, 0f00000000; @!%p6 bra NMS_SUPPRESS_LOCAL_NEXT;");
        ptx.AppendLine("    div.approx.f32 %f26, %f24, %f25;");
        ptx.AppendLine($"    setp.gt.f32 %p7, %f26, {F(threshold)}; @%p7 st.global.f32 [%rd21], %f28;");
        ptx.AppendLine($"NMS_SUPPRESS_LOCAL_NEXT: add.u32 %r7, %r7, {blockThreads}; bra NMS_SUPPRESS_LOCAL;");
        ptx.AppendLine("NMS_SUPPRESS_LOCAL_DONE:");
    }
}
