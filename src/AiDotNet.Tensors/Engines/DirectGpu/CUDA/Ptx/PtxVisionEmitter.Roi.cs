using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static partial class PtxVisionEmitter
{
    private static DirectPtxVisionDefinition EmitRoi(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        int n = spec.D0, c = spec.D1, h = spec.D2, w = spec.D3;
        int k = spec.D4, outH = spec.D5, outW = spec.D6, outputChannels = spec.D7;
        int sampling = spec.Flags & 0xff;
        bool aligned = (spec.Flags & 0x100) != 0;
        float spatialScale = PtxCompat.Int32BitsToSingle(spec.ScalarBits);
        bool positionSensitive = spec.Operation is DirectPtxVisionOperation.PsRoiAlign or
            DirectPtxVisionOperation.PsRoiPool;
        bool align = spec.Operation is DirectPtxVisionOperation.RoiAlign or
            DirectPtxVisionOperation.PsRoiAlign;
        if (!positionSensitive && (n, c, h, w, k, outH, outW) != (1, 256, 56, 56, 256, 7, 7))
            throw new NotSupportedException("RoI specialization is not emitted.");
        if (positionSensitive &&
            (n, c, h, w, k, outH, outW, outputChannels) != (1, 196, 56, 56, 256, 7, 7, 4))
            throw new NotSupportedException("Position-sensitive RoI specialization is not emitted.");
        if (align && sampling != 2)
            throw new NotSupportedException("The first RoIAlign specialization bakes samplingRatio=2.");
        if (!PtxCompat.IsFinite(spatialScale) || spatialScale <= 0)
            throw new ArgumentOutOfRangeException(nameof(spec), "Spatial scale must be finite and positive.");

        int outputC = positionSensitive ? outputChannels : c;
        int total = checked(k * outputC * outH * outW);
        var ptx = Begin(spec, ccMajor, ccMinor, "input", "boxes", "output");
        ptx.AppendLine("    .reg .pred %p<16>; .reg .b32 %r<40>; .reg .b64 %rd<24>; .reg .f32 %f<64>;");
        LoadParameters(ptx, "input", "boxes", "output"); EmitGlobalIndex(ptx, total);
        ptx.AppendLine($"    rem.u32 %r3, %r2, {outW}; div.u32 %r4, %r2, {outW}; rem.u32 %r5, %r4, {outH}; div.u32 %r6, %r4, {outH}; rem.u32 %r7, %r6, {outputC}; div.u32 %r8, %r6, {outputC};");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r8, 20; add.u64 %rd4, %rd1, %rd3; ld.global.f32 %f0, [%rd4]; cvt.rzi.s32.f32 %r9, %f0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r9, 0; setp.ge.s32 %p2, %r9, {n}; or.pred %p3, %p1, %p2; @%p3 bra ROI_ZERO;");
        ptx.AppendLine("    ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd4+4];");
        ptx.AppendLine($"    mul.rn.f32 %f1, %f1, {F(spatialScale)}; mul.rn.f32 %f2, %f2, {F(spatialScale)}; mul.rn.f32 %f3, %f3, {F(spatialScale)}; mul.rn.f32 %f4, %f4, {F(spatialScale)};");
        if (align && aligned)
        {
            ptx.AppendLine($"    sub.rn.f32 %f1, %f1, {F(0.5f)}; sub.rn.f32 %f2, %f2, {F(0.5f)}; sub.rn.f32 %f3, %f3, {F(0.5f)}; sub.rn.f32 %f4, %f4, {F(0.5f)};");
        }
        if (spec.Operation == DirectPtxVisionOperation.RoiPool)
        {
            // CUDA roundf is halfway-away-from-zero, whereas cvt.rni uses
            // ties-to-even. Emit the exact established rule before baking
            // the integer RoIPool bin boundaries.
            EmitRoundfToS32(ptx, "%f1", "%r30", 4);
            EmitRoundfToS32(ptx, "%f2", "%r31", 4);
            EmitRoundfToS32(ptx, "%f3", "%r32", 4);
            EmitRoundfToS32(ptx, "%f4", "%r33", 4);
            ptx.AppendLine("    cvt.rn.f32.s32 %f1, %r30; cvt.rn.f32.s32 %f2, %r31; cvt.rn.f32.s32 %f3, %r32; cvt.rn.f32.s32 %f4, %r33;");
        }
        ptx.AppendLine("    sub.rn.f32 %f5, %f3, %f1; sub.rn.f32 %f6, %f4, %f2;");
        if (spec.Operation == DirectPtxVisionOperation.RoiPool)
            ptx.AppendLine("    add.rn.f32 %f5, %f5, 0f3F800000; add.rn.f32 %f6, %f6, 0f3F800000;");
        if (positionSensitive)
            ptx.AppendLine($"    max.f32 %f5, %f5, {F(0.1f)}; max.f32 %f6, %f6, {F(0.1f)};");
        else if (!aligned)
            ptx.AppendLine("    max.f32 %f5, %f5, 0f3F800000; max.f32 %f6, %f6, 0f3F800000;");
        ptx.AppendLine($"    mul.rn.f32 %f7, %f5, {F(1f / outW)}; mul.rn.f32 %f8, %f6, {F(1f / outH)};");
        if (positionSensitive)
            ptx.AppendLine($"    mad.lo.u32 %r10, %r7, {outH}, %r5; mad.lo.u32 %r10, %r10, {outW}, %r3;");
        else
            ptx.AppendLine("    mov.u32 %r10, %r7;");
        ptx.AppendLine($"    mad.lo.u32 %r11, %r9, {c}, %r10; mul.lo.u32 %r11, %r11, {h * w};");

        if (align)
        {
            ptx.AppendLine("    mov.f32 %f20, 0f00000000;");
            int sampleId = 0;
            for (int iy = 0; iy < 2; iy++)
            for (int ix = 0; ix < 2; ix++)
            {
                ptx.AppendLine($"    cvt.rn.f32.u32 %f21, %r5; fma.rn.f32 %f21, %f21, %f8, %f2; fma.rn.f32 %f21, %f8, {F((iy + 0.5f) / 2f)}, %f21;");
                ptx.AppendLine($"    cvt.rn.f32.u32 %f22, %r3; fma.rn.f32 %f22, %f22, %f7, %f1; fma.rn.f32 %f22, %f7, {F((ix + 0.5f) / 2f)}, %f22;");
                EmitRoiBilinearSample(ptx, sampleId++, h, w, "%r11", "%f21", "%f22", "%f29");
                ptx.AppendLine("    add.rn.f32 %f20, %f20, %f29;");
            }
            ptx.AppendLine($"    mul.rn.f32 %f20, %f20, {F(0.25f)};");
        }
        else
        {
            // Baked per-bin integer bounds. RoIPool uses max; PSRoIPool uses
            // the established average semantics from CudaRoiKernels.
            ptx.AppendLine("    cvt.rn.f32.u32 %f21, %r5; mul.rn.f32 %f21, %f21, %f8; add.rn.f32 %f21, %f21, %f2;");
            ptx.AppendLine("    add.u32 %r12, %r5, 1; cvt.rn.f32.u32 %f22, %r12; mul.rn.f32 %f22, %f22, %f8; add.rn.f32 %f22, %f22, %f2;");
            ptx.AppendLine("    cvt.rmi.s32.f32 %r13, %f21; cvt.rpi.s32.f32 %r14, %f22;");
            ptx.AppendLine("    cvt.rn.f32.u32 %f23, %r3; mul.rn.f32 %f23, %f23, %f7; add.rn.f32 %f23, %f23, %f1;");
            ptx.AppendLine("    add.u32 %r15, %r3, 1; cvt.rn.f32.u32 %f24, %r15; mul.rn.f32 %f24, %f24, %f7; add.rn.f32 %f24, %f24, %f1;");
            ptx.AppendLine("    cvt.rmi.s32.f32 %r16, %f23; cvt.rpi.s32.f32 %r17, %f24;");
            ptx.AppendLine($"    max.s32 %r13, %r13, 0; min.s32 %r14, %r14, {h}; max.s32 %r16, %r16, 0; min.s32 %r17, %r17, {w};");
            ptx.AppendLine("    setp.ge.s32 %p4, %r13, %r14; setp.ge.s32 %p5, %r16, %r17; or.pred %p6, %p4, %p5; @%p6 bra ROI_ZERO;");
            if (positionSensitive)
                ptx.AppendLine("    mov.f32 %f20, 0f00000000; mov.u32 %r18, 0;");
            else
                ptx.AppendLine("    mov.f32 %f20, 0fFF7FFFFF;");
            ptx.AppendLine("    mov.u32 %r19, %r13;");
            ptx.AppendLine("ROI_Y_LOOP: setp.ge.s32 %p7, %r19, %r14; @%p7 bra ROI_REDUCE_DONE; mov.u32 %r20, %r16;");
            ptx.AppendLine("ROI_X_LOOP: setp.ge.s32 %p8, %r20, %r17; @%p8 bra ROI_Y_NEXT;");
            ptx.AppendLine($"    mad.lo.u32 %r21, %r19, {w}, %r20; add.u32 %r21, %r21, %r11; mul.wide.u32 %rd5, %r21, 4; add.u64 %rd6, %rd0, %rd5; ld.global.f32 %f25, [%rd6];");
            if (positionSensitive)
                ptx.AppendLine("    add.rn.f32 %f20, %f20, %f25; add.u32 %r18, %r18, 1;");
            else
                ptx.AppendLine("    max.f32 %f20, %f20, %f25;");
            ptx.AppendLine("    add.u32 %r20, %r20, 1; bra ROI_X_LOOP;");
            ptx.AppendLine("ROI_Y_NEXT: add.u32 %r19, %r19, 1; bra ROI_Y_LOOP;");
            ptx.AppendLine("ROI_REDUCE_DONE:");
            if (positionSensitive)
            {
                ptx.AppendLine("    setp.eq.u32 %p9, %r18, 0; @%p9 bra ROI_ZERO; cvt.rn.f32.u32 %f26, %r18; div.rn.f32 %f20, %f20, %f26;");
            }
        }

        ptx.AppendLine("    mul.wide.u32 %rd20, %r2, 4; add.u64 %rd21, %rd2, %rd20; st.global.f32 [%rd21], %f20; bra DONE;");
        ptx.AppendLine("ROI_ZERO: mul.wide.u32 %rd20, %r2, 4; add.u64 %rd21, %rd2, %rd20; mov.f32 %f20, 0f00000000; st.global.f32 [%rd21], %f20;");
        DirectPtxExtent inputExtent = new(n, c, h, w);
        DirectPtxExtent outputExtent = new(k, outputC, outH, outW);
        return Definition(spec, architecture,
            $"n{n}-c{c}-{h}x{w}-k{k}-{outH}x{outW}-oc{outputChannels}-s{sampling}-a{aligned}",
            [
                Tensor("input", DirectPtxPhysicalLayout.Nchw, inputExtent, DirectPtxTensorAccess.Read),
                Tensor("boxes", DirectPtxPhysicalLayout.RoiBoxes, new(k, 5), DirectPtxTensorAccess.Read),
                Tensor("output", DirectPtxPhysicalLayout.Nchw, outputExtent, DirectPtxTensorAccess.Write)
            ], Semantics(("operation", spec.Operation.ToString()),
                ("spatial-scale", spatialScale.ToString("R", System.Globalization.CultureInfo.InvariantCulture)),
                ("sampling-ratio", sampling.ToString()), ("aligned", aligned.ToString()),
                ("position-sensitive", positionSensitive.ToString())),
            Finish(ptx), total, maxRegisters: 64, minBlocksPerSm: 1);
    }

    private static void EmitRoundfToS32(
        StringBuilder ptx, string value, string result, int predicate)
    {
        ptx.AppendLine($"    setp.ge.f32 %p{predicate}, {value}, 0f00000000;");
        ptx.AppendLine($"    add.rn.f32 %f60, {value}, {F(0.5f)}; sub.rn.f32 %f61, {value}, {F(0.5f)};");
        ptx.AppendLine("    cvt.rmi.s32.f32 %r38, %f60; cvt.rpi.s32.f32 %r39, %f61;");
        ptx.AppendLine($"    selp.s32 {result}, %r38, %r39, %p{predicate};");
    }

    private static void EmitRoiBilinearSample(
        StringBuilder ptx, int id, int h, int w,
        string planeBase, string y, string x, string result)
    {
        string zero = $"ROI_SAMPLE_{id}_ZERO";
        string done = $"ROI_SAMPLE_{id}_DONE";
        ptx.AppendLine($"    setp.lt.f32 %p10, {y}, {F(-1f)}; setp.gt.f32 %p11, {y}, {F((float)h)}; or.pred %p10, %p10, %p11; setp.lt.f32 %p11, {x}, {F(-1f)}; or.pred %p10, %p10, %p11; setp.gt.f32 %p11, {x}, {F((float)w)}; or.pred %p10, %p10, %p11; @%p10 bra {zero};");
        ptx.AppendLine($"    max.f32 %f30, {y}, 0f00000000; max.f32 %f31, {x}, 0f00000000; cvt.rzi.s32.f32 %r22, %f30; cvt.rzi.s32.f32 %r23, %f31;");
        ptx.AppendLine($"    add.s32 %r24, %r22, 1; min.s32 %r24, %r24, {h - 1}; add.s32 %r25, %r23, 1; min.s32 %r25, %r25, {w - 1}; min.s32 %r22, %r22, {h - 1}; min.s32 %r23, %r23, {w - 1};");
        ptx.AppendLine("    cvt.rn.f32.s32 %f32, %r22; cvt.rn.f32.s32 %f33, %r23; sub.rn.f32 %f34, %f30, %f32; sub.rn.f32 %f35, %f31, %f33; sub.rn.f32 %f36, 0f3F800000, %f34; sub.rn.f32 %f37, 0f3F800000, %f35;");
        ptx.AppendLine($"    mad.lo.u32 %r26, %r22, {w}, %r23; add.u32 %r26, %r26, {planeBase}; mad.lo.u32 %r27, %r22, {w}, %r25; add.u32 %r27, %r27, {planeBase}; mad.lo.u32 %r28, %r24, {w}, %r23; add.u32 %r28, %r28, {planeBase}; mad.lo.u32 %r29, %r24, {w}, %r25; add.u32 %r29, %r29, {planeBase};");
        for (int lane = 0; lane < 4; lane++)
            ptx.AppendLine($"    mul.wide.u32 %rd{7 + lane}, %r{26 + lane}, 4; add.u64 %rd{7 + lane}, %rd0, %rd{7 + lane}; ld.global.f32 %f{38 + lane}, [%rd{7 + lane}];");
        ptx.AppendLine("    mul.rn.f32 %f42, %f36, %f37; mul.rn.f32 %f43, %f36, %f35; mul.rn.f32 %f44, %f34, %f37; mul.rn.f32 %f45, %f34, %f35;");
        ptx.AppendLine($"    mul.rn.f32 {result}, %f42, %f38; fma.rn.f32 {result}, %f43, %f39, {result}; fma.rn.f32 {result}, %f44, %f40, {result}; fma.rn.f32 {result}, %f45, %f41, {result}; bra {done};");
        ptx.AppendLine($"{zero}: mov.f32 {result}, 0f00000000;");
        ptx.AppendLine(done + ":");
    }
}
