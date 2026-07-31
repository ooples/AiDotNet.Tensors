using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static partial class PtxVisionEmitter
{
    private enum BoxMetric
    {
        Iou,
        GeneralizedIou,
        DistanceIou,
        CompleteIou
    }

    private static DirectPtxVisionDefinition EmitPairwiseMetric(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        int n = spec.D0, m = spec.D1;
        RequireOneOf(n, nameof(n), 256, 1024, 4096);
        RequireOneOf(m, nameof(m), 256, 1024);
        if (m == 1024 && n != 1024)
            throw new NotSupportedException("The M=1024 metric family is emitted only for N=1024.");
        int shift = m == 256 ? 8 : 10;
        var ptx = Begin(spec, ccMajor, ccMinor, "boxes_a", "boxes_b", "output");
        DeclareBoxRegisters(ptx);
        LoadParameters(ptx, "boxes_a", "boxes_b", "output");
        EmitGlobalIndex(ptx, checked(n * m));
        ptx.AppendLine($"    shr.u32 %r3, %r2, {shift};");
        ptx.AppendLine($"    and.b32 %r4, %r2, {m - 1};");
        LoadBoxPair(ptx, "%r3", "%r4");
        EmitMetric(ptx, MetricForPairwise(spec.Operation), "%f31");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f31;");
        string code = Finish(ptx);
        return Definition(spec, architecture, $"n{n}-m{m}-t256",
            [
                Tensor("boxes-a", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Read),
                Tensor("boxes-b", DirectPtxPhysicalLayout.BoxXyxy, new(m, 4), DirectPtxTensorAccess.Read),
                Tensor("metric", DirectPtxPhysicalLayout.RowMajor2D, new(n, m), DirectPtxTensorAccess.Write)
            ],
            Semantics(("coordinates", "xyxy"), ("metric", spec.Operation.ToString()),
                ("zero-union", "0"), ("layout", "contiguous exact")),
            code, checked(n * m));
    }

    private static DirectPtxVisionDefinition EmitBoxArea(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        int n = spec.D0;
        RequireOneOf(n, nameof(n), 256, 1024, 4096);
        var ptx = Begin(spec, ccMajor, ccMinor, "boxes", "output");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        LoadParameters(ptx, "boxes", "output");
        EmitGlobalIndex(ptx, n);
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd3];");
        ptx.AppendLine("    mov.f32 %f4, 0f00000000;");
        ptx.AppendLine("    sub.rn.f32 %f5, %f2, %f0;");
        ptx.AppendLine("    max.f32 %f5, %f5, %f4;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f3, %f1;");
        ptx.AppendLine("    max.f32 %f6, %f6, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f6;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f7;");
        return Definition(spec, architecture, $"n{n}-t256",
            [
                Tensor("boxes", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Read),
                Tensor("area", DirectPtxPhysicalLayout.Vector, new(n), DirectPtxTensorAccess.Write)
            ], Semantics(("coordinates", "xyxy"), ("degenerate-area", "clamp-to-zero")),
            Finish(ptx), n, maxRegisters: 16, minBlocksPerSm: 4);
    }

    private static DirectPtxVisionDefinition EmitBoxConvert(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        int n = spec.D0, from = spec.D1, to = spec.D2;
        RequireOneOf(n, nameof(n), 256, 1024, 4096);
        RequireOneOf(from, nameof(from), 0, 1, 2);
        RequireOneOf(to, nameof(to), 0, 1, 2);
        var ptx = Begin(spec, ccMajor, ccMinor, "boxes", "output");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        LoadParameters(ptx, "boxes", "output");
        EmitGlobalIndex(ptx, n);
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd3];");
        if (from == 0)
        {
            ptx.AppendLine("    mov.f32 %f4, %f0; mov.f32 %f5, %f1;");
            ptx.AppendLine("    mov.f32 %f6, %f2; mov.f32 %f7, %f3;");
        }
        else if (from == 1)
        {
            ptx.AppendLine("    mov.f32 %f4, %f0; mov.f32 %f5, %f1;");
            ptx.AppendLine("    add.rn.f32 %f6, %f0, %f2; add.rn.f32 %f7, %f1, %f3;");
        }
        else
        {
            ptx.AppendLine($"    mul.rn.f32 %f8, %f2, {F(0.5f)};");
            ptx.AppendLine($"    mul.rn.f32 %f9, %f3, {F(0.5f)};");
            ptx.AppendLine("    sub.rn.f32 %f4, %f0, %f8; sub.rn.f32 %f5, %f1, %f9;");
            ptx.AppendLine("    add.rn.f32 %f6, %f0, %f8; add.rn.f32 %f7, %f1, %f9;");
        }
        if (to == 0)
            ptx.AppendLine("    mov.f32 %f10, %f4; mov.f32 %f11, %f5; mov.f32 %f12, %f6; mov.f32 %f13, %f7;");
        else
        {
            ptx.AppendLine("    sub.rn.f32 %f12, %f6, %f4; sub.rn.f32 %f13, %f7, %f5;");
            if (to == 1)
                ptx.AppendLine("    mov.f32 %f10, %f4; mov.f32 %f11, %f5;");
            else
            {
                ptx.AppendLine($"    fma.rn.f32 %f10, %f12, {F(0.5f)}, %f4;");
                ptx.AppendLine($"    fma.rn.f32 %f11, %f13, {F(0.5f)}, %f5;");
            }
        }
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd2;");
        ptx.AppendLine("    st.global.v4.f32 [%rd4], {%f10,%f11,%f12,%f13};");
        return Definition(spec, architecture, $"n{n}-f{from}-t{to}",
            [
                Tensor("boxes", BoxLayout(from), new(n, 4), DirectPtxTensorAccess.Read),
                Tensor("output", BoxLayout(to), new(n, 4), DirectPtxTensorAccess.Write)
            ], Semantics(("from-format", from.ToString()), ("to-format", to.ToString())),
            Finish(ptx), n, maxRegisters: 24, minBlocksPerSm: 4);
    }

    private static DirectPtxPhysicalLayout BoxLayout(int format) => format switch
    {
        0 => DirectPtxPhysicalLayout.BoxXyxy,
        1 => DirectPtxPhysicalLayout.BoxXywh,
        2 => DirectPtxPhysicalLayout.BoxCxcywh,
        _ => throw new NotSupportedException($"Box format {format} has no physical layout.")
    };

    private static DirectPtxVisionDefinition EmitAlignedLoss(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        int n = spec.D0;
        RequireOneOf(n, nameof(n), 256, 1024, 4096);
        var ptx = Begin(spec, ccMajor, ccMinor, "predicted", "target", "loss");
        DeclareBoxRegisters(ptx);
        LoadParameters(ptx, "predicted", "target", "loss");
        EmitGlobalIndex(ptx, n);
        LoadBoxPair(ptx, "%r2", "%r2");
        EmitMetric(ptx, MetricForLoss(spec.Operation), "%f31");
        ptx.AppendLine($"    sub.rn.f32 %f32, {F(1f)}, %f31;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f32;");
        return Definition(spec, architecture, $"n{n}",
            [
                Tensor("predicted", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Read),
                Tensor("target", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Read),
                Tensor("loss", DirectPtxPhysicalLayout.Vector, new(n), DirectPtxTensorAccess.Write)
            ], Semantics(("loss", spec.Operation.ToString()), ("reduction", "none")),
            Finish(ptx), n);
    }

    private static void DeclareBoxRegisters(StringBuilder ptx)
    {
        ptx.AppendLine("    .reg .pred %p<16>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<80>;");
    }

    private static void LoadBoxPair(StringBuilder ptx, string aIndex, string bIndex)
    {
        ptx.AppendLine($"    mul.wide.u32 %rd3, {aIndex}, 16;");
        ptx.AppendLine($"    mul.wide.u32 %rd4, {bIndex}, 16;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd5];");
        ptx.AppendLine("    ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd6];");
    }

    /// <summary>Emits IoU-family geometry. Result is placed in result.</summary>
    private static void EmitMetric(
        StringBuilder ptx, BoxMetric metric, string result)
    {
        ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        ptx.AppendLine("    sub.rn.f32 %f9, %f2, %f0; max.f32 %f9, %f9, %f8;");
        ptx.AppendLine("    sub.rn.f32 %f10, %f3, %f1; max.f32 %f10, %f10, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f11, %f9, %f10;");
        ptx.AppendLine("    sub.rn.f32 %f12, %f6, %f4; max.f32 %f12, %f12, %f8;");
        ptx.AppendLine("    sub.rn.f32 %f13, %f7, %f5; max.f32 %f13, %f13, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f14, %f12, %f13;");
        ptx.AppendLine("    max.f32 %f15, %f0, %f4; max.f32 %f16, %f1, %f5;");
        ptx.AppendLine("    min.f32 %f17, %f2, %f6; min.f32 %f18, %f3, %f7;");
        ptx.AppendLine("    sub.rn.f32 %f17, %f17, %f15; max.f32 %f17, %f17, %f8;");
        ptx.AppendLine("    sub.rn.f32 %f18, %f18, %f16; max.f32 %f18, %f18, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f19, %f17, %f18;");
        ptx.AppendLine("    add.rn.f32 %f20, %f11, %f14; sub.rn.f32 %f20, %f20, %f19;");
        ptx.AppendLine("    mov.f32 %f21, 0f00000000; setp.gt.f32 %p1, %f20, %f8;");
        ptx.AppendLine("    @%p1 div.approx.f32 %f21, %f19, %f20;");
        if (metric == BoxMetric.Iou)
        {
            ptx.AppendLine($"    mov.f32 {result}, %f21;");
            return;
        }
        ptx.AppendLine("    min.f32 %f22, %f0, %f4; min.f32 %f23, %f1, %f5;");
        ptx.AppendLine("    max.f32 %f24, %f2, %f6; max.f32 %f25, %f3, %f7;");
        ptx.AppendLine("    sub.rn.f32 %f26, %f24, %f22; sub.rn.f32 %f27, %f25, %f23;");
        if (metric == BoxMetric.GeneralizedIou)
        {
            ptx.AppendLine("    mul.rn.f32 %f28, %f26, %f27;");
            ptx.AppendLine("    mov.f32 %f29, 0f00000000; setp.gt.f32 %p2, %f28, %f8;");
            ptx.AppendLine("    sub.rn.f32 %f30, %f28, %f20;");
            ptx.AppendLine("    @%p2 div.approx.f32 %f29, %f30, %f28;");
            ptx.AppendLine($"    sub.rn.f32 {result}, %f21, %f29;");
            return;
        }
        ptx.AppendLine($"    add.rn.f32 %f28, %f0, %f2; mul.rn.f32 %f28, %f28, {F(0.5f)};");
        ptx.AppendLine($"    add.rn.f32 %f29, %f1, %f3; mul.rn.f32 %f29, %f29, {F(0.5f)};");
        ptx.AppendLine($"    add.rn.f32 %f30, %f4, %f6; mul.rn.f32 %f30, %f30, {F(0.5f)};");
        ptx.AppendLine($"    add.rn.f32 %f31, %f5, %f7; mul.rn.f32 %f31, %f31, {F(0.5f)};");
        ptx.AppendLine("    sub.rn.f32 %f32, %f28, %f30; sub.rn.f32 %f33, %f29, %f31;");
        ptx.AppendLine("    mul.rn.f32 %f34, %f32, %f32; fma.rn.f32 %f34, %f33, %f33, %f34;");
        ptx.AppendLine("    mul.rn.f32 %f35, %f26, %f26; fma.rn.f32 %f35, %f27, %f27, %f35;");
        ptx.AppendLine("    mov.f32 %f36, 0f00000000; setp.gt.f32 %p3, %f35, %f8;");
        ptx.AppendLine("    @%p3 div.approx.f32 %f36, %f34, %f35;");
        ptx.AppendLine("    sub.rn.f32 %f37, %f21, %f36;");
        if (metric == BoxMetric.DistanceIou)
        {
            ptx.AppendLine($"    mov.f32 {result}, %f37;");
            return;
        }
        // CIoU aspect term. Width/height are non-negative. EmitPositiveAtan
        // range-reduces to [0,1] and uses a minimax polynomial whose maximum
        // absolute error is 1.15e-5 over that interval.
        EmitPositiveAtan(ptx, "%f9", "%f10", "%f38", 4);
        EmitPositiveAtan(ptx, "%f12", "%f13", "%f39", 5);
        ptx.AppendLine("    sub.rn.f32 %f40, %f38, %f39;");
        ptx.AppendLine($"    mul.rn.f32 %f41, %f40, %f40; mul.rn.f32 %f41, %f41, {F(4f / (MathF.PI * MathF.PI))};");
        ptx.AppendLine($"    sub.rn.f32 %f42, {F(1f)}, %f21; add.rn.f32 %f42, %f42, %f41;");
        ptx.AppendLine("    mov.f32 %f43, 0f00000000; setp.gt.f32 %p6, %f42, %f8;");
        ptx.AppendLine("    @%p6 div.approx.f32 %f43, %f41, %f42;");
        ptx.AppendLine("    mul.rn.f32 %f44, %f43, %f41;");
        ptx.AppendLine($"    sub.rn.f32 {result}, %f37, %f44;");
    }

    private static void EmitPositiveAtan(
        StringBuilder ptx, string numerator, string denominator, string result, int predicate)
    {
        ptx.AppendLine($"    setp.gt.f32 %p{predicate}, {denominator}, 0f00000000;");
        ptx.AppendLine($"    mov.f32 %f50, 0f00000000; @%p{predicate} div.approx.f32 %f50, {numerator}, {denominator};");
        ptx.AppendLine("    setp.gt.f32 %p7, %f50, 0f3F800000;");
        ptx.AppendLine("    mov.f32 %f51, %f50; @%p7 div.approx.f32 %f51, 0f3F800000, %f50;");
        ptx.AppendLine("    mul.rn.f32 %f53, %f51, %f51;");
        ptx.AppendLine($"    mov.f32 %f54, {F(0.0208351f)};");
        ptx.AppendLine($"    fma.rn.f32 %f54, %f54, %f53, {F(-0.0851330f)};");
        ptx.AppendLine($"    fma.rn.f32 %f54, %f54, %f53, {F(0.1801410f)};");
        ptx.AppendLine($"    fma.rn.f32 %f54, %f54, %f53, {F(-0.3302995f)};");
        ptx.AppendLine($"    fma.rn.f32 %f54, %f54, %f53, {F(0.9998660f)};");
        ptx.AppendLine("    mul.rn.f32 %f55, %f51, %f54;");
        ptx.AppendLine($"    sub.rn.f32 %f56, {F(MathF.PI / 2f)}, %f55;");
        ptx.AppendLine("    selp.f32 %f57, %f56, %f55, %p7;");
        ptx.AppendLine($"    selp.f32 {result}, %f57, 0f00000000, %p{predicate};");
    }

    private static DirectPtxVisionDefinition EmitBoxBackward(
        DirectPtxVisionSpec spec, DirectPtxArchitectureFamily architecture,
        int ccMajor, int ccMinor)
    {
        bool pairwise = spec.Operation is DirectPtxVisionOperation.IouFamilyBackwardA or
            DirectPtxVisionOperation.IouFamilyBackwardB;
        if (!pairwise)
        {
            int n = spec.D0;
            RequireOneOf(n, nameof(n), 256, 1024, 4096);
            var ptx = Begin(spec, ccMajor, ccMinor,
                "grad_output", "predicted", "target", "grad_predicted");
            DeclareBoxRegisters(ptx);
            LoadParameters(ptx, "grad_output", "predicted", "target", "grad_predicted");
            EmitGlobalIndex(ptx, n);
            // LoadBoxPair expects its inputs in rd0/rd1; backward ABI has them
            // in rd1/rd2, so use the equivalent explicit loads.
            ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 16; add.u64 %rd5, %rd1, %rd4; add.u64 %rd6, %rd2, %rd4;");
            ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd5]; ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd6];");
            ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4; add.u64 %rd8, %rd0, %rd7; ld.global.f32 %f63, [%rd8];");
            BoxMetric metric = MetricForBackward(spec.Operation);
            EmitAnalyticMetricGradient(ptx, metric, ownerA: true, negative: true,
                "%f64", "%f65", "%f66", "%f67");
            ptx.AppendLine("    add.u64 %rd9, %rd3, %rd4; st.global.v4.f32 [%rd9], {%f64,%f65,%f66,%f67};");
            return Definition(spec, architecture, $"n{n}-deterministic-analytic",
                [
                    Tensor("grad-output", DirectPtxPhysicalLayout.Vector, new(n), DirectPtxTensorAccess.Read),
                    Tensor("predicted", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Read),
                    Tensor("target", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Read),
                    Tensor("grad-predicted", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Write)
                ], Semantics(("gradient", spec.Operation.ToString()),
                    ("method", "analytical reverse mode; CIoU alpha detached"),
                    ("determinism", "one thread per box")),
                Finish(ptx), n, maxRegisters: 96, minBlocksPerSm: 1);
        }

        const uint PairBlockThreads = 128;
        int boxesA = spec.D0, boxesB = spec.D1, variant = spec.D2;
        RequireOneOf(boxesA, nameof(boxesA), 256, 1024);
        RequireOneOf(boxesB, nameof(boxesB), 256, 1024);
        RequireOneOf(variant, nameof(variant), 0, 1, 2, 3);
        bool ownerA = spec.Operation == DirectPtxVisionOperation.IouFamilyBackwardA;
        int owners = ownerA ? boxesA : boxesB;
        int other = ownerA ? boxesB : boxesA;
        var pairPtx = Begin(spec, ccMajor, ccMinor,
            "grad_output", "boxes_a", "boxes_b", ownerA ? "grad_a" : "grad_b");
        DeclareBoxRegisters(pairPtx);
        LoadParameters(pairPtx, "grad_output", "boxes_a", "boxes_b", ownerA ? "grad_a" : "grad_b");
        EmitGlobalIndex(pairPtx, owners, PairBlockThreads);
        // The owner box is loop-invariant. Hoist it exactly as the incumbent
        // CUDA compiler does so every cell fetches only the opposing box.
        if (ownerA)
        {
            pairPtx.AppendLine("    mul.wide.u32 %rd4, %r2, 16; add.u64 %rd6, %rd1, %rd4; ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd6];");
            pairPtx.AppendLine($"    mul.wide.u32 %rd8, %r2, {boxesB * 4}; add.u64 %rd9, %rd0, %rd8; mov.u64 %rd7, %rd2;");
        }
        else
        {
            pairPtx.AppendLine("    mul.wide.u32 %rd5, %r2, 16; add.u64 %rd7, %rd2, %rd5; ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd7];");
            pairPtx.AppendLine("    mul.wide.u32 %rd8, %r2, 4; add.u64 %rd9, %rd0, %rd8; mov.u64 %rd6, %rd1;");
        }
        pairPtx.AppendLine("    mov.f32 %f64, 0f00000000; mov.f32 %f65, 0f00000000; mov.f32 %f66, 0f00000000; mov.f32 %f67, 0f00000000; mov.u32 %r8, 0;");
        pairPtx.AppendLine("PAIR_GRAD_LOOP:");
        pairPtx.AppendLine($"    setp.ge.u32 %p0, %r8, {other}; @%p0 bra PAIR_GRAD_DONE;");
        if (ownerA)
            pairPtx.AppendLine("    ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd7];");
        else
            pairPtx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd6];");
        pairPtx.AppendLine("    ld.global.f32 %f63, [%rd9];");
        BoxMetric pairMetric = variant switch
        {
            0 => BoxMetric.Iou,
            1 => BoxMetric.GeneralizedIou,
            2 => BoxMetric.DistanceIou,
            3 => BoxMetric.CompleteIou,
            _ => throw new NotSupportedException($"IoU-family variant {variant} is not emitted.")
        };
        EmitAnalyticMetricGradient(pairPtx, pairMetric, ownerA, negative: false,
            "%f68", "%f69", "%f70", "%f71");
        pairPtx.AppendLine("    add.rn.f32 %f64, %f64, %f68; add.rn.f32 %f65, %f65, %f69; add.rn.f32 %f66, %f66, %f70; add.rn.f32 %f67, %f67, %f71;");
        pairPtx.AppendLine(ownerA
            ? "    add.u64 %rd7, %rd7, 16; add.u64 %rd9, %rd9, 4;"
            : $"    add.u64 %rd6, %rd6, 16; add.u64 %rd9, %rd9, {boxesB * 4};");
        pairPtx.AppendLine("    add.u32 %r8, %r8, 1; bra PAIR_GRAD_LOOP;");
        pairPtx.AppendLine("PAIR_GRAD_DONE: mul.wide.u32 %rd10, %r2, 16; add.u64 %rd11, %rd3, %rd10; st.global.v4.f32 [%rd11], {%f64,%f65,%f66,%f67};");
        return Definition(spec, architecture,
            $"n{boxesA}-m{boxesB}-v{variant}-owner-{(ownerA ? "a" : "b")}-b{PairBlockThreads}",
            [
                Tensor("grad-output", DirectPtxPhysicalLayout.RowMajor2D, new(boxesA, boxesB), DirectPtxTensorAccess.Read),
                Tensor("boxes-a", DirectPtxPhysicalLayout.BoxXyxy, new(boxesA, 4), DirectPtxTensorAccess.Read),
                Tensor("boxes-b", DirectPtxPhysicalLayout.BoxXyxy, new(boxesB, 4), DirectPtxTensorAccess.Read),
                Tensor(ownerA ? "grad-a" : "grad-b", DirectPtxPhysicalLayout.BoxXyxy, new(owners, 4), DirectPtxTensorAccess.Write)
            ], Semantics(("gradient", $"iou-family-variant-{variant}"),
                ("owner", ownerA ? "a" : "b"),
                ("method", "analytical reverse mode; CIoU alpha detached"),
                ("determinism", "one owner thread, fixed other-index order")),
            Finish(pairPtx), owners, maxRegisters: 96, minBlocksPerSm: 1, blockThreads: PairBlockThreads);
    }

    private static BoxMetric MetricForPairwise(DirectPtxVisionOperation operation) =>
        operation switch
        {
            DirectPtxVisionOperation.GeneralizedBoxIou => BoxMetric.GeneralizedIou,
            DirectPtxVisionOperation.DistanceBoxIou => BoxMetric.DistanceIou,
            DirectPtxVisionOperation.CompleteBoxIou => BoxMetric.CompleteIou,
            _ => throw new NotSupportedException($"{operation} is not a pairwise IoU metric.")
        };

    private static BoxMetric MetricForLoss(DirectPtxVisionOperation operation) =>
        operation switch
        {
            DirectPtxVisionOperation.IoULoss => BoxMetric.Iou,
            DirectPtxVisionOperation.GIoULoss => BoxMetric.GeneralizedIou,
            DirectPtxVisionOperation.DIoULoss => BoxMetric.DistanceIou,
            DirectPtxVisionOperation.CIoULoss => BoxMetric.CompleteIou,
            _ => throw new NotSupportedException($"{operation} is not an aligned IoU loss.")
        };

    private static BoxMetric MetricForBackward(DirectPtxVisionOperation operation) =>
        operation switch
        {
            DirectPtxVisionOperation.IoULossBackward => BoxMetric.Iou,
            DirectPtxVisionOperation.GIoULossBackward => BoxMetric.GeneralizedIou,
            DirectPtxVisionOperation.DIoULossBackward => BoxMetric.DistanceIou,
            DirectPtxVisionOperation.CIoULossBackward => BoxMetric.CompleteIou,
            _ => throw new NotSupportedException($"{operation} is not an aligned IoU-loss backward metric.")
        };

    /// <summary>
    /// Emits the coordinate-level reverse pass shared by aligned losses and
    /// pairwise IoU-family gradients. The generated code intentionally mirrors
    /// CpuEngine.IouFamilyBackward and compute_cell_grads_iou: CIoU's alpha is
    /// a stop-gradient, and min/max ties are owned by A.
    /// </summary>
    private static void EmitAnalyticMetricGradient(
        StringBuilder ptx,
        BoxMetric metric,
        bool ownerA,
        bool negative,
        string x1Gradient,
        string y1Gradient,
        string x2Gradient,
        string y2Gradient)
    {
        int variant = metric switch
        {
            BoxMetric.Iou => 0,
            BoxMetric.GeneralizedIou => 1,
            BoxMetric.DistanceIou => 2,
            BoxMetric.CompleteIou => 3,
            _ => throw new NotSupportedException($"Box metric {metric} is not emitted.")
        };

        ptx.AppendLine($"    mov.f32 {x1Gradient}, 0f00000000; mov.f32 {y1Gradient}, 0f00000000; mov.f32 {x2Gradient}, 0f00000000; mov.f32 {y2Gradient}, 0f00000000;");
        ptx.AppendLine($"    {(negative ? "neg" : "mov")}.f32 %f58, %f63;");
        ptx.AppendLine("    setp.eq.f32 %p1, %f58, 0f00000000; @%p1 bra ANALYTIC_GRAD_DONE;");

        // Common forward geometry. f30 and f31 accumulate the adjoints for
        // intersection and the selected owner's area respectively.
        ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        ptx.AppendLine("    sub.rn.f32 %f9, %f2, %f0; sub.rn.f32 %f10, %f3, %f1; sub.rn.f32 %f11, %f6, %f4; sub.rn.f32 %f12, %f7, %f5;");
        ptx.AppendLine("    max.f32 %f13, %f9, %f8; max.f32 %f14, %f10, %f8; max.f32 %f15, %f11, %f8; max.f32 %f16, %f12, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f13, %f14; mul.rn.f32 %f18, %f15, %f16;");
        ptx.AppendLine("    max.f32 %f19, %f0, %f4; min.f32 %f20, %f2, %f6; max.f32 %f21, %f1, %f5; min.f32 %f22, %f3, %f7;");
        ptx.AppendLine("    sub.rn.f32 %f23, %f20, %f19; sub.rn.f32 %f24, %f22, %f21; max.f32 %f25, %f23, %f8; max.f32 %f26, %f24, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f27, %f25, %f26; add.rn.f32 %f28, %f17, %f18; sub.rn.f32 %f28, %f28, %f27;");
        ptx.AppendLine("    mov.f32 %f29, 0f00000000; setp.gt.f32 %p9, %f28, %f8;");
        if (variant == 3)
            ptx.AppendLine("    @%p9 div.approx.f32 %f29, %f27, %f28;");
        ptx.AppendLine("    mov.f32 %f30, 0f00000000; mov.f32 %f31, 0f00000000;");

        if (variant == 1)
        {
            // GIoU = IoU + union/enclose - 1.
            ptx.AppendLine("    min.f32 %f32, %f0, %f4; min.f32 %f33, %f1, %f5; max.f32 %f34, %f2, %f6; max.f32 %f35, %f3, %f7;");
            ptx.AppendLine("    sub.rn.f32 %f32, %f34, %f32; sub.rn.f32 %f33, %f35, %f33; max.f32 %f34, %f32, %f8; max.f32 %f35, %f33, %f8; mul.rn.f32 %f36, %f34, %f35;");
            ptx.AppendLine("    setp.gt.f32 %p2, %f36, %f8; mov.f32 %f37, 0f00000000; @%p2 rcp.approx.f32 %f37, %f36; mul.rn.f32 %f38, %f58, %f37;");
            ptx.AppendLine("    neg.f32 %f39, %f28; mul.rn.f32 %f39, %f58, %f39; mul.rn.f32 %f39, %f39, %f37; mul.rn.f32 %f40, %f39, %f37;");
            ptx.AppendLine("    add.rn.f32 %f31, %f31, %f38; sub.rn.f32 %f30, %f30, %f38;");
            ptx.AppendLine("    mul.rn.f32 %f41, %f40, %f35; mul.rn.f32 %f42, %f40, %f34; setp.gt.f32 %p3, %f32, %f8; setp.gt.f32 %p4, %f33, %f8; selp.f32 %f41, %f41, %f8, %p3; selp.f32 %f42, %f42, %f8, %p4;");
            ptx.AppendLine("    neg.f32 %f43, %f41; neg.f32 %f44, %f42;");
            if (ownerA)
            {
                ptx.AppendLine($"    setp.le.f32 %p5, %f0, %f4; @%p5 add.rn.f32 {x1Gradient}, {x1Gradient}, %f43; setp.ge.f32 %p6, %f2, %f6; @%p6 add.rn.f32 {x2Gradient}, {x2Gradient}, %f41;");
                ptx.AppendLine($"    setp.le.f32 %p5, %f1, %f5; @%p5 add.rn.f32 {y1Gradient}, {y1Gradient}, %f44; setp.ge.f32 %p6, %f3, %f7; @%p6 add.rn.f32 {y2Gradient}, {y2Gradient}, %f42;");
            }
            else
            {
                ptx.AppendLine($"    setp.gt.f32 %p5, %f0, %f4; @%p5 add.rn.f32 {x1Gradient}, {x1Gradient}, %f43; setp.lt.f32 %p6, %f2, %f6; @%p6 add.rn.f32 {x2Gradient}, {x2Gradient}, %f41;");
                ptx.AppendLine($"    setp.gt.f32 %p5, %f1, %f5; @%p5 add.rn.f32 {y1Gradient}, {y1Gradient}, %f44; setp.lt.f32 %p6, %f3, %f7; @%p6 add.rn.f32 {y2Gradient}, {y2Gradient}, %f42;");
            }
        }
        else if (variant is 2 or 3)
        {
            // DIoU/CIoU centre distance and enclosing diagonal.
            ptx.AppendLine($"    add.rn.f32 %f32, %f0, %f2; mul.rn.f32 %f32, %f32, {F(0.5f)}; add.rn.f32 %f33, %f1, %f3; mul.rn.f32 %f33, %f33, {F(0.5f)};");
            ptx.AppendLine($"    add.rn.f32 %f34, %f4, %f6; mul.rn.f32 %f34, %f34, {F(0.5f)}; add.rn.f32 %f35, %f5, %f7; mul.rn.f32 %f35, %f35, {F(0.5f)};");
            ptx.AppendLine("    sub.rn.f32 %f36, %f32, %f34; sub.rn.f32 %f37, %f33, %f35; mul.rn.f32 %f38, %f36, %f36; fma.rn.f32 %f38, %f37, %f37, %f38;");
            ptx.AppendLine("    min.f32 %f39, %f0, %f4; min.f32 %f40, %f1, %f5; max.f32 %f41, %f2, %f6; max.f32 %f42, %f3, %f7; sub.rn.f32 %f43, %f41, %f39; sub.rn.f32 %f44, %f42, %f40;");
            ptx.AppendLine("    mul.rn.f32 %f45, %f43, %f43; fma.rn.f32 %f45, %f44, %f44, %f45; setp.gt.f32 %p2, %f45, %f8;");
            ptx.AppendLine("    mov.f32 %f47, 0f00000000; @%p2 rcp.approx.f32 %f47, %f45; neg.f32 %f46, %f58; mul.rn.f32 %f46, %f46, %f47;");
            ptx.AppendLine("    mul.rn.f32 %f59, %f58, %f38; mul.rn.f32 %f59, %f59, %f47; mul.rn.f32 %f59, %f59, %f47;");

            if (variant == 3)
            {
                // CIoU alpha is deliberately detached, matching every oracle.
                EmitPositiveAtan(ptx, "%f13", "%f14", "%f48", 10);
                EmitPositiveAtan(ptx, "%f15", "%f16", "%f49", 11);
                ptx.AppendLine("    sub.rn.f32 %f50, %f48, %f49; mul.rn.f32 %f51, %f50, %f50;");
                ptx.AppendLine($"    mul.rn.f32 %f51, %f51, {F(4f / (MathF.PI * MathF.PI))}; sub.rn.f32 %f52, {F(1f)}, %f29; add.rn.f32 %f52, %f52, %f51;");
                ptx.AppendLine("    mov.f32 %f53, 0f00000000; setp.gt.f32 %p12, %f52, %f8; @%p12 rcp.approx.f32 %f53, %f52; mul.rn.f32 %f53, %f51, %f53;");
                ptx.AppendLine($"    neg.f32 %f54, %f53; mul.rn.f32 %f54, %f58, %f54; mul.rn.f32 %f55, %f54, %f50; mul.rn.f32 %f55, %f55, {F(8f / (MathF.PI * MathF.PI))};");
                if (!ownerA) ptx.AppendLine("    neg.f32 %f55, %f55;");
                ptx.AppendLine($"    mov.f32 %f56, {(ownerA ? "%f13" : "%f15")}; mov.f32 %f57, {(ownerA ? "%f14" : "%f16")};");
                ptx.AppendLine("    mul.rn.f32 %f48, %f56, %f56; fma.rn.f32 %f48, %f57, %f57, %f48; setp.gt.f32 %p12, %f14, %f8; setp.gt.f32 %p13, %f16, %f8; and.pred %p12, %p12, %p13; setp.gt.f32 %p13, %f48, %f8; and.pred %p13, %p12, %p13;");
                ptx.AppendLine("    mov.f32 %f52, 0f00000000; @%p13 rcp.approx.f32 %f52, %f48; mul.rn.f32 %f49, %f55, %f57; mul.rn.f32 %f50, %f49, %f52; neg.f32 %f49, %f56; mul.rn.f32 %f49, %f55, %f49; mul.rn.f32 %f51, %f49, %f52;");
                ptx.AppendLine($"    setp.gt.f32 %p14, {(ownerA ? "%f9" : "%f11")}, %f8; and.pred %p14, %p13, %p14; @%p14 sub.rn.f32 {x1Gradient}, {x1Gradient}, %f50; @%p14 add.rn.f32 {x2Gradient}, {x2Gradient}, %f50;");
                ptx.AppendLine($"    setp.gt.f32 %p15, {(ownerA ? "%f10" : "%f12")}, %f8; and.pred %p15, %p13, %p15; @%p15 sub.rn.f32 {y1Gradient}, {y1Gradient}, %f51; @%p15 add.rn.f32 {y2Gradient}, {y2Gradient}, %f51;");
            }

            // Propagate centreSq to the selected owner.
            ptx.AppendLine("    mul.rn.f32 %f48, %f46, %f36; mul.rn.f32 %f49, %f46, %f37;");
            if (!ownerA) ptx.AppendLine("    neg.f32 %f48, %f48; neg.f32 %f49, %f49;");
            ptx.AppendLine($"    add.rn.f32 {x1Gradient}, {x1Gradient}, %f48; add.rn.f32 {x2Gradient}, {x2Gradient}, %f48; add.rn.f32 {y1Gradient}, {y1Gradient}, %f49; add.rn.f32 {y2Gradient}, {y2Gradient}, %f49;");

            // Propagate diagSq through enclosing min/max.
            ptx.AppendLine($"    mul.rn.f32 %f50, %f59, %f43; mul.rn.f32 %f50, %f50, {F(2f)}; mul.rn.f32 %f51, %f59, %f44; mul.rn.f32 %f51, %f51, {F(2f)}; neg.f32 %f52, %f50; neg.f32 %f53, %f51;");
            if (ownerA)
            {
                ptx.AppendLine($"    setp.le.f32 %p5, %f0, %f4; @%p5 add.rn.f32 {x1Gradient}, {x1Gradient}, %f52; setp.ge.f32 %p6, %f2, %f6; @%p6 add.rn.f32 {x2Gradient}, {x2Gradient}, %f50;");
                ptx.AppendLine($"    setp.le.f32 %p5, %f1, %f5; @%p5 add.rn.f32 {y1Gradient}, {y1Gradient}, %f53; setp.ge.f32 %p6, %f3, %f7; @%p6 add.rn.f32 {y2Gradient}, {y2Gradient}, %f51;");
            }
            else
            {
                ptx.AppendLine($"    setp.gt.f32 %p5, %f0, %f4; @%p5 add.rn.f32 {x1Gradient}, {x1Gradient}, %f52; setp.lt.f32 %p6, %f2, %f6; @%p6 add.rn.f32 {x2Gradient}, {x2Gradient}, %f50;");
                ptx.AppendLine($"    setp.gt.f32 %p5, %f1, %f5; @%p5 add.rn.f32 {y1Gradient}, {y1Gradient}, %f53; setp.lt.f32 %p6, %f3, %f7; @%p6 add.rn.f32 {y2Gradient}, {y2Gradient}, %f51;");
            }
        }

        // IoU-proper reverse pass.
        ptx.AppendLine("    mul.rn.f32 %f32, %f28, %f28; mov.f32 %f35, 0f00000000; @%p9 rcp.approx.f32 %f35, %f32; add.rn.f32 %f33, %f28, %f27; mul.rn.f32 %f34, %f58, %f33; fma.rn.f32 %f30, %f34, %f35, %f30;");
        ptx.AppendLine("    neg.f32 %f33, %f27; mul.rn.f32 %f34, %f58, %f33; fma.rn.f32 %f31, %f34, %f35, %f31;");

        ptx.AppendLine("    mul.rn.f32 %f32, %f30, %f26; mul.rn.f32 %f33, %f30, %f25; setp.gt.f32 %p3, %f23, %f8; setp.gt.f32 %p4, %f24, %f8; selp.f32 %f32, %f32, %f8, %p3; selp.f32 %f33, %f33, %f8, %p4; neg.f32 %f34, %f32; neg.f32 %f35, %f33;");
        if (ownerA)
        {
            ptx.AppendLine($"    setp.ge.f32 %p5, %f0, %f4; @%p5 add.rn.f32 {x1Gradient}, {x1Gradient}, %f34; setp.le.f32 %p6, %f2, %f6; @%p6 add.rn.f32 {x2Gradient}, {x2Gradient}, %f32;");
            ptx.AppendLine($"    setp.ge.f32 %p5, %f1, %f5; @%p5 add.rn.f32 {y1Gradient}, {y1Gradient}, %f35; setp.le.f32 %p6, %f3, %f7; @%p6 add.rn.f32 {y2Gradient}, {y2Gradient}, %f33;");
        }
        else
        {
            ptx.AppendLine($"    setp.lt.f32 %p5, %f0, %f4; @%p5 add.rn.f32 {x1Gradient}, {x1Gradient}, %f34; setp.gt.f32 %p6, %f2, %f6; @%p6 add.rn.f32 {x2Gradient}, {x2Gradient}, %f32;");
            ptx.AppendLine($"    setp.lt.f32 %p5, %f1, %f5; @%p5 add.rn.f32 {y1Gradient}, {y1Gradient}, %f35; setp.gt.f32 %p6, %f3, %f7; @%p6 add.rn.f32 {y2Gradient}, {y2Gradient}, %f33;");
        }

        string selectedWidth = ownerA ? "%f13" : "%f15";
        string selectedHeight = ownerA ? "%f14" : "%f16";
        string selectedWidthRaw = ownerA ? "%f9" : "%f11";
        string selectedHeightRaw = ownerA ? "%f10" : "%f12";
        ptx.AppendLine($"    mul.rn.f32 %f36, %f31, {selectedHeight}; mul.rn.f32 %f37, %f31, {selectedWidth}; setp.gt.f32 %p3, {selectedWidthRaw}, %f8; setp.gt.f32 %p4, {selectedHeightRaw}, %f8;");
        ptx.AppendLine($"    @%p3 sub.rn.f32 {x1Gradient}, {x1Gradient}, %f36; @%p3 add.rn.f32 {x2Gradient}, {x2Gradient}, %f36; @%p4 sub.rn.f32 {y1Gradient}, {y1Gradient}, %f37; @%p4 add.rn.f32 {y2Gradient}, {y2Gradient}, %f37;");
        ptx.AppendLine("ANALYTIC_GRAD_DONE:");
    }
}
