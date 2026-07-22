using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static partial class PtxVisionEmitter
{
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
        EmitMetric(ptx, spec.Operation, "%f31");
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
                Tensor("boxes", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Read),
                Tensor("output", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Write)
            ], Semantics(("from-format", from.ToString()), ("to-format", to.ToString())),
            Finish(ptx), n, maxRegisters: 24, minBlocksPerSm: 4);
    }

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
        DirectPtxVisionOperation metric = spec.Operation switch
        {
            DirectPtxVisionOperation.IoULoss =>
                (DirectPtxVisionOperation)((int)DirectPtxVisionOperation.GeneralizedBoxIou - 1),
            DirectPtxVisionOperation.GIoULoss => DirectPtxVisionOperation.GeneralizedBoxIou,
            DirectPtxVisionOperation.DIoULoss => DirectPtxVisionOperation.DistanceBoxIou,
            _ => DirectPtxVisionOperation.CompleteBoxIou
        };
        EmitMetric(ptx, metric, "%f31");
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
        ptx.AppendLine("    .reg .pred %p<8>;");
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
        StringBuilder ptx, DirectPtxVisionOperation operation, string result)
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
        ptx.AppendLine("    @%p1 div.rn.f32 %f21, %f19, %f20;");
        if ((int)operation < (int)DirectPtxVisionOperation.GeneralizedBoxIou)
        {
            ptx.AppendLine($"    mov.f32 {result}, %f21;");
            return;
        }
        ptx.AppendLine("    min.f32 %f22, %f0, %f4; min.f32 %f23, %f1, %f5;");
        ptx.AppendLine("    max.f32 %f24, %f2, %f6; max.f32 %f25, %f3, %f7;");
        ptx.AppendLine("    sub.rn.f32 %f26, %f24, %f22; sub.rn.f32 %f27, %f25, %f23;");
        if (operation == DirectPtxVisionOperation.GeneralizedBoxIou)
        {
            ptx.AppendLine("    mul.rn.f32 %f28, %f26, %f27;");
            ptx.AppendLine("    mov.f32 %f29, 0f00000000; setp.gt.f32 %p2, %f28, %f8;");
            ptx.AppendLine("    sub.rn.f32 %f30, %f28, %f20;");
            ptx.AppendLine("    @%p2 div.rn.f32 %f29, %f30, %f28;");
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
        ptx.AppendLine("    @%p3 div.rn.f32 %f36, %f34, %f35;");
        ptx.AppendLine("    sub.rn.f32 %f37, %f21, %f36;");
        if (operation == DirectPtxVisionOperation.DistanceBoxIou)
        {
            ptx.AppendLine($"    mov.f32 {result}, %f37;");
            return;
        }
        // CIoU aspect term. Width/height are non-negative. The rational atan
        // approximation is baked and monotonic; hardware validation owns its
        // final error gate before this unpromoted cell may ship enabled.
        EmitPositiveAtan(ptx, "%f9", "%f10", "%f38", 4);
        EmitPositiveAtan(ptx, "%f12", "%f13", "%f39", 5);
        ptx.AppendLine("    sub.rn.f32 %f40, %f38, %f39;");
        ptx.AppendLine($"    mul.rn.f32 %f41, %f40, %f40; mul.rn.f32 %f41, %f41, {F(4f / (MathF.PI * MathF.PI))};");
        ptx.AppendLine($"    sub.rn.f32 %f42, {F(1f)}, %f21; add.rn.f32 %f42, %f42, %f41;");
        ptx.AppendLine("    mov.f32 %f43, 0f00000000; setp.gt.f32 %p6, %f42, %f8;");
        ptx.AppendLine("    @%p6 div.rn.f32 %f43, %f41, %f42;");
        ptx.AppendLine("    mul.rn.f32 %f44, %f43, %f41;");
        ptx.AppendLine($"    sub.rn.f32 {result}, %f37, %f44;");
    }

    private static void EmitPositiveAtan(
        StringBuilder ptx, string numerator, string denominator, string result, int predicate)
    {
        ptx.AppendLine($"    setp.gt.f32 %p{predicate}, {denominator}, 0f00000000;");
        ptx.AppendLine($"    mov.f32 %f50, 0f00000000; @%p{predicate} div.rn.f32 %f50, {numerator}, {denominator};");
        ptx.AppendLine("    setp.gt.f32 %p7, %f50, 0f3F800000;");
        ptx.AppendLine("    rcp.approx.f32 %f51, %f50; selp.f32 %f52, %f51, %f50, %p7;");
        ptx.AppendLine($"    sub.rn.f32 %f53, %f52, {F(1f)};");
        ptx.AppendLine($"    fma.rn.f32 %f54, %f52, {F(0.0663f)}, {F(0.2447f)};");
        ptx.AppendLine($"    mul.rn.f32 %f55, %f52, {F(MathF.PI / 4f)};");
        ptx.AppendLine("    mul.rn.f32 %f56, %f52, %f53; neg.f32 %f58, %f54; fma.rn.f32 %f55, %f56, %f58, %f55;");
        ptx.AppendLine($"    sub.rn.f32 %f57, {F(MathF.PI / 2f)}, %f55;");
        ptx.AppendLine($"    selp.f32 {result}, %f57, %f55, %p7;");
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
            DirectPtxVisionOperation metric = MetricForBackward(spec.Operation);
            for (int coordinate = 0; coordinate < 4; coordinate++)
                EmitFiniteDifference(ptx, $"%f{coordinate}", $"%f{64 + coordinate}", metric, negative: true);
            ptx.AppendLine("    add.u64 %rd9, %rd3, %rd4; st.global.v4.f32 [%rd9], {%f64,%f65,%f66,%f67};");
            return Definition(spec, architecture, $"n{n}-deterministic-register-fd",
                [
                    Tensor("grad-output", DirectPtxPhysicalLayout.Vector, new(n), DirectPtxTensorAccess.Read),
                    Tensor("predicted", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Read),
                    Tensor("target", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Read),
                    Tensor("grad-predicted", DirectPtxPhysicalLayout.BoxXyxy, new(n, 4), DirectPtxTensorAccess.Write)
                ], Semantics(("gradient", spec.Operation.ToString()),
                    ("method", "symmetric register-only finite difference"),
                    ("epsilon", "0.001"), ("determinism", "one thread per box")),
                Finish(ptx), n, maxRegisters: 96, minBlocksPerSm: 1);
        }

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
        EmitGlobalIndex(pairPtx, owners);
        pairPtx.AppendLine("    mov.f32 %f64, 0f00000000; mov.f32 %f65, 0f00000000; mov.f32 %f66, 0f00000000; mov.f32 %f67, 0f00000000; mov.u32 %r8, 0;");
        pairPtx.AppendLine("PAIR_GRAD_LOOP:");
        pairPtx.AppendLine($"    setp.ge.u32 %p0, %r8, {other}; @%p0 bra PAIR_GRAD_DONE;");
        string aIndex = ownerA ? "%r2" : "%r8";
        string bIndex = ownerA ? "%r8" : "%r2";
        pairPtx.AppendLine($"    mul.wide.u32 %rd4, {aIndex}, 16; mul.wide.u32 %rd5, {bIndex}, 16; add.u64 %rd6, %rd1, %rd4; add.u64 %rd7, %rd2, %rd5;");
        pairPtx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd6]; ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd7];");
        pairPtx.AppendLine($"    mad.lo.u32 %r9, {aIndex}, {boxesB}, {bIndex}; mul.wide.u32 %rd8, %r9, 4; add.u64 %rd9, %rd0, %rd8; ld.global.f32 %f63, [%rd9];");
        DirectPtxVisionOperation pairMetric = variant switch
        {
            0 => (DirectPtxVisionOperation)((int)DirectPtxVisionOperation.GeneralizedBoxIou - 1),
            1 => DirectPtxVisionOperation.GeneralizedBoxIou,
            2 => DirectPtxVisionOperation.DistanceBoxIou,
            _ => DirectPtxVisionOperation.CompleteBoxIou
        };
        int startCoordinate = ownerA ? 0 : 4;
        for (int coordinate = 0; coordinate < 4; coordinate++)
            EmitFiniteDifference(pairPtx, $"%f{startCoordinate + coordinate}", $"%f{68 + coordinate}", pairMetric, negative: false);
        pairPtx.AppendLine("    add.rn.f32 %f64, %f64, %f68; add.rn.f32 %f65, %f65, %f69; add.rn.f32 %f66, %f66, %f70; add.rn.f32 %f67, %f67, %f71;");
        pairPtx.AppendLine("    add.u32 %r8, %r8, 1; bra PAIR_GRAD_LOOP;");
        pairPtx.AppendLine("PAIR_GRAD_DONE: mul.wide.u32 %rd10, %r2, 16; add.u64 %rd11, %rd3, %rd10; st.global.v4.f32 [%rd11], {%f64,%f65,%f66,%f67};");
        return Definition(spec, architecture,
            $"n{boxesA}-m{boxesB}-v{variant}-owner-{(ownerA ? "a" : "b")}",
            [
                Tensor("grad-output", DirectPtxPhysicalLayout.RowMajor2D, new(boxesA, boxesB), DirectPtxTensorAccess.Read),
                Tensor("boxes-a", DirectPtxPhysicalLayout.BoxXyxy, new(boxesA, 4), DirectPtxTensorAccess.Read),
                Tensor("boxes-b", DirectPtxPhysicalLayout.BoxXyxy, new(boxesB, 4), DirectPtxTensorAccess.Read),
                Tensor(ownerA ? "grad-a" : "grad-b", DirectPtxPhysicalLayout.BoxXyxy, new(owners, 4), DirectPtxTensorAccess.Write)
            ], Semantics(("gradient", $"iou-family-variant-{variant}"),
                ("owner", ownerA ? "a" : "b"),
                ("method", "symmetric register-only finite difference"),
                ("determinism", "one owner thread, fixed other-index order")),
            Finish(pairPtx), owners, maxRegisters: 96, minBlocksPerSm: 1);
    }

    private static DirectPtxVisionOperation MetricForBackward(DirectPtxVisionOperation operation) =>
        operation switch
        {
            DirectPtxVisionOperation.IoULossBackward =>
                (DirectPtxVisionOperation)((int)DirectPtxVisionOperation.GeneralizedBoxIou - 1),
            DirectPtxVisionOperation.GIoULossBackward => DirectPtxVisionOperation.GeneralizedBoxIou,
            DirectPtxVisionOperation.DIoULossBackward => DirectPtxVisionOperation.DistanceBoxIou,
            _ => DirectPtxVisionOperation.CompleteBoxIou
        };

    private static void EmitFiniteDifference(
        StringBuilder ptx,
        string coordinate,
        string result,
        DirectPtxVisionOperation metric,
        bool negative)
    {
        ptx.AppendLine($"    mov.f32 %f62, {coordinate}; add.rn.f32 {coordinate}, %f62, {F(0.001f)};");
        EmitMetric(ptx, metric, "%f60");
        ptx.AppendLine($"    sub.rn.f32 {coordinate}, %f62, {F(0.001f)};");
        EmitMetric(ptx, metric, "%f61");
        ptx.AppendLine($"    mov.f32 {coordinate}, %f62; sub.rn.f32 {result}, %f60, %f61; mul.rn.f32 {result}, {result}, {F(500f)}; mul.rn.f32 {result}, {result}, %f63;");
        if (negative) ptx.AppendLine($"    neg.f32 {result}, {result};");
    }
}
