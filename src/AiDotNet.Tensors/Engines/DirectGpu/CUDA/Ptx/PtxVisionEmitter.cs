using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal static partial class PtxVisionEmitter
{
    internal static string EntryPoint(DirectPtxVisionOperation operation) =>
        "aidotnet_vision_" + (operation switch
        {
            DirectPtxVisionOperation.GeneralizedBoxIou => "generalized_box_iou",
            DirectPtxVisionOperation.DistanceBoxIou => "distance_box_iou",
            DirectPtxVisionOperation.CompleteBoxIou => "complete_box_iou",
            DirectPtxVisionOperation.BoxArea => "box_area",
            DirectPtxVisionOperation.BoxConvert => "box_convert",
            DirectPtxVisionOperation.IoULoss => "iou_loss",
            DirectPtxVisionOperation.GIoULoss => "giou_loss",
            DirectPtxVisionOperation.DIoULoss => "diou_loss",
            DirectPtxVisionOperation.CIoULoss => "ciou_loss",
            DirectPtxVisionOperation.IoULossBackward => "iou_loss_backward",
            DirectPtxVisionOperation.GIoULossBackward => "giou_loss_backward",
            DirectPtxVisionOperation.DIoULossBackward => "diou_loss_backward",
            DirectPtxVisionOperation.CIoULossBackward => "ciou_loss_backward",
            DirectPtxVisionOperation.IouFamilyBackwardA => "iou_family_backward_a",
            DirectPtxVisionOperation.IouFamilyBackwardB => "iou_family_backward_b",
            DirectPtxVisionOperation.Nms => "nms",
            DirectPtxVisionOperation.MasksToBoxes => "masks_to_boxes",
            DirectPtxVisionOperation.RoiAlign => "roi_align",
            DirectPtxVisionOperation.RoiPool => "roi_pool",
            DirectPtxVisionOperation.PsRoiAlign => "ps_roi_align",
            DirectPtxVisionOperation.PsRoiPool => "ps_roi_pool",
            DirectPtxVisionOperation.Cross3 => "cross3",
            DirectPtxVisionOperation.Meshgrid2D => "meshgrid_2d",
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        });

    internal static DirectPtxVisionDefinition Emit(
        DirectPtxVisionSpec spec,
        DirectPtxArchitectureFamily architecture,
        int ccMajor,
        int ccMinor)
    {
        if (architecture != DirectPtxArchitectureFamily.Ampere ||
            !DirectPtxArchitecture.HasValidatedVisionBoxIou(ccMajor, ccMinor))
            throw new NotSupportedException(
                $"Vision direct PTX has no {architecture} SM {ccMajor}.{ccMinor} emitter.");
        if (!DirectPtxVisionSpecializations.IsAdmitted(spec))
            throw new NotSupportedException(
                $"Vision specialization {spec} is not in the closed admission table.");
        return spec.Operation switch
        {
            DirectPtxVisionOperation.GeneralizedBoxIou or
            DirectPtxVisionOperation.DistanceBoxIou or
            DirectPtxVisionOperation.CompleteBoxIou =>
                EmitPairwiseMetric(spec, architecture, ccMajor, ccMinor),
            DirectPtxVisionOperation.BoxArea => EmitBoxArea(spec, architecture, ccMajor, ccMinor),
            DirectPtxVisionOperation.BoxConvert => EmitBoxConvert(spec, architecture, ccMajor, ccMinor),
            DirectPtxVisionOperation.IoULoss or
            DirectPtxVisionOperation.GIoULoss or
            DirectPtxVisionOperation.DIoULoss or
            DirectPtxVisionOperation.CIoULoss => EmitAlignedLoss(spec, architecture, ccMajor, ccMinor),
            DirectPtxVisionOperation.Cross3 => EmitCross3(spec, architecture, ccMajor, ccMinor),
            DirectPtxVisionOperation.Meshgrid2D => EmitMeshgrid2D(spec, architecture, ccMajor, ccMinor),
            DirectPtxVisionOperation.MasksToBoxes => EmitMasksToBoxes(spec, architecture, ccMajor, ccMinor),
            DirectPtxVisionOperation.IoULossBackward or
            DirectPtxVisionOperation.GIoULossBackward or
            DirectPtxVisionOperation.DIoULossBackward or
            DirectPtxVisionOperation.CIoULossBackward or
            DirectPtxVisionOperation.IouFamilyBackwardA or
            DirectPtxVisionOperation.IouFamilyBackwardB =>
                EmitBoxBackward(spec, architecture, ccMajor, ccMinor),
            DirectPtxVisionOperation.Nms => EmitNms(spec, architecture, ccMajor, ccMinor),
            DirectPtxVisionOperation.RoiAlign or DirectPtxVisionOperation.RoiPool or
            DirectPtxVisionOperation.PsRoiAlign or DirectPtxVisionOperation.PsRoiPool =>
                EmitRoi(spec, architecture, ccMajor, ccMinor),
            _ => throw new NotSupportedException($"No direct PTX emitter exists for {spec.Operation}.")
        };
    }

    private static DirectPtxVisionDefinition Definition(
        DirectPtxVisionSpec spec,
        DirectPtxArchitectureFamily architecture,
        string variant,
        IReadOnlyList<DirectPtxTensorContract> tensors,
        IReadOnlyDictionary<string, string> semantics,
        string ptx,
        int totalThreads,
        int maxRegisters = 64,
        int minBlocksPerSm = 2,
        uint blockThreads = 256,
        uint dynamicSharedBytes = 0)
    {
        if (totalThreads <= 0) throw new ArgumentOutOfRangeException(nameof(totalThreads));
        uint grid = checked((uint)((totalThreads + blockThreads - 1) / blockThreads));
        var blueprint = new DirectPtxKernelBlueprint(
            Operation: "vision-" + spec.Operation,
            Version: 1,
            Architecture: architecture,
            Variant: variant,
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(
                maxRegisters, 0, 0, minBlocksPerSm),
            Semantics: semantics);
        return new DirectPtxVisionDefinition(
            blueprint, ptx, grid, 1, 1, blockThreads, 1, 1, dynamicSharedBytes);
    }

    private static DirectPtxTensorContract Tensor(
        string name,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent,
        DirectPtxTensorAccess access,
        DirectPtxPhysicalType type = DirectPtxPhysicalType.Float32,
        int alignment = 16) =>
        new(name, type, layout, extent, extent, alignment, access, DirectPtxExtentMode.Exact);

    private static Dictionary<string, string> Semantics(params (string Key, string Value)[] values)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["input"] = "fp32",
            ["accumulator"] = "fp32",
            ["output"] = "fp32",
            ["intermediate-global-bytes"] = "0"
        };
        foreach ((string key, string value) in values) result[key] = value;
        return result;
    }

    private static StringBuilder Begin(
        DirectPtxVisionSpec spec, int ccMajor, int ccMinor, params string[] parameters)
    {
        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint(spec.Operation)}(");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}_ptr{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        return ptx;
    }

    private static void LoadParameters(StringBuilder ptx, params string[] parameters)
    {
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    ld.param.u64 %rd{i}, [{parameters[i]}_ptr];");
    }

    private static void EmitGlobalIndex(StringBuilder ptx, int total, string done = "DONE")
    {
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine("    mad.lo.u32 %r2, %r0, 256, %r1;");
        if ((total & 255) != 0)
        {
            ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
            ptx.AppendLine($"    @%p0 bra {done};");
        }
    }

    private static string Finish(StringBuilder ptx, string done = "DONE")
    {
        ptx.AppendLine(done + ":");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void RequireOneOf(int value, string name, params int[] choices)
    {
        foreach (int choice in choices) if (value == choice) return;
        throw new NotSupportedException($"{name}={value} has no emitted vision specialization.");
    }

    private static string F(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
