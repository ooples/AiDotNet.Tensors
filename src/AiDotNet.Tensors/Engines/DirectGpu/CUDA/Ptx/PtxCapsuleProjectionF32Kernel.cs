#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxCapsuleProjectionOperation
{
    Predictions,
    Transform
}

/// <summary>
/// Exact capsule projection specializations. The input-dimension reduction is
/// completely unrolled and retained in one accumulator register before the
/// single global output write.
/// </summary>
internal sealed class PtxCapsuleProjectionF32Kernel : IDisposable
{
    internal const int Batch = 32;
    internal const int InputCapsules = 32;
    internal const int InputDimension = 8;
    internal const int OutputCapsules = 10;
    internal const int OutputDimension = 16;
    internal const int InputElements = Batch * InputCapsules * InputDimension;
    internal const int WeightElements = InputCapsules * OutputCapsules * InputDimension * OutputDimension;
    internal const int OutputElements = Batch * InputCapsules * OutputCapsules * OutputDimension;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxCapsuleProjectionOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCapsuleProjectionF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxCapsuleProjectionOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Capsule projection has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Operation = operation;
        EntryPoint = GetEntryPoint(operation);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, operation);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(
        int batch,
        int inputCapsules,
        int inputDimension,
        int outputCapsules,
        int outputDimension) =>
        batch == Batch && inputCapsules == InputCapsules &&
        inputDimension == InputDimension && outputCapsules == OutputCapsules &&
        outputDimension == OutputDimension;

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, input) || Overlaps(output, weights))
            throw new ArgumentException("Capsule projection output must be disjoint from both inputs.", nameof(output));
        IntPtr p0 = input.Pointer;
        IntPtr p1 = weights.Pointer;
        IntPtr p2 = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        _module.Launch(_function, OutputElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxCapsuleProjectionOperation operation)
    {
        var input = new DirectPtxExtent(Batch, InputCapsules, InputDimension);
        var output = new DirectPtxExtent(
            Batch, InputCapsules, OutputCapsules, OutputDimension);
        DirectPtxExtent weights = operation == DirectPtxCapsuleProjectionOperation.Predictions
            ? new DirectPtxExtent(InputCapsules, OutputCapsules, InputDimension, OutputDimension)
            : new DirectPtxExtent(InputCapsules, InputDimension, OutputCapsules, OutputDimension);
        DirectPtxPhysicalLayout weightLayout =
            operation == DirectPtxCapsuleProjectionOperation.Predictions
                ? DirectPtxPhysicalLayout.CapsulePredictionWeights
                : DirectPtxPhysicalLayout.CapsuleTransformWeights;
        return new DirectPtxKernelBlueprint(
            Operation: operation == DirectPtxCapsuleProjectionOperation.Predictions
                ? "capsule-predictions-f32" : "capsule-transform-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "b32-i32-id8-o10-od16",
            Tensors:
            [
                Exact("input", DirectPtxPhysicalLayout.CapsuleInput, input, DirectPtxTensorAccess.Read),
                Exact("weights", weightLayout, weights, DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalLayout.CapsulePredictions, output, DirectPtxTensorAccess.Write)
            ],
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["reduction-order"] = "fully-unrolled-ascending-input-dimension",
                ["output-write"] = "single-overwrite",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxCapsuleProjectionOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(operation)}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    and.b32 %r3, %r2, 15;");
        ptx.AppendLine("    shr.u32 %r4, %r2, 4;");
        ptx.AppendLine("    rem.u32 %r5, %r4, 10;");
        ptx.AppendLine("    div.u32 %r6, %r4, 10;");
        ptx.AppendLine("    and.b32 %r7, %r6, 31;");
        ptx.AppendLine("    shr.u32 %r8, %r6, 5;");
        ptx.AppendLine("    shl.b32 %r9, %r8, 8;");
        ptx.AppendLine("    mad.lo.u32 %r9, %r7, 8, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    mul.lo.u32 %r10, %r7, 1280;");
        if (operation == DirectPtxCapsuleProjectionOperation.Predictions)
            ptx.AppendLine("    mad.lo.u32 %r10, %r5, 128, %r10;");
        else
            ptx.AppendLine("    mad.lo.u32 %r10, %r5, 16, %r10;");
        ptx.AppendLine("    add.u32 %r10, %r10, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        int weightByteStride = operation == DirectPtxCapsuleProjectionOperation.Predictions ? 64 : 640;
        for (int k = 0; k < InputDimension; k++)
        {
            int inputByteOffset = k * sizeof(float);
            int weightByteOffset = k * weightByteStride;
            ptx.AppendLine($"    ld.global.f32 %f1, [%rd4+{inputByteOffset}];");
            ptx.AppendLine($"    ld.global.f32 %f2, [%rd6+{weightByteOffset}];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string GetEntryPoint(DirectPtxCapsuleProjectionOperation operation) =>
        operation == DirectPtxCapsuleProjectionOperation.Predictions
            ? "aidotnet_capsule_predictions_f32_b32_i32_id8_o10_od16"
            : "aidotnet_capsule_transform_f32_b32_i32_id8_o10_od16";

    private static DirectPtxTensorContract Exact(
        string name,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent,
        DirectPtxTensorAccess access) =>
        new(name, DirectPtxPhysicalType.Float32, layout, extent, extent, 16, access,
            DirectPtxExtentMode.Exact);

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer;
        nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    public void Dispose() => _module.Dispose();
}
#endif
