#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal readonly record struct DirectPtxFusedSparseLinearKey(bool HasBias, int Activation);

/// <summary>
/// Exact CSR sparse-linear specialization with a register-resident bias,
/// reduction, and activation epilogue. The packed CSR structure is consumed
/// directly and no intermediate dense product is materialized.
/// </summary>
internal sealed class PtxFusedSparseLinearF32Kernel : IDisposable
{
    internal const int Batch = 32;
    internal const int InputFeatures = 1024;
    internal const int OutputFeatures = 1024;
    internal const int NonZeros = 16_384;
    internal const int PackedCsrElements = OutputFeatures + 1 + NonZeros;
    internal const int OutputElements = Batch * OutputFeatures;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxFusedSparseLinearKey Key { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedSparseLinearF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxFusedSparseLinearKey key)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Fused sparse linear has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        if (!SupportsActivation(key.Activation)) throw new ArgumentOutOfRangeException(nameof(key));
        Key = key;
        EntryPoint = GetEntryPoint(key);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, key);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, key);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int batch, int inputFeatures, int outputFeatures, int nonZeros) =>
        batch == Batch && inputFeatures == InputFeatures &&
        outputFeatures == OutputFeatures && nonZeros == NonZeros;

    internal static bool SupportsActivation(int activation) => activation is >= 0 and <= 6;

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView packedCsr,
        DirectPtxTensorView values,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(packedCsr, Blueprint.Tensors[1], nameof(packedCsr));
        Require(values, Blueprint.Tensors[2], nameof(values));
        int outputContract = Key.HasBias ? 4 : 3;
        if (Key.HasBias) Require(bias, Blueprint.Tensors[3], nameof(bias));
        Require(output, Blueprint.Tensors[outputContract], nameof(output));
        if (Overlaps(output, input) || Overlaps(output, packedCsr) || Overlaps(output, values) ||
            (Key.HasBias && Overlaps(output, bias)))
            throw new ArgumentException("Fused sparse-linear output must be disjoint from every input.", nameof(output));

        IntPtr p0 = input.Pointer;
        IntPtr p1 = packedCsr.Pointer;
        IntPtr p2 = values.Pointer;
        IntPtr p3 = bias.Pointer;
        IntPtr p4 = output.Pointer;
        if (Key.HasBias)
        {
            void** arguments = stackalloc void*[5];
            arguments[0] = &p0; arguments[1] = &p1; arguments[2] = &p2;
            arguments[3] = &p3; arguments[4] = &p4;
            _module.Launch(_function, OutputElements / BlockThreads, 1, 1,
                BlockThreads, 1, 1, 0, arguments);
        }
        else
        {
            void** arguments = stackalloc void*[4];
            arguments[0] = &p0; arguments[1] = &p1;
            arguments[2] = &p2; arguments[3] = &p4;
            _module.Launch(_function, OutputElements / BlockThreads, 1, 1,
                BlockThreads, 1, 1, 0, arguments);
        }
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxFusedSparseLinearKey key)
    {
        if (!SupportsActivation(key.Activation)) throw new ArgumentOutOfRangeException(nameof(key));
        var input = new DirectPtxExtent(Batch, InputFeatures);
        var packed = new DirectPtxExtent(PackedCsrElements);
        var values = new DirectPtxExtent(NonZeros);
        var bias = new DirectPtxExtent(OutputFeatures);
        var output = new DirectPtxExtent(Batch, OutputFeatures);
        IReadOnlyList<DirectPtxTensorContract> tensors = key.HasBias
            ?
            [
                Exact("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    input, DirectPtxTensorAccess.Read),
                Exact("packed-csr", DirectPtxPhysicalType.Int32,
                    DirectPtxPhysicalLayout.PackedCsrRowsAndColumns, packed, DirectPtxTensorAccess.Read),
                Exact("values", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.CsrValues,
                    values, DirectPtxTensorAccess.Read),
                Exact("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, DirectPtxTensorAccess.Write)
            ]
            :
            [
                Exact("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    input, DirectPtxTensorAccess.Read),
                Exact("packed-csr", DirectPtxPhysicalType.Int32,
                    DirectPtxPhysicalLayout.PackedCsrRowsAndColumns, packed, DirectPtxTensorAccess.Read),
                Exact("values", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.CsrValues,
                    values, DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, DirectPtxTensorAccess.Write)
            ];
        return new DirectPtxKernelBlueprint(
            Operation: "fused-sparse-linear-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{Batch}-i{InputFeatures}-o{OutputFeatures}-nnz{NonZeros}-bias{(key.HasBias ? 1 : 0)}-act{key.Activation}",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(36, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["fusion"] = "csr-dot-plus-optional-bias-plus-activation",
                ["activation"] = key.Activation.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["reduction-order"] = "ascending-csr-storage-position",
                ["invalid-csr-row"] = "zero-without-activation",
                ["output-write"] = "single-overwrite",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxFusedSparseLinearKey key)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        if (!SupportsActivation(key.Activation)) throw new ArgumentOutOfRangeException(nameof(key));
        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(key)}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 packed_csr_ptr,");
        ptx.AppendLine("    .param .u64 values_ptr,");
        if (key.HasBias) ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [packed_csr_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [values_ptr];");
        int outputPointer = 3;
        if (key.HasBias)
        {
            ptx.AppendLine("    ld.param.u64 %rd3, [bias_ptr];");
            outputPointer = 4;
        }
        ptx.AppendLine($"    ld.param.u64 %rd{outputPointer}, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 10;");
        ptx.AppendLine("    and.b32 %r4, %r2, 1023;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.s32 %r5, [%rd6];");
        ptx.AppendLine("    ld.global.s32 %r6, [%rd6+4];");
        ptx.AppendLine("    setp.lt.s32 %p0, %r5, 0;");
        ptx.AppendLine("    setp.lt.s32 %p1, %r6, %r5;");
        ptx.AppendLine($"    setp.gt.s32 %p2, %r6, {NonZeros};");
        ptx.AppendLine("    or.pred %p3, %p0, %p1;");
        ptx.AppendLine("    or.pred %p3, %p3, %p2;");
        ptx.AppendLine("    @%p3 mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    @%p3 bra FSL_STORE;");
        if (key.HasBias)
        {
            ptx.AppendLine("    add.u64 %rd7, %rd3, %rd5;");
            ptx.AppendLine("    ld.global.f32 %f0, [%rd7];");
        }
        else
        {
            ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        }
        ptx.AppendLine("    mov.u32 %r7, %r5;");
        ptx.AppendLine("FSL_REDUCE:");
        ptx.AppendLine("    setp.ge.s32 %p4, %r7, %r6;");
        ptx.AppendLine("    @%p4 bra FSL_ACTIVATE;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r7, 4;");
        ptx.AppendLine($"    add.u64 %rd9, %rd1, {((OutputFeatures + 1) * sizeof(int))};");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd8;");
        ptx.AppendLine("    ld.global.u32 %r8, [%rd9];");
        ptx.AppendLine($"    setp.ge.u32 %p5, %r8, {InputFeatures};");
        ptx.AppendLine("    @%p5 bra FSL_NEXT;");
        ptx.AppendLine("    shl.b32 %r9, %r3, 10;");
        ptx.AppendLine("    add.u32 %r9, %r9, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd0, %rd10;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd11];");
        ptx.AppendLine("    add.u64 %rd12, %rd2, %rd8;");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd12];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("FSL_NEXT:");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine("    bra FSL_REDUCE;");
        ptx.AppendLine("FSL_ACTIVATE:");
        EmitActivation(ptx, key.Activation);
        ptx.AppendLine("FSL_STORE:");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r2, 4;");
        ptx.AppendLine($"    add.u64 %rd14, %rd{outputPointer}, %rd13;");
        ptx.AppendLine("    st.global.f32 [%rd14], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitActivation(StringBuilder ptx, int activation)
    {
        switch (activation)
        {
            case 0:
                return;
            case 1:
                ptx.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
                return;
            case 2:
                ptx.AppendLine("    mul.rn.f32 %f1, %f0, %f0;");
                ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f0;");
                ptx.AppendLine("    fma.rn.f32 %f1, 0f3d372713, %f1, %f0;");
                ptx.AppendLine("    mul.rn.f32 %f1, %f1, 0f3f4c422a;");
                EmitSigmoid(ptx, "%f1", "%f2", scale: 2f);
                ptx.AppendLine("    mul.rn.f32 %f0, %f0, %f2;");
                return;
            case 3:
                EmitSigmoid(ptx, "%f0", "%f1", scale: 1f);
                ptx.AppendLine("    mov.f32 %f0, %f1;");
                return;
            case 4:
                EmitSigmoid(ptx, "%f0", "%f1", scale: 2f);
                ptx.AppendLine("    mul.rn.f32 %f0, %f1, 0f40000000;");
                ptx.AppendLine("    add.rn.f32 %f0, %f0, 0fbf800000;");
                return;
            case 5:
                ptx.AppendLine("    setp.gt.f32 %p6, %f0, 0f00000000;");
                ptx.AppendLine("    mul.rn.f32 %f1, %f0, 0f3c23d70a;");
                ptx.AppendLine("    selp.f32 %f0, %f0, %f1, %p6;");
                return;
            case 6:
                EmitSigmoid(ptx, "%f0", "%f1", scale: 1f);
                ptx.AppendLine("    mul.rn.f32 %f0, %f0, %f1;");
                return;
            default:
                throw new ArgumentOutOfRangeException(nameof(activation));
        }
    }

    private static void EmitSigmoid(
        StringBuilder ptx,
        string input,
        string output,
        float scale)
    {
        ptx.AppendLine($"    mul.rn.f32 {output}, {input}, {FloatLiteral(-scale * 1.4426950408889634f)};");
        ptx.AppendLine($"    ex2.approx.f32 {output}, {output};");
        ptx.AppendLine($"    add.rn.f32 {output}, {output}, 0f3f800000;");
        ptx.AppendLine($"    rcp.approx.f32 {output}, {output};");
    }

    private static string FloatLiteral(float value) =>
        $"0f{BitConverter.SingleToInt32Bits(value):x8}";

    private static string GetEntryPoint(DirectPtxFusedSparseLinearKey key) =>
        $"aidotnet_fused_sparse_linear_f32_b32_i1024_o1024_nnz16384_bias{(key.HasBias ? 1 : 0)}_act{key.Activation}";

    private static DirectPtxTensorContract Exact(
        string name,
        DirectPtxPhysicalType type,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent,
        DirectPtxTensorAccess access) =>
        new(name, type, layout, extent, extent, 16, access, DirectPtxExtentMode.Exact);

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
