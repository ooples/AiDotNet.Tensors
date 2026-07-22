#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxStructuredSparse2x4Operation
{
    Enforce,
    Decompress,
    Gemm,
    GemmBiasRelu
}

/// <summary>
/// Exact direct-PTX 2:4 structured-sparsity family. Shape values and GEMM
/// epilogue scalars are baked into bounded module identities, so admitted
/// launches contain only tensor pointers and no dynamic layout/shape checks.
/// </summary>
internal sealed class PtxStructuredSparse2x4F32Kernel : IDisposable
{
    internal const int Rows = 256;
    internal const int Inner = 256;
    internal const int Columns = 64;
    internal const int Groups = Rows * Inner / 4;
    internal const int PackedValues = Groups * 2;
    internal const int DenseElements = Rows * Inner;
    internal const int OutputElements = Rows * Columns;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxStructuredSparse2x4Operation Operation { get; }
    internal float Alpha { get; }
    internal float Beta { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxStructuredSparse2x4F32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxStructuredSparse2x4Operation operation,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Structured 2:4 kernels have no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        if (!float.IsFinite(alpha) || !float.IsFinite(beta))
            throw new ArgumentOutOfRangeException(nameof(alpha), "GEMM epilogue scalars must be finite.");
        Operation = operation;
        Alpha = alpha;
        Beta = beta;
        EntryPoint = GetEntryPoint(operation, alpha, beta);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation, alpha, beta);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, operation, alpha, beta);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsMatrixShape(int rows, int inner) =>
        rows == Rows && inner == Inner;

    internal static bool SupportsGemmShape(int rows, int columns, int inner) =>
        rows == Rows && columns == Columns && inner == Inner;

    internal unsafe void Launch(
        DirectPtxTensorView first,
        DirectPtxTensorView second,
        DirectPtxTensorView third)
    {
        if (Operation is DirectPtxStructuredSparse2x4Operation.Gemm or
            DirectPtxStructuredSparse2x4Operation.GemmBiasRelu)
            throw new InvalidOperationException("This structured sparse operation requires a GEMM ABI.");
        Require(first, Blueprint.Tensors[0], nameof(first));
        Require(second, Blueprint.Tensors[1], nameof(second));
        Require(third, Blueprint.Tensors[2], nameof(third));
        RequireOutputDisjoint(third, first, second);
        IntPtr p0 = first.Pointer;
        IntPtr p1 = second.Pointer;
        IntPtr p2 = third.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        int blocks = Operation == DirectPtxStructuredSparse2x4Operation.Enforce
            ? Groups / BlockThreads : Groups / BlockThreads;
        _module.Launch(_function, (uint)blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal unsafe void LaunchGemm(
        DirectPtxTensorView values,
        DirectPtxTensorView metadata,
        DirectPtxTensorView denseB,
        DirectPtxTensorView output)
    {
        if (Operation != DirectPtxStructuredSparse2x4Operation.Gemm)
            throw new InvalidOperationException("This module does not expose the plain structured sparse GEMM ABI.");
        Require(values, Blueprint.Tensors[0], nameof(values));
        Require(metadata, Blueprint.Tensors[1], nameof(metadata));
        Require(denseB, Blueprint.Tensors[2], nameof(denseB));
        Require(output, Blueprint.Tensors[3], nameof(output));
        RequireOutputDisjoint(output, values, metadata, denseB);
        IntPtr p0 = values.Pointer;
        IntPtr p1 = metadata.Pointer;
        IntPtr p2 = denseB.Pointer;
        IntPtr p3 = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        _module.Launch(_function, OutputElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal unsafe void LaunchBiasRelu(
        DirectPtxTensorView values,
        DirectPtxTensorView metadata,
        DirectPtxTensorView denseB,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        if (Operation != DirectPtxStructuredSparse2x4Operation.GemmBiasRelu)
            throw new InvalidOperationException("This module does not expose the bias/ReLU structured sparse GEMM ABI.");
        Require(values, Blueprint.Tensors[0], nameof(values));
        Require(metadata, Blueprint.Tensors[1], nameof(metadata));
        Require(denseB, Blueprint.Tensors[2], nameof(denseB));
        Require(bias, Blueprint.Tensors[3], nameof(bias));
        Require(output, Blueprint.Tensors[4], nameof(output));
        RequireOutputDisjoint(output, values, metadata, denseB, bias);
        IntPtr p0 = values.Pointer;
        IntPtr p1 = metadata.Pointer;
        IntPtr p2 = denseB.Pointer;
        IntPtr p3 = bias.Pointer;
        IntPtr p4 = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        arguments[4] = &p4;
        _module.Launch(_function, OutputElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxStructuredSparse2x4Operation operation,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        var denseA = new DirectPtxExtent(Rows, Inner);
        var values = new DirectPtxExtent(PackedValues);
        var metadata = new DirectPtxExtent(Groups);
        var denseB = new DirectPtxExtent(Inner, Columns);
        var output = new DirectPtxExtent(Rows, Columns);
        IReadOnlyList<DirectPtxTensorContract> tensors = operation switch
        {
            DirectPtxStructuredSparse2x4Operation.Enforce =>
            [
                Exact("dense-input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    denseA, DirectPtxTensorAccess.Read),
                Exact("packed-values", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Values, values, DirectPtxTensorAccess.Write),
                Exact("metadata", DirectPtxPhysicalType.UInt8,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Metadata, metadata, DirectPtxTensorAccess.Write)
            ],
            DirectPtxStructuredSparse2x4Operation.Decompress =>
            [
                Exact("packed-values", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Values, values, DirectPtxTensorAccess.Read),
                Exact("metadata", DirectPtxPhysicalType.UInt8,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Metadata, metadata, DirectPtxTensorAccess.Read),
                Exact("dense-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    denseA, DirectPtxTensorAccess.Write)
            ],
            DirectPtxStructuredSparse2x4Operation.Gemm =>
            [
                Exact("packed-values", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Values, values, DirectPtxTensorAccess.Read),
                Exact("metadata", DirectPtxPhysicalType.UInt8,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Metadata, metadata, DirectPtxTensorAccess.Read),
                Exact("dense-b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    denseB, DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, beta == 0.0f ? DirectPtxTensorAccess.Write : DirectPtxTensorAccess.ReadWrite)
            ],
            DirectPtxStructuredSparse2x4Operation.GemmBiasRelu =>
            [
                Exact("packed-values", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Values, values, DirectPtxTensorAccess.Read),
                Exact("metadata", DirectPtxPhysicalType.UInt8,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Metadata, metadata, DirectPtxTensorAccess.Read),
                Exact("dense-b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    denseB, DirectPtxTensorAccess.Read),
                Exact("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    new DirectPtxExtent(Columns), DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, DirectPtxTensorAccess.Write)
            ],
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };
        return new DirectPtxKernelBlueprint(
            Operation: "structured-sparse-2x4-" + operation.ToString().ToLowerInvariant(),
            Version: 1,
            Architecture: architecture,
            Variant: operation == DirectPtxStructuredSparse2x4Operation.Gemm
                ? $"m256-n64-k256-a{BitConverter.SingleToInt32Bits(alpha):x8}-b{BitConverter.SingleToInt32Bits(beta):x8}"
                : "m256-n64-k256",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(48, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["packing"] = "two-fp32-values-plus-one-byte-position-pair-per-four",
                ["metadata"] = "low-two-bits-first-position;bits-two-three-second-position",
                ["group-order"] = "row-major",
                ["alpha-bits"] = BitConverter.SingleToInt32Bits(alpha).ToString("x8"),
                ["beta-bits"] = BitConverter.SingleToInt32Bits(beta).ToString("x8"),
                ["intermediate-global-bytes"] = "0",
                ["workspace-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxStructuredSparse2x4Operation operation,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        switch (operation)
        {
            case DirectPtxStructuredSparse2x4Operation.Enforce:
                EmitEnforce(ptx, GetEntryPoint(operation, alpha, beta));
                break;
            case DirectPtxStructuredSparse2x4Operation.Decompress:
                EmitDecompress(ptx, GetEntryPoint(operation, alpha, beta));
                break;
            case DirectPtxStructuredSparse2x4Operation.Gemm:
                EmitGemm(ptx, GetEntryPoint(operation, alpha, beta), alpha, beta, false);
                break;
            case DirectPtxStructuredSparse2x4Operation.GemmBiasRelu:
                EmitGemm(ptx, GetEntryPoint(operation, alpha, beta), 1.0f, 0.0f, true);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(operation));
        }
        return ptx.ToString();
    }

    private static void EmitEnforce(StringBuilder ptx, string entry)
    {
        EmitHeader(ptx, entry, "dense_ptr", "values_ptr", "metadata_ptr");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        EmitThreePointersAndLinearIndex(ptx, "dense_ptr", "values_ptr", "metadata_ptr");
        ptx.AppendLine("    shl.b32 %r3, %r2, 2;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd4];");
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    abs.f32 %f{i + 4}, %f{i};");
            ptx.AppendLine($"    mov.u32 %r{i + 4}, {i};");
        }
        EmitCandidateSwap(ptx, 0, 1, 0);
        EmitCandidateSwap(ptx, 0, 2, 1);
        EmitCandidateSwap(ptx, 0, 3, 2);
        EmitCandidateSwap(ptx, 1, 2, 3);
        EmitCandidateSwap(ptx, 1, 3, 4);
        ptx.AppendLine("    setp.gt.u32 %p5, %r4, %r5;");
        ptx.AppendLine("    selp.u32 %r12, %r5, %r4, %p5;");
        ptx.AppendLine("    selp.u32 %r13, %r4, %r5, %p5;");
        ptx.AppendLine("    selp.f32 %f20, %f1, %f0, %p5;");
        ptx.AppendLine("    selp.f32 %f21, %f0, %f1, %p5;");
        ptx.AppendLine("    shl.b32 %r14, %r13, 2;");
        ptx.AppendLine("    or.b32 %r14, %r14, %r12;");
        ptx.AppendLine("    shl.b32 %r15, %r2, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r15, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    st.global.v2.f32 [%rd6], {%f20,%f21};");
        ptx.AppendLine("    cvt.u64.u32 %rd7, %r2;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.u8 [%rd8], %r14;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitCandidateSwap(StringBuilder ptx, int left, int right, int predicate)
    {
        int leftValue = left;
        int rightValue = right;
        int leftAbs = left + 4;
        int rightAbs = right + 4;
        int leftIndex = left + 4;
        int rightIndex = right + 4;
        ptx.AppendLine($"    setp.gt.f32 %p{predicate}, %f{rightAbs}, %f{leftAbs};");
        ptx.AppendLine($"    mov.f32 %f16, %f{leftValue};");
        ptx.AppendLine($"    mov.f32 %f17, %f{leftAbs};");
        ptx.AppendLine($"    mov.u32 %r20, %r{leftIndex};");
        ptx.AppendLine($"    selp.f32 %f{leftValue}, %f{rightValue}, %f{leftValue}, %p{predicate};");
        ptx.AppendLine($"    selp.f32 %f{rightValue}, %f16, %f{rightValue}, %p{predicate};");
        ptx.AppendLine($"    selp.f32 %f{leftAbs}, %f{rightAbs}, %f{leftAbs}, %p{predicate};");
        ptx.AppendLine($"    selp.f32 %f{rightAbs}, %f17, %f{rightAbs}, %p{predicate};");
        ptx.AppendLine($"    selp.u32 %r{leftIndex}, %r{rightIndex}, %r{leftIndex}, %p{predicate};");
        ptx.AppendLine($"    selp.u32 %r{rightIndex}, %r20, %r{rightIndex}, %p{predicate};");
    }

    private static void EmitDecompress(StringBuilder ptx, string entry)
    {
        EmitHeader(ptx, entry, "values_ptr", "metadata_ptr", "dense_ptr");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        EmitThreePointersAndLinearIndex(ptx, "values_ptr", "metadata_ptr", "dense_ptr");
        ptx.AppendLine("    cvt.u64.u32 %rd3, %r2;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.u8 %r3, [%rd4];");
        ptx.AppendLine("    and.b32 %r4, %r3, 3;");
        ptx.AppendLine("    shr.u32 %r5, %r3, 2;");
        ptx.AppendLine("    and.b32 %r5, %r5, 3;");
        ptx.AppendLine("    shl.b32 %r6, %r2, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    ld.global.v2.f32 {%f0,%f1}, [%rd6];");
        ptx.AppendLine("    shl.b32 %r7, %r2, 2;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    st.global.v4.f32 [%rd8], {%f2,%f2,%f2,%f2};");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd8, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f0;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd8, %rd11;");
        ptx.AppendLine("    st.global.f32 [%rd12], %f1;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitGemm(
        StringBuilder ptx,
        string entry,
        float alpha,
        float beta,
        bool biasRelu)
    {
        if (biasRelu)
            EmitHeader(ptx, entry, "values_ptr", "metadata_ptr", "b_ptr", "bias_ptr", "output_ptr");
        else
            EmitHeader(ptx, entry, "values_ptr", "metadata_ptr", "b_ptr", "output_ptr");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [values_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [metadata_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [b_ptr];");
        ptx.AppendLine($"    ld.param.u64 %rd3, [{(biasRelu ? "bias_ptr" : "output_ptr")}];");
        if (biasRelu) ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {Columns - 1};");
        ptx.AppendLine("    shl.b32 %r5, %r3, 6;");
        ptx.AppendLine("    mov.u32 %r6, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("GEMM_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r6, 64;");
        ptx.AppendLine("    @%p0 bra GEMM_DONE;");
        ptx.AppendLine("    add.u32 %r7, %r5, %r6;");
        ptx.AppendLine("    cvt.u64.u32 %rd5, %r7;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.u8 %r8, [%rd6];");
        ptx.AppendLine("    and.b32 %r9, %r8, 3;");
        ptx.AppendLine("    shr.u32 %r10, %r8, 2;");
        ptx.AppendLine("    and.b32 %r10, %r10, 3;");
        ptx.AppendLine("    shl.b32 %r11, %r7, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd7;");
        ptx.AppendLine("    ld.global.v2.f32 {%f1,%f2}, [%rd8];");
        ptx.AppendLine("    shl.b32 %r12, %r6, 2;");
        ptx.AppendLine("    add.u32 %r13, %r12, %r9;");
        ptx.AppendLine("    shl.b32 %r13, %r13, 6;");
        ptx.AppendLine("    add.u32 %r13, %r13, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r13, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
        ptx.AppendLine("    ld.global.f32 %f3, [%rd10];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f3, %f0;");
        ptx.AppendLine("    add.u32 %r14, %r12, %r10;");
        ptx.AppendLine("    shl.b32 %r14, %r14, 6;");
        ptx.AppendLine("    add.u32 %r14, %r14, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r14, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd2, %rd11;");
        ptx.AppendLine("    ld.global.f32 %f4, [%rd12];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f4, %f0;");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("    bra GEMM_LOOP;");
        ptx.AppendLine("GEMM_DONE:");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r2, 4;");
        if (biasRelu)
        {
            ptx.AppendLine("    mul.wide.u32 %rd14, %r4, 4;");
            ptx.AppendLine("    add.u64 %rd15, %rd3, %rd14;");
            ptx.AppendLine("    ld.global.f32 %f5, [%rd15];");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f5;");
            ptx.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
            ptx.AppendLine("    add.u64 %rd16, %rd4, %rd13;");
            ptx.AppendLine("    st.global.f32 [%rd16], %f0;");
        }
        else
        {
            string a = FloatLiteral(alpha);
            string b = FloatLiteral(beta);
            ptx.AppendLine($"    mul.rn.f32 %f0, %f0, {a};");
            ptx.AppendLine("    add.u64 %rd16, %rd3, %rd13;");
            if (beta != 0.0f)
            {
                ptx.AppendLine("    ld.global.f32 %f5, [%rd16];");
                ptx.AppendLine($"    fma.rn.f32 %f0, %f5, {b}, %f0;");
            }
            ptx.AppendLine("    st.global.f32 [%rd16], %f0;");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitHeader(StringBuilder ptx, string entry, params string[] parameters)
    {
        ptx.AppendLine($".visible .entry {entry}(");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
    }

    private static void EmitThreePointersAndLinearIndex(
        StringBuilder ptx,
        string p0,
        string p1,
        string p2)
    {
        ptx.AppendLine($"    ld.param.u64 %rd0, [{p0}];");
        ptx.AppendLine($"    ld.param.u64 %rd1, [{p1}];");
        ptx.AppendLine($"    ld.param.u64 %rd2, [{p2}];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
    }

    private static DirectPtxTensorContract Exact(
        string name,
        DirectPtxPhysicalType type,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent,
        DirectPtxTensorAccess access) =>
        new(name, type, layout, extent, extent, 16, access, DirectPtxExtentMode.Exact);

    private static string GetEntryPoint(
        DirectPtxStructuredSparse2x4Operation operation,
        float alpha,
        float beta) => operation switch
        {
            DirectPtxStructuredSparse2x4Operation.Enforce => "aidotnet_enforce_2x4_f32_m256_k256",
            DirectPtxStructuredSparse2x4Operation.Decompress => "aidotnet_decompress_2x4_f32_m256_k256",
            DirectPtxStructuredSparse2x4Operation.Gemm =>
                $"aidotnet_sparse_gemm_2x4_f32_m256_n64_k256_a{BitConverter.SingleToInt32Bits(alpha):x8}_b{BitConverter.SingleToInt32Bits(beta):x8}",
            DirectPtxStructuredSparse2x4Operation.GemmBiasRelu =>
                "aidotnet_sparse_gemm_2x4_bias_relu_f32_m256_n64_k256",
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

    private static string FloatLiteral(float value) =>
        $"0f{BitConverter.SingleToInt32Bits(value):X8}";

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
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

    private static void RequireOutputDisjoint(
        DirectPtxTensorView output,
        DirectPtxTensorView input0,
        DirectPtxTensorView input1,
        DirectPtxTensorView input2 = default,
        DirectPtxTensorView input3 = default)
    {
        if (Overlaps(output, input0) || Overlaps(output, input1) ||
            (input2.Pointer != IntPtr.Zero && Overlaps(output, input2)) ||
            (input3.Pointer != IntPtr.Zero && Overlaps(output, input3)))
            throw new ArgumentException("Structured sparse output must be disjoint from every input.", nameof(output));
    }

    public void Dispose() => _module.Dispose();
}
#endif
