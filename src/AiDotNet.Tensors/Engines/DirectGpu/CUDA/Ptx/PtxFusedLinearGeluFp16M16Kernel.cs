#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Shape-baked M=16 FP16 linear + FP32 bias + tanh-GELU. Four warps share one
/// asynchronously staged A panel, consume four adjacent output-major weight
/// panels with Tensor Core MMA, and retain the FP32 result through the fused
/// epilogue and sole global output store.
/// </summary>
internal sealed class PtxFusedLinearGeluFp16M16Kernel : IDisposable
{
    internal const int Rows = 16;
    internal const int BlockThreads = 128;
    internal const int OutputsPerBlock = 32;
    internal const int MaxStaticSharedBytes = 24_576;
    internal const string EntryPoint = "aidotnet_fused_linear_gelu_fp16_m16";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int InputFeatures { get; }
    internal int OutputFeatures { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLinearGeluFp16M16Kernel(
        DirectPtxRuntime runtime,
        int inputFeatures,
        int outputFeatures)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedMixedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP16 Tensor Core fused-linear specialization is measured only on GA10x/SM86.");
        ValidateShape(inputFeatures, outputFeatures);

        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, inputFeatures, outputFeatures);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            inputFeatures, outputFeatures);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));
        if (Overlaps(output, input) || Overlaps(output, weights) || Overlaps(output, bias))
            throw new ArgumentException("FP16 fused-linear output may not alias input, weights, or bias.");

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &biasPointer;
        arguments[3] = &outputPointer;
        _module.Launch(
            _function,
            (uint)(OutputFeatures / OutputsPerBlock), 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int inputFeatures,
        int outputFeatures)
    {
        ValidateShape(inputFeatures, outputFeatures);
        int inputRowBytes = checked(inputFeatures * sizeof(ushort));
        int outputRowBytes = checked(outputFeatures * sizeof(float));
        int kPerPanel = GetKPerPanel(inputFeatures);
        int aPanelBytes = Rows * kPerPanel * sizeof(ushort);
        int weightPanelBytes = OutputsPerBlock * kPerPanel * sizeof(ushort);
        int bufferBytes = aPanelBytes + weightPanelBytes;
        int staticSharedBytes = 2 * bufferBytes;
        int chunks = inputFeatures / kPerPanel;
        var ptx = new StringBuilder(32_768);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b32 %a<4>;");
        ptx.AppendLine("    .reg .b32 %b<2>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %c<4>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine($"    .shared .align 16 .b8 smem[{staticSharedBytes}];");
        ptx.AppendLine();
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {OutputsPerBlock};");
        ptx.AppendLine("    shr.u32 %r5, %r1, 2;");
        ptx.AppendLine("    and.b32 %r6, %r1, 3;");
        ptx.AppendLine("    and.b32 %r7, %r0, 7;");
        ptx.AppendLine("    shr.u32 %r8, %r0, 3;");
        ptx.AppendLine("    shr.u32 %r10, %r7, 1;");
        ptx.AppendLine("    and.b32 %r19, %r7, 1;");
        ptx.AppendLine("    shl.b32 %r19, %r19, 4;");
        ptx.AppendLine("    mad.lo.u32 %r13, %r10, 512, %r19;");
        ptx.AppendLine("    mad.lo.u32 %r13, %r8, 32, %r13;");
        ptx.AppendLine("    mad.lo.u32 %r10, %r10, 1024, %r19;");
        ptx.AppendLine("    mad.lo.u32 %r10, %r8, 32, %r10;");
        ptx.AppendLine("    shl.b32 %r7, %r7, 4;");
        ptx.AppendLine($"    mad.lo.u32 %r9, %r8, {inputRowBytes}, %r7;");
        ptx.AppendLine("    add.u32 %r11, %r4, %r8;");
        ptx.AppendLine($"    mad.lo.u32 %r12, %r11, {inputRowBytes}, %r7;");
        ptx.AppendLine("    mul.lo.u32 %r14, %r5, 32;");
        ptx.AppendLine("    shl.b32 %r16, %r6, 2;");
        ptx.AppendLine("    add.u32 %r14, %r14, %r16;");
        ptx.AppendLine("    shl.b32 %r16, %r2, 8;");
        ptx.AppendLine("    mad.lo.u32 %r15, %r2, 8, %r4;");
        ptx.AppendLine("    shl.b32 %r17, %r6, 1;");
        ptx.AppendLine("    add.u32 %r15, %r15, %r17;");
        ptx.AppendLine("    mov.u64 %rd4, smem;");
        ptx.AppendLine("    cvt.u64.u32 %rd5, %r9;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        ptx.AppendLine("    cvt.u64.u32 %rd6, %r12;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        ptx.AppendLine("    cvt.u64.u32 %rd7, %r13;");
        ptx.AppendLine("    cvt.u64.u32 %rd15, %r10;");
        ptx.AppendLine("    cvt.u64.u32 %rd8, %r14;");
        ptx.AppendLine("    cvt.u64.u32 %rd9, %r16;");
        ptx.AppendLine();

        EmitAsyncPanel(
            ptx, chunk: 0, buffer: 0, inputRowBytes,
            kPerPanel, aPanelBytes, bufferBytes);
        ptx.AppendLine("    cp.async.commit_group;");
        ptx.AppendLine("    cp.async.wait_group 0;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    mov.f32 %c0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %c1, 0f00000000;");
        ptx.AppendLine("    mov.f32 %c2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %c3, 0f00000000;");
        ptx.AppendLine();

        for (int chunk = 0; chunk < chunks; chunk++)
        {
            int currentBuffer = chunk & 1;
            ptx.AppendLine($"    // ---- Tensor Core K panel {chunk} ----");
            if (chunk + 1 < chunks)
            {
                EmitAsyncPanel(
                    ptx, chunk + 1, currentBuffer ^ 1, inputRowBytes,
                    kPerPanel, aPanelBytes, bufferBytes);
                ptx.AppendLine("    cp.async.commit_group;");
            }
            for (int subchunk = 0; subchunk < kPerPanel / 16; subchunk++)
                EmitMma(ptx, currentBuffer, subchunk, aPanelBytes, bufferBytes);
            if (chunk + 1 < chunks)
            {
                ptx.AppendLine("    cp.async.wait_group 0;");
                ptx.AppendLine("    bar.sync 0;");
            }
        }

        ptx.AppendLine();
        ptx.AppendLine("    shl.b32 %r17, %r15, 2;");
        ptx.AppendLine("    cvt.u64.u32 %rd10, %r17;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd10;");
        ptx.AppendLine("    ld.global.nc.v2.f32 {%f0,%f1}, [%rd10];");
        ptx.AppendLine("    add.rn.f32 %c0, %c0, %f0;");
        ptx.AppendLine("    add.rn.f32 %c1, %c1, %f1;");
        ptx.AppendLine("    add.rn.f32 %c2, %c2, %f0;");
        ptx.AppendLine("    add.rn.f32 %c3, %c3, %f1;");
        for (int i = 0; i < 4; i++) EmitGelu(ptx, i);
        ptx.AppendLine($"    mad.lo.u32 %r18, %r5, {outputRowBytes}, %r17;");
        ptx.AppendLine("    cvt.u64.u32 %rd11, %r18;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd11;");
        ptx.AppendLine("    st.global.v2.f32 [%rd11], {%c0,%c1};");
        ptx.AppendLine($"    st.global.v2.f32 [%rd11+{8 * outputRowBytes}], {{%c2,%c3}};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitAsyncPanel(
        StringBuilder ptx,
        int chunk,
        int buffer,
        int inputRowBytes,
        int kPerPanel,
        int aPanelBytes,
        int bufferBytes)
    {
        int bufferBase = buffer * bufferBytes;
        int weightsBase = bufferBase + aPanelBytes;
        int globalOffset = chunk * kPerPanel * sizeof(ushort);
        ptx.AppendLine("    add.u64 %rd12, %rd4, %rd7;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd15;");
        ptx.AppendLine($"    cp.async.ca.shared.global [%rd12+{bufferBase}], [%rd5+{globalOffset}], 16;");
        ptx.AppendLine($"    cp.async.cg.shared.global [%rd13+{weightsBase}], [%rd6+{globalOffset}], 16;");
        ptx.AppendLine($"    cp.async.cg.shared.global [%rd13+{weightsBase + 512}], [%rd6+{globalOffset + 16 * inputRowBytes}], 16;");
        if (kPerPanel == 128)
        {
            ptx.AppendLine($"    cp.async.ca.shared.global [%rd12+{bufferBase + 2_048}], [%rd5+{globalOffset + 128}], 16;");
            ptx.AppendLine($"    cp.async.cg.shared.global [%rd13+{weightsBase + 4_096}], [%rd6+{globalOffset + 128}], 16;");
            ptx.AppendLine($"    cp.async.cg.shared.global [%rd13+{weightsBase + 4_608}], [%rd6+{globalOffset + 16 * inputRowBytes + 128}], 16;");
        }
    }

    private static void EmitMma(
        StringBuilder ptx,
        int buffer,
        int subchunk,
        int aPanelBytes,
        int bufferBytes)
    {
        int bufferBase = buffer * bufferBytes;
        int weightsBase = bufferBase + aPanelBytes;
        int aSubpanel = subchunk * 512;
        int weightSubpanel = subchunk * 1_024;
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd8;");
        ptx.AppendLine($"    ld.shared.b32 %a0, [%rd13+{bufferBase + aSubpanel}];");
        ptx.AppendLine($"    ld.shared.b32 %a2, [%rd13+{bufferBase + aSubpanel + 16}];");
        ptx.AppendLine($"    ld.shared.b32 %a1, [%rd13+{bufferBase + aSubpanel + 256}];");
        ptx.AppendLine($"    ld.shared.b32 %a3, [%rd13+{bufferBase + aSubpanel + 272}];");
        ptx.AppendLine("    add.u64 %rd14, %rd13, %rd9;");
        ptx.AppendLine($"    ld.shared.b32 %b0, [%rd14+{weightsBase + weightSubpanel}];");
        ptx.AppendLine($"    ld.shared.b32 %b1, [%rd14+{weightsBase + weightSubpanel + 16}];");
        ptx.AppendLine("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " +
            "{%c0,%c1,%c2,%c3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};");
    }

    private static void EmitGelu(StringBuilder ptx, int accumulator)
    {
        ptx.AppendLine($"    mul.rn.f32 %f2, %c{accumulator}, %c{accumulator};");
        ptx.AppendLine($"    mul.rn.f32 %f2, %f2, %c{accumulator};");
        ptx.AppendLine($"    fma.rn.f32 %f2, %f2, 0f3D372713, %c{accumulator};");
        ptx.AppendLine("    mul.rn.f32 %f2, %f2, 0f3F4C422A;");
        ptx.AppendLine("    tanh.approx.f32 %f2, %f2;");
        ptx.AppendLine("    add.rn.f32 %f2, %f2, 0f3F800000;");
        ptx.AppendLine($"    mul.rn.f32 %f2, %f2, %c{accumulator};");
        ptx.AppendLine($"    mul.rn.f32 %c{accumulator}, %f2, 0f3F000000;");
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int inputFeatures,
        int outputFeatures)
    {
        var input = new DirectPtxExtent(Rows, inputFeatures);
        var weights = new DirectPtxExtent(outputFeatures, inputFeatures);
        var bias = new DirectPtxExtent(outputFeatures);
        var output = new DirectPtxExtent(Rows, outputFeatures);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-linear-bias-gelu",
            Version: 3,
            Architecture: architecture,
            Variant: $"tensorcore-async-fp16-fp32acc-m16-k{inputFeatures}-n{outputFeatures}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float16,
                    DirectPtxPhysicalLayout.LinearWeightOutputMajor,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 48,
                MaxStaticSharedBytes: GetStaticSharedBytes(inputFeatures),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "gelu_tanh(fp32(input_fp16[16,K] @ transpose(weights_fp16[N,K])) + bias_fp32[N])",
                ["input"] = "canonical-contiguous-row-major-fp16",
                ["weights"] = "canonical-contiguous-output-major-row-major-fp16",
                ["accumulator"] = "mma-m16n8k16-fp32-register-fragment",
                ["pipeline"] = "double-buffered-cp.async-16-byte-panels",
                ["shared-A"] = "one-bank-conflict-free-M16-panel-reused-by-four-warps",
                ["epilogue"] = "bias-and-tanh-gelu-in-accumulator-registers",
                ["output"] = "canonical-contiguous-row-major-fp32-single-store",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["shape-policy"] = "exact-M16; K-multiple-16; N-multiple-32; only-measured-buckets-promoted"
            });
    }

    internal static bool IsSupportedShape(int inputFeatures, int outputFeatures) =>
        (inputFeatures, outputFeatures) is (512, 2048) or (1024, 4096);

    internal static int GetKPerPanel(int inputFeatures) => inputFeatures == 512 ? 64 : 128;

    internal static int GetStaticSharedBytes(int inputFeatures) =>
        2 * (Rows + OutputsPerBlock) * GetKPerPanel(inputFeatures) * sizeof(ushort);

    internal static bool IsPromotedShape(int inputFeatures, int outputFeatures) =>
        (inputFeatures, outputFeatures) == (1024, 4096);

    private static void ValidateShape(int inputFeatures, int outputFeatures)
    {
        if (!IsSupportedShape(inputFeatures, outputFeatures))
            throw new ArgumentOutOfRangeException(
                nameof(inputFeatures),
                "The M=16 Tensor Core experiment supports only K/N 512/2048 and 1024/4096.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer;
        nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
#endif
