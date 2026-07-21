#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Shape-baked FP16 decode-token linear + FP32 bias + tanh-GELU. Each lane
/// consumes one contiguous half2 from the input and output-major weight row;
/// accumulation and the activation remain FP32 registers until the sole store.
/// </summary>
internal sealed class PtxFusedLinearGeluFp16M1Kernel : IDisposable
{
    internal const int BlockThreads = 128;
    internal const string EntryPoint = "aidotnet_fused_linear_gelu_fp16_m1";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int InputFeatures { get; }
    internal int OutputFeatures { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLinearGeluFp16M1Kernel(
        DirectPtxRuntime runtime,
        int inputFeatures,
        int outputFeatures)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException(
                "The checked-in FP16 fused-linear specialization is validated only on Ampere.");
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
            (uint)(OutputFeatures / (2 * (BlockThreads / 32))), 1, 1,
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
        int weightRowBytes = checked(inputFeatures * sizeof(ushort));
        var ptx = new StringBuilder(8_192);
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
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b16 %h<4>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r6, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r3, %r6, {BlockThreads / 32}, %r2;");
        ptx.AppendLine("    shl.b32 %r4, %r3, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r4, {weightRowBytes};");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, %rd4;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("LINEAR_FP16_K_LOOP:");
        ptx.AppendLine("    ld.global.nc.b32 %r8, [%rd5];");
        ptx.AppendLine("    mov.b32 {%h0,%h1}, %r8;");
        ptx.AppendLine("    cvt.f32.f16 %f8, %h0;");
        ptx.AppendLine("    cvt.f32.f16 %f9, %h1;");
        for (int output = 0; output < 2; output++)
        {
            int weightOffset = checked(output * weightRowBytes);
            string suffix = weightOffset == 0 ? string.Empty : $"+{weightOffset}";
            ptx.AppendLine($"    ld.global.nc.b32 %r9, [%rd6{suffix}];");
            ptx.AppendLine("    mov.b32 {%h2,%h3}, %r9;");
            ptx.AppendLine("    cvt.f32.f16 %f10, %h2;");
            ptx.AppendLine("    cvt.f32.f16 %f11, %h3;");
            ptx.AppendLine($"    fma.rn.f32 %f{output}, %f8, %f10, %f{output};");
            ptx.AppendLine($"    fma.rn.f32 %f{output}, %f9, %f11, %f{output};");
        }
        ptx.AppendLine("    add.u64 %rd5, %rd5, 128;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 128;");
        ptx.AppendLine("    add.u32 %r5, %r5, 64;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {inputFeatures};");
        ptx.AppendLine("    @%p0 bra.uni LINEAR_FP16_K_LOOP;");
        for (int output = 0; output < 2; output++)
        {
            foreach (int delta in new[] { 16, 8, 4, 2, 1 })
            {
                ptx.AppendLine($"    mov.b32 %r10, %f{output};");
                ptx.AppendLine($"    shfl.sync.bfly.b32 %r11, %r10, {delta}, 31, 0xffffffff;");
                ptx.AppendLine("    mov.b32 %f10, %r11;");
                ptx.AppendLine($"    add.rn.f32 %f{output}, %f{output}, %f10;");
            }
        }
        ptx.AppendLine("    setp.ne.u32 %p1, %r1, 0;");
        ptx.AppendLine("    @%p1 bra LINEAR_FP16_RETURN;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd7;");
        for (int output = 0; output < 2; output++)
        {
            string suffix = output == 0 ? string.Empty : "+4";
            ptx.AppendLine($"    ld.global.nc.f32 %f9, [%rd8{suffix}];");
            ptx.AppendLine($"    add.rn.f32 %f{output}, %f{output}, %f9;");
            ptx.AppendLine($"    mul.rn.f32 %f11, %f{output}, %f{output};");
            ptx.AppendLine($"    mul.rn.f32 %f11, %f11, %f{output};");
            ptx.AppendLine($"    fma.rn.f32 %f11, %f11, 0f3D372713, %f{output};");
            ptx.AppendLine("    mul.rn.f32 %f11, %f11, 0f3F4C422A;");
            ptx.AppendLine("    tanh.approx.f32 %f11, %f11;");
            ptx.AppendLine("    add.rn.f32 %f11, %f11, 0f3F800000;");
            ptx.AppendLine($"    mul.rn.f32 %f11, %f11, %f{output};");
            ptx.AppendLine("    mul.rn.f32 %f11, %f11, 0f3F000000;");
            ptx.AppendLine($"    st.global.f32 [%rd9{suffix}], %f11;");
        }
        ptx.AppendLine("LINEAR_FP16_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int inputFeatures,
        int outputFeatures)
    {
        var input = new DirectPtxExtent(inputFeatures);
        var weights = new DirectPtxExtent(outputFeatures, inputFeatures);
        var output = new DirectPtxExtent(outputFeatures);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-linear-bias-gelu",
            Version: 2,
            Architecture: architecture,
            Variant: $"decode-fp16-fp32acc-m1-k{inputFeatures}-n{outputFeatures}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Vector,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float16,
                    DirectPtxPhysicalLayout.LinearWeightOutputMajor,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    output, output, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "gelu_tanh(fp32(input_fp16[1,K] @ transpose(weights_fp16[N,K])) + bias_fp32[N])",
                ["input"] = "fp16",
                ["weights"] = "output-major-row-major-fp16",
                ["accumulator"] = "lane-private-fp32-register",
                ["output"] = "fp32",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["shape-policy"] = "half2-simd-for-m1; tensor-core-mma-is-a-separate-m-ge-16-specialization"
            });
    }

    internal static bool IsSupportedShape(int inputFeatures, int outputFeatures) =>
        inputFeatures is 256 or 512 or 1024 or 2048 or 4096 &&
        outputFeatures is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int inputFeatures, int outputFeatures) =>
        (inputFeatures, outputFeatures) is (512, 2048) or (1024, 4096);

    private static void ValidateShape(int inputFeatures, int outputFeatures)
    {
        if (!IsSupportedShape(inputFeatures, outputFeatures))
            throw new ArgumentOutOfRangeException(
                nameof(inputFeatures),
                "Supported K and N buckets are 256, 512, 1024, 2048, and 4096.");
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
