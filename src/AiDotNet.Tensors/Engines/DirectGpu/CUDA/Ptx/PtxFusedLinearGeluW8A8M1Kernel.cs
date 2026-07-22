using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Shape-baked symmetric W8A8 decode projection. Four outputs share each
/// warp's contiguous packed INT8 activation loads; DP4A accumulates into S32,
/// and per-output dequantization, bias, and tanh-GELU remain in registers.
/// </summary>
internal sealed class PtxFusedLinearGeluW8A8M1Kernel : IDisposable
{
    internal const int BlockThreads = 128;
    internal const int OutputsPerWarp = 4;
    internal const string EntryPoint = "aidotnet_fused_linear_gelu_w8a8_m1";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int InputFeatures { get; }
    internal int OutputFeatures { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLinearGeluW8A8M1Kernel(
        DirectPtxRuntime runtime,
        int inputFeatures,
        int outputFeatures)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedMixedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in W8A8 fused-linear specialization is measured only on GA10x/SM86.");
        ValidateShape(inputFeatures, outputFeatures);

        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, inputFeatures, outputFeatures);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
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
        DirectPtxTensorView activationScale,
        DirectPtxTensorView weightScales,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(activationScale, Blueprint.Tensors[2], nameof(activationScale));
        Require(weightScales, Blueprint.Tensors[3], nameof(weightScales));
        Require(bias, Blueprint.Tensors[4], nameof(bias));
        Require(output, Blueprint.Tensors[5], nameof(output));
        if (Overlaps(output, input) || Overlaps(output, weights) ||
            Overlaps(output, activationScale) || Overlaps(output, weightScales) ||
            Overlaps(output, bias))
            throw new ArgumentException(
                "W8A8 fused-linear output may not alias any input tensor.");

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr activationScalePointer = activationScale.Pointer;
        IntPtr weightScalePointer = weightScales.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &activationScalePointer;
        arguments[3] = &weightScalePointer;
        arguments[4] = &biasPointer;
        arguments[5] = &outputPointer;
        _module.Launch(
            _function,
            (uint)(OutputFeatures / (OutputsPerWarp * (BlockThreads / 32))), 1, 1,
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
        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 activation_scale_ptr,");
        ptx.AppendLine("    .param .u64 weight_scales_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [activation_scale_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [weight_scales_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd5, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {BlockThreads / 32}, %r2;");
        ptx.AppendLine($"    mul.lo.u32 %r5, %r4, {OutputsPerWarp};");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine($"    mul.wide.u32 %rd8, %r5, {inputFeatures};");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd8;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, %rd6;");
        for (int output = 0; output < OutputsPerWarp; output++)
            ptx.AppendLine($"    mov.s32 %r{12 + output}, 0;");
        ptx.AppendLine("    mov.u32 %r6, 0;");
        ptx.AppendLine("W8A8_K_LOOP:");
        ptx.AppendLine("    ld.global.nc.u32 %r7, [%rd7];");
        for (int output = 0; output < OutputsPerWarp; output++)
        {
            string suffix = output == 0 ? string.Empty : $"+{output * inputFeatures}";
            ptx.AppendLine($"    ld.global.nc.u32 %r{8 + output}, [%rd8{suffix}];");
            ptx.AppendLine($"    dp4a.s32.s32 %r{12 + output}, %r7, %r{8 + output}, %r{12 + output};");
        }
        ptx.AppendLine("    add.u64 %rd7, %rd7, 128;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, 128;");
        ptx.AppendLine("    add.u32 %r6, %r6, 128;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r6, {inputFeatures};");
        ptx.AppendLine("    @%p0 bra.uni W8A8_K_LOOP;");
        for (int output = 0; output < OutputsPerWarp; output++)
        {
            foreach (int delta in new[] { 16, 8, 4, 2, 1 })
            {
                ptx.AppendLine($"    shfl.sync.bfly.b32 %r16, %r{12 + output}, {delta}, 31, 0xffffffff;");
                ptx.AppendLine($"    add.s32 %r{12 + output}, %r{12 + output}, %r16;");
            }
        }
        ptx.AppendLine("    setp.ne.u32 %p1, %r1, 0;");
        ptx.AppendLine("    @%p1 bra W8A8_RETURN;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd9;");
        ptx.AppendLine("    add.u64 %rd11, %rd4, %rd9;");
        ptx.AppendLine("    add.u64 %rd12, %rd5, %rd9;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd2];");
        for (int output = 0; output < OutputsPerWarp; output++)
        {
            string suffix = output == 0 ? string.Empty : $"+{output * sizeof(float)}";
            ptx.AppendLine($"    cvt.rn.f32.s32 %f1, %r{12 + output};");
            ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd10{suffix}];");
            ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f0;");
            ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f2;");
            ptx.AppendLine($"    ld.global.nc.f32 %f3, [%rd11{suffix}];");
            ptx.AppendLine("    add.rn.f32 %f1, %f1, %f3;");
            ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f1;");
            ptx.AppendLine("    mul.rn.f32 %f4, %f4, %f1;");
            ptx.AppendLine("    fma.rn.f32 %f4, %f4, 0f3D372713, %f1;");
            ptx.AppendLine("    mul.rn.f32 %f4, %f4, 0f3F4C422A;");
            ptx.AppendLine("    tanh.approx.f32 %f4, %f4;");
            ptx.AppendLine("    add.rn.f32 %f4, %f4, 0f3F800000;");
            ptx.AppendLine("    mul.rn.f32 %f4, %f4, %f1;");
            ptx.AppendLine("    mul.rn.f32 %f4, %f4, 0f3F000000;");
            ptx.AppendLine($"    st.global.f32 [%rd12{suffix}], %f4;");
        }
        ptx.AppendLine("W8A8_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int inputFeatures,
        int outputFeatures)
    {
        var scalar = new DirectPtxExtent(1);
        var input = new DirectPtxExtent(inputFeatures);
        var weights = new DirectPtxExtent(outputFeatures, inputFeatures);
        var output = new DirectPtxExtent(outputFeatures);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-linear-bias-gelu",
            Version: 4,
            Architecture: architecture,
            Variant: $"decode-w8a8-s32acc-dp4a-m1-k{inputFeatures}-n{outputFeatures}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Int8, DirectPtxPhysicalLayout.Vector,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Int8,
                    DirectPtxPhysicalLayout.LinearWeightOutputMajor,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("activation_scale", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.Vector,
                    scalar, scalar, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weight_scales", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.Vector,
                    output, output, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    output, output, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "gelu_tanh(s32(input_s8[1,K] @ transpose(weights_s8[N,K])) * activation_scale_fp32 * weight_scales_fp32[N] + bias_fp32[N])",
                ["quantization"] = "symmetric-s8; zero-points-are-exactly-zero",
                ["input"] = "contiguous-s8",
                ["weights"] = "output-major-row-major-s8",
                ["accumulator"] = "lane-private-s32-dp4a-register",
                ["epilogue"] = "per-output-fp32-dequant+bias+tanh-gelu",
                ["output"] = "fp32",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["shape-policy"] = "dp4a-for-m1; imma-is-a-separate-m-ge-16-specialization"
            });
    }

    internal static bool IsSupportedShape(int inputFeatures, int outputFeatures) =>
        (inputFeatures, outputFeatures) == (1024, 4096);

    internal static bool IsPromotedShape(int inputFeatures, int outputFeatures) =>
        (inputFeatures, outputFeatures) == (1024, 4096);

    private static void ValidateShape(int inputFeatures, int outputFeatures)
    {
        if (!IsSupportedShape(inputFeatures, outputFeatures))
            throw new ArgumentOutOfRangeException(
                nameof(inputFeatures), "The first W8A8 proof cell is exactly K=1024, N=4096.");
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
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
