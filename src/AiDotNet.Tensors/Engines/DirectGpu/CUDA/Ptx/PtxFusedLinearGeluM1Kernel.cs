using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Shape-baked FP32 decode-token linear + bias + tanh-GELU. One warp owns two
/// output rows; its lanes stream adjacent K values from canonical output-major
/// weights, and the accumulators and activation never leave registers.
/// </summary>
internal sealed class PtxFusedLinearGeluM1Kernel : IDisposable
{
    internal const int BlockThreads = 128;
    internal const string EntryPoint = "aidotnet_fused_linear_gelu_m1";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int InputFeatures { get; }
    internal int OutputFeatures { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedLinearGeluM1Kernel(
        DirectPtxRuntime runtime,
        int inputFeatures,
        int outputFeatures)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in fused-linear GELU specialization is measured only on GA10x/SM86.");
        ValidateShape(inputFeatures, outputFeatures);

        InputFeatures = inputFeatures;
        OutputFeatures = outputFeatures;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, inputFeatures, outputFeatures);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            inputFeatures, outputFeatures);
        DirectPtxLoadedKernel loaded = DirectPtxResourceInitialization.LoadKernel(
            runtime, Ptx, EntryPoint, BlockThreads, Blueprint);
        _module = loaded.Module;
        _function = loaded.Function;
        Audit = loaded.Audit;
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        PtxFusedLinearGeluShared.Require(input, Blueprint.Tensors[0], nameof(input));
        PtxFusedLinearGeluShared.Require(weights, Blueprint.Tensors[1], nameof(weights));
        PtxFusedLinearGeluShared.Require(bias, Blueprint.Tensors[2], nameof(bias));
        PtxFusedLinearGeluShared.Require(output, Blueprint.Tensors[3], nameof(output));
        if (PtxFusedLinearGeluShared.Overlaps(output, input) ||
            PtxFusedLinearGeluShared.Overlaps(output, weights) ||
            PtxFusedLinearGeluShared.Overlaps(output, bias))
            throw new ArgumentException("Fused-linear output may not alias input, weights, or bias.");

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
        int outputsPerWarp = OutputsPerWarp;
        var ptx = new StringBuilder(8_192);
        int weightRowBytes = checked(inputFeatures * sizeof(float));
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
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        EmitOutputMajorBody(ptx, inputFeatures, weightRowBytes, outputsPerWarp);
        ptx.AppendLine("LINEAR_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitOutputMajorBody(
        StringBuilder ptx,
        int inputFeatures,
        int weightRowBytes,
        int outputsPerWarp)
    {
        bool vectorizedK = inputFeatures > 256;
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    and.b32 %r1, %r0, 31;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r6, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r3, %r6, {BlockThreads / 32}, %r2;");
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {outputsPerWarp};");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r1, {(vectorizedK ? 16 : 4)};");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r4, {weightRowBytes};");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, %rd4;");
        for (int accumulator = 0; accumulator < outputsPerWarp; accumulator++)
            ptx.AppendLine($"    mov.f32 %f{accumulator}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("LINEAR_K_LOOP:");
        if (vectorizedK)
        {
            ptx.AppendLine("    ld.global.nc.v4.f32 {%f8,%f9,%f10,%f11}, [%rd5];");
            for (int output = 0; output < outputsPerWarp; output++)
            {
                int weightOffset = checked(output * weightRowBytes);
                string suffix = weightOffset == 0 ? string.Empty : $"+{weightOffset}";
                ptx.AppendLine(
                    $"    ld.global.nc.v4.f32 {{%f12,%f13,%f14,%f15}}, [%rd6{suffix}];");
                for (int component = 0; component < 4; component++)
                    ptx.AppendLine(
                        $"    fma.rn.f32 %f{output}, %f{8 + component}, " +
                        $"%f{12 + component}, %f{output};");
            }
        }
        else
        {
            ptx.AppendLine("    ld.global.nc.f32 %f8, [%rd5];");
            for (int output = 0; output < outputsPerWarp; output++)
            {
                int weightOffset = checked(output * weightRowBytes);
                string suffix = weightOffset == 0 ? string.Empty : $"+{weightOffset}";
                ptx.AppendLine($"    ld.global.nc.f32 %f9, [%rd6{suffix}];");
                ptx.AppendLine($"    fma.rn.f32 %f{output}, %f8, %f9, %f{output};");
            }
        }
        int laneStepElements = vectorizedK ? 128 : 32;
        int laneStepBytes = laneStepElements * sizeof(float);
        ptx.AppendLine($"    add.u64 %rd5, %rd5, {laneStepBytes};");
        ptx.AppendLine($"    add.u64 %rd6, %rd6, {laneStepBytes};");
        ptx.AppendLine($"    add.u32 %r5, %r5, {laneStepElements};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {inputFeatures};");
        ptx.AppendLine("    @%p0 bra.uni LINEAR_K_LOOP;");
        for (int output = 0; output < outputsPerWarp; output++)
            PtxFusedLinearGeluShared.EmitFp32WarpButterflyReduction(
                ptx, $"%f{output}", "%r10", "%r11", "%f10");
        ptx.AppendLine("    setp.ne.u32 %p1, %r1, 0;");
        ptx.AppendLine("    @%p1 bra LINEAR_RETURN;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd7;");
        for (int output = 0; output < outputsPerWarp; output++)
        {
            int outputOffset = output * sizeof(float);
            string suffix = outputOffset == 0 ? string.Empty : $"+{outputOffset}";
            ptx.AppendLine($"    ld.global.nc.f32 %f9, [%rd8{suffix}];");
            ptx.AppendLine($"    add.rn.f32 %f{output}, %f{output}, %f9;");
            PtxFusedLinearGeluShared.EmitTanhGeluEpilogue(
                ptx, $"%f{output}", "%f11", $"%rd9{suffix}");
        }
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
            Version: 1,
            Architecture: architecture,
            Variant: $"decode-fp32-m1-k{inputFeatures}-n{outputFeatures}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32,
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
                ["formula"] = "gelu_tanh(input[1,K] @ transpose(weights[N,K]) + bias[N])",
                ["weights"] = "output-major-row-major-fp32",
                ["activation"] = "shape-baked-tanh-gelu",
                ["accumulator"] = "lane-private-fp32-register",
                ["input-stream"] = "lane-contiguous-read-only-cache-loads",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int inputFeatures, int outputFeatures) =>
        inputFeatures is 256 or 512 or 1024 or 2048 or 4096 &&
        outputFeatures is 256 or 512 or 1024 or 2048 or 4096;

    // The final clean three-run issue matrix did not reproduce a release-gate
    // win for any supported shape. Keep every specialization experiment-only.
    internal static bool IsPromotedShape(int inputFeatures, int outputFeatures) => false;

    private const int OutputsPerWarp = 2;

    private static void ValidateShape(int inputFeatures, int outputFeatures)
    {
        if (!IsSupportedShape(inputFeatures, outputFeatures))
            throw new ArgumentOutOfRangeException(
                nameof(inputFeatures),
                "Supported K and N buckets are 256, 512, 1024, 2048, and 4096.");
    }

}
