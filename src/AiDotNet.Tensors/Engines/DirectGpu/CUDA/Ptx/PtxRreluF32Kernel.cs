#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxRreluKind
{
    Forward,
    Backward
}

/// <summary>Exact-shape direct PTX ports of the saved-noise RReLU kernels.</summary>
internal sealed class PtxRreluF32Kernel : IDisposable
{
    internal const int ThreadsPerBlock = 128;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int ElementCount { get; }
    internal DirectPtxRreluKind Kind { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxRreluF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxRreluKind kind,
        int elementCount)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental saved-noise RReLU specialization targets SM86 only.");
        ValidateElementCount(elementCount);
        ElementCount = elementCount;
        Kind = kind;
        EntryPoint = kind == DirectPtxRreluKind.Forward
            ? "aidotnet_rrelu_f32"
            : "aidotnet_rrelu_backward_f32";
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, kind, elementCount);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, kind, elementCount);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, ThreadsPerBlock);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, ThreadsPerBlock, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            ThreadsPerBlock, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void LaunchForward(
        DirectPtxTensorView input,
        DirectPtxTensorView noise,
        DirectPtxTensorView output)
    {
        if (Kind != DirectPtxRreluKind.Forward)
            throw new InvalidOperationException("This is not an RReLU forward module.");
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(noise, Blueprint.Tensors[1], nameof(noise));
        Require(output, Blueprint.Tensors[2], nameof(output));
        RequireNoAlias(input, noise, output);
        IntPtr inputPointer = input.Pointer;
        IntPtr noisePointer = noise.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &noisePointer;
        arguments[2] = &outputPointer;
        Launch(arguments);
    }

    internal unsafe void LaunchBackward(
        DirectPtxTensorView gradientOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView noise,
        DirectPtxTensorView gradientInput)
    {
        if (Kind != DirectPtxRreluKind.Backward)
            throw new InvalidOperationException("This is not an RReLU backward module.");
        Require(gradientOutput, Blueprint.Tensors[0], nameof(gradientOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(noise, Blueprint.Tensors[2], nameof(noise));
        Require(gradientInput, Blueprint.Tensors[3], nameof(gradientInput));
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(gradientInput, gradientOutput) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(gradientInput, input) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(gradientInput, noise))
            throw new ArgumentException("RReLU backward output may not alias an input.");
        IntPtr gradientOutputPointer = gradientOutput.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr noisePointer = noise.Pointer;
        IntPtr gradientInputPointer = gradientInput.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &gradientOutputPointer;
        arguments[1] = &inputPointer;
        arguments[2] = &noisePointer;
        arguments[3] = &gradientInputPointer;
        Launch(arguments);
    }

    private unsafe void Launch(void** arguments)
    {
        uint groups = checked((uint)(ElementCount / 4));
        _module.Launch(
            _function, groups / ThreadsPerBlock, 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedElementCount(int elementCount) =>
        PtxFusedPhiloxRreluF32Kernel.IsSupportedElementCount(elementCount);

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxRreluKind kind,
        int elementCount)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental saved-noise RReLU emitter targets SM86 only.");
        ValidateElementCount(elementCount);
        string entryPoint = kind == DirectPtxRreluKind.Forward
            ? "aidotnet_rrelu_f32"
            : "aidotnet_rrelu_backward_f32";
        var ptx = new StringBuilder(5_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entryPoint}(");
        if (kind == DirectPtxRreluKind.Forward)
        {
            ptx.AppendLine("    .param .u64 input_ptr,");
            ptx.AppendLine("    .param .u64 noise_ptr,");
            ptx.AppendLine("    .param .u64 output_ptr");
        }
        else
        {
            ptx.AppendLine("    .param .u64 grad_output_ptr,");
            ptx.AppendLine("    .param .u64 input_ptr,");
            ptx.AppendLine("    .param .u64 noise_ptr,");
            ptx.AppendLine("    .param .u64 grad_input_ptr");
        }
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<4>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        int parameter = 0;
        if (kind == DirectPtxRreluKind.Backward)
            ptx.AppendLine($"    ld.param.u64 %rd{parameter++}, [grad_output_ptr];");
        ptx.AppendLine($"    ld.param.u64 %rd{parameter++}, [input_ptr];");
        ptx.AppendLine($"    ld.param.u64 %rd{parameter++}, [noise_ptr];");
        ptx.AppendLine($"    ld.param.u64 %rd{parameter}, [{(kind == DirectPtxRreluKind.Forward ? "output_ptr" : "grad_input_ptr")}];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {ThreadsPerBlock}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 16;");
        int inputPointer = kind == DirectPtxRreluKind.Forward ? 0 : 1;
        int noisePointer = inputPointer + 1;
        int outputPointer = noisePointer + 1;
        ptx.AppendLine($"    add.u64 %rd8, %rd{inputPointer}, %rd4;");
        ptx.AppendLine($"    add.u64 %rd9, %rd{noisePointer}, %rd4;");
        ptx.AppendLine($"    add.u64 %rd10, %rd{outputPointer}, %rd4;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd8];");
        ptx.AppendLine("    ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd9];");
        if (kind == DirectPtxRreluKind.Backward)
        {
            ptx.AppendLine("    add.u64 %rd11, %rd0, %rd4;");
            ptx.AppendLine("    ld.global.v4.f32 {%f8,%f9,%f10,%f11}, [%rd11];");
        }
        for (int i = 0; i < 4; i++)
        {
            ptx.AppendLine($"    setp.ge.f32 %p{i}, %f{i}, 0f00000000;");
            ptx.AppendLine($"    selp.f32 %f{12 + i}, 0f3F800000, %f{4 + i}, %p{i};");
            int value = kind == DirectPtxRreluKind.Forward ? i : 8 + i;
            ptx.AppendLine($"    mul.rn.f32 %f{16 + i}, %f{value}, %f{12 + i};");
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd10], {%f16,%f17,%f18,%f19};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_elements={elementCount}; no_tail_branch=1");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxRreluKind kind,
        int elementCount)
    {
        var extent = new DirectPtxExtent(elementCount);
        DirectPtxTensorContract Contract(string name, DirectPtxTensorAccess access) =>
            new(name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                extent, extent, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract[] tensors = kind == DirectPtxRreluKind.Forward
            ? [Contract("input", DirectPtxTensorAccess.Read),
               Contract("saved-noise", DirectPtxTensorAccess.Read),
               Contract("output", DirectPtxTensorAccess.Write)]
            : [Contract("grad-output", DirectPtxTensorAccess.Read),
               Contract("input", DirectPtxTensorAccess.Read),
               Contract("saved-noise", DirectPtxTensorAccess.Read),
               Contract("grad-input", DirectPtxTensorAccess.Write)];
        return new DirectPtxKernelBlueprint(
            Operation: kind == DirectPtxRreluKind.Forward
                ? "rrelu-saved-noise-forward-f32"
                : "rrelu-saved-noise-backward-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-n{elementCount}-float4-t{ThreadsPerBlock}",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["rrelu"] = kind == DirectPtxRreluKind.Forward
                    ? "x>=0 ? x : savedNoise*x"
                    : "gradInput=gradOutput*(x>=0 ? 1 : savedNoise)",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateElementCount(int elementCount)
    {
        if (!IsSupportedElementCount(elementCount))
            throw new ArgumentOutOfRangeException(
                nameof(elementCount), "Supported exact element counts are 4096, 65536, and 1048576.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    private static void RequireNoAlias(
        DirectPtxTensorView input,
        DirectPtxTensorView noise,
        DirectPtxTensorView output)
    {
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(input, noise) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(input, output) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(noise, output))
            throw new ArgumentException("RReLU forward tensors may not alias.");
    }
}
#endif
