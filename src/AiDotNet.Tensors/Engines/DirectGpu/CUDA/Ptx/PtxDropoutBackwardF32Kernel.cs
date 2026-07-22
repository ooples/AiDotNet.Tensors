#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape saved-mask dropout backward. The forward mask already contains
/// inverted-dropout scaling, so backward is one coalesced float4 multiply with
/// no shape, stride, tail, or dropout-rate work in the admitted kernel.
/// </summary>
internal sealed class PtxDropoutBackwardF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_dropout_backward_f32";
    internal const int ThreadsPerBlock = 128;
    internal const int ValuesPerThread = 4;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int ElementCount { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDropoutBackwardF32Kernel(DirectPtxRuntime runtime, int elementCount)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental dropout-backward specialization targets SM86 only.");
        ValidateElementCount(elementCount);
        ElementCount = elementCount;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, elementCount);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, elementCount);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, ThreadsPerBlock);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, ThreadsPerBlock, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            ThreadsPerBlock, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView mask,
        DirectPtxTensorView gradInput)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(mask, Blueprint.Tensors[1], nameof(mask));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(gradOutput, gradInput) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(mask, gradInput))
            throw new ArgumentException("Dropout backward output may not alias either input.");

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr maskPointer = mask.Pointer;
        IntPtr gradInputPointer = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &maskPointer;
        arguments[2] = &gradInputPointer;
        uint groups = checked((uint)(ElementCount / ValuesPerThread));
        _module.Launch(
            _function,
            groups / ThreadsPerBlock, 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedElementCount(int elementCount) =>
        PtxFusedPhiloxDropoutF32Kernel.IsSupportedElementCount(elementCount);

    internal static string EmitPtx(int ccMajor, int ccMinor, int elementCount)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental dropout-backward emitter targets SM86 only.");
        ValidateElementCount(elementCount);
        int groups = elementCount / ValuesPerThread;
        int blocks = groups / ThreadsPerBlock;
        var ptx = new StringBuilder(2_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 grad_input_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<4>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_input_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {ThreadsPerBlock}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd4];");
        ptx.AppendLine("    ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd5];");
        ptx.AppendLine("    mul.rn.f32 %f8, %f0, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f9, %f1, %f5;");
        ptx.AppendLine("    mul.rn.f32 %f10, %f2, %f6;");
        ptx.AppendLine("    mul.rn.f32 %f11, %f3, %f7;");
        ptx.AppendLine("    st.global.v4.f32 [%rd6], {%f8,%f9,%f10,%f11};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_elements={elementCount}; exact_blocks={blocks}; no_tail_branch=1");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int elementCount)
    {
        var extent = new DirectPtxExtent(elementCount);
        return new DirectPtxKernelBlueprint(
            Operation: "dropout-backward-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-n{elementCount}-float4-t{ThreadsPerBlock}",
            Tensors:
            [
                new("grad-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("saved-mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad-input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 12),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["mask"] = "saved forward mask already contains inverse-keep scale",
                ["output"] = "gradOutput*savedMask",
                ["global-reads"] = "one grad-output float4 plus one saved-mask float4 per thread",
                ["global-writes"] = "one grad-input float4 per thread",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateElementCount(int elementCount)
    {
        if (!IsSupportedElementCount(elementCount))
            throw new ArgumentOutOfRangeException(
                nameof(elementCount),
                "Supported exact element buckets are 4096, 65536, and 1048576.");
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
}
#endif
