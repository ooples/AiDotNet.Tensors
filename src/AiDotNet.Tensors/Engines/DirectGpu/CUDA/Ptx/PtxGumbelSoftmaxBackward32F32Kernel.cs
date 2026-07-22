using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact one-warp Gumbel-softmax backward over 32 contiguous classes.
/// The softmax Jacobian dot product and inverse-temperature scaling remain in
/// registers; no reduction temporary is written to global memory.
/// </summary>
internal sealed class PtxGumbelSoftmaxBackward32F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_gumbel_softmax_backward32_f32";
    internal const int Classes = 32;
    internal const int ThreadsPerBlock = Classes;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxGumbelSoftmaxBackward32F32Kernel(DirectPtxRuntime runtime, int rows)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The experimental Gumbel-softmax backward specialization targets SM86 only.");
        ValidateRows(rows);
        Rows = rows;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, rows);
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
        DirectPtxTensorView softOutput,
        DirectPtxTensorView gradInput,
        float temperature)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(softOutput, Blueprint.Tensors[1], nameof(softOutput));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));
        if (!PtxCompat.IsFinite(temperature) || temperature <= 0f)
            throw new ArgumentOutOfRangeException(nameof(temperature));
        if (PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(gradOutput, gradInput) ||
            PtxFusedPhiloxDropoutF32Kernel.RangesOverlap(softOutput, gradInput))
            throw new ArgumentException("Gumbel-softmax backward output may not alias either input.");
        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr softOutputPointer = softOutput.Pointer;
        IntPtr gradInputPointer = gradInput.Pointer;
        float inverseTemperature = 1f / temperature;
        void** arguments = stackalloc void*[4];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &softOutputPointer;
        arguments[2] = &gradInputPointer;
        arguments[3] = &inverseTemperature;
        _module.Launch(
            _function,
            checked((uint)Rows), 1, 1,
            ThreadsPerBlock, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static bool IsSupportedShape(int rows, int classes) =>
        classes == Classes && rows is 128 or 2_048 or 32_768;

    internal static string EmitPtx(int ccMajor, int ccMinor, int rows)
    {
        if (!DirectPtxArchitecture.HasExperimentalRngDropout(ccMajor, ccMinor))
            throw new PlatformNotSupportedException(
                "The experimental Gumbel-softmax backward emitter targets SM86 only.");
        ValidateRows(rows);
        var ptx = new StringBuilder(3_500);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 soft_output_ptr,");
        ptx.AppendLine("    .param .u64 grad_input_ptr,");
        ptx.AppendLine("    .param .f32 inverse_temperature");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {ThreadsPerBlock}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [soft_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_input_ptr];");
        ptx.AppendLine("    ld.param.f32 %f0, [inverse_temperature];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    mad.lo.u32 %r2, %r1, 32, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd4];");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd5];");
        ptx.AppendLine("    mul.rn.f32 %f3, %f1, %f2;");
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine("    mov.b32 %r3, %f3;");
            ptx.AppendLine($"    shfl.sync.down.b32 %r4, %r3, {offset}, 0x1f, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f4, %r4;");
            ptx.AppendLine("    add.rn.f32 %f3, %f3, %f4;");
        }
        ptx.AppendLine("    mov.b32 %r3, %f3;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r4, %r3, 0, 0x1f, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f5, %r4;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f1, %f5;");
        ptx.AppendLine("    mul.rn.f32 %f6, %f6, %f2;");
        ptx.AppendLine("    mul.rn.f32 %f6, %f6, %f0;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f6;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        ptx.AppendLine($"// exact_shape=[{rows},32]; one_warp_per_row=1; no_tail_branch=1");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows)
    {
        var extent = new DirectPtxExtent(rows, Classes);
        return new DirectPtxKernelBlueprint(
            Operation: "gumbel-softmax-backward32-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"exact-r{rows}-c32-warp",
            Tensors:
            [
                new("grad-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("soft-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad-input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 4, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 16),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["output"] = "softOutput*(gradOutput-sum(gradOutput*softOutput))/temperature",
                ["global-reads"] = "one grad-output plus one soft-output scalar per lane",
                ["global-writes"] = "one grad-input scalar per lane",
                ["register-resident"] = "Jacobian dot reduction and scaled gradient",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-and-tail-parameters"] = "none"
            });
    }

    private static void ValidateRows(int rows)
    {
        if (!IsSupportedShape(rows, Classes))
            throw new ArgumentOutOfRangeException(
                nameof(rows), "Supported exact row counts are 128, 2048, and 32768 for 32 classes.");
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
