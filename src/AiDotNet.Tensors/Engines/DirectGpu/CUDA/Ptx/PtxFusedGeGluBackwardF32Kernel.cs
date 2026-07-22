using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 tanh-GeGLU backward specialization over split rows.
/// Four derivatives remain lane-private from three vector loads through the
/// two final vector stores; shape and stride data never enter the launch ABI.
/// </summary>
internal sealed class PtxFusedGeGluBackwardF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_geglu_backward_f32x4";
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int OuterSize { get; }
    internal int HalfDimension { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedGeGluBackwardF32Kernel(
        DirectPtxRuntime runtime,
        int outerSize,
        int halfDimension)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedGatedGlu(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP32 GeGLU-backward specialization is measured only on GA10x/SM86.");
        Validate(outerSize, halfDimension);
        OuterSize = outerSize;
        HalfDimension = halfDimension;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, outerSize, halfDimension);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            outerSize, halfDimension);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView gradInput)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));
        if (Overlaps(gradInput, gradOutput) || Overlaps(gradInput, input))
            throw new ArgumentException("The GeGLU input gradient may not alias either read tensor.");

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr gradInputPointer = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &inputPointer;
        arguments[2] = &gradInputPointer;
        uint vectorColumns = (uint)(HalfDimension / 4);
        _module.Launch(
            _function,
            (vectorColumns + BlockThreads - 1) / BlockThreads,
            (uint)OuterSize,
            1,
            BlockThreads,
            1,
            1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int outerSize,
        int halfDimension)
    {
        Validate(outerSize, halfDimension);
        int vectorColumns = halfDimension / 4;
        int splitRowBytes = checked(2 * halfDimension * sizeof(float));
        int halfRowBytes = checked(halfDimension * sizeof(float));
        var ptx = new StringBuilder(12_288);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 grad_input_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<28>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_input_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {vectorColumns};");
        ptx.AppendLine("    @%p0 bra GEGLU_BACKWARD_RETURN;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.y;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r3, {halfRowBytes};");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r3, {splitRowBytes};");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd5;");
        ptx.AppendLine($"    add.u64 %rd8, %rd7, {halfRowBytes};");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd4;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd5;");
        ptx.AppendLine($"    add.u64 %rd10, %rd9, {halfRowBytes};");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f0,%f1,%f2,%f3}, [%rd6];");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f4,%f5,%f6,%f7}, [%rd7];");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f8,%f9,%f10,%f11}, [%rd8];");
        for (int lane = 0; lane < 4; lane++)
        {
            int grad = lane;
            int value = 4 + lane;
            int gate = 8 + lane;
            int gradValue = 12 + lane;
            int gradGate = 16 + lane;
            ptx.AppendLine($"    mul.rn.f32 %f20, %f{gate}, %f{gate};");
            ptx.AppendLine($"    mul.rn.f32 %f21, %f20, %f{gate};");
            ptx.AppendLine($"    fma.rn.f32 %f21, %f21, 0f3D372713, %f{gate};");
            ptx.AppendLine("    mul.rn.f32 %f21, %f21, 0f3F4C422A;");
            ptx.AppendLine("    tanh.approx.f32 %f22, %f21;");
            ptx.AppendLine("    add.rn.f32 %f23, %f22, 0f3F800000;");
            ptx.AppendLine($"    mul.rn.f32 %f24, %f23, %f{gate};");
            ptx.AppendLine("    mul.rn.f32 %f24, %f24, 0f3F000000;");
            ptx.AppendLine($"    mul.rn.f32 %f{gradValue}, %f{grad}, %f24;");
            ptx.AppendLine("    mul.rn.f32 %f24, %f22, %f22;");
            ptx.AppendLine("    sub.rn.f32 %f24, 0f3F800000, %f24;");
            ptx.AppendLine("    fma.rn.f32 %f25, %f20, 0f3E095D4F, 0f3F800000;");
            ptx.AppendLine("    mul.rn.f32 %f24, %f24, %f25;");
            ptx.AppendLine("    mul.rn.f32 %f24, %f24, 0f3F4C422A;");
            ptx.AppendLine($"    mul.rn.f32 %f24, %f24, %f{gate};");
            ptx.AppendLine("    mul.rn.f32 %f24, %f24, 0f3F000000;");
            ptx.AppendLine("    fma.rn.f32 %f24, %f23, 0f3F000000, %f24;");
            ptx.AppendLine($"    mul.rn.f32 %f25, %f{grad}, %f{value};");
            ptx.AppendLine($"    mul.rn.f32 %f{gradGate}, %f25, %f24;");
        }
        ptx.AppendLine("    st.global.v4.f32 [%rd9], {%f12,%f13,%f14,%f15};");
        ptx.AppendLine("    st.global.v4.f32 [%rd10], {%f16,%f17,%f18,%f19};");
        ptx.AppendLine("GEGLU_BACKWARD_RETURN:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int outerSize,
        int halfDimension)
    {
        var halfExtent = new DirectPtxExtent(outerSize, halfDimension);
        var splitExtent = new DirectPtxExtent(outerSize, checked(2 * halfDimension));
        return new DirectPtxKernelBlueprint(
            Operation: "geglu-backward-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"split-last-x4-o{outerSize}-d{halfDimension}",
            Tensors:
            [
                new("grad-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    halfExtent, halfExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    splitExtent, splitExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad-input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    splitExtent, splitExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 36,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dvalue=grad*gelu_tanh(gate); dgate=grad*value*gelu_tanh_prime(gate)",
                ["split"] = "input and grad-input rows split [value|gate]",
                ["mode"] = "training-backward-tanh-approximation",
                ["input"] = "fp32",
                ["accumulator"] = "lane-private-fp32x4-register",
                ["output"] = "fp32",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int outerSize, int halfDimension) =>
        PtxFusedSwiGluF32Kernel.IsSupportedShape(outerSize, halfDimension);

    internal static bool IsPromotedShape(int outerSize, int halfDimension) => false;

    private static void Validate(int outerSize, int halfDimension)
    {
        if (!IsSupportedShape(outerSize, halfDimension))
            throw new ArgumentOutOfRangeException(nameof(outerSize),
                "The first GeGLU-backward family supports exact (outer,D) buckets " +
                "(1,4096), (32,4096), (256,4096), and (256,11008).");
        if ((halfDimension & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(halfDimension));
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
