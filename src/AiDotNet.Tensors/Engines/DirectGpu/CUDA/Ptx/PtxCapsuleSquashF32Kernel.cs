#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxCapsuleSquashOperation
{
    Forward,
    Backward
}

/// <summary>
/// Exact capsule squash forward/backward specializations. Each thread retains
/// one complete 16-element capsule in registers across FP64 reductions and
/// commits only the final vector to global memory.
/// </summary>
internal sealed class PtxCapsuleSquashF32Kernel : IDisposable
{
    internal const int Capsules = 320;
    internal const int Dimension = 16;
    internal const int Elements = Capsules * Dimension;
    internal const int BlockThreads = 256;
    internal const float DefaultEpsilon = 1e-8f;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxCapsuleSquashOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCapsuleSquashF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxCapsuleSquashOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Capsule squash has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Operation = operation;
        EntryPoint = GetEntryPoint(operation);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, operation);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int capsules, int dimension, float epsilon) =>
        capsules == Capsules && dimension == Dimension &&
        BitConverter.SingleToInt32Bits(epsilon) == BitConverter.SingleToInt32Bits(DefaultEpsilon);

    internal unsafe void LaunchForward(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        if (Operation != DirectPtxCapsuleSquashOperation.Forward)
            throw new InvalidOperationException("This module does not expose capsule squash forward.");
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));
        if (Overlaps(output, input))
            throw new ArgumentException("Capsule squash output must be disjoint from input.", nameof(output));
        IntPtr p0 = input.Pointer;
        IntPtr p1 = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &p0;
        arguments[1] = &p1;
        _module.Launch(_function, 2, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal unsafe void LaunchBackward(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView gradInput)
    {
        if (Operation != DirectPtxCapsuleSquashOperation.Backward)
            throw new InvalidOperationException("This module does not expose capsule squash backward.");
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));
        if (Overlaps(gradInput, gradOutput) || Overlaps(gradInput, input))
            throw new ArgumentException("Capsule squash gradient output must be disjoint from inputs.", nameof(gradInput));
        IntPtr p0 = gradOutput.Pointer;
        IntPtr p1 = input.Pointer;
        IntPtr p2 = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        _module.Launch(_function, 2, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxCapsuleSquashOperation operation)
    {
        var vectors = new DirectPtxExtent(Capsules, Dimension);
        IReadOnlyList<DirectPtxTensorContract> tensors =
            operation == DirectPtxCapsuleSquashOperation.Forward
                ?
                [
                    Exact("input", DirectPtxPhysicalLayout.CapsuleVectors, vectors, DirectPtxTensorAccess.Read),
                    Exact("output", DirectPtxPhysicalLayout.CapsuleVectors, vectors, DirectPtxTensorAccess.Write)
                ]
                :
                [
                    Exact("grad-output", DirectPtxPhysicalLayout.CapsuleGradients, vectors, DirectPtxTensorAccess.Read),
                    Exact("input", DirectPtxPhysicalLayout.CapsuleVectors, vectors, DirectPtxTensorAccess.Read),
                    Exact("grad-input", DirectPtxPhysicalLayout.CapsuleGradients, vectors, DirectPtxTensorAccess.Write)
                ];
        return new DirectPtxKernelBlueprint(
            Operation: operation == DirectPtxCapsuleSquashOperation.Forward
                ? "capsule-squash-f32" : "capsule-squash-backward-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"c320-d16-e{BitConverter.SingleToInt32Bits(DefaultEpsilon):x8}",
            Tensors: tensors,
            ResourceBudget: operation == DirectPtxCapsuleSquashOperation.Forward
                ? new DirectPtxResourceBudget(64, 0, 0, 3)
                : new DirectPtxResourceBudget(96, 0, 0, 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["reduction-precision"] = "fp64",
                ["reduction-order"] = "fully-unrolled-ascending-dimension",
                ["capsule-residency"] = operation == DirectPtxCapsuleSquashOperation.Forward
                    ? "16-input-fp32-registers" : "16-input-plus-16-gradient-fp32-registers",
                ["epsilon-bits"] = BitConverter.SingleToInt32Bits(DefaultEpsilon).ToString("x8"),
                ["output-write"] = "single-overwrite-per-element",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxCapsuleSquashOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        if (operation == DirectPtxCapsuleSquashOperation.Forward)
            EmitForward(ptx);
        else
            EmitBackward(ptx);
        return ptx.ToString();
    }

    private static void EmitForward(StringBuilder ptx)
    {
        EmitHeader(ptx, GetEntryPoint(DirectPtxCapsuleSquashOperation.Forward),
            "input_ptr", "output_ptr");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<4>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    .reg .f64 %fd<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        EmitCapsuleIndexAndBase(ptx, "CAPSULE_SQUASH_FORWARD_DONE");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    mov.f64 %fd0, 0d0000000000000000;");
        for (int d = 0; d < Dimension; d++)
        {
            ptx.AppendLine($"    ld.global.f32 %f{d}, [%rd4+{d * sizeof(float)}];");
            ptx.AppendLine($"    cvt.f64.f32 %fd1, %f{d};");
            ptx.AppendLine("    fma.rn.f64 %fd0, %fd1, %fd1, %fd0;");
        }
        ptx.AppendLine("    sqrt.rn.f64 %fd2, %fd0;");
        ptx.AppendLine($"    mov.f64 %fd3, 0d{BitConverter.DoubleToInt64Bits(DefaultEpsilon):x16};");
        ptx.AppendLine("    add.rn.f64 %fd4, %fd0, 0d3ff0000000000000;");
        ptx.AppendLine("    add.rn.f64 %fd5, %fd2, %fd3;");
        ptx.AppendLine("    mul.rn.f64 %fd6, %fd4, %fd5;");
        ptx.AppendLine("    setp.ne.f64 %p1, %fd6, 0d0000000000000000;");
        ptx.AppendLine("    div.rn.f64 %fd7, %fd0, %fd6;");
        ptx.AppendLine("    selp.f64 %fd7, %fd7, 0d0000000000000000, %p1;");
        for (int d = 0; d < Dimension; d++)
        {
            ptx.AppendLine($"    cvt.f64.f32 %fd8, %f{d};");
            ptx.AppendLine("    mul.rn.f64 %fd8, %fd7, %fd8;");
            ptx.AppendLine("    cvt.rn.f32.f64 %f18, %fd8;");
            ptx.AppendLine($"    st.global.f32 [%rd5+{d * sizeof(float)}], %f18;");
        }
        ptx.AppendLine("CAPSULE_SQUASH_FORWARD_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitBackward(StringBuilder ptx)
    {
        EmitHeader(ptx, GetEntryPoint(DirectPtxCapsuleSquashOperation.Backward),
            "grad_output_ptr", "input_ptr", "grad_input_ptr");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<4>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<34>;");
        ptx.AppendLine("    .reg .f64 %fd<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_input_ptr];");
        EmitCapsuleIndexAndBase(ptx, "CAPSULE_SQUASH_BACKWARD_DONE");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        ptx.AppendLine("    mov.f64 %fd0, 0d0000000000000000;");
        ptx.AppendLine("    mov.f64 %fd1, 0d0000000000000000;");
        for (int d = 0; d < Dimension; d++)
        {
            ptx.AppendLine($"    ld.global.f32 %f{d}, [%rd5+{d * sizeof(float)}];");
            ptx.AppendLine($"    ld.global.f32 %f{Dimension + d}, [%rd4+{d * sizeof(float)}];");
            ptx.AppendLine($"    cvt.f64.f32 %fd2, %f{d};");
            ptx.AppendLine($"    cvt.f64.f32 %fd3, %f{Dimension + d};");
            ptx.AppendLine("    fma.rn.f64 %fd0, %fd2, %fd2, %fd0;");
            ptx.AppendLine("    fma.rn.f64 %fd1, %fd2, %fd3, %fd1;");
        }
        ptx.AppendLine("    sqrt.rn.f64 %fd4, %fd0;");
        ptx.AppendLine($"    mov.f64 %fd5, 0d{BitConverter.DoubleToInt64Bits(DefaultEpsilon):x16};");
        ptx.AppendLine("    add.rn.f64 %fd6, %fd4, %fd5;");
        ptx.AppendLine("    add.rn.f64 %fd7, %fd0, 0d3ff0000000000000;");
        ptx.AppendLine("    mul.rn.f64 %fd8, %fd7, %fd6;");
        ptx.AppendLine("    setp.ne.f64 %p1, %fd8, 0d0000000000000000;");
        ptx.AppendLine("    div.rn.f64 %fd9, %fd0, %fd8;");
        ptx.AppendLine("    selp.f64 %fd9, %fd9, 0d0000000000000000, %p1;");
        ptx.AppendLine("    add.rn.f64 %fd10, %fd5, %fd5;");
        ptx.AppendLine("    add.rn.f64 %fd10, %fd4, %fd10;");
        ptx.AppendLine("    neg.f64 %fd11, %fd0;");
        ptx.AppendLine("    fma.rn.f64 %fd10, %fd11, %fd4, %fd10;");
        ptx.AppendLine("    mul.rn.f64 %fd8, %fd8, %fd8;");
        ptx.AppendLine("    div.rn.f64 %fd10, %fd10, %fd8;");
        ptx.AppendLine("    selp.f64 %fd10, %fd10, 0d0000000000000000, %p1;");
        for (int d = 0; d < Dimension; d++)
        {
            ptx.AppendLine($"    cvt.f64.f32 %fd2, %f{Dimension + d};");
            ptx.AppendLine("    mul.rn.f64 %fd2, %fd9, %fd2;");
            ptx.AppendLine($"    cvt.f64.f32 %fd3, %f{d};");
            ptx.AppendLine("    mul.rn.f64 %fd3, %fd10, %fd3;");
            ptx.AppendLine("    fma.rn.f64 %fd2, %fd3, %fd1, %fd2;");
            ptx.AppendLine("    cvt.rn.f32.f64 %f32, %fd2;");
            ptx.AppendLine($"    st.global.f32 [%rd6+{d * sizeof(float)}], %f32;");
        }
        ptx.AppendLine("CAPSULE_SQUASH_BACKWARD_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitHeader(StringBuilder ptx, string entryPoint, params string[] pointers)
    {
        ptx.AppendLine($".visible .entry {entryPoint}(");
        for (int i = 0; i < pointers.Length; i++)
            ptx.AppendLine($"    .param .u64 {pointers[i]}{(i + 1 == pointers.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
    }

    private static void EmitCapsuleIndexAndBase(StringBuilder ptx, string doneLabel)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {Capsules};");
        ptx.AppendLine($"    @%p0 bra {doneLabel};");
        ptx.AppendLine("    shl.b32 %r3, %r2, 6;");
        ptx.AppendLine("    cvt.u64.u32 %rd3, %r3;");
    }

    private static string GetEntryPoint(DirectPtxCapsuleSquashOperation operation) =>
        (operation == DirectPtxCapsuleSquashOperation.Forward
            ? "aidotnet_capsule_squash_f32_c320_d16_e"
            : "aidotnet_capsule_squash_backward_f32_c320_d16_e") +
        BitConverter.SingleToInt32Bits(DefaultEpsilon).ToString("x8");

    private static DirectPtxTensorContract Exact(
        string name,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent,
        DirectPtxTensorAccess access) =>
        new(name, DirectPtxPhysicalType.Float32, layout, extent, extent, 16, access,
            DirectPtxExtentMode.Exact);

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
