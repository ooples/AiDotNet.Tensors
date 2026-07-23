using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous interleaved-complex FP32 multiplication for issue #850.
/// Every thread loads one <c>(real,imag)</c> pair from each input, keeps the
/// four components and both products in registers, and writes one final pair.
/// The element count is baked into the module and the launch ABI contains only
/// three pointers. The initial candidate is disabled and unpromoted.
/// </summary>
internal sealed class PtxFusedComplexMultiplyF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_complex_multiply_f32";
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumPairs { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedComplexMultiplyF32Kernel(DirectPtxRuntime runtime, int numPairs)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in complex-multiply specialization is admitted only on SM86.");
        Validate(numPairs);
        NumPairs = numPairs;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numPairs);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, numPairs);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads,
            activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView left,
        DirectPtxTensorView right,
        DirectPtxTensorView output)
    {
        Require(left, Blueprint.Tensors[0], nameof(left));
        Require(right, Blueprint.Tensors[1], nameof(right));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, left) || Overlaps(output, right))
            throw new ArgumentException(
                "The complex-multiply specialization requires output to be disjoint from both inputs.");

        IntPtr leftPointer = left.Pointer;
        IntPtr rightPointer = right.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &leftPointer;
        arguments[1] = &rightPointer;
        arguments[2] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)(NumPairs / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int numPairs)
    {
        Validate(numPairs);
        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape pairs={numPairs} block={BlockThreads} layout=interleaved-complex-fp32");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 left_ptr,");
        ptx.AppendLine("    .param .u64 right_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<9>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [left_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [right_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
                // Inputs are streamed once and never revisited, so they go through the
        // read-only data cache rather than displacing L1 lines the store path
        // can use.
        ptx.AppendLine("    ld.global.nc.v2.f32 {%f0,%f1}, [%rd4];");
                ptx.AppendLine("    ld.global.nc.v2.f32 {%f2,%f3}, [%rd5];");
        ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f3;");
        ptx.AppendLine("    neg.f32 %f4, %f4;");
        ptx.AppendLine("    fma.rn.f32 %f4, %f0, %f2, %f4;");
        ptx.AppendLine("    mul.rn.f32 %f5, %f1, %f2;");
        ptx.AppendLine("    fma.rn.f32 %f5, %f0, %f3, %f5;");
        ptx.AppendLine("    st.global.v2.f32 [%rd6], {%f4,%f5};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    internal static bool IsSupportedShape(int numPairs) =>
        numPairs is 65536 or 262144 or 1048576 or 4194304;

    internal static bool IsPromotedShape(int numPairs) => false;

    private static void Validate(int numPairs)
    {
        if (!IsSupportedShape(numPairs))
            throw new ArgumentOutOfRangeException(nameof(numPairs),
                "The first complex-multiply family supports exact pair counts 65536, 262144, 1048576, and 4194304.");
        if (numPairs % BlockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(numPairs),
                "The exact-shape kernel requires a full final block.");
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int numPairs)
    {
        var extent = new DirectPtxExtent(numPairs, 2);
        return new DirectPtxKernelBlueprint(
            Operation: "interleaved-complex-multiply-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"pairwise-v2-n{numPairs}",
            Tensors:
            [
                new("left", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("right", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "(ar+j*ai)*(br+j*bi)",
                ["mode"] = "fp32-interleaved-pairwise-forward",
                ["input-layout"] = "canonical-interleaved-[pair,real-imag]",
                ["output-layout"] = "canonical-interleaved-[pair,real-imag]",
                ["arithmetic"] = "two-mul-two-fma-register-resident",
                ["global-input-reads"] = "one-fp32x2-per-input-per-pair",
                ["global-output-writes"] = "one-fp32x2-per-pair",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["alias-policy"] = "inputs-may-alias-output-disjoint",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero ||
            view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
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
