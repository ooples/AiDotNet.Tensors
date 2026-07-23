using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Elementwise complex conjugate <c>out[i] = (real, -imag)</c> over interleaved
/// <c>[real, imag]</c> pairs (issue #854). One thread owns one pair — no shared memory,
/// no reduction, exact. 256 threads/block, grid = pairs/256 (positive multiple of 256).
/// </summary>
internal sealed class PtxComplexConjugateKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxPairs = 2048 * 4096;
    internal const string EntryPoint = "aidotnet_complex_conjugate";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Pairs { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxComplexConjugateKernel(DirectPtxRuntime runtime, int pairs)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in complex-conjugate specialization is measured only on GA10x/SM86.");
        ValidateShape(pairs);
        Pairs = pairs;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, pairs);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, pairs);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(_function, (uint)(Pairs / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int pairs)
    {
        ValidateShape(pairs);
        var ptx = new StringBuilder(3_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// complex-conjugate pairs={pairs}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd2;");
        // Coalesced 8-byte transactions on the 8-byte-aligned interleaved [re,im] pair.
        ptx.AppendLine("    ld.global.nc.v2.f32 {%f0, %f1}, [%rd3];");   // real, imag
        ptx.AppendLine("    neg.f32 %f2, %f1;");                         // -imag
        ptx.AppendLine("    st.global.v2.f32 [%rd4], {%f0, %f2};");      // real, -imag
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int pairs)
    {
        var extent = new DirectPtxExtent(pairs, 2);
        return new DirectPtxKernelBlueprint(
            Operation: "complex-conjugate",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-pairs{pairs}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 12,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "out = (real, -imag), interleaved [re,im]",
                ["layout"] = "interleaved-complex-pairs",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedPairs(int pairs) =>
        pairs > 0 && pairs % BlockThreads == 0 && pairs <= MaxPairs;

    internal static bool IsPromotedPairs(int pairs) => false;

    private static void ValidateShape(int pairs)
    {
        if (!IsSupportedPairs(pairs))
            throw new ArgumentOutOfRangeException(
                nameof(pairs),
                $"Complex conjugate supports a positive pair count that is a multiple of {BlockThreads} up to {MaxPairs}.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
