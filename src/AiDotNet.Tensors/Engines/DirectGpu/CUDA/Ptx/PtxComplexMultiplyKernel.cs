using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Elementwise complex multiply <c>out[i] = a[i] * b[i]</c> over interleaved
/// <c>[real, imag]</c> pairs (issue #854): <c>(ar+ai·j)(br+bi·j) = (ar·br - ai·bi) +
/// (ar·bi + ai·br)·j</c>. One thread owns one complex pair — no shared memory, no reduction,
/// no global intermediate. The result is exact.
///
/// 256 threads/block, grid = pairs/256; supported pair counts are positive multiples of 256.
/// Buffers are contiguous interleaved complex of length <c>2·pairs</c> floats.
/// </summary>
internal sealed class PtxComplexMultiplyKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxPairs = 2048 * 4096;
    internal const string EntryPoint = "aidotnet_complex_multiply";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Pairs { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxComplexMultiplyKernel(DirectPtxRuntime runtime, int pairs)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in complex-multiply specialization is measured only on GA10x/SM86.");
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

    internal unsafe void Launch(DirectPtxTensorView a, DirectPtxTensorView b, DirectPtxTensorView output)
    {
        Require(a, Blueprint.Tensors[0], nameof(a));
        Require(b, Blueprint.Tensors[1], nameof(b));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, (uint)(Pairs / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int pairs)
    {
        ValidateShape(pairs);
        var ptx = new StringBuilder(4_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// complex-multiply pairs={pairs}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");    // pair index
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 8;");                    // byte offset = pair*2*4
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        // Coalesced 8-byte transactions: each interleaved [re,im] pair is 8-byte aligned
        // (16-byte-aligned base + pair*8), so a single v2.f32 loads/stores both components.
        ptx.AppendLine("    ld.global.nc.v2.f32 {%f0, %f1}, [%rd4];");       // ar, ai
        ptx.AppendLine("    ld.global.nc.v2.f32 {%f2, %f3}, [%rd5];");       // br, bi
        // real = ar*br - ai*bi ; imag = ar*bi + ai*br
        ptx.AppendLine("    mul.rn.f32 %f4, %f0, %f2;");                     // ar*br
        ptx.AppendLine("    mul.rn.f32 %f5, %f1, %f3;");                     // ai*bi
        ptx.AppendLine("    sub.rn.f32 %f6, %f4, %f5;");                     // real
        ptx.AppendLine("    mul.rn.f32 %f7, %f0, %f3;");                     // ar*bi
        ptx.AppendLine("    fma.rn.f32 %f8, %f1, %f2, %f7;");                // ai*br + ar*bi = imag
        ptx.AppendLine("    st.global.v2.f32 [%rd6], {%f6, %f8};");          // real, imag
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int pairs)
    {
        var extent = new DirectPtxExtent(pairs, 2); // [pairs, 2] interleaved complex
        return new DirectPtxKernelBlueprint(
            Operation: "complex-multiply",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-pairs{pairs}",
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "out = (ar*br - ai*bi) + (ar*bi + ai*br) j, interleaved [re,im]",
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
                $"Complex multiply supports a positive pair count that is a multiple of {BlockThreads} up to {MaxPairs}.");
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
