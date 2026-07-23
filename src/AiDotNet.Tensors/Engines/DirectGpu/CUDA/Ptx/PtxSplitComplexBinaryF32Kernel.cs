using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Which split-buffer binary complex map a <see cref="PtxSplitComplexBinaryF32Kernel"/> applies. Both
/// read two split real/imag operands and write a split real/imag result, so each is a separate module
/// and a separate issue-#850 coverage cell.
/// </summary>
internal enum DirectPtxSplitComplexBinaryOp
{
    /// <summary>out = a * b: (ar*br - ai*bi, ar*bi + ai*br).</summary>
    Multiply,

    /// <summary>out = a + b: (ar+br, ai+bi).</summary>
    Add,

    /// <summary>out = a * conj(b): (ar*br + ai*bi, ai*br - ar*bi) - the cross-spectral product.</summary>
    CrossSpectral
}

/// <summary>
/// Exact contiguous split real/imag FP32 binary complex maps for issue #850, the split siblings of the
/// interleaved <see cref="PtxFusedComplexMultiplyF32Kernel"/>. One thread owns one element, reads the
/// four input lanes from their separate buffers, keeps everything in registers, and writes the two
/// output lanes once. The launch covers exactly the element count, so neither op needs the reference's
/// bounds guard and neither emits a branch. Six pointers reach the launch ABI.
///
/// The multiply operator forms <c>ar*br - ai*bi</c> and <c>ar*bi + ai*br</c> with the same
/// multiply-then-fma contraction the interleaved multiply kernel uses, matching the reference's default
/// fused evaluation exactly. The add operator is two <c>add.rn</c> lanes.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs
/// clear the release gate.
/// </summary>
internal sealed class PtxSplitComplexBinaryF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxSplitComplexBinaryOp Op { get; }
    internal int Count { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal static string EntryPointFor(DirectPtxSplitComplexBinaryOp op) => op switch
    {
        DirectPtxSplitComplexBinaryOp.Multiply => "aidotnet_split_complex_multiply_f32",
        DirectPtxSplitComplexBinaryOp.Add => "aidotnet_split_complex_add_f32",
        DirectPtxSplitComplexBinaryOp.CrossSpectral => "aidotnet_split_complex_cross_spectral_f32",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    internal PtxSplitComplexBinaryF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxSplitComplexBinaryOp op,
        int count,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in split-complex binary specializations are admitted only on SM86.");
        ValidateOp(op);
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        Op = op;
        Count = count;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, op, count, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, op, count, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPointFor(op), out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPointFor(op), info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView aReal, DirectPtxTensorView aImag,
        DirectPtxTensorView bReal, DirectPtxTensorView bImag,
        DirectPtxTensorView outReal, DirectPtxTensorView outImag)
    {
        Require(aReal, Blueprint.Tensors[0], nameof(aReal));
        Require(aImag, Blueprint.Tensors[1], nameof(aImag));
        Require(bReal, Blueprint.Tensors[2], nameof(bReal));
        Require(bImag, Blueprint.Tensors[3], nameof(bImag));
        Require(outReal, Blueprint.Tensors[4], nameof(outReal));
        Require(outImag, Blueprint.Tensors[5], nameof(outImag));

        IntPtr aRealPointer = aReal.Pointer, aImagPointer = aImag.Pointer;
        IntPtr bRealPointer = bReal.Pointer, bImagPointer = bImag.Pointer;
        IntPtr outRealPointer = outReal.Pointer, outImagPointer = outImag.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &aRealPointer;
        arguments[1] = &aImagPointer;
        arguments[2] = &bRealPointer;
        arguments[3] = &bImagPointer;
        arguments[4] = &outRealPointer;
        arguments[5] = &outImagPointer;
        _module.Launch(
            _function,
            checked((uint)(Count / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, DirectPtxSplitComplexBinaryOp op, int count, int blockThreads = DefaultBlockThreads)
    {
        ValidateOp(op);
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        bool isMultiply = op == DirectPtxSplitComplexBinaryOp.Multiply;
        string entryPoint = EntryPointFor(op);

        var ptx = new StringBuilder(2_560);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape count={count} block={blockThreads} layout=split-complex-fp32 op={OpTag(op)}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entryPoint}(");
        ptx.AppendLine("    .param .u64 a_real_ptr,");
        ptx.AppendLine("    .param .u64 a_imag_ptr,");
        ptx.AppendLine("    .param .u64 b_real_ptr,");
        ptx.AppendLine("    .param .u64 b_imag_ptr,");
        ptx.AppendLine("    .param .u64 out_real_ptr,");
        ptx.AppendLine("    .param .u64 out_imag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [a_imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [b_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [b_imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [out_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd5, [out_imag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd6;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd6;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd8];");     // ar
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd9];");     // ai
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd10];");    // br
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd11];");    // bi
        if (op == DirectPtxSplitComplexBinaryOp.Multiply)
        {
            // outReal = ar*br - ai*bi, outImag = ar*bi + ai*br, matching the interleaved
            // multiply's multiply-then-fma contraction (the reference's default fused form).
            ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f3;");     // ai*bi
            ptx.AppendLine("    neg.f32 %f4, %f4;");
            ptx.AppendLine("    fma.rn.f32 %f4, %f0, %f2, %f4;"); // ar*br - ai*bi
            ptx.AppendLine("    mul.rn.f32 %f5, %f1, %f2;");     // ai*br
            ptx.AppendLine("    fma.rn.f32 %f5, %f0, %f3, %f5;"); // ar*bi + ai*br
        }
        else if (op == DirectPtxSplitComplexBinaryOp.CrossSpectral)
        {
            // x * conj(y): outReal = xr*yr + xi*yi, outImag = xi*yr - xr*yi, using the same
            // multiply-then-fma contraction as the reference's default fused form (a=x, b=y).
            ptx.AppendLine("    mul.rn.f32 %f4, %f1, %f3;");     // xi*yi
            ptx.AppendLine("    fma.rn.f32 %f4, %f0, %f2, %f4;"); // xr*yr + xi*yi
            ptx.AppendLine("    mul.rn.f32 %f5, %f0, %f3;");     // xr*yi
            ptx.AppendLine("    neg.f32 %f5, %f5;");
            ptx.AppendLine("    fma.rn.f32 %f5, %f1, %f2, %f5;"); // xi*yr - xr*yi
        }
        else
        {
            ptx.AppendLine("    add.rn.f32 %f4, %f0, %f2;");     // ar + br
            ptx.AppendLine("    add.rn.f32 %f5, %f1, %f3;");     // ai + bi
        }
        ptx.AppendLine("    add.u64 %rd12, %rd4, %rd6;");
        ptx.AppendLine("    add.u64 %rd13, %rd5, %rd6;");
        ptx.AppendLine("    st.global.f32 [%rd12], %f4;");
        ptx.AppendLine("    st.global.f32 [%rd13], %f5;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string OpTag(DirectPtxSplitComplexBinaryOp op) => op switch
    {
        DirectPtxSplitComplexBinaryOp.Multiply => "split-complex-multiply",
        DirectPtxSplitComplexBinaryOp.Add => "split-complex-add",
        DirectPtxSplitComplexBinaryOp.CrossSpectral => "split-complex-cross-spectral",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, DirectPtxSplitComplexBinaryOp op, int count, int blockThreads)
    {
        var extent = new DirectPtxExtent(count);
        DirectPtxTensorContract In(string name) => new(name, DirectPtxPhysicalType.Float32,
            DirectPtxPhysicalLayout.Vector, extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract Out(string name) => new(name, DirectPtxPhysicalType.Float32,
            DirectPtxPhysicalLayout.Vector, extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact);
        return new DirectPtxKernelBlueprint(
            Operation: $"{OpTag(op)}-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"split-b{blockThreads}-n{count}",
            Tensors:
            [
                In("aReal"), In("aImag"), In("bReal"), In("bImag"), Out("outReal"), Out("outImag")
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // Measured by the offline gate at sm86: 20 registers for Add and
                // 22 for Multiply. The budget was 16, so ResourceBudget.Validate
                // would have thrown at construction on a real device for BOTH
                // operators - the kernel takes four input streams and holds two
                // output components live, which costs more than the unary kernels
                // this budget was copied from.
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = op switch
                {
                    DirectPtxSplitComplexBinaryOp.Multiply => "out = (ar*br - ai*bi, ar*bi + ai*br)",
                    DirectPtxSplitComplexBinaryOp.CrossSpectral => "out = a*conj(b) = (ar*br + ai*bi, ai*br - ar*bi)",
                    _ => "out = (ar + br, ai + bi)"
                },
                ["mode"] = $"inference-forward-{OpTag(op)}",
                ["input-layout"] = "canonical-split-real-imag-contiguous",
                ["output-layout"] = "canonical-split-real-imag-contiguous",
                ["arithmetic"] = op == DirectPtxSplitComplexBinaryOp.Add
                    ? "two add.rn lanes"
                    : "multiply-then-fma contraction matching the interleaved multiply and the reference's fused form",
                ["global-input-reads"] = "four-fp32-per-element",
                ["global-output-writes"] = "two-fp32-per-element",
                ["bounds-check"] = "none - the launch covers exactly the element count",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int count) =>
        count is 65_536 or 262_144 or 1_048_576 or 4_194_304;

    internal static bool IsPromotedShape(int count) => false;

    internal static bool IsSupportedOp(DirectPtxSplitComplexBinaryOp op) =>
        op is DirectPtxSplitComplexBinaryOp.Multiply or DirectPtxSplitComplexBinaryOp.Add
            or DirectPtxSplitComplexBinaryOp.CrossSpectral;

    private static void ValidateOp(DirectPtxSplitComplexBinaryOp op)
    {
        if (!IsSupportedOp(op))
            throw new ArgumentOutOfRangeException(nameof(op),
                "The split complex binary family covers Multiply, Add, and CrossSpectral.");
    }

    private static void Validate(int count)
    {
        if (!IsSupportedShape(count))
            throw new ArgumentOutOfRangeException(nameof(count),
                "The first split-complex binary family supports exact counts 65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int count, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || count % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Split complex binary block threads must be 128, 256, or 512 and evenly tile the element count.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
