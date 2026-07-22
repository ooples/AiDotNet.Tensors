using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Which unary map a <see cref="PtxFusedComplexUnaryF32Kernel"/> applies to an
/// interleaved complex FP32 buffer. The two operators write different output
/// extents - conjugate is pair-to-pair, magnitude is pair-to-scalar - so each
/// is a separate module and a separate issue-#850 coverage cell.
/// </summary>
internal enum DirectPtxComplexUnaryOp
{
    /// <summary>output[i] = (re, -im). Same extent in and out.</summary>
    Conjugate,

    /// <summary>output[i] = sqrt(re*re + im*im). Half the output extent.</summary>
    Magnitude
}

/// <summary>
/// Exact contiguous interleaved-complex FP32 unary maps for issue #850, the
/// conjugate and magnitude siblings of
/// <see cref="PtxFusedComplexMultiplyF32Kernel"/>.
///
/// Both read the same canonical <c>[pairs, 2]</c> interleaved layout with a
/// single <c>v2</c> load per thread, keep everything in registers, and write
/// once. The launch covers exactly the pair count, so neither needs the
/// reference's bounds guard and neither emits a branch. Two pointers reach the
/// launch ABI.
///
/// The magnitude operator follows the established <c>sqrtf(re*re + im*im)</c>
/// exactly rather than reaching for a fused or approximate form: <c>sqrt.rn</c>
/// is the IEEE-correct square root that <c>sqrtf</c> compiles to, and the sum
/// is formed with an unfused multiply-multiply-add so it rounds the same way.
/// Using <c>fma</c> here would be faster and MORE accurate, which is precisely
/// why it is not used - it would silently disagree with the kernel it replaces.
///
/// Conjugate negates with <c>neg.f32</c>, which flips the sign bit, so it
/// carries NaN payloads and signed zeros through unchanged exactly as the
/// reference's unary minus does.
///
/// The specialization stays disabled by default and fails closed until three
/// clean promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedComplexUnaryF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxComplexUnaryOp Op { get; }
    internal int NumPairs { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal static string EntryPointFor(DirectPtxComplexUnaryOp op) => op switch
    {
        DirectPtxComplexUnaryOp.Conjugate => "aidotnet_fused_complex_conjugate_f32",
        DirectPtxComplexUnaryOp.Magnitude => "aidotnet_fused_complex_magnitude_f32",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    internal PtxFusedComplexUnaryF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxComplexUnaryOp op,
        int numPairs,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in complex unary specializations are admitted only on SM86.");
        ValidateOp(op);
        Validate(numPairs);
        ValidateBlockThreads(numPairs, blockThreads);
        Op = op;
        NumPairs = numPairs;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, op, numPairs, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            op, numPairs, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPointFor(op), out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPointFor(op), info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
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
        _module.Launch(
            _function,
            checked((uint)(NumPairs / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxComplexUnaryOp op,
        int numPairs,
        int blockThreads = DefaultBlockThreads)
    {
        ValidateOp(op);
        Validate(numPairs);
        ValidateBlockThreads(numPairs, blockThreads);
        bool isConjugate = op == DirectPtxComplexUnaryOp.Conjugate;
        string entryPoint = EntryPointFor(op);

        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape pairs={numPairs} block={blockThreads} layout=interleaved-complex-fp32 op={OpTag(op)}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        // Input strides 8 bytes per pair; the output stride depends on the op.
        ptx.AppendLine("    mul.wide.u32 %rd2, %r2, 8;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");
        ptx.AppendLine("    ld.global.ca.v2.f32 {%f0, %f1}, [%rd4];");
        if (isConjugate)
        {
            // Sign-bit flip: NaN payloads and signed zeros pass through as the
            // reference's unary minus leaves them.
            ptx.AppendLine("    neg.f32 %f2, %f1;");
            ptx.AppendLine("    add.u64 %rd6, %rd1, %rd2;");
            ptx.AppendLine("    st.global.v2.f32 [%rd6], {%f0, %f2};");
        }
        else
        {
            // sqrtf(re*re + im*im), unfused so the rounding matches the
            // reference. An fma would be faster and more accurate, and would
            // therefore disagree with the kernel this replaces.
            ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f0;");
            ptx.AppendLine("    mul.rn.f32 %f3, %f1, %f1;");
            ptx.AppendLine("    add.rn.f32 %f2, %f2, %f3;");
            ptx.AppendLine("    sqrt.rn.f32 %f2, %f2;");
            ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
            ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
            ptx.AppendLine("    st.global.f32 [%rd6], %f2;");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string OpTag(DirectPtxComplexUnaryOp op) => op switch
    {
        DirectPtxComplexUnaryOp.Conjugate => "complex-conjugate",
        DirectPtxComplexUnaryOp.Magnitude => "complex-magnitude",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxComplexUnaryOp op,
        int numPairs,
        int blockThreads)
    {
        var pairExtent = new DirectPtxExtent(numPairs, 2);
        var scalarExtent = new DirectPtxExtent(numPairs);
        bool isConjugate = op == DirectPtxComplexUnaryOp.Conjugate;
        return new DirectPtxKernelBlueprint(
            Operation: $"{OpTag(op)}-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"pairwise-v2-b{blockThreads}-n{numPairs}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    pairExtent, pairExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                isConjugate
                    ? new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                        pairExtent, pairExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
                    : new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                        scalarExtent, scalarExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = isConjugate
                    ? "output[i] = (input[i].re, -input[i].im)"
                    : "output[i] = sqrt(re*re + im*im)",
                ["mode"] = $"inference-forward-{OpTag(op)}",
                ["input-layout"] = "canonical-interleaved-[pair,real-imag]",
                ["output-layout"] = isConjugate
                    ? "canonical-interleaved-[pair,real-imag]"
                    : "contiguous-scalar-[pair]",
                ["arithmetic"] = isConjugate
                    ? "sign-bit-flip; NaN payloads and signed zeros preserved"
                    : "unfused mul-mul-add then sqrt.rn, matching sqrtf(re*re + im*im) rounding",
                ["fma-use"] = isConjugate
                    ? "n/a"
                    : "deliberately none - an fma would be more accurate and would disagree with the reference",
                ["global-input-reads"] = "one-fp32x2-per-pair",
                ["global-output-writes"] = isConjugate ? "one-fp32x2-per-pair" : "one-fp32-per-pair",
                ["bounds-check"] = "none - the launch covers exactly the pair count",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    /// <summary>Mirrors the complex-multiply family's exact pair counts.</summary>
    internal static bool IsSupportedShape(int numPairs) =>
        numPairs is 65_536 or 262_144 or 1_048_576 or 4_194_304;

    internal static bool IsPromotedShape(int numPairs) => false;

    internal static bool IsSupportedOp(DirectPtxComplexUnaryOp op) =>
        op is DirectPtxComplexUnaryOp.Conjugate or DirectPtxComplexUnaryOp.Magnitude;

    private static void ValidateOp(DirectPtxComplexUnaryOp op)
    {
        if (!IsSupportedOp(op))
            throw new ArgumentOutOfRangeException(nameof(op),
                "The complex unary family covers Conjugate and Magnitude.");
    }

    private static void Validate(int numPairs)
    {
        if (!IsSupportedShape(numPairs))
            throw new ArgumentOutOfRangeException(nameof(numPairs),
                "The first complex unary family supports exact pair counts " +
                "65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int numPairs, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || numPairs % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Complex unary block threads must be 128, 256, or 512 and evenly tile the pair count.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
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
