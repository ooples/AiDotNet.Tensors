using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FP16 transpose-B GEMM <c>C[M,No] = A[M,Kc] @ transpose(B[No,Kc])</c> with FP16 A, B
/// and FP16 output, FP32 accumulation. This is the input-gradient (<c>dA</c>) half of
/// <c>MatMulBackwardFp16Fused</c>: with (Kc, No) = (N, K) it computes
/// <c>dA[M,K] = dC[M,N] @ transpose(B[K,N])</c>. Combines the transpose-B staging of
/// <see cref="PtxMatMulTransposedKernel"/> (both operands row-major over the contracted
/// dimension) with the FP16 convert-in / convert-out epilogue of
/// <see cref="PtxHgemmKernel"/>.
///
/// Tile BM=BN=64, BK=8, TM=TN=4 → 256 threads, 16 FP32 accumulators/thread, 4 KiB
/// shared. Grid = (No/BN, M/BM).
/// </summary>
internal sealed class PtxGemmFp16TransposedKernel : IDisposable
{
    internal const int BlockM = 64;
    internal const int BlockN = 64;
    internal const int BlockK = 8;
    internal const int ThreadM = 4;
    internal const int ThreadN = 4;
    internal const int BlockThreads = (BlockM / ThreadM) * (BlockN / ThreadN); // 256
    internal const string EntryPoint = "aidotnet_gemm_fp16_transposed";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int Kc { get; }
    internal int No { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxGemmFp16TransposedKernel(DirectPtxRuntime runtime, int m, int kc, int no)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP16 transpose-B specialization is measured only on GA10x/SM86.");
        ValidateShape(m, kc, no);
        M = m;
        Kc = kc;
        No = no;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, kc, no);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, kc, no);
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
        if (Overlaps(output, a) || Overlaps(output, b))
            throw new ArgumentException("FP16 transpose-B output may not alias A or B.");

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, (uint)(No / BlockN), (uint)(M / BlockM), 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int kc, int no)
    {
        ValidateShape(m, kc, no);
        int kHalfBytes = checked(kc * sizeof(ushort)); // A and B row stride (over Kc)
        int noHalfBytes = checked(no * sizeof(ushort)); // C row stride (over No)

        var ptx = new StringBuilder(24_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// gemm-fp16-transposed M={m} Kc={kc} No={no} tile={BlockM}x{BlockN}x{BlockK}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b16 %h<4>;");
        ptx.AppendLine("    .reg .b32 %r<40>;");
        ptx.AppendLine("    .reg .b64 %rd<48>;");
        ptx.AppendLine("    .reg .f32 %f<48>;");
        ptx.AppendLine($"    .shared .align 16 .b8 a_tile[{BlockM * BlockK * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 b_tile[{BlockN * BlockK * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, a_tile;");
        ptx.AppendLine("    mov.u64 %rd5, b_tile;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r2, %ctaid.y;");
        ptx.AppendLine("    shr.u32 %r3, %r0, 4;");
        ptx.AppendLine("    and.b32 %r4, %r0, 15;");
        ptx.AppendLine($"    mul.lo.u32 %r5, %r2, {BlockM};");
        ptx.AppendLine($"    mul.lo.u32 %r6, %r1, {BlockN};");
        ptx.AppendLine($"    mul.lo.u32 %r7, %r3, {ThreadM};");
        ptx.AppendLine($"    mul.lo.u32 %r8, %r4, {ThreadN};");
        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");
        ptx.AppendLine("K_TILE_LOOP:");
        EmitCooperativeLoad(ptx, kHalfBytes);
        ptx.AppendLine("    bar.sync 0;");
        EmitInnerAccumulate(ptx);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {kc};");
        ptx.AppendLine("    @%p0 bra.uni K_TILE_LOOP;");
        EmitEpilogue(ptx, noHalfBytes);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // Both A (rows over M) and B (rows over No) are row-major over the contracted Kc and
    // loaded as FP16 -> converted to FP32 in the shared tile.
    private static void EmitCooperativeLoad(StringBuilder ptx, int kHalfBytes)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            ptx.AppendLine($"    shr.u32 %r12, {e}, 3;");                   // row within tile
            ptx.AppendLine($"    and.b32 %r13, {e}, 7;");                   // kk over Kc
            ptx.AppendLine("    add.u32 %r14, %r5, %r12;");                 // A global row (M)
            ptx.AppendLine("    add.u32 %r15, %r9, %r13;");                 // Kc index
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kHalfBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 2;");
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.u16 %h0, [%rd10];");
            ptx.AppendLine("    cvt.f32.f16 %f16, %h0;");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
            // B: output-major row over No; global row = nBlock + row.
            ptx.AppendLine("    add.u32 %r14, %r6, %r12;");
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kHalfBytes};");
            ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.u16 %h0, [%rd10];");
            ptx.AppendLine("    cvt.f32.f16 %f16, %h0;");
            ptx.AppendLine("    add.u64 %rd12, %rd5, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
        }
    }

    private static void EmitInnerAccumulate(StringBuilder ptx)
    {
        ptx.AppendLine($"    mul.lo.u32 %r16, %r7, {BlockK};");
        ptx.AppendLine($"    mul.lo.u32 %r17, %r8, {BlockK};");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r16, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd13;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r17, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd5, %rd14;");
        for (int k = 0; k < BlockK; k++)
        {
            for (int i = 0; i < ThreadM; i++)
                ptx.AppendLine($"    ld.shared.f32 %f{16 + i}, [%rd13+{(i * BlockK + k) * sizeof(float)}];");
            for (int j = 0; j < ThreadN; j++)
                ptx.AppendLine($"    ld.shared.f32 %f{20 + j}, [%rd14+{(j * BlockK + k) * sizeof(float)}];");
            for (int i = 0; i < ThreadM; i++)
                for (int j = 0; j < ThreadN; j++)
                    ptx.AppendLine(
                        $"    fma.rn.f32 %f{i * ThreadN + j}, %f{16 + i}, %f{20 + j}, %f{i * ThreadN + j};");
        }
    }

    // FP16 output: narrow each accumulator and store 2 bytes.
    private static void EmitEpilogue(StringBuilder ptx, int noHalfBytes)
    {
        ptx.AppendLine("    add.u32 %r18, %r5, %r7;");
        ptx.AppendLine("    add.u32 %r19, %r6, %r8;");
        for (int i = 0; i < ThreadM; i++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r18, {i};");
            ptx.AppendLine($"    mul.wide.u32 %rd15, %r20, {noHalfBytes};");
            ptx.AppendLine("    add.u64 %rd16, %rd3, %rd15;");
            for (int j = 0; j < ThreadN; j++)
            {
                int acc = i * ThreadN + j;
                ptx.AppendLine($"    add.u32 %r21, %r19, {j};");
                ptx.AppendLine("    mul.wide.u32 %rd17, %r21, 2;");
                ptx.AppendLine("    add.u64 %rd19, %rd16, %rd17;");
                ptx.AppendLine($"    cvt.rn.f16.f32 %h0, %f{acc};");
                ptx.AppendLine("    st.global.u16 [%rd19], %h0;");
            }
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int kc, int no)
    {
        var a = new DirectPtxExtent(m, kc);
        var b = new DirectPtxExtent(no, kc);
        var output = new DirectPtxExtent(m, no);
        return new DirectPtxKernelBlueprint(
            Operation: "gemm-fp16-transposed",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp16-m{m}-kc{kc}-no{no}",
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    a, a, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("b", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.LinearWeightOutputMajor,
                    b, b, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 96,
                MaxStaticSharedBytes: (BlockM + BlockN) * BlockK * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "C_fp16[M,No] = A_fp16[M,Kc] @ transpose(B_fp16[No,Kc])",
                ["role"] = "dInput (dA=dC@transpose(B)) half of MatMulBackwardFp16Fused",
                ["accumulate"] = "fp32-then-cvt-rn-f16-f32-output",
                ["b-layout"] = "output-major-row-major-fp16",
                ["tile"] = $"{BlockM}x{BlockN}x{BlockK}-register-{ThreadM}x{ThreadN}",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int kc, int no) =>
        m > 0 && m % BlockM == 0 &&
        no > 0 && no % BlockN == 0 &&
        kc > 0 && kc % BlockK == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        kc is 256 or 512 or 1024 or 2048 or 4096 &&
        no is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int kc, int no) => false;

    private static void ValidateShape(int m, int kc, int no)
    {
        if (!IsSupportedShape(m, kc, no))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "FP16 transpose-B supports M in {64,128,256,512,1024,2048}, Kc/No in {256,512,1024,2048,4096}.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
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
