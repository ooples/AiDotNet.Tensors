using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FP16 contract-M GEMM <c>C_fp16[P,Q] = transpose(A_fp16[M,P]) @ B_fp16[M,Q]</c>, i.e.
/// <c>C[p,q] = sum_m A[m,p] * B[m,q]</c> contracting over M with FP32 accumulation. This
/// is the weight-gradient (<c>dB</c>) half of <c>MatMulBackwardFp16Fused</c>: with
/// (P,Q) = (K,N), A = A[M,K] and B = dC[M,N] it computes
/// <c>dB[K,N] = transpose(A) @ dC</c>. It reuses the coalesced <c>[contract][out-index]</c>
/// shared staging of <see cref="PtxGemmContractMKernel"/> (both global loads contiguous
/// over the output index) with the FP16 convert-in / convert-out of
/// <see cref="PtxHgemmKernel"/>.
///
/// Tile BP=BQ=64 over (P,Q), BM=8 over M, TM=TN=4 → 256 threads, 16 FP32
/// accumulators/thread, 4 KiB shared. Grid = (Q/BQ, P/BP).
/// </summary>
internal sealed class PtxGemmFp16ContractMKernel : IDisposable
{
    internal const int BlockP = 64;
    internal const int BlockQ = 64;
    internal const int BlockM = 8;
    internal const int ThreadM = 4;
    internal const int ThreadN = 4;
    internal const int TileWidth = 64;
    internal const int BlockThreads = (BlockP / ThreadM) * (BlockQ / ThreadN); // 256
    internal const string EntryPoint = "aidotnet_gemm_fp16_contract_m";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int P { get; }
    internal int Q { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxGemmFp16ContractMKernel(DirectPtxRuntime runtime, int m, int p, int q)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP16 contract-M specialization is measured only on GA10x/SM86.");
        ValidateShape(m, p, q);
        M = m;
        P = p;
        Q = q;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, p, q);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, p, q);
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
            throw new ArgumentException("FP16 contract-M output may not alias A or B.");

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, (uint)(Q / BlockQ), (uint)(P / BlockP), 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int p, int q)
    {
        ValidateShape(m, p, q);
        int pHalfBytes = checked(p * sizeof(ushort)); // A row stride (over M)
        int qHalfBytes = checked(q * sizeof(ushort)); // B row stride (over M) and C row stride (over P)

        var ptx = new StringBuilder(24_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// gemm-fp16-contract-m M={m} P={p} Q={q} tile(P,Q,M)={BlockP}x{BlockQ}x{BlockM}");
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
        ptx.AppendLine($"    .shared .align 16 .b8 a_tile[{BlockM * TileWidth * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 b_tile[{BlockM * TileWidth * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, a_tile;");
        ptx.AppendLine("    mov.u64 %rd5, b_tile;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // Q block
        ptx.AppendLine("    mov.u32 %r2, %ctaid.y;");                 // P block
        ptx.AppendLine("    shr.u32 %r3, %r0, 4;");                   // threadRow (over P)
        ptx.AppendLine("    and.b32 %r4, %r0, 15;");                  // threadCol (over Q)
        ptx.AppendLine($"    mul.lo.u32 %r5, %r2, {BlockP};");        // pBlock
        ptx.AppendLine($"    mul.lo.u32 %r6, %r1, {BlockQ};");        // qBlock
        ptx.AppendLine($"    mul.lo.u32 %r7, %r3, {ThreadM};");
        ptx.AppendLine($"    mul.lo.u32 %r8, %r4, {ThreadN};");
        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");                        // m0 contract counter
        ptx.AppendLine("M_TILE_LOOP:");
        EmitCooperativeLoad(ptx, pHalfBytes, qHalfBytes);
        ptx.AppendLine("    bar.sync 0;");
        EmitInnerAccumulate(ptx);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockM};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {m};");
        ptx.AppendLine("    @%p0 bra.uni M_TILE_LOOP;");
        EmitEpilogue(ptx, qHalfBytes);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitCooperativeLoad(StringBuilder ptx, int pHalfBytes, int qHalfBytes)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            ptx.AppendLine($"    shr.u32 %r12, {e}, 6;");                   // contract c
            ptx.AppendLine($"    and.b32 %r13, {e}, 63;");                  // out-index
            ptx.AppendLine("    add.u32 %r14, %r9, %r12;");                 // global M row
            // A[m, pBlock + idx]
            ptx.AppendLine("    add.u32 %r15, %r5, %r13;");
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {pHalfBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 2;");
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.u16 %h0, [%rd10];");
            ptx.AppendLine("    cvt.f32.f16 %f16, %h0;");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
            // B[m, qBlock + idx]
            ptx.AppendLine("    add.u32 %r15, %r6, %r13;");
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {qHalfBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 2;");
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
        ptx.AppendLine("    mul.wide.u32 %rd13, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd13;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd5, %rd14;");
        for (int c = 0; c < BlockM; c++)
        {
            for (int i = 0; i < ThreadM; i++)
                ptx.AppendLine($"    ld.shared.f32 %f{16 + i}, [%rd13+{(c * TileWidth + i) * sizeof(float)}];");
            for (int j = 0; j < ThreadN; j++)
                ptx.AppendLine($"    ld.shared.f32 %f{20 + j}, [%rd14+{(c * TileWidth + j) * sizeof(float)}];");
            for (int i = 0; i < ThreadM; i++)
                for (int j = 0; j < ThreadN; j++)
                    ptx.AppendLine(
                        $"    fma.rn.f32 %f{i * ThreadN + j}, %f{16 + i}, %f{20 + j}, %f{i * ThreadN + j};");
        }
    }

    // FP16 output C[P,Q] row-major: p = pBlock + threadRow*TM + i, q = qBlock + threadCol*TN + j.
    private static void EmitEpilogue(StringBuilder ptx, int qHalfBytes)
    {
        ptx.AppendLine("    add.u32 %r18, %r5, %r7;");                       // p0
        ptx.AppendLine("    add.u32 %r19, %r6, %r8;");                       // q0
        for (int i = 0; i < ThreadM; i++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r18, {i};");
            ptx.AppendLine($"    mul.wide.u32 %rd15, %r20, {qHalfBytes};");
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
        DirectPtxArchitectureFamily architecture, int m, int p, int q)
    {
        var a = new DirectPtxExtent(m, p);
        var b = new DirectPtxExtent(m, q);
        var output = new DirectPtxExtent(p, q);
        return new DirectPtxKernelBlueprint(
            Operation: "gemm-fp16-contract-m",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp16-m{m}-p{p}-q{q}",
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    a, a, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("b", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    b, b, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 96,
                MaxStaticSharedBytes: 2 * BlockM * TileWidth * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "C_fp16[P,Q] = transpose(A_fp16[M,P]) @ B_fp16[M,Q]",
                ["role"] = "dWeight (dB=transpose(A)@dC) half of MatMulBackwardFp16Fused",
                ["contract"] = "over-M",
                ["accumulate"] = "fp32-then-cvt-rn-f16-f32-output",
                ["shared-layout"] = "contract-major-coalesced-global-load",
                ["tile"] = $"{BlockP}x{BlockQ}x{BlockM}-register-{ThreadM}x{ThreadN}",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int p, int q) =>
        m > 0 && m % BlockM == 0 &&
        p > 0 && p % BlockP == 0 &&
        q > 0 && q % BlockQ == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        p is 256 or 512 or 1024 or 2048 or 4096 &&
        q is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int p, int q) => false;

    private static void ValidateShape(int m, int p, int q)
    {
        if (!IsSupportedShape(m, p, q))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "FP16 contract-M supports M in {64,128,256,512,1024,2048}, P/Q in {256,512,1024,2048,4096}.");
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
