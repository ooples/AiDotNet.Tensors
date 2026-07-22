using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FP16 GEMM <c>C[M,N] = A[M,K] @ B[K,N]</c> with FP16 A, B and FP16 output (Hgemm /
/// GemmFp16HalfOut). Identical to <see cref="PtxGemmFp16Kernel"/> — FP16 operands
/// converted to FP32 for an FP32-accumulated register-blocked tile — but the
/// epilogue narrows each accumulator back to FP16 (<c>cvt.rn.f16.f32</c>) before the
/// coalesced FP16 store. Tile BM=BN=64, BK=8, TM=TN=4 → 256 threads, 4 KiB shared.
/// </summary>
internal sealed class PtxHgemmKernel : IDisposable
{
    internal const int BlockM = 64;
    internal const int BlockN = 64;
    internal const int BlockK = 8;
    internal const int ThreadM = 4;
    internal const int ThreadN = 4;
    internal const int BlockThreads = (BlockM / ThreadM) * (BlockN / ThreadN); // 256
    internal const string EntryPoint = "aidotnet_hgemm";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int K { get; }
    internal int N { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxHgemmKernel(DirectPtxRuntime runtime, int m, int k, int n)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in Hgemm specialization is measured only on GA10x/SM86.");
        ValidateShape(m, k, n);
        M = m;
        K = k;
        N = n;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, k, n);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, k, n);
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
            throw new ArgumentException("Hgemm output may not alias A or B.");

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, (uint)(N / BlockN), (uint)(M / BlockM), 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int k, int n)
    {
        ValidateShape(m, k, n);
        int kHalfBytes = checked(k * sizeof(ushort));
        int nHalfBytes = checked(n * sizeof(ushort));

        var ptx = new StringBuilder(24_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// hgemm M={m} K={k} N={n} tile={BlockM}x{BlockN}x{BlockK} thread={ThreadM}x{ThreadN}");
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
        ptx.AppendLine($"    .shared .align 16 .b8 b_tile[{BlockK * BlockN * sizeof(float)}];");
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
        EmitCooperativeLoad(ptx, kHalfBytes, nHalfBytes);
        ptx.AppendLine("    bar.sync 0;");
        EmitInnerAccumulate(ptx);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {k};");
        ptx.AppendLine("    @%p0 bra.uni K_TILE_LOOP;");
        EmitEpilogue(ptx, nHalfBytes);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitCooperativeLoad(StringBuilder ptx, int kHalfBytes, int nHalfBytes)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            ptx.AppendLine($"    shr.u32 %r12, {e}, 3;");
            ptx.AppendLine($"    and.b32 %r13, {e}, 7;");
            ptx.AppendLine("    add.u32 %r14, %r5, %r12;");
            ptx.AppendLine("    add.u32 %r15, %r9, %r13;");
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kHalfBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 2;");
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.u16 %h0, [%rd10];");
            ptx.AppendLine("    cvt.f32.f16 %f16, %h0;");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
            ptx.AppendLine($"    shr.u32 %r16, {e}, 6;");
            ptx.AppendLine($"    and.b32 %r17, {e}, 63;");
            ptx.AppendLine("    add.u32 %r18, %r9, %r16;");
            ptx.AppendLine("    add.u32 %r19, %r6, %r17;");
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r18, {nHalfBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r19, 2;");
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
        ptx.AppendLine($"    mul.lo.u32 %r20, %r7, {BlockK};");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r20, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd13;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd5, %rd14;");
        for (int k = 0; k < BlockK; k++)
        {
            for (int i = 0; i < ThreadM; i++)
                ptx.AppendLine($"    ld.shared.f32 %f{16 + i}, [%rd13+{(i * BlockK + k) * sizeof(float)}];");
            for (int j = 0; j < ThreadN; j++)
                ptx.AppendLine($"    ld.shared.f32 %f{20 + j}, [%rd14+{(k * BlockN + j) * sizeof(float)}];");
            for (int i = 0; i < ThreadM; i++)
                for (int j = 0; j < ThreadN; j++)
                    ptx.AppendLine(
                        $"    fma.rn.f32 %f{i * ThreadN + j}, %f{16 + i}, %f{20 + j}, %f{i * ThreadN + j};");
        }
    }

    // FP16 output: narrow each accumulator and store 2 bytes.
    private static void EmitEpilogue(StringBuilder ptx, int nHalfBytes)
    {
        ptx.AppendLine("    add.u32 %r18, %r5, %r7;");
        ptx.AppendLine("    add.u32 %r19, %r6, %r8;");
        for (int i = 0; i < ThreadM; i++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r18, {i};");
            ptx.AppendLine($"    mul.wide.u32 %rd15, %r20, {nHalfBytes};");
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
        DirectPtxArchitectureFamily architecture, int m, int k, int n)
    {
        var a = new DirectPtxExtent(m, k);
        var b = new DirectPtxExtent(k, n);
        var output = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "hgemm-fp16",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp16-m{m}-k{k}-n{n}",
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
                MaxStaticSharedBytes: (BlockM + BlockN) * BlockK * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "C_fp16[M,N] = A_fp16[M,K] @ B_fp16[K,N]",
                ["accumulate"] = "fp32-then-cvt-rn-f16-f32-output",
                ["tile"] = $"{BlockM}x{BlockN}x{BlockK}-register-{ThreadM}x{ThreadN}",
                ["tensor-cores"] = "no-simt-cvt-path-mma-sync-is-followup",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int k, int n) =>
        m > 0 && m % BlockM == 0 &&
        n > 0 && n % BlockN == 0 &&
        k > 0 && k % BlockK == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        k is 256 or 512 or 1024 or 2048 or 4096 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int k, int n) => false;

    private static void ValidateShape(int m, int k, int n)
    {
        if (!IsSupportedShape(m, k, n))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "Hgemm supports M in {64,128,256,512,1024,2048}, K/N in {256,512,1024,2048,4096}.");
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
