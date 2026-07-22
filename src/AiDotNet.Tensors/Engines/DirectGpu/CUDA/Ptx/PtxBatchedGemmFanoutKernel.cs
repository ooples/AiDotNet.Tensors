using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FP32 batched pointer-fanout GEMM: a single shared <c>A[M,K]</c> multiplied against a
/// batch of independent right operands <c>C_i[M,N] = A @ B_i[K,N]</c>, where the B_i and
/// C_i bases are supplied as device arrays of pointers (the cuBLAS batched-pointer ABI)
/// rather than a single strided tensor. Each block resolves its batch slice from
/// <c>%ctaid.z</c>, dereferences <c>B_i</c> / <c>C_i</c> from the pointer arrays, then runs
/// the same register-blocked tile as <see cref="PtxGemmKernel"/>. A is loaded once per
/// block and reused across the batch dimension.
///
/// Tile BM=BN=64, BK=8, TM=TN=4 → 256 threads, 16 FP32 accumulators/thread, 4 KiB
/// shared. Grid = (N/BN, M/BM, batch).
/// </summary>
internal sealed class PtxBatchedGemmFanoutKernel : IDisposable
{
    internal const int BlockM = 64;
    internal const int BlockN = 64;
    internal const int BlockK = 8;
    internal const int ThreadM = 4;
    internal const int ThreadN = 4;
    internal const int BlockThreads = (BlockM / ThreadM) * (BlockN / ThreadN); // 256
    internal const string EntryPoint = "aidotnet_batched_gemm_fanout";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int K { get; }
    internal int N { get; }
    internal int Batch { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxBatchedGemmFanoutKernel(DirectPtxRuntime runtime, int m, int k, int n, int batch)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in batched-fanout specialization is measured only on GA10x/SM86.");
        ValidateShape(m, k, n, batch);
        M = m;
        K = k;
        N = n;
        Batch = batch;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, k, n, batch);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, k, n);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    /// <summary>
    /// Launches the fanout. <paramref name="a"/> is the shared left operand; the two
    /// pointer arrays are device buffers each holding <see cref="Batch"/> little-endian
    /// 64-bit device addresses (B_i / C_i bases), matching the cuBLAS batched-pointer ABI.
    /// </summary>
    internal unsafe void Launch(
        DirectPtxTensorView a, IntPtr bPointerArray, IntPtr cPointerArray)
    {
        Require(a, Blueprint.Tensors[0], nameof(a));
        if (bPointerArray == IntPtr.Zero || cPointerArray == IntPtr.Zero)
            throw new ArgumentException("Fanout pointer arrays must be non-null device buffers.");

        IntPtr aPointer = a.Pointer;
        IntPtr bArray = bPointerArray;
        IntPtr cArray = cPointerArray;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bArray;
        arguments[2] = &cArray;
        _module.Launch(
            _function, (uint)(N / BlockN), (uint)(M / BlockM), (uint)Batch, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int k, int n)
    {
        ValidateShape(m, k, n, 1);
        int kBytes = checked(k * sizeof(float));
        int nBytes = checked(n * sizeof(float));

        var ptx = new StringBuilder(24_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// batched-gemm-fanout M={m} K={k} N={n} tile={BlockM}x{BlockN}x{BlockK}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 a_ptr,");
        ptx.AppendLine("    .param .u64 b_ptr_array,");
        ptx.AppendLine("    .param .u64 c_ptr_array");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<40>;");
        ptx.AppendLine("    .reg .b64 %rd<48>;");
        ptx.AppendLine("    .reg .f32 %f<48>;");
        ptx.AppendLine($"    .shared .align 16 .b8 a_tile[{BlockM * BlockK * sizeof(float)}];");
        ptx.AppendLine($"    .shared .align 16 .b8 b_tile[{BlockK * BlockN * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [a_ptr];");             // shared A
        ptx.AppendLine("    ld.param.u64 %rd20, [b_ptr_array];");
        ptx.AppendLine("    ld.param.u64 %rd21, [c_ptr_array];");
        // Resolve this block's batch slice from ctaid.z.
        ptx.AppendLine("    mov.u32 %r22, %ctaid.z;");
        ptx.AppendLine("    mul.wide.u32 %rd22, %r22, 8;");
        ptx.AppendLine("    add.u64 %rd23, %rd20, %rd22;");
        ptx.AppendLine("    ld.global.nc.u64 %rd1, [%rd23];");         // B_z base
        ptx.AppendLine("    add.u64 %rd23, %rd21, %rd22;");
        ptx.AppendLine("    ld.global.nc.u64 %rd3, [%rd23];");         // C_z base
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
        EmitCooperativeLoad(ptx, kBytes, nBytes);
        ptx.AppendLine("    bar.sync 0;");
        EmitInnerAccumulate(ptx);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockK};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {k};");
        ptx.AppendLine("    @%p0 bra.uni K_TILE_LOOP;");
        EmitEpilogue(ptx, nBytes);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitCooperativeLoad(StringBuilder ptx, int kBytes, int nBytes)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            ptx.AppendLine($"    shr.u32 %r12, {e}, 3;");
            ptx.AppendLine($"    and.b32 %r13, {e}, 7;");
            ptx.AppendLine("    add.u32 %r14, %r5, %r12;");
            ptx.AppendLine("    add.u32 %r15, %r9, %r13;");
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
            ptx.AppendLine($"    shr.u32 %r16, {e}, 6;");
            ptx.AppendLine($"    and.b32 %r17, {e}, 63;");
            ptx.AppendLine("    add.u32 %r18, %r9, %r16;");
            ptx.AppendLine("    add.u32 %r19, %r6, %r17;");
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r18, {nBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r19, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
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

    private static void EmitEpilogue(StringBuilder ptx, int nBytes)
    {
        ptx.AppendLine("    add.u32 %r18, %r5, %r7;");
        ptx.AppendLine("    add.u32 %r19, %r6, %r8;");
        for (int i = 0; i < ThreadM; i++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r18, {i};");
            ptx.AppendLine($"    mul.wide.u32 %rd15, %r20, {nBytes};");
            ptx.AppendLine("    add.u64 %rd16, %rd3, %rd15;");
            for (int j = 0; j < ThreadN; j++)
            {
                int acc = i * ThreadN + j;
                ptx.AppendLine($"    add.u32 %r21, %r19, {j};");
                ptx.AppendLine("    mul.wide.u32 %rd17, %r21, 4;");
                ptx.AppendLine("    add.u64 %rd19, %rd16, %rd17;");
                ptx.AppendLine($"    st.global.f32 [%rd19], %f{acc};");
            }
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int m, int k, int n, int batch)
    {
        var a = new DirectPtxExtent(m, k);
        var b = new DirectPtxExtent(k, n);
        var output = new DirectPtxExtent(m, n);
        return new DirectPtxKernelBlueprint(
            Operation: "batched-gemm-fanout",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-k{k}-n{n}-batch{batch}",
            Tensors:
            [
                new("a", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    a, a, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("b_element", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    b, b, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("c_element", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 96,
                MaxStaticSharedBytes: (BlockM + BlockN) * BlockK * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "C_i[M,N] = A[M,K] @ B_i[K,N] for i in [0,batch)",
                ["batch-abi"] = "device-pointer-arrays-for-B-and-C-indexed-by-ctaid-z",
                ["shared-operand"] = "A-loaded-once-reused-across-batch",
                ["tile"] = $"{BlockM}x{BlockN}x{BlockK}-register-{ThreadM}x{ThreadN}",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int k, int n, int batch) =>
        batch is > 0 and <= 256 &&
        m > 0 && m % BlockM == 0 &&
        n > 0 && n % BlockN == 0 &&
        k > 0 && k % BlockK == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        k is 256 or 512 or 1024 or 2048 or 4096 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int k, int n, int batch) => false;

    private static void ValidateShape(int m, int k, int n, int batch)
    {
        if (!IsSupportedShape(m, k, n, batch))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "Batched fanout supports M in {64,128,256,512,1024,2048}, K/N in {256,512,1024,2048,4096}, batch 1..256.");
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
