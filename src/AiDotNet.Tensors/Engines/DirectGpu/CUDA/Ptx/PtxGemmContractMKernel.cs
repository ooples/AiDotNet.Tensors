using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Weight-gradient GEMM <c>dW[N,K] = transpose(dZ[M,N]) @ X[M,K]</c>, i.e.
/// <c>dW[n,k] = sum_m dZ[m,n] * X[m,k]</c>, contracting over the batch/token dimension
/// M. This is the <c>dWeight</c> stage of the FusedLinear*Backward family and produces
/// weights in the same output-major [N,K] layout the forward tiles consume, so the
/// gradient can update the weight buffer in place. Both operands are staged in a
/// <c>[contract][out-index]</c> shared layout so the global loads stay coalesced (dZ is
/// contiguous over N, X over K), while the register-blocked inner product mirrors the
/// forward tiles.
///
/// Tile BM=BN=64 over (N,K), BK=8 over M, TM=TN=4 → 256 threads, 16 FP32
/// accumulators/thread, 4 KiB shared. Grid = (K/BN, N/BM).
/// </summary>
internal sealed class PtxGemmContractMKernel : IDisposable
{
    internal const int BlockN = 64; // output rows over N (dZ columns)
    internal const int BlockK = 64; // output cols over K (X columns)
    internal const int BlockM = 8;  // contract slice over M
    internal const int ThreadM = 4;
    internal const int ThreadN = 4;
    internal const int TileWidth = 64;
    internal const int BlockThreads = (BlockN / ThreadM) * (BlockK / ThreadN); // 256
    internal const string EntryPoint = "aidotnet_dweight_contract_m";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int M { get; }
    internal int N { get; }
    internal int K { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxGemmContractMKernel(DirectPtxRuntime runtime, int m, int n, int k)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in dWeight specialization is measured only on GA10x/SM86.");
        ValidateShape(m, n, k);
        M = m;
        N = n;
        K = k;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, m, n, k);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, n, k);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(
        DirectPtxTensorView dz, DirectPtxTensorView x, DirectPtxTensorView dw)
    {
        Require(dz, Blueprint.Tensors[0], nameof(dz));
        Require(x, Blueprint.Tensors[1], nameof(x));
        Require(dw, Blueprint.Tensors[2], nameof(dw));
        if (Overlaps(dw, dz) || Overlaps(dw, x))
            throw new ArgumentException("dWeight output may not alias dZ or X.");

        IntPtr dzPointer = dz.Pointer;
        IntPtr xPointer = x.Pointer;
        IntPtr dwPointer = dw.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &dzPointer;
        arguments[1] = &xPointer;
        arguments[2] = &dwPointer;
        _module.Launch(_function, (uint)(K / BlockK), (uint)(N / BlockN), 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n, int k)
    {
        ValidateShape(m, n, k);
        int nBytes = checked(n * sizeof(float)); // dZ row stride (over M)
        int kBytes = checked(k * sizeof(float)); // X row stride (over M) and dW row stride (over N)

        var ptx = new StringBuilder(24_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// dweight M={m} N={n} K={k} tile(N,K,M)={BlockN}x{BlockK}x{BlockM}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 dz_ptr,");
        ptx.AppendLine("    .param .u64 x_ptr,");
        ptx.AppendLine("    .param .u64 dw_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<40>;");
        ptx.AppendLine("    .reg .b64 %rd<48>;");
        ptx.AppendLine("    .reg .f32 %f<48>;");
        ptx.AppendLine($"    .shared .align 16 .b8 a_tile[{BlockM * TileWidth * sizeof(float)}];"); // dZ [M][N]
        ptx.AppendLine($"    .shared .align 16 .b8 b_tile[{BlockM * TileWidth * sizeof(float)}];"); // X  [M][K]
        ptx.AppendLine("    ld.param.u64 %rd0, [dz_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [x_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [dw_ptr];");
        ptx.AppendLine("    mov.u64 %rd4, a_tile;");
        ptx.AppendLine("    mov.u64 %rd5, b_tile;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                 // K block
        ptx.AppendLine("    mov.u32 %r2, %ctaid.y;");                 // N block
        ptx.AppendLine("    shr.u32 %r3, %r0, 4;");                   // threadRow (over N)
        ptx.AppendLine("    and.b32 %r4, %r0, 15;");                  // threadCol (over K)
        ptx.AppendLine($"    mul.lo.u32 %r5, %r2, {BlockN};");        // nBlock
        ptx.AppendLine($"    mul.lo.u32 %r6, %r1, {BlockK};");        // kBlock
        ptx.AppendLine($"    mul.lo.u32 %r7, %r3, {ThreadM};");       // threadRow*TM
        ptx.AppendLine($"    mul.lo.u32 %r8, %r4, {ThreadN};");       // threadCol*TN
        for (int i = 0; i < ThreadM * ThreadN; i++)
            ptx.AppendLine($"    mov.f32 %f{i}, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");                        // m0 contract counter
        ptx.AppendLine("M_TILE_LOOP:");
        EmitCooperativeLoad(ptx, nBytes, kBytes);
        ptx.AppendLine("    bar.sync 0;");
        EmitInnerAccumulate(ptx);
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine($"    add.u32 %r9, %r9, {BlockM};");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r9, {m};");
        ptx.AppendLine("    @%p0 bra.uni M_TILE_LOOP;");
        EmitEpilogue(ptx, kBytes);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // Load an 8(M)x64 slice of dZ (cols over N) and of X (cols over K), both in
    // [contract][out-index] layout so consecutive threads read consecutive N/K columns.
    private static void EmitCooperativeLoad(StringBuilder ptx, int nBytes, int kBytes)
    {
        for (int slot = 0; slot < 2; slot++)
        {
            string e = slot == 0 ? "%r10" : "%r11";
            ptx.AppendLine($"    add.u32 {e}, %r0, {slot * BlockThreads};");
            ptx.AppendLine($"    shr.u32 %r12, {e}, 6;");                   // contract c = e / 64
            ptx.AppendLine($"    and.b32 %r13, {e}, 63;");                  // out-index within tile
            ptx.AppendLine("    add.u32 %r14, %r9, %r12;");                 // global M row = m0 + c
            // dZ[m, nBlock + idx]
            ptx.AppendLine("    add.u32 %r15, %r5, %r13;");
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {nBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd0, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine($"    mul.wide.u32 %rd11, {e}, 4;");
            ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
            // X[m, kBlock + idx]
            ptx.AppendLine("    add.u32 %r15, %r6, %r13;");
            ptx.AppendLine($"    mul.wide.u32 %rd8, %r14, {kBytes};");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
            ptx.AppendLine("    add.u64 %rd10, %rd10, %rd9;");
            ptx.AppendLine("    ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine("    add.u64 %rd12, %rd5, %rd11;");
            ptx.AppendLine("    st.shared.f32 [%rd12], %f16;");
        }
    }

    // [contract][out-index] layout: fragment element at contract c = shared[c*64 + out].
    private static void EmitInnerAccumulate(StringBuilder ptx)
    {
        ptx.AppendLine("    mul.wide.u32 %rd13, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd13;");                   // &a_sh[threadRow*TM]
        ptx.AppendLine("    mul.wide.u32 %rd14, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd5, %rd14;");                   // &b_sh[threadCol*TN]
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

    // dW[n,k] row-major [N,K]: n = nBlock + threadRow*TM + i, k = kBlock + threadCol*TN + j.
    private static void EmitEpilogue(StringBuilder ptx, int kBytes)
    {
        ptx.AppendLine("    add.u32 %r18, %r5, %r7;");                       // n0
        ptx.AppendLine("    add.u32 %r19, %r6, %r8;");                       // k0
        for (int i = 0; i < ThreadM; i++)
        {
            ptx.AppendLine($"    add.u32 %r20, %r18, {i};");
            ptx.AppendLine($"    mul.wide.u32 %rd15, %r20, {kBytes};");
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
        DirectPtxArchitectureFamily architecture, int m, int n, int k)
    {
        var dz = new DirectPtxExtent(m, n);
        var x = new DirectPtxExtent(m, k);
        var dw = new DirectPtxExtent(n, k);
        return new DirectPtxKernelBlueprint(
            Operation: "dweight-contract-m",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-m{m}-n{n}-k{k}",
            Tensors:
            [
                new("dz", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    dz, dz, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("x", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    x, x, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("dw", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.LinearWeightOutputMajor,
                    dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 96,
                MaxStaticSharedBytes: 2 * BlockM * TileWidth * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "dW[N,K] = transpose(dZ[M,N]) @ X[M,K]",
                ["contract"] = "over-M-batch-token-dimension",
                ["dw-layout"] = "output-major-row-major-fp32",
                ["shared-layout"] = "contract-major-coalesced-global-load",
                ["tile"] = $"{BlockN}x{BlockK}x{BlockM}-register-{ThreadM}x{ThreadN}",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int m, int n, int k) =>
        m > 0 && m % BlockM == 0 &&
        n > 0 && n % BlockN == 0 &&
        k > 0 && k % BlockK == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        n is 256 or 512 or 1024 or 2048 or 4096 &&
        k is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int m, int n, int k) => false;

    private static void ValidateShape(int m, int n, int k)
    {
        if (!IsSupportedShape(m, n, k))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                "dWeight supports M in {64,128,256,512,1024,2048}, N/K in {256,512,1024,2048,4096}.");
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
