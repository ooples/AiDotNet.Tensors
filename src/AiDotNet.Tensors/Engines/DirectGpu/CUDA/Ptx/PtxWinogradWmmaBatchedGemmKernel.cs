using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// FP16 Tensor-Core batched GEMM for the Winograd F(2,3) 3x3 convolution path.
/// Computes the 16 position GEMMs M[xi] = U[xi] * V[xi] on the Ampere Tensor
/// Cores via <c>wmma.mma.sync m16n16k16 f16 -&gt; f32</c>. Operands are the fp16
/// transforms: U[16, K, C] (A, row-major per position) and V[16, P, C] (B, stored
/// row-major so the col-major WMMA load yields V[xi]^T, giving M = U*V). Output
/// M[16, K, P] is fp32 and feeds the existing fp32 output transform unchanged.
/// One CTA computes a blockTile x blockTile output tile of one position (grid.z =
/// 16). This is the accuracy-matched fp16 counterpart of the fp32 batched GEMM;
/// the multiply-heavy GEMM stage runs on Tensor Cores while the numerically
/// sensitive transforms stay fp32.
/// </summary>
internal sealed class PtxWinogradWmmaBatchedGemmKernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_winograd_wmma_batched_gemm";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;
    private readonly int _blockTile;

    internal int OutputChannels { get; }   // M
    internal int InputChannels { get; }    // K (contraction)
    internal int Tiles { get; }            // N = P
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal long UBytes => (long)16 * OutputChannels * InputChannels * sizeof(ushort);
    internal long VBytes => (long)16 * Tiles * InputChannels * sizeof(ushort);
    internal long MBytes => (long)16 * OutputChannels * Tiles * sizeof(float);

    internal PtxWinogradWmmaBatchedGemmKernel(
        DirectPtxRuntime runtime, int outputChannels, int inputChannels, int tiles)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Winograd WMMA GEMM has no experimental non-SM86 specialization.");
        if (runtime.ComputeCapabilityMajor < 7)
            throw new NotSupportedException("WMMA Tensor Core PTX requires compute capability 7.0 or newer.");
        if (outputChannels <= 0 || (outputChannels & 31) != 0)
            throw new ArgumentOutOfRangeException(nameof(outputChannels), "K (output channels) must be a positive multiple of 32.");
        if (tiles <= 0 || (tiles & 31) != 0)
            throw new ArgumentOutOfRangeException(nameof(tiles), "P (tile count) must be a positive multiple of 32.");
        if (inputChannels <= 0 || (inputChannels & 15) != 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "C (input channels) must be a positive multiple of 16.");

        OutputChannels = outputChannels;
        InputChannels = inputChannels;
        Tiles = tiles;
        _blockTile = outputChannels % 64 == 0 && tiles % 64 == 0 ? 64 : 32;

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, outputChannels, inputChannels, tiles, _blockTile);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, outputChannels, inputChannels, tiles);
        _module = runtime.LoadModule(
            Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int blockThreads = _blockTile == 64 ? 512 : 128;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, blockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, blockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, blockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int k, int c, int p, int blockTile)
    {
        var u = new DirectPtxExtent(16, k, c);
        var v = new DirectPtxExtent(16, p, c);
        var m = new DirectPtxExtent(16, k, p);
        return new DirectPtxKernelBlueprint(
            Operation: "winograd-wmma-batched-gemm",
            Version: 1,
            Architecture: architecture,
            Variant: FormattableString.Invariant($"k{k}-c{c}-p{p}-m16n16k16-f16f32-tile{blockTile}"),
            Tensors:
            [
                new("U", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    u, u, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("V", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    v, v, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("M", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    m, m, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 96, MaxStaticSharedBytes: 8192,
                MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "M[xi] = U[xi] * V[xi] (16 batched GEMMs)",
                ["mma"] = "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32",
                ["operands"] = "A=U[16,K,C] fp16 row; B=V[16,P,C] fp16 (col-load = V^T); D=M[16,K,P] fp32",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView u, DirectPtxTensorView v, DirectPtxTensorView m)
    {
        Require(u, Blueprint.Tensors[0], nameof(u));
        Require(v, Blueprint.Tensors[1], nameof(v));
        Require(m, Blueprint.Tensors[2], nameof(m));
        IntPtr uPtr = u.Pointer, vPtr = v.Pointer, mPtr = m.Pointer;
        void** args = stackalloc void*[3];
        args[0] = &uPtr; args[1] = &vPtr; args[2] = &mPtr;
        _module.Launch(
            _function,
            (uint)(Tiles / _blockTile), (uint)(OutputChannels / _blockTile), 16,
            (uint)(_blockTile == 64 ? 512 : 128), 1, 1,
            0,
            args);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    // WMMA batched GEMM: M[xi][ko,pp] = sum_c U[xi][ko,c] * V[xi][pp,c]. A=U row-major
    // [K,C]; B=V row-major [P,C] loaded col-major (= V^T) so the mma yields U*V.
    // Mirrors the proven Q*K^T tensor-core loop with m=K, n=P, k=C, batch=16, no scale.
    internal static string EmitPtx(int major, int minor, int k, int c, int p)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 WMMA GEMM emitter exists.");
        if ((k & 31) != 0 || (p & 31) != 0 || (c & 15) != 0)
            throw new ArgumentException("K and P must be multiples of 32; C must be a multiple of 16.");

        int m = k;              // WMMA M axis = output channels
        int n = p;              // WMMA N axis = tiles
        int kk = c;             // WMMA K axis = input channels (contraction)
        int aBatchBytes = checked(m * kk * sizeof(ushort));
        int bBatchBytes = checked(n * kk * sizeof(ushort));
        int cBatchBytes = checked(m * n * sizeof(float));
        int blockTile = m % 64 == 0 && n % 64 == 0 ? 64 : 32;
        int warpsPerAxis = blockTile / 16;
        int warpAxisShift = warpsPerAxis == 4 ? 2 : 1;
        int aBlockRowBytes = checked(blockTile * kk * sizeof(ushort));
        int bBlockRowBytes = checked(blockTile * kk * sizeof(ushort));
        int cBlockRowBytes = checked(blockTile * n * sizeof(float));
        int sharedOperandBytes = blockTile * 16 * sizeof(ushort);
        int chunksPerOperand = blockTile * 2;

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{major}{minor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 u_ptr,");
        ptx.AppendLine("    .param .u64 v_ptr,");
        ptx.AppendLine("    .param .u64 m_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %a<8>;");
        ptx.AppendLine("    .reg .b32 %b<8>;");
        ptx.AppendLine("    .reg .f32 %d<8>;");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine($"    .shared .align 16 .b8 smem[{sharedOperandBytes * 2}];");
        ptx.AppendLine();
        ptx.AppendLine("    ld.param.u64 %rd0, [u_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [v_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [m_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r2, %ctaid.y;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.z;");
        ptx.AppendLine("    shr.u32 %r4, %r0, 5;");
        ptx.AppendLine($"    shr.u32 %r5, %r4, {warpAxisShift};");
        ptx.AppendLine($"    and.b32 %r6, %r4, {warpsPerAxis - 1};");
        ptx.AppendLine();
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r3, {aBatchBytes};");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r2, {aBlockRowBytes};");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd5, %rd4;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r3, {bBatchBytes};");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r1, {bBlockRowBytes};");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, %rd4;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r3, {cBatchBytes};");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r2, {cBlockRowBytes};");
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r1, {blockTile * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd3;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, %rd4;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, %rd7;");
        ptx.AppendLine($"    mul.wide.u32 %rd9, %r5, {16 * n * sizeof(float)};");
        ptx.AppendLine($"    mul.wide.u32 %rd10, %r6, {16 * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd8, %rd8, %rd9;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, %rd10;");
        ptx.AppendLine("    mov.u64 %rd11, smem;");
        ptx.AppendLine();
        for (int i = 0; i < 8; i++)
            ptx.AppendLine($"    mov.f32 %d{i}, 0f00000000;");

        for (int tileK = 0; tileK < kk; tileK += 16)
        {
            int kByteOffset = tileK * sizeof(ushort);
            string copyK = $"COPY_K_{tileK}";
            string copyDone = $"COPY_DONE_{tileK}";
            ptx.AppendLine();
            ptx.AppendLine($"    setp.ge.u32 %p1, %r0, {chunksPerOperand * 2};");
            ptx.AppendLine($"    @%p1 bra {copyDone};");
            ptx.AppendLine($"    setp.ge.u32 %p0, %r0, {chunksPerOperand};");
            ptx.AppendLine($"    @%p0 bra {copyK};");
            ptx.AppendLine("    shr.u32 %r7, %r0, 1;");
            ptx.AppendLine("    and.b32 %r8, %r0, 1;");
            ptx.AppendLine($"    mul.wide.u32 %rd12, %r7, {kk * sizeof(ushort)};");
            ptx.AppendLine("    add.u64 %rd12, %rd5, %rd12;");
            ptx.AppendLine($"    add.u64 %rd12, %rd12, {kByteOffset};");
            ptx.AppendLine("    mul.wide.u32 %rd13, %r8, 16;");
            ptx.AppendLine("    add.u64 %rd12, %rd12, %rd13;");
            ptx.AppendLine("    mul.wide.u32 %rd13, %r7, 32;");
            ptx.AppendLine("    add.u64 %rd13, %rd11, %rd13;");
            ptx.AppendLine("    mul.wide.u32 %rd14, %r8, 16;");
            ptx.AppendLine("    add.u64 %rd13, %rd13, %rd14;");
            ptx.AppendLine("    ld.global.v4.u32 {%r9,%r10,%r11,%r12}, [%rd12];");
            ptx.AppendLine("    st.shared.v4.u32 [%rd13], {%r9,%r10,%r11,%r12};");
            ptx.AppendLine($"    bra {copyDone};");
            ptx.AppendLine($"{copyK}:");
            ptx.AppendLine($"    sub.u32 %r7, %r0, {chunksPerOperand};");
            ptx.AppendLine("    shr.u32 %r8, %r7, 1;");
            ptx.AppendLine("    and.b32 %r7, %r7, 1;");
            ptx.AppendLine($"    mul.wide.u32 %rd12, %r8, {kk * sizeof(ushort)};");
            ptx.AppendLine("    add.u64 %rd12, %rd6, %rd12;");
            ptx.AppendLine($"    add.u64 %rd12, %rd12, {kByteOffset};");
            ptx.AppendLine("    mul.wide.u32 %rd13, %r7, 16;");
            ptx.AppendLine("    add.u64 %rd12, %rd12, %rd13;");
            ptx.AppendLine("    mul.wide.u32 %rd13, %r8, 32;");
            ptx.AppendLine($"    add.u64 %rd13, %rd13, {sharedOperandBytes};");
            ptx.AppendLine("    add.u64 %rd13, %rd11, %rd13;");
            ptx.AppendLine("    mul.wide.u32 %rd14, %r7, 16;");
            ptx.AppendLine("    add.u64 %rd13, %rd13, %rd14;");
            ptx.AppendLine("    ld.global.v4.u32 {%r9,%r10,%r11,%r12}, [%rd12];");
            ptx.AppendLine("    st.shared.v4.u32 [%rd13], {%r9,%r10,%r11,%r12};");
            ptx.AppendLine($"{copyDone}:");
            ptx.AppendLine("    bar.sync 0;");
            ptx.AppendLine("    mul.wide.u32 %rd15, %r5, 512;");
            ptx.AppendLine("    add.u64 %rd15, %rd11, %rd15;");
            ptx.AppendLine("    mul.wide.u32 %rd16, %r6, 512;");
            ptx.AppendLine($"    add.u64 %rd16, %rd16, {sharedOperandBytes};");
            ptx.AppendLine("    add.u64 %rd16, %rd11, %rd16;");
            ptx.AppendLine("    wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 " +
                "{%a0,%a1,%a2,%a3,%a4,%a5,%a6,%a7}, [%rd15], 16;");
            ptx.AppendLine("    wmma.load.b.sync.aligned.col.m16n16k16.shared.f16 " +
                "{%b0,%b1,%b2,%b3,%b4,%b5,%b6,%b7}, [%rd16], 16;");
            ptx.AppendLine("    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 " +
                "{%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, " +
                "{%a0,%a1,%a2,%a3,%a4,%a5,%a6,%a7}, " +
                "{%b0,%b1,%b2,%b3,%b4,%b5,%b6,%b7}, " +
                "{%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7};");
            ptx.AppendLine("    bar.sync 0;");
        }

        ptx.AppendLine();
        ptx.AppendLine("    wmma.store.d.sync.aligned.row.m16n16k16.global.f32 " +
            "[%rd8], {%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, " + n + ";");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
