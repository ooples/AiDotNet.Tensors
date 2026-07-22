#if NET5_0_OR_GREATER
using System;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted PTX Tensor-Core kernel for the Q*K^T stage of multi-head
/// attention. A and K are row-major FP16; output is row-major FP32.
/// </summary>
/// <remarks>
/// M, N, K, batch count, scale, strides, and loop trip counts are specialized
/// into the PTX. The execution path is Driver API only: PTX text -&gt;
/// cuModuleLoadData -&gt; cuLaunchKernel. No NVRTC or vendor math library is used.
/// </remarks>
internal sealed class PtxWmmaBatchedQkKernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_wmma_batched_qk";

    private readonly DirectPtxRuntime _runtime;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;
    private readonly int _blockTile;

    internal int M { get; }
    internal int N { get; }
    internal int K { get; }
    internal int BatchCount { get; }
    internal float Scale { get; }
    internal string Ptx { get; }

    internal PtxWmmaBatchedQkKernel(
        DirectPtxRuntime runtime, int m, int n, int k, int batchCount, float scale)
    {
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        if (m <= 0 || (m & 31) != 0) throw new ArgumentOutOfRangeException(nameof(m), "M must be a positive multiple of 32.");
        if (n <= 0 || (n & 31) != 0) throw new ArgumentOutOfRangeException(nameof(n), "N must be a positive multiple of 32.");
        if (k <= 0 || (k & 15) != 0) throw new ArgumentOutOfRangeException(nameof(k), "K must be a positive multiple of 16.");
        if (batchCount <= 0 || batchCount > 65535) throw new ArgumentOutOfRangeException(nameof(batchCount));
        if (!float.IsFinite(scale)) throw new ArgumentOutOfRangeException(nameof(scale));
        if (runtime.ComputeCapabilityMajor < 7)
            throw new NotSupportedException("WMMA Tensor Core PTX requires compute capability 7.0 or newer.");

        M = m;
        N = n;
        K = k;
        BatchCount = batchCount;
        Scale = scale;
        _blockTile = m % 64 == 0 && n % 64 == 0 ? 64 : 32;
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, m, n, k, scale);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint);
    }

    internal nuint ABytes => checked((nuint)BatchCount * (nuint)M * (nuint)K * sizeof(ushort));
    internal nuint BBytes => checked((nuint)BatchCount * (nuint)N * (nuint)K * sizeof(ushort));
    internal nuint CBytes => checked((nuint)BatchCount * (nuint)M * (nuint)N * sizeof(float));

    internal unsafe void Launch(DirectPtxBuffer a, DirectPtxBuffer b, DirectPtxBuffer c)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(c);
        if (a.ByteLength < ABytes || b.ByteLength < BBytes || c.ByteLength < CBytes)
            throw new ArgumentException("A PTX kernel buffer is smaller than its specialized tensor shape.");

        IntPtr aPointer = a.Pointer;
        IntPtr bPointer = b.Pointer;
        IntPtr cPointer = c.Pointer;
        void** args = stackalloc void*[3];
        args[0] = &aPointer;
        args[1] = &bPointer;
        args[2] = &cPointer;

        _module.Launch(
            _function,
            (uint)(N / _blockTile), (uint)(M / _blockTile), (uint)BatchCount,
            (uint)(_blockTile == 64 ? 512 : 128), 1, 1,
            0,
            args);
    }

    internal double EffectiveTflops(float milliseconds)
    {
        double flops = 2.0 * BatchCount * M * N * K;
        return flops / (milliseconds * 1e-3) / 1e12;
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int m, int n, int k, float scale)
    {
        if ((m & 31) != 0 || (n & 31) != 0 || (k & 15) != 0)
            throw new ArgumentException("M and N must be multiples of 32; K must be a multiple of 16.");

        // PTX 7.1 includes the sm_86 target used by Ampere consumer GPUs.
        // Target the actual device family so the Driver
        // JIT produces architecture-specific SASS for this context.
        string target = $"sm_{ccMajor}{ccMinor}";
        string scaleHex = "0f" + BitConverter.SingleToInt32Bits(scale).ToString("X8", CultureInfo.InvariantCulture);
        int aBatchBytes = checked(m * k * sizeof(ushort));
        int bBatchBytes = checked(n * k * sizeof(ushort));
        int cBatchBytes = checked(m * n * sizeof(float));
        int blockTile = m % 64 == 0 && n % 64 == 0 ? 64 : 32;
        int warpsPerAxis = blockTile / 16;
        int warpAxisShift = warpsPerAxis == 4 ? 2 : 1;
        int aBlockRowBytes = checked(blockTile * k * sizeof(ushort));
        int bBlockRowBytes = checked(blockTile * k * sizeof(ushort));
        int cBlockRowBytes = checked(blockTile * n * sizeof(float));
        int sharedOperandBytes = blockTile * 16 * sizeof(ushort);
        int chunksPerOperand = blockTile * 2;

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target {target}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 q_ptr,");
        ptx.AppendLine("    .param .u64 k_ptr,");
        ptx.AppendLine("    .param .u64 scores_ptr");
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
        ptx.AppendLine("    ld.param.u64 %rd0, [q_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [k_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [scores_ptr];");
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

        for (int tileK = 0; tileK < k; tileK += 16)
        {
            int kByteOffset = tileK * sizeof(ushort);
            string copyK = $"COPY_K_{tileK}";
            string copyDone = $"COPY_DONE_{tileK}";
            ptx.AppendLine();
            // Aligned 16-byte copies stage one blockTile x 16 tile from
            // each operand. The 64x64 variant uses 16 warps and reuses
            // every Q/K tile four times; its remaining threads skip copy.
            ptx.AppendLine($"    setp.ge.u32 %p1, %r0, {chunksPerOperand * 2};");
            ptx.AppendLine($"    @%p1 bra {copyDone};");
            ptx.AppendLine($"    setp.ge.u32 %p0, %r0, {chunksPerOperand};");
            ptx.AppendLine($"    @%p0 bra {copyK};");
            ptx.AppendLine("    shr.u32 %r7, %r0, 1;");
            ptx.AppendLine("    and.b32 %r8, %r0, 1;");
            ptx.AppendLine($"    mul.wide.u32 %rd12, %r7, {k * sizeof(ushort)};");
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
            ptx.AppendLine($"    mul.wide.u32 %rd12, %r8, {k * sizeof(ushort)};");
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
        for (int i = 0; i < 8; i++)
            ptx.AppendLine($"    mul.rn.f32 %d{i}, %d{i}, {scaleHex};");
        ptx.AppendLine("    wmma.store.d.sync.aligned.row.m16n16k16.global.f32 " +
            "[%rd8], {%d0,%d1,%d2,%d3,%d4,%d5,%d6,%d7}, " + n + ";");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }
}
#endif
