using System;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Shape-specialized fused attention-forward proof for S=32, D=16.
/// Q/K/V are row-major FP16, output is row-major FP32. Scores and
/// probabilities exist only in shared memory; both Q*K^T and P*V use WMMA.
/// </summary>
internal sealed class PtxWmmaFusedAttention32x16Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_wmma_fused_attention_32x16";
    internal const int SequenceLength = 32;
    internal const int HeadDimension = 16;

    private const int InputBytesPerHead = SequenceLength * HeadDimension * sizeof(ushort);
    private const int OutputBytesPerHead = SequenceLength * HeadDimension * sizeof(float);

    private readonly DirectPtxRuntime _runtime;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchHeads { get; }
    internal bool IsCausal { get; }
    internal float Scale { get; }
    internal string Ptx { get; }

    internal nuint QBytes => checked((nuint)BatchHeads * InputBytesPerHead);
    internal nuint KBytes => QBytes;
    internal nuint VBytes => QBytes;
    internal nuint OutputBytes => checked((nuint)BatchHeads * OutputBytesPerHead);

    internal PtxWmmaFusedAttention32x16Kernel(
        DirectPtxRuntime runtime, int batchHeads, bool isCausal, float scale = 0.25f)
    {
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        if (batchHeads <= 0 || batchHeads > 65535) throw new ArgumentOutOfRangeException(nameof(batchHeads));
        if (!PtxCompat.IsFinite(scale)) throw new ArgumentOutOfRangeException(nameof(scale));
        if (runtime.ComputeCapabilityMajor < 7)
            throw new NotSupportedException("WMMA Tensor Core PTX requires compute capability 7.0 or newer.");

        BatchHeads = batchHeads;
        IsCausal = isCausal;
        Scale = scale;
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, isCausal, scale);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint);
    }

    internal unsafe void Launch(
        DirectPtxBuffer query, DirectPtxBuffer key, DirectPtxBuffer value, DirectPtxBuffer output)
    {
        PtxCompat.ThrowIfNull(query, nameof(query));
        PtxCompat.ThrowIfNull(key, nameof(key));
        PtxCompat.ThrowIfNull(value, nameof(value));
        PtxCompat.ThrowIfNull(output, nameof(output));
        if (query.ByteLength < QBytes || key.ByteLength < KBytes ||
            value.ByteLength < VBytes || output.ByteLength < OutputBytes)
            throw new ArgumentException("A fused-attention buffer is smaller than the specialized shape.");

        IntPtr qPointer = query.Pointer;
        IntPtr kPointer = key.Pointer;
        IntPtr vPointer = value.Pointer;
        IntPtr oPointer = output.Pointer;
        void** args = stackalloc void*[4];
        args[0] = &qPointer;
        args[1] = &kPointer;
        args[2] = &vPointer;
        args[3] = &oPointer;

        _module.Launch(
            _function,
            (uint)BatchHeads, 1, 1,
            128, 1, 1,
            0,
            args);
    }

    internal double EffectiveTflops(float milliseconds)
    {
        double flops = 4.0 * BatchHeads * SequenceLength * SequenceLength * HeadDimension;
        return flops / (milliseconds * 1e-3) / 1e12;
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, bool isCausal, float scale)
    {
        string target = $"sm_{ccMajor}{ccMinor}";
        string scaleHex = FloatLiteral(scale);
        const string log2E = "0f3FB8AA3B";
        const int qShared = 0;
        const int kShared = 1024;
        const int vShared = 2048;
        const int scoreShared = 3072;
        const int invSumShared = scoreShared + 2048;
        const int probabilityShared = 7168;
        const int sharedBytes = 9216;

        var ptx = new StringBuilder(24 * 1024);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target {target}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 q_ptr,");
        ptx.AppendLine("    .param .u64 k_ptr,");
        ptx.AppendLine("    .param .u64 v_ptr,");
        ptx.AppendLine("    .param .u64 o_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b16 %h<2>;");
        ptx.AppendLine("    .reg .b32 %a<8>;");
        ptx.AppendLine("    .reg .b32 %b<8>;");
        ptx.AppendLine("    .reg .f32 %d<8>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine($"    .shared .align 16 .b8 smem[{sharedBytes}];");
        ptx.AppendLine();
        ptx.AppendLine("    ld.param.u64 %rd0, [q_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [k_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [v_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [o_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    mov.u64 %rd4, smem;");
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r1, {InputBytesPerHead};");
        ptx.AppendLine("    add.u64 %rd0, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd1, %rd1, %rd5;");
        ptx.AppendLine("    add.u64 %rd2, %rd2, %rd5;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r1, {OutputBytesPerHead};");
        ptx.AppendLine("    add.u64 %rd3, %rd3, %rd6;");
        ptx.AppendLine();

        // Sixty-four aligned 16-byte chunks cover each 32x16 FP16 operand.
        ptx.AppendLine("    setp.ge.u32 %p0, %r0, 64;");
        ptx.AppendLine("    @%p0 bra LOAD_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r0, 16;");
        EmitVectorCopy(ptx, "%rd0", qShared);
        EmitVectorCopy(ptx, "%rd1", kShared);
        EmitVectorCopy(ptx, "%rd2", vShared);
        ptx.AppendLine("LOAD_DONE:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine();

        // Warps 0 and 1 compute the two 16-row score tiles. Each warp emits
        // two 16x16 fragments, covering all 32 keys.
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    setp.ge.u32 %p0, %r2, 2;");
        ptx.AppendLine("    @%p0 bra SCORE_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 512;");
        ptx.AppendLine($"    add.u64 %rd9, %rd4, {qShared};");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd8;");
        ptx.AppendLine($"    add.u64 %rd10, %rd4, {kShared};");
        ptx.AppendLine($"    add.u64 %rd11, %rd4, {scoreShared};");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r2, 2048;");
        ptx.AppendLine("    add.u64 %rd11, %rd11, %rd12;");
        ptx.AppendLine("    wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 " + Fragment("a") + ", [%rd9], 16;");
        EmitZeroAccumulator(ptx);
        ptx.AppendLine("    wmma.load.b.sync.aligned.col.m16n16k16.shared.f16 " + Fragment("b") + ", [%rd10], 16;");
        EmitMma(ptx, "row", "col");
        ptx.AppendLine("    wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [%rd11], " + Fragment("d") + ", 32;");
        EmitZeroAccumulator(ptx);
        ptx.AppendLine("    add.u64 %rd10, %rd10, 512;");
        ptx.AppendLine("    wmma.load.b.sync.aligned.col.m16n16k16.shared.f16 " + Fragment("b") + ", [%rd10], 16;");
        EmitMma(ptx, "row", "col");
        ptx.AppendLine("    add.u64 %rd11, %rd11, 64;");
        ptx.AppendLine("    wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [%rd11], " + Fragment("d") + ", 32;");
        ptx.AppendLine("SCORE_DONE:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine();

        // One thread owns each complete score row. Scores are scaled and
        // normalized in shared memory; the FP16 probability tile feeds the
        // second Tensor Core operation without a global-memory round trip.
        ptx.AppendLine("    setp.ge.u32 %p0, %r0, 32;");
        ptx.AppendLine("    @%p0 bra SOFTMAX_DONE;");
        ptx.AppendLine($"    add.u64 %rd10, %rd4, {scoreShared};");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r0, 128;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, %rd11;");
        ptx.AppendLine($"    add.u64 %rd12, %rd4, {probabilityShared};");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r0, 64;");
        ptx.AppendLine("    add.u64 %rd12, %rd12, %rd13;");
        ptx.AppendLine("    mov.f32 %f0, 0fFF800000;");
        for (int column = 0; column < SequenceLength; column++)
        {
            if (isCausal)
                ptx.AppendLine($"    setp.gt.u32 %p1, {column}, %r0;");
            ptx.AppendLine($"    ld.shared.f32 %f1, [%rd10+{column * sizeof(float)}];");
            ptx.AppendLine($"    mul.rn.f32 %f1, %f1, {scaleHex};");
            if (isCausal)
                ptx.AppendLine("    @!%p1 max.f32 %f0, %f0, %f1;");
            else
                ptx.AppendLine("    max.f32 %f0, %f0, %f1;");
        }
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        for (int column = 0; column < SequenceLength; column++)
        {
            if (isCausal)
            {
                ptx.AppendLine($"    setp.gt.u32 %p1, {column}, %r0;");
                ptx.AppendLine("    @%p1 mov.f32 %f2, 0f00000000;");
                ptx.AppendLine($"    @!%p1 ld.shared.f32 %f1, [%rd10+{column * sizeof(float)}];");
                ptx.AppendLine($"    @!%p1 mul.rn.f32 %f1, %f1, {scaleHex};");
                ptx.AppendLine("    @!%p1 sub.rn.f32 %f2, %f1, %f0;");
                ptx.AppendLine($"    @!%p1 mul.rn.f32 %f2, %f2, {log2E};");
                ptx.AppendLine("    @!%p1 ex2.approx.f32 %f2, %f2;");
            }
            else
            {
                ptx.AppendLine($"    ld.shared.f32 %f1, [%rd10+{column * sizeof(float)}];");
                ptx.AppendLine($"    mul.rn.f32 %f1, %f1, {scaleHex};");
                ptx.AppendLine("    sub.rn.f32 %f2, %f1, %f0;");
                ptx.AppendLine($"    mul.rn.f32 %f2, %f2, {log2E};");
                ptx.AppendLine("    ex2.approx.f32 %f2, %f2;");
            }
            ptx.AppendLine("    add.rn.f32 %f3, %f3, %f2;");
            ptx.AppendLine("    cvt.rn.f16.f32 %h0, %f2;");
            ptx.AppendLine($"    st.shared.u16 [%rd12+{column * sizeof(ushort)}], %h0;");
        }
        ptx.AppendLine("    rcp.approx.f32 %f4, %f3;");
        ptx.AppendLine($"    add.u64 %rd14, %rd4, {invSumShared};");
        ptx.AppendLine("    mul.wide.u32 %rd15, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd14, %rd15;");
        ptx.AppendLine("    st.shared.f32 [%rd14], %f4;");
        ptx.AppendLine("SOFTMAX_DONE:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine();

        // Warps 0 and 1 compute unnormalized P*V for 16 query rows each.
        // The normalization factor is applied during the final coalesced copy.
        ptx.AppendLine("    setp.ge.u32 %p0, %r2, 2;");
        ptx.AppendLine("    @%p0 bra PV_DONE;");
        ptx.AppendLine($"    add.u64 %rd9, %rd4, {probabilityShared};");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 1024;");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd8;");
        ptx.AppendLine($"    add.u64 %rd10, %rd4, {vShared};");
        EmitZeroAccumulator(ptx);
        ptx.AppendLine("    wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 " + Fragment("a") + ", [%rd9], 32;");
        ptx.AppendLine("    wmma.load.b.sync.aligned.row.m16n16k16.shared.f16 " + Fragment("b") + ", [%rd10], 16;");
        EmitMma(ptx, "row", "row");
        ptx.AppendLine("    add.u64 %rd9, %rd9, 32;");
        ptx.AppendLine("    add.u64 %rd10, %rd10, 512;");
        ptx.AppendLine("    wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 " + Fragment("a") + ", [%rd9], 32;");
        ptx.AppendLine("    wmma.load.b.sync.aligned.row.m16n16k16.shared.f16 " + Fragment("b") + ", [%rd10], 16;");
        EmitMma(ptx, "row", "row");
        ptx.AppendLine($"    add.u64 %rd11, %rd4, {scoreShared};");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r2, 1024;");
        ptx.AppendLine("    add.u64 %rd11, %rd11, %rd12;");
        ptx.AppendLine("    wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [%rd11], " + Fragment("d") + ", 16;");
        ptx.AppendLine("PV_DONE:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine();

        // Four coalesced passes write the 32x16 result. scoreShared is reused
        // as the P*V output scratch after all score consumers have finished.
        for (int pass = 0; pass < 4; pass++)
        {
            int add = pass * 128;
            if (add == 0) ptx.AppendLine("    mov.u32 %r3, %r0;");
            else ptx.AppendLine($"    add.u32 %r3, %r0, {add};");
            ptx.AppendLine("    shr.u32 %r4, %r3, 4;");
            ptx.AppendLine($"    add.u64 %rd9, %rd4, {scoreShared};");
            ptx.AppendLine("    mul.wide.u32 %rd10, %r3, 4;");
            ptx.AppendLine("    add.u64 %rd9, %rd9, %rd10;");
            ptx.AppendLine("    ld.shared.f32 %f1, [%rd9];");
            ptx.AppendLine($"    add.u64 %rd11, %rd4, {invSumShared};");
            ptx.AppendLine("    mul.wide.u32 %rd12, %r4, 4;");
            ptx.AppendLine("    add.u64 %rd11, %rd11, %rd12;");
            ptx.AppendLine("    ld.shared.f32 %f2, [%rd11];");
            ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f2;");
            ptx.AppendLine("    add.u64 %rd13, %rd3, %rd10;");
            ptx.AppendLine("    st.global.f32 [%rd13], %f1;");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitVectorCopy(StringBuilder ptx, string globalBase, int sharedOffset)
    {
        ptx.AppendLine($"    add.u64 %rd8, {globalBase}, %rd7;");
        ptx.AppendLine($"    add.u64 %rd9, %rd4, {sharedOffset};");
        ptx.AppendLine("    add.u64 %rd9, %rd9, %rd7;");
        ptx.AppendLine("    ld.global.v4.u32 {%r8,%r9,%r10,%r11}, [%rd8];");
        ptx.AppendLine("    st.shared.v4.u32 [%rd9], {%r8,%r9,%r10,%r11};");
    }

    private static void EmitZeroAccumulator(StringBuilder ptx)
    {
        for (int i = 0; i < 8; i++) ptx.AppendLine($"    mov.f32 %d{i}, 0f00000000;");
    }

    private static void EmitMma(StringBuilder ptx, string aLayout, string bLayout)
    {
        ptx.AppendLine($"    wmma.mma.sync.aligned.{aLayout}.{bLayout}.m16n16k16.f32.f32 " +
            Fragment("d") + ", " + Fragment("a") + ", " + Fragment("b") + ", " + Fragment("d") + ";");
    }

    private static string Fragment(string prefix)
    {
        var result = new StringBuilder("{");
        for (int i = 0; i < 8; i++)
        {
            if (i != 0) result.Append(',');
            result.Append('%').Append(prefix).Append(i);
        }
        return result.Append('}').ToString();
    }

    private static string FloatLiteral(float value) =>
        "0f" + PtxCompat.SingleToInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);
}
