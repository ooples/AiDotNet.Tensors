#if NET5_0_OR_GREATER
using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Warp-per-row softmax used to construct a strong, decomposed cuBLAS
/// attention baseline for S in {16,32,64,128}. Input is FP32 and output is FP16.
/// </summary>
internal sealed class PtxAttentionSoftmax32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_attention_softmax_32";
    internal const int DefaultSequenceLength = 32;

    private readonly DirectPtxRuntime _runtime;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int BatchHeads { get; }
    internal int SequenceLength { get; }
    internal bool IsCausal { get; }
    internal nuint ScoreBytes => checked(
        (nuint)BatchHeads * (nuint)SequenceLength * (nuint)SequenceLength * sizeof(float));
    internal nuint ProbabilityBytes => checked(
        (nuint)BatchHeads * (nuint)SequenceLength * (nuint)SequenceLength * sizeof(ushort));
    internal string Ptx { get; }

    internal PtxAttentionSoftmax32Kernel(
        DirectPtxRuntime runtime, int batchHeads, bool isCausal,
        int sequenceLength = DefaultSequenceLength)
    {
        _runtime = runtime ?? throw new ArgumentNullException(nameof(runtime));
        if (batchHeads <= 0 || batchHeads > 65535) throw new ArgumentOutOfRangeException(nameof(batchHeads));
        if (!PtxOnlineFusedAttention128x64Kernel.IsSupportedSequenceLength(sequenceLength))
            throw new ArgumentOutOfRangeException(nameof(sequenceLength));

        BatchHeads = batchHeads;
        SequenceLength = sequenceLength;
        IsCausal = isCausal;
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            isCausal, sequenceLength);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint);
    }

    internal unsafe void Launch(DirectPtxBuffer scores, DirectPtxBuffer probabilities)
    {
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(probabilities);
        if (scores.ByteLength < ScoreBytes || probabilities.ByteLength < ProbabilityBytes)
            throw new ArgumentException("A softmax buffer is smaller than the specialized shape.");

        IntPtr scorePointer = scores.Pointer;
        IntPtr probabilityPointer = probabilities.Pointer;
        void** args = stackalloc void*[2];
        args[0] = &scorePointer;
        args[1] = &probabilityPointer;
        _module.Launch(
            _function,
            checked((uint)((BatchHeads * SequenceLength + 3) / 4)), 1, 1,
            128, 1, 1,
            0,
            args);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, bool isCausal,
        int sequenceLength = DefaultSequenceLength)
    {
        if (!PtxOnlineFusedAttention128x64Kernel.IsSupportedSequenceLength(sequenceLength))
            throw new ArgumentOutOfRangeException(nameof(sequenceLength));
        int valuesPerLane = (sequenceLength + 31) / 32;
        var ptx = new StringBuilder(16 * 1024);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 score_ptr,");
        ptx.AppendLine("    .param .u64 probability_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b16 %h<4>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [score_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [probability_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine("    mad.lo.u32 %r4, %r1, 4, %r2;"); // row
        if (isCausal)
            ptx.AppendLine($"    rem.u32 %r6, %r4, {sequenceLength};");
        ptx.AppendLine("    mov.f32 %f8, 0fFF800000;");
        for (int i = 0; i < valuesPerLane; i++)
        {
            int columnOffset = i * 32;
            ptx.AppendLine($"    add.u32 %r5, %r3, {columnOffset};");
            ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {sequenceLength};");
            ptx.AppendLine($"    mad.lo.u32 %r7, %r4, {sequenceLength}, %r5;");
            ptx.AppendLine("    mul.wide.u32 %rd2, %r7, 4;");
            ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
            ptx.AppendLine($"    @%p0 ld.global.f32 %f{i}, [%rd3];");
            ptx.AppendLine($"    @!%p0 mov.f32 %f{i}, 0fFF800000;");
            if (isCausal)
            {
                ptx.AppendLine("    setp.gt.u32 %p1, %r5, %r6;");
                ptx.AppendLine($"    @%p1 mov.f32 %f{i}, 0fFF800000;");
            }
            ptx.AppendLine($"    max.f32 %f8, %f8, %f{i};");
        }
        EmitShuffleReduction(ptx, "max.f32", "%f8");
        ptx.AppendLine("    mov.f32 %f9, 0f00000000;");
        for (int i = 0; i < valuesPerLane; i++)
        {
            ptx.AppendLine($"    sub.rn.f32 %f{i}, %f{i}, %f8;");
            ptx.AppendLine($"    mul.rn.f32 %f{i}, %f{i}, 0f3FB8AA3B;");
            ptx.AppendLine($"    ex2.approx.f32 %f{i}, %f{i};");
            ptx.AppendLine($"    add.rn.f32 %f9, %f9, %f{i};");
        }
        EmitShuffleReduction(ptx, "add.rn.f32", "%f9");
        ptx.AppendLine("    rcp.approx.f32 %f10, %f9;");
        for (int i = 0; i < valuesPerLane; i++)
        {
            int columnOffset = i * 32;
            ptx.AppendLine($"    add.u32 %r5, %r3, {columnOffset};");
            ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {sequenceLength};");
            ptx.AppendLine($"    mul.rn.f32 %f{i}, %f{i}, %f10;");
            ptx.AppendLine($"    cvt.rn.f16.f32 %h0, %f{i};");
            ptx.AppendLine($"    mad.lo.u32 %r7, %r4, {sequenceLength}, %r5;");
            ptx.AppendLine("    mul.wide.u32 %rd4, %r7, 2;");
            ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
            ptx.AppendLine("    @%p0 st.global.u16 [%rd5], %h0;");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitShuffleReduction(StringBuilder ptx, string operation, string accumulator)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r18, {accumulator};");
            ptx.AppendLine($"    shfl.sync.bfly.b32 %r19, %r18, {delta}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f18, %r19;");
            ptx.AppendLine($"    {operation} {accumulator}, {accumulator}, %f18;");
        }
    }
}
#endif
