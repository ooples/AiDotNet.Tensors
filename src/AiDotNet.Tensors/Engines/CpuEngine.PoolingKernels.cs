using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// One plane of the tiling 2x2 stride-2 max-pool backward (<see cref="MaxPool2DBackwardRecomputeInto{T}"/>), eight
    /// windows per AVX step: the two input rows are de-interleaved into the four taps, each window's winner is chosen
    /// with the scalar rule (start at float.MinValue, strict ordered greater-than in tap order, so the first maximum
    /// wins and NaN never does), and the four output cells are written as <c>0 + g</c> on the winner and +0 elsewhere,
    /// re-interleaved into the two rows. A window with no winner sends its gradient to plane cell 0, added in window
    /// order after its block is stored: the same additions in the same order as the scalar loop, so the plane is
    /// bit-identical. The columns past the last full block of eight windows run the scalar loop.
    /// </summary>
    [MethodImpl(Compatibility.MethodImplHelper.Hot)]
    private static unsafe void MaxPool2x2Stride2TilesBackwardPlane(
        float[] x, int xBase, float[] g, int gBase, float[] d, int dBase, int width, int outH, int outW)
    {
        fixed (float* px = &x[xBase])
        fixed (float* pg = &g[gBase])
        fixed (float* pd = &d[dBase])
        {
            for (int oh = 0; oh < outH; oh++)
            {
                float* x0 = px + 2 * oh * width, x1 = x0 + width;
                float* d0 = pd + 2 * oh * width, d1 = d0 + width;
                float* gr = pg + oh * outW;
                int ow = 0;
#if NET5_0_OR_GREATER
                if (Avx2.IsSupported)
                {
                    var minV = Vector256.Create(float.MinValue);
                    var zero = Vector256<float>.Zero;
                    var none = Vector256.Create(-1f);
                    var t0 = Vector256.Create(0f); var t1 = Vector256.Create(1f);
                    var t2 = Vector256.Create(2f); var t3 = Vector256.Create(3f);
                    for (; ow + 8 <= outW; ow += 8)
                    {
                        int c = 2 * ow;
                        var r0lo = Avx.LoadVector256(x0 + c);
                        var r0hi = Avx.LoadVector256(x0 + c + 8);
                        var r1lo = Avx.LoadVector256(x1 + c);
                        var r1hi = Avx.LoadVector256(x1 + c + 8);
                        var a = DeinterleaveEven(r0lo, r0hi);
                        var b = DeinterleaveOdd(r0lo, r0hi);
                        var cc = DeinterleaveEven(r1lo, r1hi);
                        var dd = DeinterleaveOdd(r1lo, r1hi);
                        var m = minV;
                        var sel = none;
                        var gt = Avx.Compare(a, m, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                        m = Avx.BlendVariable(m, a, gt); sel = Avx.BlendVariable(sel, t0, gt);
                        gt = Avx.Compare(b, m, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                        m = Avx.BlendVariable(m, b, gt); sel = Avx.BlendVariable(sel, t1, gt);
                        gt = Avx.Compare(cc, m, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                        m = Avx.BlendVariable(m, cc, gt); sel = Avx.BlendVariable(sel, t2, gt);
                        gt = Avx.Compare(dd, m, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                        sel = Avx.BlendVariable(sel, t3, gt);
                        var gv = Avx.Add(zero, Avx.LoadVector256(gr + ow));
                        var dA = Avx.And(gv, Avx.Compare(sel, t0, FloatComparisonMode.OrderedEqualNonSignaling));
                        var dB = Avx.And(gv, Avx.Compare(sel, t1, FloatComparisonMode.OrderedEqualNonSignaling));
                        var dC = Avx.And(gv, Avx.Compare(sel, t2, FloatComparisonMode.OrderedEqualNonSignaling));
                        var dD = Avx.And(gv, Avx.Compare(sel, t3, FloatComparisonMode.OrderedEqualNonSignaling));
                        StoreInterleaved(d0 + c, dA, dB);
                        StoreInterleaved(d1 + c, dC, dD);
                        int orphan = Avx.MoveMask(Avx.Compare(sel, none, FloatComparisonMode.OrderedEqualNonSignaling));
                        if (orphan != 0)
                            for (int lane = 0; lane < 8; lane++)
                                if ((orphan & (1 << lane)) != 0) pd[0] += gr[ow + lane];
                    }
                }
#endif
                for (; ow < outW; ow++)
                {
                    int iw0 = 2 * ow;
                    float maxVal = float.MinValue;
                    int winner = -1;
                    float v = x0[iw0]; if (v > maxVal) { maxVal = v; winner = 0; }
                    v = x0[iw0 + 1]; if (v > maxVal) { maxVal = v; winner = 1; }
                    v = x1[iw0]; if (v > maxVal) { maxVal = v; winner = 2; }
                    v = x1[iw0 + 1]; if (v > maxVal) { winner = 3; }
                    d0[iw0] = 0f; d0[iw0 + 1] = 0f; d1[iw0] = 0f; d1[iw0 + 1] = 0f;
                    float* target = winner switch { 0 => d0 + iw0, 1 => d0 + iw0 + 1, 2 => d1 + iw0, 3 => d1 + iw0 + 1, _ => pd };
                    *target += gr[ow];
                }
            }
        }
    }

#if NET5_0_OR_GREATER
    /// <summary>Elements 0, 2, 4, ... of the 16 floats <paramref name="lo"/>:<paramref name="hi"/>, in order.</summary>
    [MethodImpl(Compatibility.MethodImplHelper.HotInline)]
    private static Vector256<float> DeinterleaveEven(Vector256<float> lo, Vector256<float> hi)
        => Avx2.Permute4x64(Avx.Shuffle(lo, hi, 0b10_00_10_00).AsDouble(), 0b11_01_10_00).AsSingle();

    /// <summary>Elements 1, 3, 5, ... of the 16 floats <paramref name="lo"/>:<paramref name="hi"/>, in order.</summary>
    [MethodImpl(Compatibility.MethodImplHelper.HotInline)]
    private static Vector256<float> DeinterleaveOdd(Vector256<float> lo, Vector256<float> hi)
        => Avx2.Permute4x64(Avx.Shuffle(lo, hi, 0b11_01_11_01).AsDouble(), 0b11_01_10_00).AsSingle();

    /// <summary>Stores a0 b0 a1 b1 ... a7 b7 (16 floats) at <paramref name="dst"/>.</summary>
    [MethodImpl(Compatibility.MethodImplHelper.HotInline)]
    private static unsafe void StoreInterleaved(float* dst, Vector256<float> a, Vector256<float> b)
    {
        var lo = Avx.UnpackLow(a, b);
        var hi = Avx.UnpackHigh(a, b);
        Avx.Store(dst, Avx.Permute2x128(lo, hi, 0x20));
        Avx.Store(dst + 8, Avx.Permute2x128(lo, hi, 0x31));
    }
#endif

    /// <summary>
    /// One output row of the unpadded 2x2 stride-2 max pool: <c>dst[ow] = max</c> over the window at columns
    /// 2ow, 2ow+1 of input rows <paramref name="r0"/> and <paramref name="r1"/>, eight windows per AVX step. Same rule
    /// as the scalar kernel (start from the window's first tap, take a later tap only when strictly greater, so a NaN
    /// first tap stays and a later NaN never replaces), so the output is bit-identical.
    /// </summary>
    [MethodImpl(Compatibility.MethodImplHelper.Hot)]
    internal static unsafe void MaxPool2x2Stride2Row(float* r0, float* r1, float* dst, int outW)
    {
        int ow = 0;
#if NET5_0_OR_GREATER
        if (Avx2.IsSupported)
        {
            for (; ow + 8 <= outW; ow += 8)
            {
                int c = 2 * ow;
                var r0lo = Avx.LoadVector256(r0 + c);
                var r0hi = Avx.LoadVector256(r0 + c + 8);
                var r1lo = Avx.LoadVector256(r1 + c);
                var r1hi = Avx.LoadVector256(r1 + c + 8);
                var m = DeinterleaveEven(r0lo, r0hi);
                var v = DeinterleaveOdd(r0lo, r0hi);
                m = Avx.BlendVariable(m, v, Avx.Compare(v, m, FloatComparisonMode.OrderedGreaterThanNonSignaling));
                v = DeinterleaveEven(r1lo, r1hi);
                m = Avx.BlendVariable(m, v, Avx.Compare(v, m, FloatComparisonMode.OrderedGreaterThanNonSignaling));
                v = DeinterleaveOdd(r1lo, r1hi);
                m = Avx.BlendVariable(m, v, Avx.Compare(v, m, FloatComparisonMode.OrderedGreaterThanNonSignaling));
                Avx.Store(dst + ow, m);
            }
        }
#endif
        for (; ow < outW; ow++)
        {
            int iw = 2 * ow;
            float m = r0[iw];
            float v = r0[iw + 1]; if (v > m) m = v;
            v = r1[iw]; if (v > m) m = v;
            v = r1[iw + 1]; if (v > m) m = v;
            dst[ow] = m;
        }
    }

    /// <summary>
    /// Adaptive-pool window bounds for one axis: for output index <c>o</c> of <paramref name="outSize"/>,
    /// <c>bins[2o] = floor(o * inSize / outSize)</c> and <c>bins[2o + 1] = ceil((o + 1) * inSize / outSize)</c>,
    /// computed in double exactly as the per-element loops computed them.
    /// </summary>
    internal static int[] AdaptivePoolBins(int inSize, int outSize)
    {
        var bins = new int[2 * outSize];
        for (int o = 0; o < outSize; o++)
        {
            bins[2 * o] = (int)Math.Floor((double)o * inSize / outSize);
            bins[2 * o + 1] = (int)Math.Ceiling((double)(o + 1) * inSize / outSize);
        }
        return bins;
    }

    /// <summary>
    /// Float adaptive average pool of <paramref name="planes"/> contiguous [iH, iW] planes into [oH, oW] planes. Each
    /// output is <c>sum / count</c>, the sum running from +0 over its window's rows, then columns, and <c>count</c>
    /// the window's element count: the order and arithmetic of the scalar bin loop, so the result is bit-identical
    /// to it. The window bounds are computed once per call rather than per output, the planes run in contiguous
    /// runs (a few per pool participant) instead of one dispatch each, and the loops index through pointers.
    /// </summary>
    internal static unsafe void AdaptiveAvgPool2DFloat(
        float[] input, int inOff, float[] output, int outOff, int planes, int iH, int iW, int oH, int oW)
    {
        if (planes <= 0 || oH <= 0 || oW <= 0) return;
        if (iH <= 0 || iW <= 0)
            throw new ArgumentException("Adaptive average pooling needs a non-empty input plane.");
        var hBins = AdaptivePoolBins(iH, oH);
        var wBins = AdaptivePoolBins(iW, oW);
        int inPlane = iH * iW, outPlane = oH * oW;
        if (inOff < 0 || outOff < 0
            || (long)inOff + (long)planes * inPlane > input.Length
            || (long)outOff + (long)planes * outPlane > output.Length)
            throw new ArgumentException("Adaptive average pool buffers are smaller than the planes they hold.");
        int tasks = AdaptivePoolTaskCount(planes, inPlane);
        CpuParallelSettings.ParallelForOrSerial(0, tasks, (long)planes * inPlane, [MethodImpl(Compatibility.MethodImplHelper.Hot)] (int task) =>
        {
            int p0 = (int)((long)task * planes / tasks), p1 = (int)((long)(task + 1) * planes / tasks);
            fixed (float* pin = input)
            fixed (float* pout = output)
            fixed (int* hb = hBins)
            fixed (int* wb = wBins)
            {
                for (int p = p0; p < p1; p++)
                {
                    float* src = pin + inOff + (long)p * inPlane;
                    float* dst = pout + outOff + (long)p * outPlane;
                    for (int oh = 0; oh < oH; oh++)
                    {
                        int hs = hb[2 * oh], he = hb[2 * oh + 1];
                        for (int ow = 0; ow < oW; ow++)
                        {
                            int ws = wb[2 * ow], we = wb[2 * ow + 1];
                            float sum = 0f;
                            for (int ih = hs; ih < he; ih++)
                            {
                                float* row = src + ih * iW;
                                for (int iw = ws; iw < we; iw++) sum += row[iw];
                            }
                            dst[oh * oW + ow] = sum / ((he - hs) * (we - ws));
                        }
                    }
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>
    /// Float adaptive average pool backward over <paramref name="planes"/> planes: each output gradient is divided by
    /// its window's element count and added to every cell of its window, in output order (rows, then columns), onto
    /// a plane that starts at +0 -- the additions and order of the scalar loop, so overlapping windows sum to the
    /// identical value. Overwrites the input-gradient planes; with <paramref name="accumulate"/> each plane is built
    /// in scratch and then added to the existing gradient once, as accumulating a separately computed gradient does.
    /// </summary>
    internal static unsafe void AdaptiveAvgPool2DBackwardFloat(
        float[] gradOutput, int gOff, float[] gradInput, int giOff, int planes, int iH, int iW, int oH, int oW,
        bool accumulate)
    {
        if (planes <= 0 || iH <= 0 || iW <= 0) return;
        if (oH <= 0 || oW <= 0)
            throw new ArgumentException("Adaptive average pooling needs a non-empty output plane.");
        var hBins = AdaptivePoolBins(iH, oH);
        var wBins = AdaptivePoolBins(iW, oW);
        int inPlane = iH * iW, outPlane = oH * oW;
        if (gOff < 0 || giOff < 0
            || (long)giOff + (long)planes * inPlane > gradInput.Length
            || (long)gOff + (long)planes * outPlane > gradOutput.Length)
            throw new ArgumentException("Adaptive average pool gradient buffers are smaller than the planes they hold.");
        int tasks = AdaptivePoolTaskCount(planes, inPlane);
        CpuParallelSettings.ParallelForOrSerial(0, tasks, (long)planes * inPlane, [MethodImpl(Compatibility.MethodImplHelper.Hot)] (int task) =>
        {
            int p0 = (int)((long)task * planes / tasks), p1 = (int)((long)(task + 1) * planes / tasks);
            float[] scratch = accumulate ? System.Buffers.ArrayPool<float>.Shared.Rent(inPlane) : gradInput;
            try
            {
                fixed (float* pg = gradOutput)
                fixed (float* pd = gradInput)
                fixed (float* ps = scratch)
                fixed (int* hb = hBins)
                fixed (int* wb = wBins)
                {
                    for (int p = p0; p < p1; p++)
                    {
                        float* src = pg + gOff + (long)p * outPlane;
                        float* target = pd + giOff + (long)p * inPlane;
                        float* dst = accumulate ? ps : target;
                        AdaptiveAvgPoolBackwardPlane(src, dst, iH, iW, oH, oW, hb, wb);
                        if (accumulate)
                            for (int i = 0; i < inPlane; i++) target[i] += dst[i];
                    }
                }
            }
            finally
            {
                if (accumulate) System.Buffers.ArrayPool<float>.Shared.Return(scratch);
            }
        }, deterministicSafe: true);
    }

    /// <summary>
    /// One plane of <see cref="AdaptiveAvgPool2DBackwardFloat"/>: clears <paramref name="dst"/> ([iH, iW]) to +0, then
    /// adds each output gradient divided by its window's element count to every cell of its window, in output order
    /// (rows, then columns). <paramref name="hb"/> / <paramref name="wb"/> are <see cref="AdaptivePoolBins"/>.
    /// </summary>
    [MethodImpl(Compatibility.MethodImplHelper.Hot)]
    private static unsafe void AdaptiveAvgPoolBackwardPlane(
        float* src, float* dst, int iH, int iW, int oH, int oW, int* hb, int* wb)
    {
        new Span<float>(dst, iH * iW).Clear();
        for (int oh = 0; oh < oH; oh++)
        {
            int hs = hb[2 * oh], he = hb[2 * oh + 1];
            for (int ow = 0; ow < oW; ow++)
            {
                int ws = wb[2 * ow], we = wb[2 * ow + 1];
                float g = src[oh * oW + ow] / ((he - hs) * (we - ws));
                for (int ih = hs; ih < he; ih++)
                {
                    float* row = dst + ih * iW;
                    for (int iw = ws; iw < we; iw++) row[iw] += g;
                }
            }
        }
    }

    /// <summary>
    /// The backward of <c>AdaptiveAvgPool2D(ReLU(z))</c> down to <c>z</c> and the bias, for a fused conv chain whose
    /// activation feeds only an adaptive average pool: per channel, per image, the pool gradient is spread over the
    /// dz plane and the ReLU mask and bias reduction run over that plane while it is in cache, so the activation's
    /// gradient is never written to memory and re-read.
    /// </summary>
    /// <remarks>
    /// Bit-identical to <see cref="AdaptiveAvgPool2DBackwardFloat"/> (not accumulating) into a separate buffer
    /// followed by <see cref="ChannelBiasActivationBackwardInto"/> with ReLU: the same per-plane code
    /// (<see cref="AdaptiveAvgPoolBackwardPlane"/>, <see cref="ReluBiasBackwardPlane"/>), and each channel's bias sum
    /// runs over its planes in batch order.
    /// </remarks>
    internal unsafe void AdaptiveAvgPoolReluBiasBackwardInto(
        Tensor<float> gradInput, Tensor<float> gradBias, Tensor<float> gradPoolOutput, Tensor<float> activation,
        bool accumulateBias)
    {
        if (gradInput == null) throw new ArgumentNullException(nameof(gradInput));
        if (gradBias == null) throw new ArgumentNullException(nameof(gradBias));
        if (gradPoolOutput == null) throw new ArgumentNullException(nameof(gradPoolOutput));
        if (activation == null) throw new ArgumentNullException(nameof(activation));
        if (activation.Rank != 4 || gradPoolOutput.Rank != 4)
            throw new ArgumentException("AdaptiveAvgPoolReluBiasBackwardInto needs rank-4 tensors.");
        int batch = activation._shape[0], channels = activation._shape[1];
        int iH = activation._shape[2], iW = activation._shape[3];
        int oH = gradPoolOutput._shape[2], oW = gradPoolOutput._shape[3];
        if (gradPoolOutput._shape[0] != batch || gradPoolOutput._shape[1] != channels || oH <= 0 || oW <= 0
            || iH <= 0 || iW <= 0 || gradInput.Length != activation.Length || gradBias.Length != channels)
            throw new ArgumentException("AdaptiveAvgPoolReluBiasBackwardInto: the pool gradient, input gradient and bias gradient do not match the activation.");
        var y = activation.GetCpuBackingForStridedRead(out int yOff);
        var g = gradPoolOutput.GetCpuBackingForStridedRead(out int gOff);
        var dz = gradInput.GetCpuBackingForContiguousWrite(out int dzOff);
        var db = gradBias.GetCpuBackingForContiguousWrite(out int dbOff);
        if (y is null || g is null || dz is null || db is null || !activation.IsContiguous || !gradPoolOutput.IsContiguous)
            throw new ArgumentException("AdaptiveAvgPoolReluBiasBackwardInto needs contiguous CPU tensors.");
        var hBins = AdaptivePoolBins(iH, oH);
        var wBins = AdaptivePoolBins(iW, oW);
        int inPlane = iH * iW, outPlane = oH * oW;
        if (channels == 0) return;
        CpuParallelSettings.ParallelForOrSerial(0, channels, (long)batch * channels * inPlane, [MethodImpl(Compatibility.MethodImplHelper.Hot)] (int c) =>
        {
            float acc = 0f;
            fixed (float* py0 = y)
            fixed (float* pg0 = g)
            fixed (float* pz0 = dz)
            fixed (int* hb = hBins)
            fixed (int* wb = wBins)
            {
                for (int n = 0; n < batch; n++)
                {
                    int plane = n * channels + c;
                    float* pz = pz0 + dzOff + (long)plane * inPlane;
                    AdaptiveAvgPoolBackwardPlane(pg0 + gOff + (long)plane * outPlane, pz, iH, iW, oH, oW, hb, wb);
                    ReluBiasBackwardPlane(pz, py0 + yOff + (long)plane * inPlane, pz, inPlane, relu: true, ref acc);
                }
            }
            db[dbOff + c] = accumulateBias ? db[dbOff + c] + acc : acc;
        });
    }

    /// <summary>
    /// Task count for the adaptive pool kernels: about four contiguous plane runs per pool participant, and no run
    /// under ~4K input elements, so a tiny pool stays on one task.
    /// </summary>
    private static int AdaptivePoolTaskCount(int planes, int inPlane)
    {
        long participants = Math.Max(1, CpuParallelSettings.MaxDegreeOfParallelism);
        long byWork = Math.Max(1, (long)planes * inPlane / 4096);
        return (int)Math.Max(1, Math.Min(planes, Math.Min(4 * participants, byWork)));
    }
}