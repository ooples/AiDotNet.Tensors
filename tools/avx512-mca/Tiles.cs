using System.Runtime.Intrinsics;

// Hand-reduced copies of DirectConvAvx2's AVX-512 weight-gradient inner loop, before (#1445: two 256-bit loads
// joined per 16 lanes) and after (one 512-bit load). Not verbatim: each keeps the shipped loop's per-iteration loads,
// joins and FMAs, but the surrounding tap loops are flattened or fixed (JoinedThree fixes columns at 3 and takes
// planeIn; the sextets run one loop over n). They drop [MethodImpl(AggressiveOptimization)] so crossgen2 precompiles
// them.
public static unsafe class Tiles
{
    const int Block = 8;

    // ForwardSextetRun512's innermost loop (one input lane per iteration: two 16-lane weight vectors x six positions),
    // with the AVX2 blocked weights (two 256-bit loads joined per vector) and with tile-interleaved weights.
    public static void SextetBlocked(float* w, float* p0, float* p1, float* p2, float* p3, float* p4, float* p5,
        long blockStride, int n, float* slot)
    {
        var a0l = Vector512.Load(slot); var a0h = Vector512.Load(slot + 16); var a1l = Vector512.Load(slot + 32); var a1h = Vector512.Load(slot + 48);
        var a2l = Vector512.Load(slot + 64); var a2h = Vector512.Load(slot + 80); var a3l = Vector512.Load(slot + 96); var a3h = Vector512.Load(slot + 112);
        var a4l = Vector512.Load(slot + 128); var a4h = Vector512.Load(slot + 144); var a5l = Vector512.Load(slot + 160); var a5h = Vector512.Load(slot + 176);
        long stride2 = 2 * blockStride, stride3 = 3 * blockStride;
        for (int c = 0; c < n; c++)
        {
            var wl = Vector512.Create(Vector256.Load(w), Vector256.Load(w + blockStride));
            var wh = Vector512.Create(Vector256.Load(w + stride2), Vector256.Load(w + stride3));
            var b = Vector512.Create(p0[c]); a0l = Vector512.FusedMultiplyAdd(b, wl, a0l); a0h = Vector512.FusedMultiplyAdd(b, wh, a0h);
            b = Vector512.Create(p1[c]); a1l = Vector512.FusedMultiplyAdd(b, wl, a1l); a1h = Vector512.FusedMultiplyAdd(b, wh, a1h);
            b = Vector512.Create(p2[c]); a2l = Vector512.FusedMultiplyAdd(b, wl, a2l); a2h = Vector512.FusedMultiplyAdd(b, wh, a2h);
            b = Vector512.Create(p3[c]); a3l = Vector512.FusedMultiplyAdd(b, wl, a3l); a3h = Vector512.FusedMultiplyAdd(b, wh, a3h);
            b = Vector512.Create(p4[c]); a4l = Vector512.FusedMultiplyAdd(b, wl, a4l); a4h = Vector512.FusedMultiplyAdd(b, wh, a4h);
            b = Vector512.Create(p5[c]); a5l = Vector512.FusedMultiplyAdd(b, wl, a5l); a5h = Vector512.FusedMultiplyAdd(b, wh, a5h);
            w += Block;
        }
        a0l.Store(slot); a0h.Store(slot + 16); a1l.Store(slot + 32); a1h.Store(slot + 48); a2l.Store(slot + 64); a2h.Store(slot + 80);
        a3l.Store(slot + 96); a3h.Store(slot + 112); a4l.Store(slot + 128); a4h.Store(slot + 144); a5l.Store(slot + 160); a5h.Store(slot + 176);
    }

    public static void SextetInterleaved(float* w, float* p0, float* p1, float* p2, float* p3, float* p4, float* p5,
        long blockStride, int n, float* slot)
    {
        var a0l = Vector512.Load(slot); var a0h = Vector512.Load(slot + 16); var a1l = Vector512.Load(slot + 32); var a1h = Vector512.Load(slot + 48);
        var a2l = Vector512.Load(slot + 64); var a2h = Vector512.Load(slot + 80); var a3l = Vector512.Load(slot + 96); var a3h = Vector512.Load(slot + 112);
        var a4l = Vector512.Load(slot + 128); var a4h = Vector512.Load(slot + 144); var a5l = Vector512.Load(slot + 160); var a5h = Vector512.Load(slot + 176);
        for (int c = 0; c < n; c++)
        {
            var wl = Vector512.Load(w);
            var wh = Vector512.Load(w + 2 * Block);
            var b = Vector512.Create(p0[c]); a0l = Vector512.FusedMultiplyAdd(b, wl, a0l); a0h = Vector512.FusedMultiplyAdd(b, wh, a0h);
            b = Vector512.Create(p1[c]); a1l = Vector512.FusedMultiplyAdd(b, wl, a1l); a1h = Vector512.FusedMultiplyAdd(b, wh, a1h);
            b = Vector512.Create(p2[c]); a2l = Vector512.FusedMultiplyAdd(b, wl, a2l); a2h = Vector512.FusedMultiplyAdd(b, wh, a2h);
            b = Vector512.Create(p3[c]); a3l = Vector512.FusedMultiplyAdd(b, wl, a3l); a3h = Vector512.FusedMultiplyAdd(b, wh, a3h);
            b = Vector512.Create(p4[c]); a4l = Vector512.FusedMultiplyAdd(b, wl, a4l); a4h = Vector512.FusedMultiplyAdd(b, wh, a4h);
            b = Vector512.Create(p5[c]); a5l = Vector512.FusedMultiplyAdd(b, wl, a5l); a5h = Vector512.FusedMultiplyAdd(b, wh, a5h);
            w += Block * 4;
        }
        a0l.Store(slot); a0h.Store(slot + 16); a1l.Store(slot + 32); a1h.Store(slot + 48); a2l.Store(slot + 64); a2h.Store(slot + 80);
        a3l.Store(slot + 96); a3h.Store(slot + 112); a4l.Store(slot + 128); a4h.Store(slot + 144); a5l.Store(slot + 160); a5h.Store(slot + 176);
    }

    public static void JoinedThree(float* park, float* xTap, float* gq, int* oImage, int q0, int q1, int step, int planeIn)
    {
        var a00 = Vector512.Load(park); var a01 = Vector512.Load(park + 16); var a02 = Vector512.Load(park + 32);
        var a10 = Vector512.Load(park + 48); var a11 = Vector512.Load(park + 64); var a12 = Vector512.Load(park + 80);
        var a20 = Vector512.Load(park + 96); var a21 = Vector512.Load(park + 112); var a22 = Vector512.Load(park + 128);
        var a30 = Vector512.Load(park + 144); var a31 = Vector512.Load(park + 160); var a32 = Vector512.Load(park + 176);
        var a40 = Vector512.Load(park + 192); var a41 = Vector512.Load(park + 208); var a42 = Vector512.Load(park + 224);
        var a50 = Vector512.Load(park + 240); var a51 = Vector512.Load(park + 256); var a52 = Vector512.Load(park + 272);
        var a60 = Vector512.Load(park + 288); var a61 = Vector512.Load(park + 304); var a62 = Vector512.Load(park + 320);
        var a70 = Vector512.Load(park + 336); var a71 = Vector512.Load(park + 352); var a72 = Vector512.Load(park + 368);
        int columns = 3;
        for (int q = q0; q < q1; q++, gq += Block)
        {
            float* xs = xTap + oImage[q];
            var x0v = Vector512.Create(Vector256.Load(xs), Vector256.Load(xs + planeIn));
            var x1v = columns > 1 ? Vector512.Create(Vector256.Load(xs + step), Vector256.Load(xs + step + planeIn)) : Vector512<float>.Zero;
            var x2v = columns > 2 ? Vector512.Create(Vector256.Load(xs + 2 * step), Vector256.Load(xs + 2 * step + planeIn)) : Vector512<float>.Zero;
            var g = Vector512.Create(gq[0]);
            a00 = Vector512.FusedMultiplyAdd(x0v, g, a00); a01 = Vector512.FusedMultiplyAdd(x1v, g, a01); a02 = Vector512.FusedMultiplyAdd(x2v, g, a02);
            g = Vector512.Create(gq[1]);
            a10 = Vector512.FusedMultiplyAdd(x0v, g, a10); a11 = Vector512.FusedMultiplyAdd(x1v, g, a11); a12 = Vector512.FusedMultiplyAdd(x2v, g, a12);
            g = Vector512.Create(gq[2]);
            a20 = Vector512.FusedMultiplyAdd(x0v, g, a20); a21 = Vector512.FusedMultiplyAdd(x1v, g, a21); a22 = Vector512.FusedMultiplyAdd(x2v, g, a22);
            g = Vector512.Create(gq[3]);
            a30 = Vector512.FusedMultiplyAdd(x0v, g, a30); a31 = Vector512.FusedMultiplyAdd(x1v, g, a31); a32 = Vector512.FusedMultiplyAdd(x2v, g, a32);
            g = Vector512.Create(gq[4]);
            a40 = Vector512.FusedMultiplyAdd(x0v, g, a40); a41 = Vector512.FusedMultiplyAdd(x1v, g, a41); a42 = Vector512.FusedMultiplyAdd(x2v, g, a42);
            g = Vector512.Create(gq[5]);
            a50 = Vector512.FusedMultiplyAdd(x0v, g, a50); a51 = Vector512.FusedMultiplyAdd(x1v, g, a51); a52 = Vector512.FusedMultiplyAdd(x2v, g, a52);
            g = Vector512.Create(gq[6]);
            a60 = Vector512.FusedMultiplyAdd(x0v, g, a60); a61 = Vector512.FusedMultiplyAdd(x1v, g, a61); a62 = Vector512.FusedMultiplyAdd(x2v, g, a62);
            g = Vector512.Create(gq[7]);
            a70 = Vector512.FusedMultiplyAdd(x0v, g, a70); a71 = Vector512.FusedMultiplyAdd(x1v, g, a71); a72 = Vector512.FusedMultiplyAdd(x2v, g, a72);
        }
        Vector512.Store(a00, park); Vector512.Store(a01, park + 16); Vector512.Store(a02, park + 32);
        Vector512.Store(a10, park + 48); Vector512.Store(a11, park + 64); Vector512.Store(a12, park + 80);
        Vector512.Store(a20, park + 96); Vector512.Store(a21, park + 112); Vector512.Store(a22, park + 128);
        Vector512.Store(a30, park + 144); Vector512.Store(a31, park + 160); Vector512.Store(a32, park + 176);
        Vector512.Store(a40, park + 192); Vector512.Store(a41, park + 208); Vector512.Store(a42, park + 224);
        Vector512.Store(a50, park + 240); Vector512.Store(a51, park + 256); Vector512.Store(a52, park + 272);
        Vector512.Store(a60, park + 288); Vector512.Store(a61, park + 304); Vector512.Store(a62, park + 320);
        Vector512.Store(a70, park + 336); Vector512.Store(a71, park + 352); Vector512.Store(a72, park + 368);
    }

    // The AVX2 weight-gradient tile's three-column loop (four output channels x three columns x eight input lanes).
    public static void Avx2Three(float* park, float* xTap, float* gq, int* oImage, int q0, int q1, int step)
    {
        var a00 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park); var a01 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 8); var a02 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 16);
        var a10 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 24); var a11 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 32); var a12 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 40);
        var a20 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 48); var a21 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 56); var a22 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 64);
        var a30 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 72); var a31 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 80); var a32 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(park + 88);
        for (int q = q0; q < q1; q++, gq += Block)
        {
            float* xs = xTap + oImage[q];
            var x0v = System.Runtime.Intrinsics.X86.Avx.LoadVector256(xs);
            var x1v = System.Runtime.Intrinsics.X86.Avx.LoadVector256(xs + step);
            var x2v = System.Runtime.Intrinsics.X86.Avx.LoadVector256(xs + 2 * step);
            var g0 = Vector256.Create(gq[0]);
            a00 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x0v, g0, a00); a01 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x1v, g0, a01); a02 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x2v, g0, a02);
            var g1 = Vector256.Create(gq[1]);
            a10 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x0v, g1, a10); a11 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x1v, g1, a11); a12 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x2v, g1, a12);
            var g2 = Vector256.Create(gq[2]);
            a20 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x0v, g2, a20); a21 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x1v, g2, a21); a22 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x2v, g2, a22);
            var g3 = Vector256.Create(gq[3]);
            a30 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x0v, g3, a30); a31 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x1v, g3, a31); a32 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(x2v, g3, a32);
        }
        System.Runtime.Intrinsics.X86.Avx.Store(park, a00); System.Runtime.Intrinsics.X86.Avx.Store(park + 8, a01); System.Runtime.Intrinsics.X86.Avx.Store(park + 16, a02);
        System.Runtime.Intrinsics.X86.Avx.Store(park + 24, a10); System.Runtime.Intrinsics.X86.Avx.Store(park + 32, a11); System.Runtime.Intrinsics.X86.Avx.Store(park + 40, a12);
        System.Runtime.Intrinsics.X86.Avx.Store(park + 48, a20); System.Runtime.Intrinsics.X86.Avx.Store(park + 56, a21); System.Runtime.Intrinsics.X86.Avx.Store(park + 64, a22);
        System.Runtime.Intrinsics.X86.Avx.Store(park + 72, a30); System.Runtime.Intrinsics.X86.Avx.Store(park + 80, a31); System.Runtime.Intrinsics.X86.Avx.Store(park + 88, a32);
    }

    public static void WideThree(float* park, float* xTap, float* gq, int* oImage, int q0, int q1, int step)
    {
        var a00 = Vector512.Load(park); var a01 = Vector512.Load(park + 16); var a02 = Vector512.Load(park + 32);
        var a10 = Vector512.Load(park + 48); var a11 = Vector512.Load(park + 64); var a12 = Vector512.Load(park + 80);
        var a20 = Vector512.Load(park + 96); var a21 = Vector512.Load(park + 112); var a22 = Vector512.Load(park + 128);
        var a30 = Vector512.Load(park + 144); var a31 = Vector512.Load(park + 160); var a32 = Vector512.Load(park + 176);
        var a40 = Vector512.Load(park + 192); var a41 = Vector512.Load(park + 208); var a42 = Vector512.Load(park + 224);
        var a50 = Vector512.Load(park + 240); var a51 = Vector512.Load(park + 256); var a52 = Vector512.Load(park + 272);
        var a60 = Vector512.Load(park + 288); var a61 = Vector512.Load(park + 304); var a62 = Vector512.Load(park + 320);
        var a70 = Vector512.Load(park + 336); var a71 = Vector512.Load(park + 352); var a72 = Vector512.Load(park + 368);
        for (int q = q0; q < q1; q++, gq += Block)
        {
            float* xs = xTap + oImage[q];
            var x0v = Vector512.Load(xs);
            var x1v = Vector512.Load(xs + step);
            var x2v = Vector512.Load(xs + 2 * step);
            var g = Vector512.Create(gq[0]);
            a00 = Vector512.FusedMultiplyAdd(x0v, g, a00); a01 = Vector512.FusedMultiplyAdd(x1v, g, a01); a02 = Vector512.FusedMultiplyAdd(x2v, g, a02);
            g = Vector512.Create(gq[1]);
            a10 = Vector512.FusedMultiplyAdd(x0v, g, a10); a11 = Vector512.FusedMultiplyAdd(x1v, g, a11); a12 = Vector512.FusedMultiplyAdd(x2v, g, a12);
            g = Vector512.Create(gq[2]);
            a20 = Vector512.FusedMultiplyAdd(x0v, g, a20); a21 = Vector512.FusedMultiplyAdd(x1v, g, a21); a22 = Vector512.FusedMultiplyAdd(x2v, g, a22);
            g = Vector512.Create(gq[3]);
            a30 = Vector512.FusedMultiplyAdd(x0v, g, a30); a31 = Vector512.FusedMultiplyAdd(x1v, g, a31); a32 = Vector512.FusedMultiplyAdd(x2v, g, a32);
            g = Vector512.Create(gq[4]);
            a40 = Vector512.FusedMultiplyAdd(x0v, g, a40); a41 = Vector512.FusedMultiplyAdd(x1v, g, a41); a42 = Vector512.FusedMultiplyAdd(x2v, g, a42);
            g = Vector512.Create(gq[5]);
            a50 = Vector512.FusedMultiplyAdd(x0v, g, a50); a51 = Vector512.FusedMultiplyAdd(x1v, g, a51); a52 = Vector512.FusedMultiplyAdd(x2v, g, a52);
            g = Vector512.Create(gq[6]);
            a60 = Vector512.FusedMultiplyAdd(x0v, g, a60); a61 = Vector512.FusedMultiplyAdd(x1v, g, a61); a62 = Vector512.FusedMultiplyAdd(x2v, g, a62);
            g = Vector512.Create(gq[7]);
            a70 = Vector512.FusedMultiplyAdd(x0v, g, a70); a71 = Vector512.FusedMultiplyAdd(x1v, g, a71); a72 = Vector512.FusedMultiplyAdd(x2v, g, a72);
        }
        Vector512.Store(a00, park); Vector512.Store(a01, park + 16); Vector512.Store(a02, park + 32);
        Vector512.Store(a10, park + 48); Vector512.Store(a11, park + 64); Vector512.Store(a12, park + 80);
        Vector512.Store(a20, park + 96); Vector512.Store(a21, park + 112); Vector512.Store(a22, park + 128);
        Vector512.Store(a30, park + 144); Vector512.Store(a31, park + 160); Vector512.Store(a32, park + 176);
        Vector512.Store(a40, park + 192); Vector512.Store(a41, park + 208); Vector512.Store(a42, park + 224);
        Vector512.Store(a50, park + 240); Vector512.Store(a51, park + 256); Vector512.Store(a52, park + 272);
        Vector512.Store(a60, park + 288); Vector512.Store(a61, park + 304); Vector512.Store(a62, park + 320);
        Vector512.Store(a70, park + 336); Vector512.Store(a71, park + 352); Vector512.Store(a72, park + 368);
    }
}
