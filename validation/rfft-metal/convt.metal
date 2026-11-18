#include <metal_stdlib>
using namespace metal;

kernel void convt_gemm_regroup(device const float* src [[buffer(0)]], device float* xt [[buffer(1)]], constant int& nb [[buffer(2)]], constant int& C [[buffer(3)]], constant int& L [[buffer(4)]], constant int& srcOffset [[buffer(5)]], uint gidU [[thread_position_in_grid]])
{
    int gid = (int)gidU;

    int total = nb * C * L;
    if (gid < total)
    {
        int l = gid % L;
        int r = gid / L;
        int c = r % C;
        int b = r / C;
        xt[c * (nb * L) + b * L + l] = src[srcOffset + gid];
    }
}

kernel void convt_gemm_overlap_add(device const float* col [[buffer(0)]], device float* dst [[buffer(1)]], constant int& nb [[buffer(2)]], constant int& OC [[buffer(3)]], constant int& OH [[buffer(4)]], constant int& OW [[buffer(5)]], constant int& KH [[buffer(6)]], constant int& KW [[buffer(7)]], constant int& SH [[buffer(8)]], constant int& SW [[buffer(9)]], constant int& PH [[buffer(10)]], constant int& PW [[buffer(11)]], constant int& IH [[buffer(12)]], constant int& IW [[buffer(13)]], constant int& dstOffset [[buffer(14)]], uint gidU [[thread_position_in_grid]])
{
    int gid = (int)gidU;

    int total = nb * OC * OH * OW;
    if (gid < total)
    {
        int ow = gid % OW;
        int r = gid / OW;
        int oh = r % OH;
        r = r / OH;
        int c = r % OC;
        int b = r / OC;
        int n = nb * IH * IW;
        float sum = 0.0f;
        for (int kh = (oh + PH) % SH; kh < KH; kh += SH)
        {
            int ih = (oh + PH - kh) / SH;
            if (ih >= 0 && ih < IH)
            {
                for (int kw = (ow + PW) % SW; kw < KW; kw += SW)
                {
                    int iw = (ow + PW - kw) / SW;
                    if (iw >= 0 && iw < IW) sum += col[((c * KH + kh) * KW + kw) * n + (b * IH + ih) * IW + iw];
                }
            }
        }
        dst[dstOffset + gid] = sum;
    }
}
