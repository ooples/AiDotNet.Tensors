namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #775: Metal (MSL) mirrors of the OpenCL extended-conv/geometry kernels. Kept in one library compiled
// by MetalBackend; the per-family capability interfaces are implemented in MetalBackend.ExtendedConv.cs.
// The per-element arithmetic is kept byte-identical to the OpenCL reference so the cross-backend
// source-parity tests can assert it; only the kernel signature/dispatch differs per shader language.
internal static class MetalExtendedConvKernels
{
    public const string Source = @"
#include <metal_stdlib>
using namespace metal;

kernel void trilinear_interpolate(
    device const float* grid [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& C [[buffer(6)]],
    constant int& P [[buffer(7)]],
    constant float& upperEps [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= P * C) return;
    int c = idx % C;
    int n = idx / C;
    float z = fmax(0.0f, fmin((float)(D - 1) - upperEps, positions[n * 3 + 0]));
    float y = fmax(0.0f, fmin((float)(H - 1) - upperEps, positions[n * 3 + 1]));
    float x = fmax(0.0f, fmin((float)(W - 1) - upperEps, positions[n * 3 + 2]));
    int z0 = (int)floor(z), y0 = (int)floor(y), x0 = (int)floor(x);
    int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
    float fz = z - z0, fy = y - y0, fx = x - x0;
    float w000 = (1 - fz) * (1 - fy) * (1 - fx), w001 = (1 - fz) * (1 - fy) * fx;
    float w010 = (1 - fz) * fy * (1 - fx),       w011 = (1 - fz) * fy * fx;
    float w100 = fz * (1 - fy) * (1 - fx),       w101 = fz * (1 - fy) * fx;
    float w110 = fz * fy * (1 - fx),             w111 = fz * fy * fx;
    output[n * C + c] =
        w000 * grid[(((z0 * H + y0) * W + x0) * C) + c] + w001 * grid[(((z0 * H + y0) * W + x1) * C) + c] +
        w010 * grid[(((z0 * H + y1) * W + x0) * C) + c] + w011 * grid[(((z0 * H + y1) * W + x1) * C) + c] +
        w100 * grid[(((z1 * H + y0) * W + x0) * C) + c] + w101 * grid[(((z1 * H + y0) * W + x1) * C) + c] +
        w110 * grid[(((z1 * H + y1) * W + x0) * C) + c] + w111 * grid[(((z1 * H + y1) * W + x1) * C) + c];
}

kernel void trilinear_interpolate_backward(
    device const float* gradOutput [[buffer(0)]],
    device const float* positions [[buffer(1)]],
    device float* gradGrid [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant int& H [[buffer(4)]],
    constant int& W [[buffer(5)]],
    constant int& C [[buffer(6)]],
    constant int& P [[buffer(7)]],
    constant float& upperEps [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= D * H * W * C) return;
    int c = idx % C;
    int gx = (idx / C) % W;
    int gy = (idx / (C * W)) % H;
    int gz = idx / (C * W * H);
    float sum = 0.0f;
    for (int n = 0; n < P; n++) {
        float z = fmax(0.0f, fmin((float)(D - 1) - upperEps, positions[n * 3 + 0]));
        float y = fmax(0.0f, fmin((float)(H - 1) - upperEps, positions[n * 3 + 1]));
        float x = fmax(0.0f, fmin((float)(W - 1) - upperEps, positions[n * 3 + 2]));
        int z0 = (int)floor(z), y0 = (int)floor(y), x0 = (int)floor(x);
        int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float fz = z - z0, fy = y - y0, fx = x - x0;
        float wz = (gz == z0 ? (1.0f - fz) : 0.0f) + (gz == z1 ? fz : 0.0f);
        if (wz == 0.0f) continue;
        float wy = (gy == y0 ? (1.0f - fy) : 0.0f) + (gy == y1 ? fy : 0.0f);
        if (wy == 0.0f) continue;
        float wx = (gx == x0 ? (1.0f - fx) : 0.0f) + (gx == x1 ? fx : 0.0f);
        sum += wz * wy * wx * gradOutput[n * C + c];
    }
    gradGrid[idx] = sum;
}

kernel void conv_transpose3d(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& inC [[buffer(4)]],
    constant int& iD [[buffer(5)]], constant int& iH [[buffer(6)]], constant int& iW [[buffer(7)]],
    constant int& outC [[buffer(8)]], constant int& outD [[buffer(9)]],
    constant int& outH [[buffer(10)]], constant int& outW [[buffer(11)]],
    constant int& kD [[buffer(12)]], constant int& kH [[buffer(13)]], constant int& kW [[buffer(14)]],
    constant int& strideD [[buffer(15)]], constant int& strideH [[buffer(16)]], constant int& strideW [[buffer(17)]],
    constant int& padD [[buffer(18)]], constant int& padH [[buffer(19)]], constant int& padW [[buffer(20)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= N * outC * outD * outH * outW) return;
    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int od = (idx / (outW * outH)) % outD;
    int oc = (idx / (outW * outH * outD)) % outC;
    int n = idx / (outW * outH * outD * outC);
    float sum = 0.0f;
    for (int kd = 0; kd < kD; kd++) {
        int td = od + padD - kd;
        if (td < 0 || (td % strideD) != 0) continue;
        int id = td / strideD;
        if (id < 0 || id >= iD) continue;
        for (int kh = 0; kh < kH; kh++) {
            int th = oh + padH - kh;
            if (th < 0 || (th % strideH) != 0) continue;
            int ih = th / strideH;
            if (ih < 0 || ih >= iH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int tw = ow + padW - kw;
                if (tw < 0 || (tw % strideW) != 0) continue;
                int iw = tw / strideW;
                if (iw < 0 || iw >= iW) continue;
                for (int ic = 0; ic < inC; ic++) {
                    sum += input[(((n * inC + ic) * iD + id) * iH + ih) * iW + iw]
                         * weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)];
                }
            }
        }
    }
    output[idx] = sum;
}

kernel void conv_transpose3d_backward_input(
    device const float* gradOutput [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& inC [[buffer(4)]],
    constant int& iD [[buffer(5)]], constant int& iH [[buffer(6)]], constant int& iW [[buffer(7)]],
    constant int& outC [[buffer(8)]], constant int& outD [[buffer(9)]],
    constant int& outH [[buffer(10)]], constant int& outW [[buffer(11)]],
    constant int& kD [[buffer(12)]], constant int& kH [[buffer(13)]], constant int& kW [[buffer(14)]],
    constant int& strideD [[buffer(15)]], constant int& strideH [[buffer(16)]], constant int& strideW [[buffer(17)]],
    constant int& padD [[buffer(18)]], constant int& padH [[buffer(19)]], constant int& padW [[buffer(20)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= N * inC * iD * iH * iW) return;
    int iw = idx % iW;
    int ih = (idx / iW) % iH;
    int id = (idx / (iW * iH)) % iD;
    int ic = (idx / (iW * iH * iD)) % inC;
    int n = idx / (iW * iH * iD * inC);
    float sum = 0.0f;
    for (int kd = 0; kd < kD; kd++) {
        int od = id * strideD - padD + kd;
        if (od < 0 || od >= outD) continue;
        for (int kh = 0; kh < kH; kh++) {
            int oh = ih * strideH - padH + kh;
            if (oh < 0 || oh >= outH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int ow = iw * strideW - padW + kw;
                if (ow < 0 || ow >= outW) continue;
                for (int oc = 0; oc < outC; oc++) {
                    sum += gradOutput[(((n * outC + oc) * outD + od) * outH + oh) * outW + ow]
                         * weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)];
                }
            }
        }
    }
    gradInput[idx] = sum;
}

kernel void conv_transpose3d_backward_weights(
    device const float* gradOutput [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* gradWeights [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& inC [[buffer(4)]],
    constant int& iD [[buffer(5)]], constant int& iH [[buffer(6)]], constant int& iW [[buffer(7)]],
    constant int& outC [[buffer(8)]], constant int& outD [[buffer(9)]],
    constant int& outH [[buffer(10)]], constant int& outW [[buffer(11)]],
    constant int& kD [[buffer(12)]], constant int& kH [[buffer(13)]], constant int& kW [[buffer(14)]],
    constant int& strideD [[buffer(15)]], constant int& strideH [[buffer(16)]], constant int& strideW [[buffer(17)]],
    constant int& padD [[buffer(18)]], constant int& padH [[buffer(19)]], constant int& padW [[buffer(20)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= inC * outC * kD * kH * kW) return;
    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int kd = (idx / (kW * kH)) % kD;
    int oc = (idx / (kW * kH * kD)) % outC;
    int ic = idx / (kW * kH * kD * outC);
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int id = 0; id < iD; id++) {
            int od = id * strideD - padD + kd;
            if (od < 0 || od >= outD) continue;
            for (int ih = 0; ih < iH; ih++) {
                int oh = ih * strideH - padH + kh;
                if (oh < 0 || oh >= outH) continue;
                for (int iw = 0; iw < iW; iw++) {
                    int ow = iw * strideW - padW + kw;
                    if (ow < 0 || ow >= outW) continue;
                    sum += input[(((n * inC + ic) * iD + id) * iH + ih) * iW + iw]
                         * gradOutput[(((n * outC + oc) * outD + od) * outH + oh) * outW + ow];
                }
            }
        }
    }
    gradWeights[idx] = sum;
}

kernel void spiral_conv(
    device const float* vertexFeatures [[buffer(0)]],
    device const int* spiralIndices [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& V [[buffer(5)]], constant int& inC [[buffer(6)]],
    constant int& spiralLength [[buffer(7)]], constant int& outC [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= V * outC) return;
    int oc = idx % outC;
    int v = idx / outC;
    int gatheredSize = inC * spiralLength;
    float sum = biases[oc];
    for (int s = 0; s < spiralLength; s++) {
        int neighborIdx = spiralIndices[v * spiralLength + s];
        if (neighborIdx < 0 || neighborIdx >= V) continue;
        int gatherOffset = s * inC;
        for (int c = 0; c < inC; c++) {
            sum += vertexFeatures[neighborIdx * inC + c] * weights[oc * gatheredSize + gatherOffset + c];
        }
    }
    output[idx] = sum;
}

kernel void spiral_conv_backward_input(
    device const float* gradOutput [[buffer(0)]],
    device const int* spiralIndices [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* gradVertexFeatures [[buffer(3)]],
    constant int& V [[buffer(4)]], constant int& inC [[buffer(5)]],
    constant int& spiralLength [[buffer(6)]], constant int& outC [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= V * inC) return;
    int ic = idx % inC;
    int nbr = idx / inC;
    int gatheredSize = inC * spiralLength;
    float sum = 0.0f;
    for (int v = 0; v < V; v++) {
        for (int s = 0; s < spiralLength; s++) {
            if (spiralIndices[v * spiralLength + s] != nbr) continue;
            for (int oc = 0; oc < outC; oc++) {
                sum += gradOutput[v * outC + oc] * weights[oc * gatheredSize + s * inC + ic];
            }
        }
    }
    gradVertexFeatures[idx] = sum;
}

kernel void spiral_conv_backward_weights(
    device const float* gradOutput [[buffer(0)]],
    device const float* vertexFeatures [[buffer(1)]],
    device const int* spiralIndices [[buffer(2)]],
    device float* gradWeights [[buffer(3)]],
    constant int& V [[buffer(4)]], constant int& inC [[buffer(5)]],
    constant int& spiralLength [[buffer(6)]], constant int& outC [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    int gatheredSize = inC * spiralLength;
    if (idx >= outC * gatheredSize) return;
    int g = idx % gatheredSize;
    int oc = idx / gatheredSize;
    int s = g / inC;
    int ic = g % inC;
    float sum = 0.0f;
    for (int v = 0; v < V; v++) {
        int neighborIdx = spiralIndices[v * spiralLength + s];
        if (neighborIdx < 0 || neighborIdx >= V) continue;
        sum += gradOutput[v * outC + oc] * vertexFeatures[neighborIdx * inC + ic];
    }
    gradWeights[idx] = sum;
}

kernel void adaptive_max_pool2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]], constant int& channels [[buffer(3)]],
    constant int& inHeight [[buffer(4)]], constant int& inWidth [[buffer(5)]],
    constant int& outHeight [[buffer(6)]], constant int& outWidth [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= batch * channels * outHeight * outWidth) return;
    int ow = idx % outWidth;
    int oh = (idx / outWidth) % outHeight;
    int c = (idx / (outWidth * outHeight)) % channels;
    int b = idx / (outWidth * outHeight * channels);
    int hStart = (oh * inHeight) / outHeight;
    int hEnd = ((oh + 1) * inHeight) / outHeight;
    int wStart = (ow * inWidth) / outWidth;
    int wEnd = ((ow + 1) * inWidth) / outWidth;
    float maxV = -INFINITY;
    for (int ih = hStart; ih < hEnd; ih++) {
        for (int iw = wStart; iw < wEnd; iw++) {
            float v = input[((b * channels + c) * inHeight + ih) * inWidth + iw];
            if (v > maxV) maxV = v;
        }
    }
    output[((b * channels + c) * outHeight + oh) * outWidth + ow] = maxV;
}

kernel void conv3d_backward_input(
    device const float* gradOutput [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& inC [[buffer(4)]],
    constant int& D [[buffer(5)]], constant int& H [[buffer(6)]], constant int& W [[buffer(7)]],
    constant int& outC [[buffer(8)]], constant int& outD [[buffer(9)]],
    constant int& outH [[buffer(10)]], constant int& outW [[buffer(11)]],
    constant int& kD [[buffer(12)]], constant int& kH [[buffer(13)]], constant int& kW [[buffer(14)]],
    constant int& strideD [[buffer(15)]], constant int& strideH [[buffer(16)]], constant int& strideW [[buffer(17)]],
    constant int& padD [[buffer(18)]], constant int& padH [[buffer(19)]], constant int& padW [[buffer(20)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    int totalSize = N * inC * D * H * W;
    if (idx >= totalSize) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int ic = (idx / (W * H * D)) % inC;
    int n = idx / (W * H * D * inC);
    float sum = 0.0f;
    for (int oc = 0; oc < outC; oc++) {
        for (int kd = 0; kd < kD; kd++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int od = (d + padD - kd);
                    int oh = (h + padH - kh);
                    int ow = (w + padW - kw);
                    if (od % strideD == 0 && oh % strideH == 0 && ow % strideW == 0) {
                        od /= strideD;
                        oh /= strideH;
                        ow /= strideW;
                        if (od >= 0 && od < outD && oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                            int gradOutIdx = ((n * outC + oc) * outD + od) * outH * outW + oh * outW + ow;
                            int kernelIdx = ((oc * inC + ic) * kD + kd) * kH * kW + kh * kW + kw;
                            sum += gradOutput[gradOutIdx] * weights[kernelIdx];
                        }
                    }
                }
            }
        }
    }
    gradInput[idx] = sum;
}

kernel void conv3d_backward_weights(
    device const float* gradOutput [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* gradKernel [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& inC [[buffer(4)]],
    constant int& D [[buffer(5)]], constant int& H [[buffer(6)]], constant int& W [[buffer(7)]],
    constant int& outC [[buffer(8)]], constant int& outD [[buffer(9)]],
    constant int& outH [[buffer(10)]], constant int& outW [[buffer(11)]],
    constant int& kD [[buffer(12)]], constant int& kH [[buffer(13)]], constant int& kW [[buffer(14)]],
    constant int& strideD [[buffer(15)]], constant int& strideH [[buffer(16)]], constant int& strideW [[buffer(17)]],
    constant int& padD [[buffer(18)]], constant int& padH [[buffer(19)]], constant int& padW [[buffer(20)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    int totalKernelSize = outC * inC * kD * kH * kW;
    if (idx >= totalKernelSize) return;
    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int kd = (idx / (kW * kH)) % kD;
    int ic = (idx / (kW * kH * kD)) % inC;
    int oc = idx / (kW * kH * kD * inC);
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int od = 0; od < outD; od++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int d = od * strideD + kd - padD;
                    int h = oh * strideH + kh - padH;
                    int w = ow * strideW + kw - padW;
                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        int gradOutIdx = ((n * outC + oc) * outD + od) * outH * outW + oh * outW + ow;
                        int inputIdx = ((n * inC + ic) * D + d) * H * W + h * W + w;
                        sum += gradOutput[gradOutIdx] * input[inputIdx];
                    }
                }
            }
        }
    }
    gradKernel[idx] = sum;
}

kernel void depthwise_conv2d_backward_input(
    device const float* gradOutput [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& inC [[buffer(4)]],
    constant int& H [[buffer(5)]], constant int& W [[buffer(6)]], constant int& M [[buffer(7)]],
    constant int& outH [[buffer(8)]], constant int& outW [[buffer(9)]],
    constant int& kH [[buffer(10)]], constant int& kW [[buffer(11)]],
    constant int& strideH [[buffer(12)]], constant int& strideW [[buffer(13)]],
    constant int& padH [[buffer(14)]], constant int& padW [[buffer(15)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    int total = N * inC * H * W;
    if (idx >= total) return;
    int iw = idx % W;
    int ih = (idx / W) % H;
    int ic = (idx / (W * H)) % inC;
    int b = idx / (W * H * inC);
    int outC = inC * M;
    float sum = 0.0f;
    for (int m = 0; m < M; m++) {
        int oc = ic * M + m;
        for (int kh = 0; kh < kH; kh++) {
            int t = ih + padH - kh;
            if (t < 0 || (t % strideH) != 0) continue;
            int oh = t / strideH;
            if (oh < 0 || oh >= outH) continue;
            for (int kw = 0; kw < kW; kw++) {
                int tw = iw + padW - kw;
                if (tw < 0 || (tw % strideW) != 0) continue;
                int ow = tw / strideW;
                if (ow < 0 || ow >= outW) continue;
                sum += weights[(oc * kH + kh) * kW + kw]
                     * gradOutput[((b * outC + oc) * outH + oh) * outW + ow];
            }
        }
    }
    gradInput[idx] = sum;
}

kernel void depthwise_conv2d_backward_weights(
    device const float* gradOutput [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* gradKernel [[buffer(2)]],
    constant int& N [[buffer(3)]], constant int& inC [[buffer(4)]],
    constant int& H [[buffer(5)]], constant int& W [[buffer(6)]], constant int& M [[buffer(7)]],
    constant int& outH [[buffer(8)]], constant int& outW [[buffer(9)]],
    constant int& kH [[buffer(10)]], constant int& kW [[buffer(11)]],
    constant int& strideH [[buffer(12)]], constant int& strideW [[buffer(13)]],
    constant int& padH [[buffer(14)]], constant int& padW [[buffer(15)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    int outC = inC * M;
    int total = outC * kH * kW;
    if (idx >= total) return;
    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int oc = idx / (kW * kH);
    int ic = oc / M;
    float sum = 0.0f;
    for (int b = 0; b < N; b++) {
        for (int oh = 0; oh < outH; oh++) {
            int ih = oh * strideH - padH + kh;
            if (ih < 0 || ih >= H) continue;
            for (int ow = 0; ow < outW; ow++) {
                int iw = ow * strideW - padW + kw;
                if (iw < 0 || iw >= W) continue;
                sum += input[((b * inC + ic) * H + ih) * W + iw]
                     * gradOutput[((b * outC + oc) * outH + oh) * outW + ow];
            }
        }
    }
    gradKernel[idx] = sum;
}
";
}
