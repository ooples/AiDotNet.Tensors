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

kernel void avgpool3d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& batch [[buffer(2)]], constant int& channels [[buffer(3)]],
    constant int& inDepth [[buffer(4)]], constant int& inHeight [[buffer(5)]], constant int& inWidth [[buffer(6)]],
    constant int& outDepth [[buffer(7)]], constant int& outHeight [[buffer(8)]], constant int& outWidth [[buffer(9)]],
    constant int& kernelD [[buffer(10)]], constant int& kernelH [[buffer(11)]], constant int& kernelW [[buffer(12)]],
    constant int& strideD [[buffer(13)]], constant int& strideH [[buffer(14)]], constant int& strideW [[buffer(15)]],
    constant int& countIncludePad [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= batch * channels * outDepth * outHeight * outWidth) return;
    int ow = idx % outWidth;
    int oh = (idx / outWidth) % outHeight;
    int od = (idx / (outWidth * outHeight)) % outDepth;
    int c = (idx / (outWidth * outHeight * outDepth)) % channels;
    int b = idx / (outWidth * outHeight * outDepth * channels);
    float sum = 0.0f;
    int count = 0;
    for (int kd = 0; kd < kernelD; kd++) {
        int id = od * strideD + kd;
        if (id >= inDepth) continue;
        for (int kh = 0; kh < kernelH; kh++) {
            int ih = oh * strideH + kh;
            if (ih >= inHeight) continue;
            for (int kw = 0; kw < kernelW; kw++) {
                int iw = ow * strideW + kw;
                if (iw >= inWidth) continue;
                int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth + ih * inWidth + iw;
                sum += input[inputIdx];
                count++;
            }
        }
    }
    int divisor = countIncludePad ? (kernelD * kernelH * kernelW) : count;
    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth + oh * outWidth + ow;
    output[outIdx] = sum / (float)max(divisor, 1);
}

kernel void avgpool3d_backward(
    device const float* gradOutput [[buffer(0)]],
    device float* gradInput [[buffer(1)]],
    constant int& batch [[buffer(2)]], constant int& channels [[buffer(3)]],
    constant int& inDepth [[buffer(4)]], constant int& inHeight [[buffer(5)]], constant int& inWidth [[buffer(6)]],
    constant int& outDepth [[buffer(7)]], constant int& outHeight [[buffer(8)]], constant int& outWidth [[buffer(9)]],
    constant int& kernelD [[buffer(10)]], constant int& kernelH [[buffer(11)]], constant int& kernelW [[buffer(12)]],
    constant int& strideD [[buffer(13)]], constant int& strideH [[buffer(14)]], constant int& strideW [[buffer(15)]],
    constant int& countIncludePad [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= batch * channels * inDepth * inHeight * inWidth) return;
    int iw = idx % inWidth;
    int ih = (idx / inWidth) % inHeight;
    int id = (idx / (inWidth * inHeight)) % inDepth;
    int c = (idx / (inWidth * inHeight * inDepth)) % channels;
    int b = idx / (inWidth * inHeight * inDepth * channels);
    float sum = 0.0f;
    for (int od = 0; od < outDepth; od++) {
        int dStart = od * strideD;
        int dEnd = dStart + kernelD;
        if (id < dStart || id >= dEnd) continue;
        for (int oh = 0; oh < outHeight; oh++) {
            int hStart = oh * strideH;
            int hEnd = hStart + kernelH;
            if (ih < hStart || ih >= hEnd) continue;
            for (int ow = 0; ow < outWidth; ow++) {
                int wStart = ow * strideW;
                int wEnd = wStart + kernelW;
                if (iw < wStart || iw >= wEnd) continue;
                int poolSize;
                if (countIncludePad) {
                    poolSize = kernelD * kernelH * kernelW;
                } else {
                    int dEndClamp = min(dEnd, inDepth);
                    int hEndClamp = min(hEnd, inHeight);
                    int wEndClamp = min(wEnd, inWidth);
                    poolSize = (dEndClamp - dStart) * (hEndClamp - hStart) * (wEndClamp - wStart);
                }
                int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth + oh * outWidth + ow;
                sum += gradOutput[outIdx] / (float)max(poolSize, 1);
            }
        }
    }
    int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth + ih * inWidth + iw;
    gradInput[inputIdx] = sum;
}

kernel void gaussian_covariance(
    device const float* rotations [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device float* covariances [[buffer(2)]],
    constant int& numGaussians [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int i = int(gid);
    if (i >= numGaussians) return;
    float qw = rotations[i * 4], qx = rotations[i * 4 + 1], qy = rotations[i * 4 + 2], qz = rotations[i * 4 + 3];
    float qNorm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qNorm > 0.0f) { float inv = 1.0f / qNorm; qw *= inv; qx *= inv; qy *= inv; qz *= inv; }
    float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    float r01 = 2.0f * (qx * qy - qw * qz);
    float r02 = 2.0f * (qx * qz + qw * qy);
    float r10 = 2.0f * (qx * qy + qw * qz);
    float r11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    float r12 = 2.0f * (qy * qz - qw * qx);
    float r20 = 2.0f * (qx * qz - qw * qy);
    float r21 = 2.0f * (qy * qz + qw * qx);
    float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);
    float sx = fmax(1e-6f, fabs(scales[i * 3])); float sx2 = sx * sx;
    float sy = fmax(1e-6f, fabs(scales[i * 3 + 1])); float sy2 = sy * sy;
    float sz = fmax(1e-6f, fabs(scales[i * 3 + 2])); float sz2 = sz * sz;
    float m00 = r00 * sx2, m01 = r01 * sy2, m02 = r02 * sz2;
    float m10 = r10 * sx2, m11 = r11 * sy2, m12 = r12 * sz2;
    float m20 = r20 * sx2, m21 = r21 * sy2, m22 = r22 * sz2;
    int o = i * 6;
    covariances[o]     = m00 * r00 + m01 * r01 + m02 * r02;
    covariances[o + 1] = m00 * r10 + m01 * r11 + m02 * r12;
    covariances[o + 2] = m00 * r20 + m01 * r21 + m02 * r22;
    covariances[o + 3] = m10 * r10 + m11 * r11 + m12 * r12;
    covariances[o + 4] = m10 * r20 + m11 * r21 + m12 * r22;
    covariances[o + 5] = m20 * r20 + m21 * r21 + m22 * r22;
}

kernel void spherical_harmonics(
    device const float* shCoefficients [[buffer(0)]],
    device const float* viewDirections [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& numPoints [[buffer(3)]], constant int& basisCount [[buffer(4)]],
    constant int& numChannels [[buffer(5)]], constant int& degree [[buffer(6)]], constant int& broadcastDir [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= numPoints * numChannels) return;
    int ch = idx % numChannels;
    int i = idx / numChannels;
    int dirIdx = broadcastDir ? 0 : i;
    float dx = viewDirections[dirIdx * 3], dy = viewDirections[dirIdx * 3 + 1], dz = viewDirections[dirIdx * 3 + 2];
    float norm = sqrt(dx * dx + dy * dy + dz * dz);
    if (norm > 0.0f) { float inv = 1.0f / norm; dx *= inv; dy *= inv; dz *= inv; }
    float basis[16];
    basis[0] = 0.282095f;
    if (degree >= 1) { basis[1] = 0.488603f * dy; basis[2] = 0.488603f * dz; basis[3] = 0.488603f * dx; }
    if (degree >= 2) {
        basis[4] = 1.092548f * dx * dy; basis[5] = 1.092548f * dy * dz;
        basis[6] = 0.315392f * (3.0f * dz * dz - 1.0f);
        basis[7] = 1.092548f * dx * dz; basis[8] = 0.546274f * (dx * dx - dy * dy);
    }
    if (degree >= 3) {
        basis[9]  = 0.590044f * dy * (3.0f * dx * dx - dy * dy);
        basis[10] = 2.890611f * dx * dy * dz;
        basis[11] = 0.457046f * dy * (5.0f * dz * dz - 1.0f);
        basis[12] = 0.373176f * dz * (5.0f * dz * dz - 3.0f);
        basis[13] = 0.457046f * dx * (5.0f * dz * dz - 1.0f);
        basis[14] = 1.445306f * dz * (dx * dx - dy * dy);
        basis[15] = 0.590044f * dx * (dx * dx - 3.0f * dy * dy);
    }
    float color = 0.0f;
    for (int b = 0; b < basisCount; b++)
        color += shCoefficients[i * basisCount * numChannels + b * numChannels + ch] * basis[b];
    output[i * numChannels + ch] = fmin(fmax(color, 0.0f), 1.0f);
}

kernel void spherical_harmonics_backward(
    device const float* shCoefficients [[buffer(0)]],
    device const float* viewDirections [[buffer(1)]],
    device const float* outputGradient [[buffer(2)]],
    device float* shGrad [[buffer(3)]],
    constant int& numPoints [[buffer(4)]], constant int& basisCount [[buffer(5)]],
    constant int& numChannels [[buffer(6)]], constant int& degree [[buffer(7)]], constant int& broadcastDir [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    int idx = int(gid);
    if (idx >= numPoints * basisCount * numChannels) return;
    int ch = idx % numChannels;
    int b = (idx / numChannels) % basisCount;
    int i = idx / (basisCount * numChannels);
    int dirIdx = broadcastDir ? 0 : i;
    float dx = viewDirections[dirIdx * 3], dy = viewDirections[dirIdx * 3 + 1], dz = viewDirections[dirIdx * 3 + 2];
    float norm = sqrt(dx * dx + dy * dy + dz * dz);
    if (norm > 0.0f) { float inv = 1.0f / norm; dx *= inv; dy *= inv; dz *= inv; }
    float basis[16];
    basis[0] = 0.282095f;
    if (degree >= 1) { basis[1] = 0.488603f * dy; basis[2] = 0.488603f * dz; basis[3] = 0.488603f * dx; }
    if (degree >= 2) {
        basis[4] = 1.092548f * dx * dy; basis[5] = 1.092548f * dy * dz;
        basis[6] = 0.315392f * (3.0f * dz * dz - 1.0f);
        basis[7] = 1.092548f * dx * dz; basis[8] = 0.546274f * (dx * dx - dy * dy);
    }
    if (degree >= 3) {
        basis[9]  = 0.590044f * dy * (3.0f * dx * dx - dy * dy);
        basis[10] = 2.890611f * dx * dy * dz;
        basis[11] = 0.457046f * dy * (5.0f * dz * dz - 1.0f);
        basis[12] = 0.373176f * dz * (5.0f * dz * dz - 3.0f);
        basis[13] = 0.457046f * dx * (5.0f * dz * dz - 1.0f);
        basis[14] = 1.445306f * dz * (dx * dx - dy * dy);
        basis[15] = 0.590044f * dx * (dx * dx - 3.0f * dy * dy);
    }
    float preclamp = 0.0f;
    for (int bb = 0; bb < basisCount; bb++)
        preclamp += shCoefficients[i * basisCount * numChannels + bb * numChannels + ch] * basis[bb];
    float colorGrad = outputGradient[i * numChannels + ch];
    if (preclamp < 0.0f || preclamp > 1.0f) colorGrad = 0.0f;
    shGrad[idx] = colorGrad * basis[b];
}
";
}
