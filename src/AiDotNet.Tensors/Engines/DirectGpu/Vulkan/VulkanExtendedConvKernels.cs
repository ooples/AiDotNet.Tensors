namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #775: Vulkan (GLSL compute) mirrors of the OpenCL extended-conv/geometry kernels. Each kernel is a
// standalone GLSL shader string (compiled to SPIR-V on demand by GlslDispatchN). The per-element
// arithmetic is kept as close to the OpenCL reference as GLSL allows (GLSL float literals drop the `f`
// suffix and fmin/fmax/fabs become min/max/abs), so the source-parity tests assert GLSL-form markers for
// the Vulkan rows. Int params travel in the push-constant block; a float param (upperEps) is passed as
// its raw bit pattern in the same uint[] and declared `float` in the block.
internal static class VulkanExtendedConvKernels
{
    public const string TrilinearInterpolate = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Grid { float grid[]; };
layout(set=0,binding=1) readonly buffer Positions { float positions[]; };
layout(set=0,binding=2) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform PC { int D; int H; int W; int C; int P; float upperEps; };
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= P * C) return;
    int c = idx % C;
    int n = idx / C;
    float z = max(0.0, min(float(D - 1) - upperEps, positions[n * 3 + 0]));
    float y = max(0.0, min(float(H - 1) - upperEps, positions[n * 3 + 1]));
    float x = max(0.0, min(float(W - 1) - upperEps, positions[n * 3 + 2]));
    int z0 = int(floor(z)), y0 = int(floor(y)), x0 = int(floor(x));
    int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
    float fz = z - float(z0), fy = y - float(y0), fx = x - float(x0);
    float w000 = (1 - fz) * (1 - fy) * (1 - fx), w001 = (1 - fz) * (1 - fy) * fx;
    float w010 = (1 - fz) * fy * (1 - fx),       w011 = (1 - fz) * fy * fx;
    float w100 = fz * (1 - fy) * (1 - fx),       w101 = fz * (1 - fy) * fx;
    float w110 = fz * fy * (1 - fx),             w111 = fz * fy * fx;
    output_[n * C + c] =
        w000 * grid[(((z0 * H + y0) * W + x0) * C) + c] + w001 * grid[(((z0 * H + y0) * W + x1) * C) + c] +
        w010 * grid[(((z0 * H + y1) * W + x0) * C) + c] + w011 * grid[(((z0 * H + y1) * W + x1) * C) + c] +
        w100 * grid[(((z1 * H + y0) * W + x0) * C) + c] + w101 * grid[(((z1 * H + y0) * W + x1) * C) + c] +
        w110 * grid[(((z1 * H + y1) * W + x0) * C) + c] + w111 * grid[(((z1 * H + y1) * W + x1) * C) + c];
}";

    public const string TrilinearInterpolateBackward = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Positions { float positions[]; };
layout(set=0,binding=2) writeonly buffer GradGrid { float gradGrid[]; };
layout(push_constant) uniform PC { int D; int H; int W; int C; int P; float upperEps; };
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= D * H * W * C) return;
    int c = idx % C;
    int gx = (idx / C) % W;
    int gy = (idx / (C * W)) % H;
    int gz = idx / (C * W * H);
    float sum = 0.0;
    for (int n = 0; n < P; n++) {
        float z = max(0.0, min(float(D - 1) - upperEps, positions[n * 3 + 0]));
        float y = max(0.0, min(float(H - 1) - upperEps, positions[n * 3 + 1]));
        float x = max(0.0, min(float(W - 1) - upperEps, positions[n * 3 + 2]));
        int z0 = int(floor(z)), y0 = int(floor(y)), x0 = int(floor(x));
        int z1 = min(z0 + 1, D - 1), y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
        float fz = z - float(z0), fy = y - float(y0), fx = x - float(x0);
        float wz = (gz == z0 ? (1.0 - fz) : 0.0) + (gz == z1 ? fz : 0.0);
        if (wz == 0.0) continue;
        float wy = (gy == y0 ? (1.0 - fy) : 0.0) + (gy == y1 ? fy : 0.0);
        if (wy == 0.0) continue;
        float wx = (gx == x0 ? (1.0 - fx) : 0.0) + (gx == x1 ? fx : 0.0);
        sum += wz * wy * wx * gradOutput[n * C + c];
    }
    gradGrid[idx] = sum;
}";

    private const string ConvT3DParams =
        "layout(push_constant) uniform PC { int N; int inC; int iD; int iH; int iW; int outC; int outD; int outH; int outW; int kD; int kH; int kW; int strideD; int strideH; int strideW; int padD; int padH; int padW; };";

    public static readonly string ConvTranspose3D = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Input { float input_[]; };
layout(set=0,binding=1) readonly buffer Weights { float weights[]; };
layout(set=0,binding=2) writeonly buffer Output { float output_[]; };
" + ConvT3DParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= N * outC * outD * outH * outW) return;
    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int od = (idx / (outW * outH)) % outD;
    int oc = (idx / (outW * outH * outD)) % outC;
    int n = idx / (outW * outH * outD * outC);
    float sum = 0.0;
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
                    sum += input_[(((n * inC + ic) * iD + id) * iH + ih) * iW + iw]
                         * weights[((((ic * outC + oc) * kD + kd) * kH + kh) * kW + kw)];
                }
            }
        }
    }
    output_[idx] = sum;
}";

    public static readonly string ConvTranspose3DBackwardInput = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Weights { float weights[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
" + ConvT3DParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= N * inC * iD * iH * iW) return;
    int iw = idx % iW;
    int ih = (idx / iW) % iH;
    int id = (idx / (iW * iH)) % iD;
    int ic = (idx / (iW * iH * iD)) % inC;
    int n = idx / (iW * iH * iD * inC);
    float sum = 0.0;
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
}";

    public static readonly string ConvTranspose3DBackwardWeights = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Input { float input_[]; };
layout(set=0,binding=2) writeonly buffer GradWeights { float gradWeights[]; };
" + ConvT3DParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= inC * outC * kD * kH * kW) return;
    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int kd = (idx / (kW * kH)) % kD;
    int oc = (idx / (kW * kH * kD)) % outC;
    int ic = idx / (kW * kH * kD * outC);
    float sum = 0.0;
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
                    sum += input_[(((n * inC + ic) * iD + id) * iH + ih) * iW + iw]
                         * gradOutput[(((n * outC + oc) * outD + od) * outH + oh) * outW + ow];
                }
            }
        }
    }
    gradWeights[idx] = sum;
}";

    private const string SpiralParams =
        "layout(push_constant) uniform PC { int V; int inC; int spiralLength; int outC; };";

    public static readonly string SpiralConv = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer VertexFeatures { float vertexFeatures[]; };
layout(set=0,binding=1) readonly buffer SpiralIndices { int spiralIndices[]; };
layout(set=0,binding=2) readonly buffer Weights { float weights[]; };
layout(set=0,binding=3) readonly buffer Biases { float biases[]; };
layout(set=0,binding=4) writeonly buffer Output { float output_[]; };
" + SpiralParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
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
    output_[idx] = sum;
}";

    public static readonly string SpiralConvBackwardInput = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer SpiralIndices { int spiralIndices[]; };
layout(set=0,binding=2) readonly buffer Weights { float weights[]; };
layout(set=0,binding=3) writeonly buffer GradVertexFeatures { float gradVertexFeatures[]; };
" + SpiralParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= V * inC) return;
    int ic = idx % inC;
    int nbr = idx / inC;
    int gatheredSize = inC * spiralLength;
    float sum = 0.0;
    for (int v = 0; v < V; v++) {
        for (int s = 0; s < spiralLength; s++) {
            if (spiralIndices[v * spiralLength + s] != nbr) continue;
            for (int oc = 0; oc < outC; oc++) {
                sum += gradOutput[v * outC + oc] * weights[oc * gatheredSize + s * inC + ic];
            }
        }
    }
    gradVertexFeatures[idx] = sum;
}";

    public static readonly string SpiralConvBackwardWeights = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer VertexFeatures { float vertexFeatures[]; };
layout(set=0,binding=2) readonly buffer SpiralIndices { int spiralIndices[]; };
layout(set=0,binding=3) writeonly buffer GradWeights { float gradWeights[]; };
" + SpiralParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int gatheredSize = inC * spiralLength;
    if (idx >= outC * gatheredSize) return;
    int g = idx % gatheredSize;
    int oc = idx / gatheredSize;
    int s = g / inC;
    int ic = g % inC;
    float sum = 0.0;
    for (int v = 0; v < V; v++) {
        int neighborIdx = spiralIndices[v * spiralLength + s];
        if (neighborIdx < 0 || neighborIdx >= V) continue;
        sum += gradOutput[v * outC + oc] * vertexFeatures[neighborIdx * inC + ic];
    }
    gradWeights[idx] = sum;
}";

    public static readonly string AdaptiveMaxPool2D = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Input { float input_[]; };
layout(set=0,binding=1) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform PC { int batch; int channels; int inHeight; int inWidth; int outHeight; int outWidth; };
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= batch * channels * outHeight * outWidth) return;
    int ow = idx % outWidth;
    int oh = (idx / outWidth) % outHeight;
    int c = (idx / (outWidth * outHeight)) % channels;
    int b = idx / (outWidth * outHeight * channels);
    int hStart = (oh * inHeight) / outHeight;
    int hEnd = ((oh + 1) * inHeight) / outHeight;
    int wStart = (ow * inWidth) / outWidth;
    int wEnd = ((ow + 1) * inWidth) / outWidth;
    float maxV = -3.402823466e38;
    for (int ih = hStart; ih < hEnd; ih++) {
        for (int iw = wStart; iw < wEnd; iw++) {
            float v = input_[((b * channels + c) * inHeight + ih) * inWidth + iw];
            if (v > maxV) maxV = v;
        }
    }
    output_[((b * channels + c) * outHeight + oh) * outWidth + ow] = maxV;
}";

    private const string Conv3DBackwardParams =
        "layout(push_constant) uniform PC { int N; int inC; int D; int H; int W; int outC; int outD; int outH; int outW; int kD; int kH; int kW; int strideD; int strideH; int strideW; int padD; int padH; int padW; };";

    public static readonly string Conv3DBackwardInput = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Weights { float weights[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
" + Conv3DBackwardParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int totalSize = N * inC * D * H * W;
    if (idx >= totalSize) return;
    int w = idx % W;
    int h = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int ic = (idx / (W * H * D)) % inC;
    int n = idx / (W * H * D * inC);
    float sum = 0.0;
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
}";

    public static readonly string Conv3DBackwardWeights = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Input { float input_[]; };
layout(set=0,binding=2) writeonly buffer GradKernel { float gradKernel[]; };
" + Conv3DBackwardParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int totalKernelSize = outC * inC * kD * kH * kW;
    if (idx >= totalKernelSize) return;
    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int kd = (idx / (kW * kH)) % kD;
    int ic = (idx / (kW * kH * kD)) % inC;
    int oc = idx / (kW * kH * kD * inC);
    float sum = 0.0;
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
                        sum += gradOutput[gradOutIdx] * input_[inputIdx];
                    }
                }
            }
        }
    }
    gradKernel[idx] = sum;
}";

    private const string DepthwiseParams =
        "layout(push_constant) uniform PC { int N; int inC; int H; int W; int M; int outH; int outW; int kH; int kW; int strideH; int strideW; int padH; int padW; };";

    public static readonly string DepthwiseConv2DBackwardInput = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Weights { float weights[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
" + DepthwiseParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int total = N * inC * H * W;
    if (idx >= total) return;
    int iw = idx % W;
    int ih = (idx / W) % H;
    int ic = (idx / (W * H)) % inC;
    int b = idx / (W * H * inC);
    int outC = inC * M;
    float sum = 0.0;
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
}";

    public static readonly string DepthwiseConv2DBackwardWeights = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Input { float input_[]; };
layout(set=0,binding=2) writeonly buffer GradKernel { float gradKernel[]; };
" + DepthwiseParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int outC = inC * M;
    int total = outC * kH * kW;
    if (idx >= total) return;
    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int oc = idx / (kW * kH);
    int ic = oc / M;
    float sum = 0.0;
    for (int b = 0; b < N; b++) {
        for (int oh = 0; oh < outH; oh++) {
            int ih = oh * strideH - padH + kh;
            if (ih < 0 || ih >= H) continue;
            for (int ow = 0; ow < outW; ow++) {
                int iw = ow * strideW - padW + kw;
                if (iw < 0 || iw >= W) continue;
                sum += input_[((b * inC + ic) * H + ih) * W + iw]
                     * gradOutput[((b * outC + oc) * outH + oh) * outW + ow];
            }
        }
    }
    gradKernel[idx] = sum;
}";
}
