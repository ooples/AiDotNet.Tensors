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

    private const string Pool3DParams =
        "layout(push_constant) uniform PC { int batch; int channels; int inDepth; int inHeight; int inWidth; int outDepth; int outHeight; int outWidth; int kernelD; int kernelH; int kernelW; int strideD; int strideH; int strideW; int countIncludePad; };";

    public static readonly string AvgPool3D = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Input { float input_[]; };
layout(set=0,binding=1) writeonly buffer Output { float output_[]; };
" + Pool3DParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= batch * channels * outDepth * outHeight * outWidth) return;
    int ow = idx % outWidth;
    int oh = (idx / outWidth) % outHeight;
    int od = (idx / (outWidth * outHeight)) % outDepth;
    int c = (idx / (outWidth * outHeight * outDepth)) % channels;
    int b = idx / (outWidth * outHeight * outDepth * channels);
    float sum = 0.0;
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
                sum += input_[inputIdx];
                count++;
            }
        }
    }
    int divisor = countIncludePad != 0 ? (kernelD * kernelH * kernelW) : count;
    int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth + oh * outWidth + ow;
    output_[outIdx] = sum / float(max(divisor, 1));
}";

    public static readonly string AvgPool3DBackward = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) writeonly buffer GradInput { float gradInput[]; };
" + Pool3DParams + @"
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= batch * channels * inDepth * inHeight * inWidth) return;
    int iw = idx % inWidth;
    int ih = (idx / inWidth) % inHeight;
    int id = (idx / (inWidth * inHeight)) % inDepth;
    int c = (idx / (inWidth * inHeight * inDepth)) % channels;
    int b = idx / (inWidth * inHeight * inDepth * channels);
    float sum = 0.0;
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
                if (countIncludePad != 0) {
                    poolSize = kernelD * kernelH * kernelW;
                } else {
                    int dEndClamp = min(dEnd, inDepth);
                    int hEndClamp = min(hEnd, inHeight);
                    int wEndClamp = min(wEnd, inWidth);
                    poolSize = (dEndClamp - dStart) * (hEndClamp - hStart) * (wEndClamp - wStart);
                }
                int outIdx = ((b * channels + c) * outDepth + od) * outHeight * outWidth + oh * outWidth + ow;
                sum += gradOutput[outIdx] / float(max(poolSize, 1));
            }
        }
    }
    int inputIdx = ((b * channels + c) * inDepth + id) * inHeight * inWidth + ih * inWidth + iw;
    gradInput[inputIdx] = sum;
}";

    private const string ShBasis = @"
    float basis[16];
    basis[0] = 0.282095;
    if (degree >= 1) { basis[1] = 0.488603 * dy; basis[2] = 0.488603 * dz; basis[3] = 0.488603 * dx; }
    if (degree >= 2) {
        basis[4] = 1.092548 * dx * dy; basis[5] = 1.092548 * dy * dz;
        basis[6] = 0.315392 * (3.0 * dz * dz - 1.0);
        basis[7] = 1.092548 * dx * dz; basis[8] = 0.546274 * (dx * dx - dy * dy);
    }
    if (degree >= 3) {
        basis[9]  = 0.590044 * dy * (3.0 * dx * dx - dy * dy);
        basis[10] = 2.890611 * dx * dy * dz;
        basis[11] = 0.457046 * dy * (5.0 * dz * dz - 1.0);
        basis[12] = 0.373176 * dz * (5.0 * dz * dz - 3.0);
        basis[13] = 0.457046 * dx * (5.0 * dz * dz - 1.0);
        basis[14] = 1.445306 * dz * (dx * dx - dy * dy);
        basis[15] = 0.590044 * dx * (dx * dx - 3.0 * dy * dy);
    }";

    public static readonly string GaussianCovariance = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer Rotations { float rotations[]; };
layout(set=0,binding=1) readonly buffer Scales { float scales[]; };
layout(set=0,binding=2) writeonly buffer Covariances { float covariances[]; };
layout(push_constant) uniform PC { int numGaussians; };
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= numGaussians) return;
    float qw = rotations[i * 4], qx = rotations[i * 4 + 1], qy = rotations[i * 4 + 2], qz = rotations[i * 4 + 3];
    float qNorm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qNorm > 0.0) { float inv = 1.0 / qNorm; qw *= inv; qx *= inv; qy *= inv; qz *= inv; }
    float r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
    float r01 = 2.0 * (qx * qy - qw * qz);
    float r02 = 2.0 * (qx * qz + qw * qy);
    float r10 = 2.0 * (qx * qy + qw * qz);
    float r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
    float r12 = 2.0 * (qy * qz - qw * qx);
    float r20 = 2.0 * (qx * qz - qw * qy);
    float r21 = 2.0 * (qy * qz + qw * qx);
    float r22 = 1.0 - 2.0 * (qx * qx + qy * qy);
    float sx = max(1e-6, abs(scales[i * 3])); float sx2 = sx * sx;
    float sy = max(1e-6, abs(scales[i * 3 + 1])); float sy2 = sy * sy;
    float sz = max(1e-6, abs(scales[i * 3 + 2])); float sz2 = sz * sz;
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
}";

    public static readonly string SphericalHarmonics = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer ShCoefficients { float shCoefficients[]; };
layout(set=0,binding=1) readonly buffer ViewDirections { float viewDirections[]; };
layout(set=0,binding=2) writeonly buffer Output { float output_[]; };
layout(push_constant) uniform PC { int numPoints; int basisCount; int numChannels; int degree; int broadcastDir; };
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= numPoints * numChannels) return;
    int ch = idx % numChannels;
    int i = idx / numChannels;
    int dirIdx = broadcastDir != 0 ? 0 : i;
    float dx = viewDirections[dirIdx * 3], dy = viewDirections[dirIdx * 3 + 1], dz = viewDirections[dirIdx * 3 + 2];
    float norm = sqrt(dx * dx + dy * dy + dz * dz);
    if (norm > 0.0) { float inv = 1.0 / norm; dx *= inv; dy *= inv; dz *= inv; }" + ShBasis + @"
    float color = 0.0;
    for (int b = 0; b < basisCount; b++)
        color += shCoefficients[i * basisCount * numChannels + b * numChannels + ch] * basis[b];
    output_[i * numChannels + ch] = min(max(color, 0.0), 1.0);
}";

    public static readonly string SphericalHarmonicsBackward = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer ShCoefficients { float shCoefficients[]; };
layout(set=0,binding=1) readonly buffer ViewDirections { float viewDirections[]; };
layout(set=0,binding=2) readonly buffer OutputGradient { float outputGradient[]; };
layout(set=0,binding=3) writeonly buffer ShGrad { float shGrad[]; };
layout(push_constant) uniform PC { int numPoints; int basisCount; int numChannels; int degree; int broadcastDir; };
void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= numPoints * basisCount * numChannels) return;
    int ch = idx % numChannels;
    int b = (idx / numChannels) % basisCount;
    int i = idx / (basisCount * numChannels);
    int dirIdx = broadcastDir != 0 ? 0 : i;
    float dx = viewDirections[dirIdx * 3], dy = viewDirections[dirIdx * 3 + 1], dz = viewDirections[dirIdx * 3 + 2];
    float norm = sqrt(dx * dx + dy * dy + dz * dz);
    if (norm > 0.0) { float inv = 1.0 / norm; dx *= inv; dy *= inv; dz *= inv; }" + ShBasis + @"
    float preclamp = 0.0;
    for (int bb = 0; bb < basisCount; bb++)
        preclamp += shCoefficients[i * basisCount * numChannels + bb * numChannels + ch] * basis[bb];
    float colorGrad = outputGradient[i * numChannels + ch];
    if (preclamp < 0.0 || preclamp > 1.0) colorGrad = 0.0;
    shGrad[idx] = colorGrad * basis[b];
}";
}
