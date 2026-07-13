namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// GLSL audit kernels (translated from verified CUDA-C). Validated on Vulkan HW later.
public static class VulkanAuditKernels
{
    public static string ReflectPad1d => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float outp[]; };
layout(push_constant) uniform PC {
    int batch;
    int L;
    int Lp;
    int pad;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch*Lp) return;
    int j = idx % Lp; int b = idx / Lp; int src;
    if (j < pad) src = pad - j; else if (j < pad + L) src = j - pad; else src = L - 2 - (j - pad - L);
    outp[idx] = inp[b*L + src];

}";

    public static string StftMagPhase => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float padded[]; };
layout(set=0, binding=1) buffer B1 { float window[]; };
layout(set=0, binding=2) buffer B2 { float mag[]; };
layout(set=0, binding=3) buffer B3 { float phase[]; };
layout(push_constant) uniform PC {
    int batch;
    int Lp;
    int nFft;
    int hop;
    int numFrames;
    int numFreqs;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; int total = batch*numFreqs*numFrames; if (idx >= total) return;
    int frame = idx % numFrames; int tmp = idx / numFrames; int k = tmp % numFreqs; int b = tmp / numFreqs;
    int start = frame * hop; int inOff = b * Lp; float re = 0.0, im = 0.0;
    float bk = -2.0 * float(M_PI) * float(k) / float(nFft);
    for (int i = 0; i < nFft; i++) { float x = padded[inOff + start + i] * window[i]; float a = bk * float(i); re += x*cos(a); im += x*sin(a); }
    int outOff = b*numFreqs*numFrames + k*numFrames + frame;
    mag[outOff] = sqrt(re*re + im*im); phase[outOff] = atan(im, re);

}";

    public static string PhaseVocoder => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float mag[]; };
layout(set=0, binding=1) buffer B1 { float phase[]; };
layout(set=0, binding=2) buffer B2 { float newMag[]; };
layout(set=0, binding=3) buffer B3 { float newPhase[]; };
layout(push_constant) uniform PC {
    int leading;
    int nFramesV;
    int nFreqV;
    int outFrames;
    float rate;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= leading*nFreqV) return;
    int f = idx % nFreqV; int b = idx / nFreqV; int stride = nFramesV*nFreqV; int outStride = outFrames*nFreqV;
    float accPhase = 0.0;
    for (int t = 0; t < outFrames; t++) {
        float srcT = float(t) * rate; int t0 = int(floor(srcT)); int t1 = min(t0+1, nFramesV-1); float frac = srcT - float(t0);
        float m0 = mag[b*stride + t0*nFreqV + f]; float m1 = mag[b*stride + t1*nFreqV + f];
        newMag[b*outStride + t*nFreqV + f] = (1.0-frac)*m0 + frac*m1;
        float dp = 0.0;
        if (t0+1 < nFramesV) { dp = phase[b*stride + (t0+1)*nFreqV + f] - phase[b*stride + t0*nFreqV + f]; dp -= 2.0*float(M_PI) * round(dp/(2.0*float(M_PI))); }
        accPhase += dp; newPhase[b*outStride + t*nFreqV + f] = accPhase;
    }

}";

    public static string BuildSpectrum => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float mag[]; };
layout(set=0, binding=1) buffer B1 { float phase[]; };
layout(set=0, binding=2) buffer B2 { float specRe[]; };
layout(set=0, binding=3) buffer B3 { float specIm[]; };
layout(push_constant) uniform PC {
    int batch;
    int numFreqs;
    int numFrames;
    int nFft;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch*numFrames) return;
    int frame = idx % numFrames; int b = idx / numFrames; int magOff = b*numFreqs*numFrames; int specOff = idx * nFft;
    for (int k = 0; k < nFft; k++) { specRe[specOff+k] = 0.0; specIm[specOff+k] = 0.0; }
    for (int k = 0; k < numFreqs && k < nFft; k++) { float m = mag[magOff + k*numFrames + frame]; float p = phase[magOff + k*numFrames + frame]; specRe[specOff+k] = m*cos(p); specIm[specOff+k] = m*sin(p); }
    for (int k = 1; k < numFreqs - 1; k++) { int dst = nFft - k; if (dst >= 0 && dst < nFft && k < nFft) { specRe[specOff+dst] = specRe[specOff+k]; specIm[specOff+dst] = -specIm[specOff+k]; } }

}";

    public static string IstftFromSpectrum => @"#version 450
#extension GL_EXT_shader_atomic_float : enable
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float specRe[]; };
layout(set=0, binding=1) buffer B1 { float specIm[]; };
layout(set=0, binding=2) buffer B2 { float window[]; };
layout(set=0, binding=3) buffer B3 { uint result[]; };
layout(set=0, binding=4) buffer B4 { uint windowSum[]; };
layout(push_constant) uniform PC {
    int batch;
    int numFrames;
    int nFft;
    int hop;
    int outputLength;
    int center;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; int total = batch*numFrames*nFft; if (idx >= total) return;
    int i = idx % nFft; int tmp = idx / nFft; int frame = tmp % numFrames; int b = tmp / numFrames; int specOff = (b*numFrames + frame) * nFft;
    float acc = 0.0;
    for (int k = 0; k < nFft; k++) { float a = 2.0*float(M_PI)*float(k)*float(i)/float(nFft); acc += specRe[specOff+k]*cos(a) - specIm[specOff+k]*sin(a); }
    int writeStart = center ? max(0, frame*hop - nFft/2) : frame*hop; int outIdx = writeStart + i;
    if (outIdx >= 0 && outIdx < outputLength) { float w = window[i]; atomicAdd(result[b*outputLength + outIdx], acc*(1.0/float(nFft))*w); atomicAdd(windowSum[b*outputLength + outIdx], w*w); }

}";

    public static string IstftNormalize => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float result[]; };
layout(set=0, binding=1) buffer B1 { float windowSum[]; };
layout(push_constant) uniform PC {
    int total;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= total) return;
    float ws = windowSum[idx]; if (ws > 1e-8) result[idx] = result[idx] / ws;

}";

    public static string LogicalOp => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float a[]; };
layout(set=0, binding=1) buffer B1 { float b[]; };
layout(set=0, binding=2) buffer B2 { float out[]; };
layout(push_constant) uniform PC {
    int mode;
    int n;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int i = int(gl_GlobalInvocationID.x); if (i >= n) return;
    int ba = (a[i] != 0.0); int bb = (b[i] != 0.0);
    int r = (mode == 0) ? (ba && bb) : (mode == 1) ? (ba || bb) : (ba != bb);
    out[i] = r ? 1.0 : 0.0;

}";

    public static string LogicalNot => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float a[]; };
layout(set=0, binding=1) buffer B1 { float out[]; };
layout(push_constant) uniform PC {
    int n;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int i = int(gl_GlobalInvocationID.x); if (i >= n) return;
    out[i] = (a[i] != 0.0) ? 0.0 : 1.0;

}";

    public static string ShiftedDiff => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float x[]; };
layout(set=0, binding=1) buffer B1 { float mask[]; };
layout(push_constant) uniform PC {
    int n;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int i = int(gl_GlobalInvocationID.x); if (i >= n) return;
    mask[i] = (i == 0 || x[i] != x[i-1]) ? 1.0 : 0.0;

}";

    public static string MasksToBoxes => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float masks[]; };
layout(set=0, binding=1) buffer B1 { float out[]; };
layout(push_constant) uniform PC {
    int N;
    int H;
    int W;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int n = int(gl_GlobalInvocationID.x); if (n >= N) return;
    int xMin = W, yMin = H, xMax = -1, yMax = -1; int planeOff = n * H * W;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (masks[planeOff + y*W + x] != 0.0) { if (x<xMin)xMin=x; if (x>xMax)xMax=x; if (y<yMin)yMin=y; if (y>yMax)yMax=y; }
    int o = n * 4;
    if (xMax < 0) { out[o]=0.0; out[o+1]=0.0; out[o+2]=0.0; out[o+3]=0.0; }
    else { out[o]=float(xMin); out[o+1]=float(yMin); out[o+2]=float(xMax); out[o+3]=float(yMax); }

}";

    public static string PairwiseIou => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float boxes[]; };
layout(set=0, binding=1) buffer B1 { float iou[]; };
layout(push_constant) uniform PC {
    int N;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx >= N*N) return; int j = idx % N; int i = idx / N;
    float ix1=boxes[i*4],iy1=boxes[i*4+1],ix2=boxes[i*4+2],iy2=boxes[i*4+3];
    float jx1=boxes[j*4],jy1=boxes[j*4+1],jx2=boxes[j*4+2],jy2=boxes[j*4+3];
    float ai=max(0.0,ix2-ix1)*max(0.0,iy2-iy1); float aj=max(0.0,jx2-jx1)*max(0.0,jy2-jy1);
    float iw=max(0.0,min(ix2,jx2)-max(ix1,jx1)); float ih=max(0.0,min(iy2,jy2)-max(iy1,jy1));
    float inter=iw*ih; float uni=ai+aj-inter; iou[idx]=(uni>0.0)?inter/uni:0.0;

}";

    public static string Histogramdd => @"#version 450
#extension GL_EXT_shader_atomic_float : enable
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float samples[]; };
layout(set=0, binding=1) buffer B1 { uint hist[]; };
layout(set=0, binding=2) buffer B2 { int bins[]; };
layout(set=0, binding=3) buffer B3 { float mins[]; };
layout(set=0, binding=4) buffer B4 { float maxs[]; };
layout(push_constant) uniform PC {
    int n;
    int d;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int i = int(gl_GlobalInvocationID.x); if (i >= n) return; int linIdx = 0; int valid = 1;
    for (int k = 0; k < d; k++) { float v = samples[i*d + k]; float mn = mins[k]; float mx = maxs[k]; if (!(v >= mn && v <= mx)) { valid = 0; break; } float width = (mx - mn) / float(bins[k]); int kIdx = int(floor((v - mn) / width)); if (kIdx >= bins[k]) kIdx = bins[k] - 1; if (kIdx < 0) kIdx = 0; linIdx = linIdx * bins[k] + kIdx; }
    if (valid) atomicAdd(hist[linIdx], 1.0);

}";

    public static string GridsampleBackwardInput => @"#version 450
#extension GL_EXT_shader_atomic_float : enable
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float gradOut[]; };
layout(set=0, binding=1) buffer B1 { float grid[]; };
layout(set=0, binding=2) buffer B2 { uint gradIn[]; };
layout(push_constant) uniform PC {
    int batch;
    int H;
    int W;
    int C;
    int outH;
    int outW;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); int total = batch*outH*outW*C; if (idx >= total) return;
    int ow = idx % outW; int tmp = idx / outW; int oh = tmp % outH; tmp /= outH; int c = tmp % C; int b = tmp / C;
    int gridBase = ((b*outH + oh)*outW + ow)*2; float gx = grid[gridBase]; float gy = grid[gridBase+1];
    float srcH = (gy + 1.0) * 0.5 * float((H - 1)); float srcW = (gx + 1.0) * 0.5 * float((W - 1));
    if (srcH <= -1.0 || srcH >= float(H) || srcW <= -1.0 || srcW >= float(W)) return;
    int h0 = int(floor(srcH)); int h1 = h0 + 1; int w0 = int(floor(srcW)); int w1 = w0 + 1;
    float lh = srcH - float(h0); float lw = srcW - float(w0); float g = gradOut[idx];
    if (h0>=0 && h0<H && w0>=0 && w0<W) atomicAdd(gradIn[((b*C+c)*H+h0)*W+w0], g*(1.0-lh)*(1.0-lw));
    if (h0>=0 && h0<H && w1>=0 && w1<W) atomicAdd(gradIn[((b*C+c)*H+h0)*W+w1], g*(1.0-lh)*lw);
    if (h1>=0 && h1<H && w0>=0 && w0<W) atomicAdd(gradIn[((b*C+c)*H+h1)*W+w0], g*lh*(1.0-lw));
    if (h1>=0 && h1<H && w1>=0 && w1<W) atomicAdd(gradIn[((b*C+c)*H+h1)*W+w1], g*lh*lw);

}";

    public static string GridsampleBackwardGrid => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float gradOut[]; };
layout(set=0, binding=1) buffer B1 { float input[]; };
layout(set=0, binding=2) buffer B2 { float grid[]; };
layout(set=0, binding=3) buffer B3 { float gradGrid[]; };
layout(push_constant) uniform PC {
    int batch;
    int H;
    int W;
    int C;
    int outH;
    int outW;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx >= batch*outH*outW) return;
    int ow = idx % outW; int tmp = idx / outW; int oh = tmp % outH; int b = tmp / outH;
    int gridBase = ((b*outH+oh)*outW+ow)*2; float gx = grid[gridBase]; float gy = grid[gridBase+1];
    float srcH = (gy+1.0)*0.5*float((H-1)); float srcW = (gx+1.0)*0.5*float((W-1));
    float gradGx = 0.0; float gradGy = 0.0;
    if (!(srcH <= -1.0 || srcH >= float(H) || srcW <= -1.0 || srcW >= float(W))) {
        int h0 = int(floor(srcH)); int h1 = h0+1; int w0 = int(floor(srcW)); int w1 = w0+1;
        float lh = srcH-float(h0); float lw = srcW-float(w0);
        int in00 = (h0>=0&&h0<H&&w0>=0&&w0<W); int in01 = (h0>=0&&h0<H&&w1>=0&&w1<W);
        int in10 = (h1>=0&&h1<H&&w0>=0&&w0<W); int in11 = (h1>=0&&h1<H&&w1>=0&&w1<W);
        for (int c = 0; c < C; c++) {
            float v00 = in00 ? input[((b*C+c)*H+h0)*W+w0] : 0.0; float v01 = in01 ? input[((b*C+c)*H+h0)*W+w1] : 0.0;
            float v10 = in10 ? input[((b*C+c)*H+h1)*W+w0] : 0.0; float v11 = in11 ? input[((b*C+c)*H+h1)*W+w1] : 0.0;
            float dH = (1.0-lw)*(v10-v00) + lw*(v11-v01); float dW = (1.0-lh)*(v01-v00) + lh*(v11-v10);
            float go = gradOut[((b*C+c)*outH+oh)*outW+ow];
            gradGx += go * dW * float((W-1))*0.5; gradGy += go * dH * float((H-1))*0.5;
        }
    }
    gradGrid[gridBase] = gradGx; gradGrid[gridBase+1] = gradGy;

}";

    public static string TakeAlongDimF => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float input[]; };
layout(set=0, binding=1) buffer B1 { float indices[]; };
layout(set=0, binding=2) buffer B2 { float output[]; };
layout(push_constant) uniform PC {
    int outerSize;
    int axisOut;
    int innerSize;
    int axisIn;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); int total = outerSize*axisOut*innerSize; if (idx >= total) return;
    int inner = idx % innerSize; int outer = (idx / innerSize) / axisOut; int srcJ = int(indices[idx]);
    if (srcJ < 0 || srcJ >= axisIn) { output[idx] = 0.0; return; }
    output[idx] = input[(outer * axisIn + srcJ) * innerSize + inner];

}";

    public static string Cross3 => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float a[]; };
layout(set=0, binding=1) buffer B1 { float b[]; };
layout(set=0, binding=2) buffer B2 { float output[]; };
layout(push_constant) uniform PC {
    int outerSize;
    int innerSize;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx >= outerSize * innerSize) return;
    int inner = idx % innerSize; int outer = idx / innerSize; int p = outer * 3 * innerSize + inner;
    float a0=a[p],a1=a[p+innerSize],a2=a[p+2*innerSize]; float b0=b[p],b1=b[p+innerSize],b2=b[p+2*innerSize];
    output[p]=a1*b2-a2*b1; output[p+innerSize]=a2*b0-a0*b2; output[p+2*innerSize]=a0*b1-a1*b0;

}";

    public static string Ldexp => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float input[]; };
layout(set=0, binding=1) buffer B1 { int exponents[]; };
layout(set=0, binding=2) buffer B2 { float output[]; };
layout(push_constant) uniform PC {
    int size;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx >= size) return; output[idx] = ldexp(input[idx], exponents[idx]);

}";

    public static string Kron2d => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float a[]; };
layout(set=0, binding=1) buffer B1 { float b[]; };
layout(set=0, binding=2) buffer B2 { float output[]; };
layout(push_constant) uniform PC {
    int am;
    int an;
    int bp;
    int bq;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); int outCols = an*bq; int total=(am*bp)*outCols; if (idx>=total) return;
    int oc=idx%outCols; int orow=idx/outCols; int i=orow/bp; int k=orow%bp; int j=oc/bq; int l=oc%bq;
    output[idx] = a[i*an+j] * b[k*bq+l];

}";

    public static string SearchSorted => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float seq[]; };
layout(set=0, binding=1) buffer B1 { float values[]; };
layout(set=0, binding=2) buffer B2 { float output[]; };
layout(push_constant) uniform PC {
    int seqLen;
    int numValues;
    int right;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx >= numValues) return; float v = values[idx]; int lo=0, hi=seqLen;
    while (lo < hi) { int mid=(lo+hi)>>1; int cond=(right!=0)?(seq[mid]<=v):(seq[mid]<v); if (cond) lo=mid+1; else hi=mid; }
    output[idx] = float(lo);

}";

    public static string NextAfter => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float a[]; };
layout(set=0, binding=1) buffer B1 { float b[]; };
layout(set=0, binding=2) buffer B2 { float output[]; };
layout(push_constant) uniform PC {
    int size;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx >= size) return; float av=a[idx], bv=b[idx];
    uint ua=floatBitsToUint(av), ub=floatBitsToUint(bv);
    int aNan=(((ua>>23)&0xFFu)==0xFFu)&&((ua&0x7FFFFFu)!=0u); int bNan=(((ub>>23)&0xFFu)==0xFFu)&&((ub&0x7FFFFFu)!=0u);
    if (aNan||bNan) { output[idx]=uintBitsToFloat(0x7FC00000u); return; }
    if (av==bv) { output[idx]=bv; return; }
    if (av==0.0) { output[idx]=uintBitsToFloat(bv>0.0?0x00000001u:0x80000001u); return; }
    uint r=ua;
    if (bv>av) r=(av>0.0)?(r+1u):(r-1u); else r=(av>0.0)?(r-1u):(r+1u);
    output[idx]=uintBitsToFloat(r);

}";

    public static string IndexWrite => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float output[]; };
layout(set=0, binding=1) buffer B1 { int indices[]; };
layout(set=0, binding=2) buffer B2 { float source[]; };
layout(push_constant) uniform PC {
    float fillValue;
    int mode;
    int outerSize;
    int idxAxis;
    int innerSize;
    int dstAxis;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); int total=outerSize*idxAxis*innerSize; if (idx>=total) return;
    int inner=idx%innerSize; int j=(idx/innerSize)%idxAxis; int outer=(idx/innerSize)/idxAxis; int dstJ=indices[j];
    if (dstJ<0||dstJ>=dstAxis) return; float v=(mode==0)?source[idx]:fillValue;
    output[(outer*dstAxis+dstJ)*innerSize+inner]=v;

}";

    public static string Cdist => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float x1[]; };
layout(set=0, binding=1) buffer B1 { float x2[]; };
layout(set=0, binding=2) buffer B2 { float output[]; };
layout(push_constant) uniform PC {
    int m;
    int n;
    int d;
    float p;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx>=m*n) return; int j=idx%n; int i=idx/n; float sum=0.0;
    for (int k=0;k<d;k++){ float diff=abs(x1[i*d+k]-x2[j*d+k]); if (p==1.0) sum+=diff; else if (p==2.0) sum+=diff*diff; else sum+=pow(diff,p); }
    output[idx]=(p==1.0)?sum:(p==2.0)?sqrt(sum):pow(sum,1.0/p);

}";

    public static string Pdist => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float input[]; };
layout(set=0, binding=1) buffer B1 { float output[]; };
layout(push_constant) uniform PC {
    int n;
    int d;
    float p;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int flat = int(gl_GlobalInvocationID.x); if (flat>=n*n) return; int j=flat%n; int i=flat/n; if (i>=j) return; float sum=0.0;
    for (int k=0;k<d;k++){ float diff=abs(input[i*d+k]-input[j*d+k]); if (p==1.0) sum+=diff; else if (p==2.0) sum+=diff*diff; else sum+=pow(diff,p); }
    float dist=(p==1.0)?sum:(p==2.0)?sqrt(sum):pow(sum,1.0/p); int outIdx=i*n-(i*(i+1))/2+(j-i-1); output[outIdx]=dist;

}";

    public static string Histc => @"#version 450
#extension GL_EXT_shader_atomic_float : enable
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float input[]; };
layout(set=0, binding=1) buffer B1 { uint hist[]; };
layout(push_constant) uniform PC {
    int n;
    int bins;
    float mn;
    float mx;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx>=n) return; float x=input[idx]; if (!(x>=mn&&x<=mx)) return;
    float bw=(mx-mn)/float(bins); int b=int(((x-mn)/bw)); if (b>=bins) b=bins-1; if (b<0) b=0; atomicAdd(hist[b], 1.0);

}";

    public static string BitonicStep => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float values[]; };
layout(set=0, binding=1) buffer B1 { float indices[]; };
layout(push_constant) uniform PC {
    int rowLen;
    int k;
    int j;
    int numRows;
    int descending;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int gid = int(gl_GlobalInvocationID.x); if (gid>=numRows*rowLen) return; int i=gid%rowLen; int ixj=i^j; if (ixj<=i) return;
    int base=(gid/rowLen)*rowLen; float a=values[base+i], b=values[base+ixj];
    uint ua=floatBitsToUint(a), ub=floatBitsToUint(b);
    float ka=(((ua>>23)&0xFFu)==0xFFu&&(ua&0x7FFFFFu)!=0u)?P210_INF:a; float kb=(((ub>>23)&0xFFu)==0xFFu&&(ub&0x7FFFFFu)!=0u)?P210_INF:b;
    int up=((i&k)==0); if (descending!=0) up=!up; int doSwap=up?(ka>kb):(ka<kb);
    if (doSwap) { values[base+i]=b; values[base+ixj]=a; float t=indices[base+i]; indices[base+i]=indices[base+ixj]; indices[base+ixj]=t; }

}";

    public static string CopyRows => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float src[]; };
layout(set=0, binding=1) buffer B1 { float dst[]; };
layout(push_constant) uniform PC {
    int srcRowLen;
    int dstRowLen;
    int numRows;
    int copyLen;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int gid = int(gl_GlobalInvocationID.x); if (gid>=numRows*copyLen) return; int i=gid%copyLen; int r=gid/copyLen; dst[r*dstRowLen+i]=src[r*srcRowLen+i];

}";

    public static string IotaPad => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float idx[]; };
layout(push_constant) uniform PC {
    int L;
    int P;
    int numRows;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int gid = int(gl_GlobalInvocationID.x); if (gid>=numRows*P) return; int i=gid%P; idx[gid]=(i<L)?float(i):-1.0;

}";

    public static string HsoftmaxPaths => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float acts[]; };
layout(set=0, binding=1) buffer B1 { float out[]; };
layout(push_constant) uniform PC {
    int rows;
    int treeDepth;
    int numClasses;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx>=rows*numClasses) return; int c=idx%numClasses; int r=idx/numClasses; int gbase=r*treeDepth; float prob=1.0; int node=1;
    for (int level=0; level<treeDepth; level++) { int goRight=(c&(1<<(treeDepth-level-1)))!=0; float g=1.0/(1.0+exp(-acts[gbase+level])); prob*=goRight?g:(1.0-g); node=node*2+(goRight?1:0); if (node>=numClasses) break; }
    out[idx]=prob;

}";

    public static string Isin => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float elements[]; };
layout(set=0, binding=1) buffer B1 { float sortedTest[]; };
layout(set=0, binding=2) buffer B2 { float mask[]; };
layout(push_constant) uniform PC {
    int numElements;
    int testLen;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx>=numElements) return; float v=elements[idx]; int lo=0, hi=testLen;
    while (lo<hi) { int mid=(lo+hi)>>1; if (sortedTest[mid]<v) lo=mid+1; else hi=mid; } mask[idx]=(lo<testLen&&sortedTest[lo]==v)?1.0:0.0;

}";

    public static string CopyBlock2d => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float block[]; };
layout(set=0, binding=1) buffer B1 { float output[]; };
layout(push_constant) uniform PC {
    int blockRows;
    int blockCols;
    int totalCols;
    int rowOff;
    int colOff;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx>=blockRows*blockCols) return; int j=idx%blockCols; int i=idx/blockCols; output[(rowOff+i)*totalCols+(colOff+j)]=block[i*blockCols+j];

}";

    public static string ScatterReduce => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { uint output[]; };
layout(set=0, binding=1) buffer B1 { float source[]; };
layout(set=0, binding=2) buffer B2 { int indexb[]; };
layout(push_constant) uniform PC { int outerSize; int srcDim; int dstDim; int innerSize; int mode; };
void main() {
    int idx = int(gl_GlobalInvocationID.x); int total = outerSize*srcDim*innerSize; if (idx>=total) return;
    int inner=idx%innerSize; int tmp=idx/innerSize; int outer=tmp/srcDim; int t=indexb[idx];
    if (t<0||t>=dstDim) return; int dst=(outer*dstDim+t)*innerSize+inner; float val=source[idx];
    uint old=output[dst]; bool ok=false;
    while(!ok){ float pf=uintBitsToFloat(old); float nv=(mode==0)?(pf+val):(mode==1)?(pf*val):(mode==2)?max(pf,val):min(pf,val);
        uint des=floatBitsToUint(nv); uint prev=atomicCompSwap(output[dst], old, des); ok=(prev==old); old=prev; }
}
";

    public static string Unfold => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float src[]; };
layout(set=0, binding=1) buffer B1 { float dst[]; };
layout(push_constant) uniform PC {
    int outerSize;
    int dimSize;
    int innerSize;
    int nWindows;
    int size;
    int step;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); int total=outerSize*nWindows*innerSize*size; if (idx>=total) return; int s=idx%size; int tmp=idx/size; int inner=tmp%innerSize; tmp/=innerSize; int w=tmp%nWindows; int outer=tmp/nWindows;
    dst[idx]=src[(outer*dimSize+(w*step+s))*innerSize+inner];

}";

    public static string ClassifyFloat => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float a[]; };
layout(set=0, binding=1) buffer B1 { float output[]; };
layout(push_constant) uniform PC {
    int mode;
    int size;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int idx = int(gl_GlobalInvocationID.x); if (idx>=size) return; uint bits=floatBitsToUint(a[idx]); uint expo=(bits>>23)&0xFFu; uint mant=bits&0x7FFFFFu;
    int isNan=(expo==0xFFu)&&(mant!=0u); int isInf=(expo==0xFFu)&&(mant==0u); float r;
    if (mode==0) r=isNan?1.0:0.0; else if (mode==1) r=isInf?1.0:0.0; else r=(!isNan&&!isInf)?1.0:0.0; output[idx]=r;

}";

    public static string Zeta => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float x[]; };
layout(set=0, binding=1) buffer B1 { float q[]; };
layout(set=0, binding=2) buffer B2 { float out[]; };
layout(push_constant) uniform PC {
    int size;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}


float p210_zeta_scalar(float x, float q) {
    if (x == 1.0) return P210_INF;
    if (q <= 0.0 && q == floor(q)) return P210_INF;
    float sum = 0.0;
    for (int k = 0; k < 12; k++) sum += pow(q + float(k), -x);
    float Nq = 12.0 + q; float lnNq = log(Nq);
    float cont = exp((1.0 - x) * lnNq) / (x - 1.0);
    float halfTerm = 0.5 * exp(-x * lnNq);
    float[8] b2k = float[8](1.0/6.0, -1.0/30.0, 1.0/42.0, -1.0/30.0, 5.0/66.0, -691.0/2730.0, 7.0/6.0, -3617.0/510.0);
    float[8] fact2k = float[8](2.0, 24.0, 720.0, 40320.0, 3628800.0, 479001600.0, 87178291200.0, 20922789888000.0);
    float corr = 0.0; float xPow = exp(-x * lnNq) / Nq; float invNq2 = 1.0 / (Nq * Nq); float rising = x;
    for (int j = 1; j <= 8; j++) {
        if (j > 1) { rising *= (x + 2.0*float(j-1) - 2.0) * (x + 2.0*float(j-1) - 1.0); xPow *= invNq2; }
        corr += (b2k[j-1] / fact2k[j-1]) * rising * xPow;
    }
    return sum + cont + halfTerm + corr;
}

void main() {

    int i = int(gl_GlobalInvocationID.x); if (i>=size) return; out[i]=p210_zeta_scalar(x[i], q[i]);

}";

    public static string Polygamma => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float x[]; };
layout(set=0, binding=1) buffer B1 { float out[]; };
layout(push_constant) uniform PC {
    int n;
    int size;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}


float p210_polygamma_scalar(int n, float x) {
    if (x <= 0.0 && x == floor(x)) return P210_INF;
    float recurrence = 0.0; float xd = x; int signN = (n & 1) == 1 ? 1 : -1;
    while (xd < 10.0) { recurrence += float(signN) * exp(p210_lgamma(float(n+1)) - float(n+1) * log(xd)); xd += 1.0; }
    float lnX = log(xd);
    float asympt = exp(p210_lgamma(float(n)) - float(n) * lnX) + 0.5 * exp(p210_lgamma(float(n+1)) - float(n+1) * lnX);
    float[8] b2k = float[8](1.0/6.0, -1.0/30.0, 1.0/42.0, -1.0/30.0, 5.0/66.0, -691.0/2730.0, 7.0/6.0, -3617.0/510.0);
    float invX2 = 1.0 / (xd * xd); float xPow = exp(-float(n) * lnX);
    for (int k = 1; k <= 8; k++) { xPow *= invX2; asympt += b2k[k-1] * exp(p210_lgamma(float(2*k+n)) - p210_lgamma(float(2*k+1))) * xPow; }
    return recurrence + float(signN) * asympt;
}

void main() {

    int i = int(gl_GlobalInvocationID.x); if (i>=size) return; out[i]=p210_polygamma_scalar(n, x[i]);

}";

    public static string Rwkv7Forward => @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float R[]; };
layout(set=0, binding=1) buffer B1 { float K[]; };
layout(set=0, binding=2) buffer B2 { float V[]; };
layout(set=0, binding=3) buffer B3 { float A[]; };
layout(set=0, binding=4) buffer B4 { float B[]; };
layout(set=0, binding=5) buffer B5 { float outp[]; };
layout(set=0, binding=6) buffer B6 { float Sbuf[]; };
layout(push_constant) uniform PC {
    int batch;
    int seqLen;
    int modelDim;
    int numHeads;
    int headDim;
};

#define M_PI 3.14159265358979323846
#define P210_INF (uintBitsToFloat(0x7F800000u))
float p210_lgamma(float x) {
    float[9] c = float[9](0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7);
    float xx = x - 1.0; float a = c[0]; float t = xx + 7.5;
    for (int i = 1; i < 9; i++) a += c[i] / (xx + float(i));
    return 0.5*log(2.0*M_PI) + (xx+0.5)*log(t) - t + log(a);
}

void main() {

    int bh = int(gl_GlobalInvocationID.x); if (bh>=batch*numHeads) return; int b=bh/numHeads; int h=bh%numHeads; int hOff=h*headDim; int hh=headDim*headDim; float* S=Sbuf+bh*hh;
    for (int i=0;i<hh;i++) S[i]=0.0;
    for (int t=0;t<seqLen;t++) {
        int baseOff=(b*seqLen+t)*modelDim+hOff;
        for (int di=0;di<headDim;di++) { float ga=1.0/(1.0+exp(-A[baseOff+di])); float gbk=(1.0/(1.0+exp(-B[baseOff+di])))*K[baseOff+di]; int srow=di*headDim; for (int vi=0;vi<headDim;vi++) S[srow+vi]=ga*S[srow+vi]+gbk*V[baseOff+vi]; }
        for (int di=0;di<headDim;di++) { int srow=di*headDim; float sk=0.0; for (int vi=0;vi<headDim;vi++) sk+=S[srow+vi]*K[baseOff+vi]; outp[baseOff+di]=(1.0/(1.0+exp(-R[baseOff+di])))*sk; }
    }

}";
}
