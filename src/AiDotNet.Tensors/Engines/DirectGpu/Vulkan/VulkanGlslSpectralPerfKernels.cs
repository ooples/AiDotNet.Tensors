// Copyright (c) AiDotNet. All rights reserved.
// GLSL compute shader sources for Issue #160 spectral perf kernels.
// Compiled at runtime to SPIR-V via shaderc.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan
{
    internal static class VulkanGlslSpectralPerfKernels
    {
        private const string Header = @"#version 450
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
";
        private const string TwoBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer InputA { float a[]; };
layout(set = 0, binding = 1) writeonly buffer OutputB { float b[]; };
";
        private const string ThreeBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer InputA { float a[]; };
layout(set = 0, binding = 1) readonly buffer InputB { float b[]; };
layout(set = 0, binding = 2) writeonly buffer OutputC { float c[]; };
";

        public static string Atan2Elementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    c[idx] = atan(a[idx], b[idx]);
}";

        public static string NormalizeRowsFused => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint rows; uint cols; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= rows) return;
    uint rowOff = row * cols;
    float sumSq = 0.0;
    for (uint c = 0; c < cols; c++) { float v = a[rowOff + c]; sumSq += v * v; }
    float invNorm = (sumSq > 0.0) ? inversesqrt(sumSq) : 0.0;
    for (uint c = 0; c < cols; c++) b[rowOff + c] = a[rowOff + c] * invNorm;
}";

        // Single-buffer variant: writes b[idx] = a[idx] * gain.
        // Used twice (once for real, once for imag) to implement the analytic-signal mask
        // since the GLSL helper is constrained to two buffers.
        public static string AnalyticSignalMaskScalar => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint fftSize; uint binLow; uint binHigh; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = batch * fftSize;
    if (idx >= total) return;
    uint k = idx % fftSize;
    uint halfN = fftSize >> 1;
    float gain;
    if (k == 0u || k == halfN) gain = (k < binLow || k >= binHigh) ? 0.0 : 1.0;
    else if (k < halfN)        gain = (k < binLow || k >= binHigh) ? 0.0 : 2.0;
    else                       gain = 0.0;
    b[idx] = a[idx] * gain;
}";

        public static string BispectrumReal => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint maxF1; uint maxF2; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = maxF1 * maxF2;
    if (idx >= total) return;
    uint f1 = idx / maxF2;
    uint f2 = idx % maxF2;
    uint sumIdx = f1 + f2;
    float ar = a[f1], ai = b[f1];
    float br = a[f2], bi = b[f2];
    float cr = a[sumIdx], ci = -b[sumIdx];
    float abr = ar * br - ai * bi;
    float abi = ar * bi + ai * br;
    c[idx] = abr * cr - abi * ci;
}";

        public static string BispectrumImag => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint maxF1; uint maxF2; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = maxF1 * maxF2;
    if (idx >= total) return;
    uint f1 = idx / maxF2;
    uint f2 = idx % maxF2;
    uint sumIdx = f1 + f2;
    float ar = a[f1], ai = b[f1];
    float br = a[f2], bi = b[f2];
    float cr = a[sumIdx], ci = -b[sumIdx];
    float abr = ar * br - ai * bi;
    float abi = ar * bi + ai * br;
    c[idx] = abr * ci + abi * cr;
}";

        public static string TrispectrumReal => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint maxF1; uint maxF2; uint maxF3; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = maxF1 * maxF2 * maxF3;
    if (idx >= total) return;
    uint f1 = idx / (maxF2 * maxF3);
    uint rem = idx - f1 * maxF2 * maxF3;
    uint f2 = rem / maxF3;
    uint f3 = rem - f2 * maxF3;
    uint sumIdx = f1 + f2 + f3;
    float ar = a[f1], ai = b[f1];
    float br = a[f2], bi = b[f2];
    float cr = a[f3], ci = b[f3];
    float dr = a[sumIdx], di = -b[sumIdx];
    float t1r = ar * br - ai * bi;
    float t1i = ar * bi + ai * br;
    float t2r = t1r * cr - t1i * ci;
    float t2i = t1r * ci + t1i * cr;
    c[idx] = t2r * dr - t2i * di;
}";

        public static string TrispectrumImag => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint maxF1; uint maxF2; uint maxF3; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = maxF1 * maxF2 * maxF3;
    if (idx >= total) return;
    uint f1 = idx / (maxF2 * maxF3);
    uint rem = idx - f1 * maxF2 * maxF3;
    uint f2 = rem / maxF3;
    uint f3 = rem - f2 * maxF3;
    uint sumIdx = f1 + f2 + f3;
    float ar = a[f1], ai = b[f1];
    float br = a[f2], bi = b[f2];
    float cr = a[f3], ci = b[f3];
    float dr = a[sumIdx], di = -b[sumIdx];
    float t1r = ar * br - ai * bi;
    float t1i = ar * bi + ai * br;
    float t2r = t1r * cr - t1i * ci;
    float t2i = t1r * ci + t1i * cr;
    c[idx] = t2r * di + t2i * dr;
}";

        public static string CavityBounceReal => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint total; uint invNBits; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= total) return;
    float invN = uintBitsToFloat(invNBits);
    float r = a[idx] * invN;
    r = clamp(r, -20.0, 20.0);
    b[idx] = tanh(r);
}";

        public static string ZeroBuffer => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint total; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= total) return;
    b[idx] = 0.0;
}";

        public static string WidebandLogBinPool => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint totalSegBatch; uint fftSize; uint numBins; uint usable; };
void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint total = totalSegBatch * numBins;
    if (outIdx >= total) return;
    uint seg = outIdx / numBins;
    uint k = outIdx % numBins;
    float r0 = float(k) / float(numBins);
    float r1 = float(k + 1u) / float(numBins);
    int binStart = 1 + int(r0 * r0 * float(usable - 1u));
    int binEnd = 1 + int(r1 * r1 * float(usable - 1u));
    if (binEnd <= binStart) binEnd = binStart + 1;
    if (binEnd > int(usable)) binEnd = int(usable);
    uint magOff = seg * fftSize;
    float sum = 0.0; int cnt = 0;
    for (int i = binStart; i < binEnd; i++) { sum += a[magOff + uint(i)]; cnt++; }
    float avg = (cnt > 0) ? (sum / float(cnt)) : 0.0;
    b[outIdx] = log(1.0 + avg);
}";

        public static string MelFilterbankApply => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint totalSegBatch; uint specBins; uint melBins; };
void main() {
    uint outIdx = gl_GlobalInvocationID.x;
    uint total = totalSegBatch * melBins;
    if (outIdx >= total) return;
    uint seg = outIdx / melBins;
    uint m = outIdx % melBins;
    uint powerOff = seg * specBins;
    uint filtOff = m * specBins;
    float sum = 0.0;
    for (uint i = 0u; i < specBins; i++) sum += a[powerOff + i] * b[filtOff + i];
    c[outIdx] = sum;
}";

        public static string MfccLog1p => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    b[idx] = log(1.0 + a[idx]);
}";

        public static string PacPhaseBinMi => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint numSamples; uint numGammaBands; uint gammaIdx; };
void main() {
    uint b_idx = gl_GlobalInvocationID.x;
    if (b_idx >= batch) return;
    const uint NUM_PHASE_BINS = 18u;
    float binSum[18];
    float binCount[18];
    for (uint k = 0u; k < NUM_PHASE_BINS; k++) { binSum[k] = 0.0; binCount[k] = 0.0; }
    uint sampleOff = b_idx * numSamples;
    uint gammaOff = (gammaIdx * batch + b_idx) * numSamples;
    for (uint i = 0u; i < numSamples; i++) {
        float phase = a[sampleOff + i];
        float amp = b[gammaOff + i];
        float fbin = (phase + 3.14159265358979) / (2.0 * 3.14159265358979) * float(NUM_PHASE_BINS);
        int bin = int(fbin);
        if (bin < 0) bin = 0;
        if (bin >= int(NUM_PHASE_BINS)) bin = int(NUM_PHASE_BINS) - 1;
        binSum[bin] += amp;
        binCount[bin] += 1.0;
    }
    float totalAmp = 0.0;
    for (uint k = 0u; k < NUM_PHASE_BINS; k++) {
        float avg = (binCount[k] > 0.0) ? (binSum[k] / binCount[k]) : 0.0;
        totalAmp += avg;
    }
    float mi = 0.0;
    if (totalAmp > 0.0) {
        float entropy = 0.0;
        for (uint k = 0u; k < NUM_PHASE_BINS; k++) {
            float avg = (binCount[k] > 0.0) ? (binSum[k] / binCount[k]) : 0.0;
            float p = avg / totalAmp;
            if (p > 1e-12) entropy -= p * log(p);
        }
        mi = (log(float(NUM_PHASE_BINS)) - entropy) / log(float(NUM_PHASE_BINS));
    }
    c[b_idx * numGammaBands + gammaIdx] = mi;
}";
    }
}
