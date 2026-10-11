namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels that <see cref="IGpuBatchExecution"/> needs and no other OpenCL program provides (#1142):
/// whole-tensor reductions that overwrite their output, the reduction backwards, axis slices, the crop
/// backward, an unbounded sparsemax and the per-row MSE loss. Every name carries a <c>bx_</c> prefix so
/// none can shadow a same-named kernel registered by another program.
/// </summary>
public static class BatchExecutionKernels
{
    /// <summary>Work-group size of <c>bx_full_reduce</c>; the tree reduction needs a power of two.</summary>
    public const int FullReduceGroupSize = 256;

    public static string GetSource()
    {
        return @"
// mode: 0 sum of squares, 1 L2 norm, 2 product, 3 max |x|, 4 min |x|, 5 log-sum-exp around maxVal.
// One work-group strides over the whole input, so the result is deterministic and always overwrites output[0].
__kernel void bx_full_reduce(__global const float* input, __global float* output, int size, int mode, float maxVal) {
    __local float scratch[256];
    int lid = get_local_id(0); int ls = get_local_size(0);
    float acc = mode == 2 ? 1.0f : (mode == 4 ? INFINITY : 0.0f);
    for (int i = lid; i < size; i += ls) {
        float v = input[i];
        if (mode <= 1) acc += v * v;
        else if (mode == 2) acc *= v;
        else if (mode == 3) acc = fmax(acc, fabs(v));
        else if (mode == 4) acc = fmin(acc, fabs(v));
        else acc += exp(v - maxVal);
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ls / 2; s > 0; s >>= 1) {
        if (lid < s) {
            float a = scratch[lid]; float b = scratch[lid + s];
            scratch[lid] = mode == 2 ? a * b : (mode == 3 ? fmax(a, b) : (mode == 4 ? fmin(a, b) : a + b));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float r = scratch[0];
        output[0] = mode == 1 ? sqrt(r) : (mode == 5 ? maxVal + log(r) : r);
    }
}

__kernel void bx_reduce_max_backward(__global const float* gradOutput, __global const float* input, __global const float* maxValues,
    __global float* gradInput, int outerSize, int reduceSize) {
    int idx = get_global_id(0); if (idx >= outerSize * reduceSize) return;
    int o = idx / reduceSize;
    gradInput[idx] = input[idx] == maxValues[o] ? gradOutput[o] : 0.0f;
}

__kernel void bx_reduce_variance_backward(__global const float* gradOutput, __global const float* input, __global const float* means,
    __global float* gradInput, int outerSize, int reduceSize) {
    int idx = get_global_id(0); if (idx >= outerSize * reduceSize) return;
    int o = idx / reduceSize;
    gradInput[idx] = gradOutput[o] * 2.0f * (input[idx] - means[o]) / (float)reduceSize;
}

__kernel void bx_reduce_log_variance(__global const float* input, __global float* output, int outerSize, int reduceSize) {
    int o = get_global_id(0); if (o >= outerSize) return;
    __global const float* row = input + o * reduceSize;
    float sum = 0.0f;
    for (int j = 0; j < reduceSize; j++) sum += row[j];
    float mean = sum / (float)reduceSize;
    float sq = 0.0f;
    for (int j = 0; j < reduceSize; j++) { float d = row[j] - mean; sq += d * d; }
    output[o] = log(sq / (float)reduceSize + 1e-8f);
}

__kernel void bx_reduce_log_variance_backward(__global const float* gradOutput, __global const float* input, __global const float* means,
    __global const float* variances, __global float* gradInput, int outerSize, int reduceSize) {
    int idx = get_global_id(0); if (idx >= outerSize * reduceSize) return;
    int o = idx / reduceSize;
    gradInput[idx] = gradOutput[o] / (variances[o] + 1e-8f) * 2.0f * (input[idx] - means[o]) / (float)reduceSize;
}

__kernel void bx_sincos(__global const float* input, __global float* sinOutput, __global float* cosOutput, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float v = input[idx];
    sinOutput[idx] = sin(v);
    cosOutput[idx] = cos(v);
}

__kernel void bx_slice_axis(__global const float* input, __global float* output, int outerSize, int axisSize, int stride, int index) {
    int idx = get_global_id(0); if (idx >= outerSize * stride) return;
    int o = idx / stride; int s = idx % stride;
    output[idx] = input[(o * axisSize + index) * stride + s];
}

__kernel void bx_set_slice_axis(__global float* output, __global const float* values, int outerSize, int axisSize, int stride, int index) {
    int idx = get_global_id(0); if (idx >= outerSize * stride) return;
    int o = idx / stride; int s = idx % stride;
    output[(o * axisSize + index) * stride + s] = values[idx];
}

// One work-item per input-gradient cell, so cells outside the crop window are written as zero.
__kernel void bx_crop_2d_backward(__global const float* gradOutput, __global float* gradInput, int batch, int channels,
    int inH, int inW, int outH, int outW, int offsetH, int offsetW) {
    int idx = get_global_id(0); if (idx >= batch * channels * inH * inW) return;
    int w = idx % inW - offsetW; int h = idx / inW % inH - offsetH; int bc = idx / (inW * inH);
    gradInput[idx] = (h >= 0 && h < outH && w >= 0 && w < outW) ? gradOutput[(bc * outH + h) * outW + w] : 0.0f;
}

// Sparsemax without a sort or a row-length cap: z_j is in the support when 1 + k_j*z_j > s_j, where k_j and s_j
// count and sum the entries >= z_j. Ties share k_j, which matches the sorted formulation.
__kernel void bx_sparsemax(__global const float* input, __global float* output, int outerSize, int innerSize) {
    int o = get_global_id(0); if (o >= outerSize) return;
    __global const float* row = input + o * innerSize;
    float supportSum = 0.0f; int supportCount = 0;
    for (int j = 0; j < innerSize; j++) {
        float z = row[j]; int k = 0; float s = 0.0f;
        for (int i = 0; i < innerSize; i++) { float v = row[i]; if (v >= z) { k++; s += v; } }
        if (1.0f + (float)k * z > s) { supportSum += z; supportCount++; }
    }
    float tau = (supportSum - 1.0f) / (float)supportCount;
    for (int j = 0; j < innerSize; j++) output[o * innerSize + j] = fmax(row[j] - tau, 0.0f);
}

__kernel void bx_mse_loss_rows(__global const float* predictions, __global const float* targets, __global float* loss,
    int batchSize, int numFeatures) {
    int b = get_global_id(0); if (b >= batchSize) return;
    float sum = 0.0f;
    for (int f = 0; f < numFeatures; f++) { float d = predictions[b * numFeatures + f] - targets[b * numFeatures + f]; sum += d * d; }
    loss[b] = sum / (float)numFeatures;
}

__kernel void bx_meshgrid2d(__global const float* source0, __global const float* source1, __global float* output0,
    __global float* output1, int n0, int n1, int xy) {
    int i = get_global_id(0); if (i >= n0 * n1) return;
    int cols = xy ? n0 : n1; int r = i / cols; int c = i % cols;
    output0[i] = source0[xy ? c : r];
    output1[i] = source1[xy ? r : c];
}

// ---- Generic FFT (#962): Bluestein for any length, 16-bit storage packed two values per 32-bit word. ----
uint bx_mulmod(uint a, uint b, uint m) {
    uint r = 0u;
    while (b != 0u) { if (b & 1u) { r += a; if (r >= m) r -= m; } a += a; if (a >= m) a -= m; b >>= 1; }
    return r;
}

__kernel void bx_fft_chirp(__global float* chirpRe, __global float* chirpIm, __global float* filterRe, __global float* filterIm,
    int n, int m, int inverse) {
    int k = get_global_id(0); if (k >= m) return;
    uint twoN = 2u * (uint)n;
    float sgn = inverse ? 1.0f : -1.0f;
    if (k < n) {
        float a = M_PI_F * (float)bx_mulmod((uint)k, (uint)k, twoN) / (float)n;
        chirpRe[k] = cos(a); chirpIm[k] = sgn * sin(a);
    }
    int t = k < n ? k : (k > m - n ? m - k : -1);
    if (t < 0) { filterRe[k] = 0.0f; filterIm[k] = 0.0f; return; }
    float b = M_PI_F * (float)bx_mulmod((uint)t, (uint)t, twoN) / (float)n;
    filterRe[k] = cos(b); filterIm[k] = -sgn * sin(b);
}

__kernel void bx_fft_bluestein_pre(__global const float* re, __global const float* im, __global const float* chirpRe,
    __global const float* chirpIm, __global float* padRe, __global float* padIm, int batch, int n, int m) {
    int i = get_global_id(0); if (i >= batch * m) return;
    int b = i / m; int k = i % m;
    if (k >= n) { padRe[i] = 0.0f; padIm[i] = 0.0f; return; }
    float xr = re[b * n + k]; float xi = im[b * n + k]; float cr = chirpRe[k]; float ci = chirpIm[k];
    padRe[i] = xr * cr - xi * ci;
    padIm[i] = xr * ci + xi * cr;
}

__kernel void bx_fft_cmul(__global float* aRe, __global float* aIm, __global const float* bRe, __global const float* bIm, int batch, int m) {
    int i = get_global_id(0); if (i >= batch * m) return;
    int k = i % m;
    float xr = aRe[i]; float xi = aIm[i]; float yr = bRe[k]; float yi = bIm[k];
    aRe[i] = xr * yr - xi * yi;
    aIm[i] = xr * yi + xi * yr;
}

__kernel void bx_fft_bluestein_post(__global const float* padRe, __global const float* padIm, __global const float* chirpRe,
    __global const float* chirpIm, __global float* re, __global float* im, int batch, int n, int m, float scale) {
    int i = get_global_id(0); if (i >= batch * n) return;
    int b = i / n; int k = i % n;
    float xr = padRe[b * m + k]; float xi = padIm[b * m + k]; float cr = chirpRe[k]; float ci = chirpIm[k];
    re[i] = (xr * cr - xi * ci) * scale;
    im[i] = (xr * ci + xi * cr) * scale;
}

uint bx_to_16(float x, int mode) {
    uint f = as_uint(x);
    if (mode == 2) {
        if ((f & 0x7fffffffu) > 0x7f800000u) return (f >> 16) | 0x40u;
        return (f + 0x7fffu + ((f >> 16) & 1u)) >> 16;
    }
    uint sgn = (f >> 16) & 0x8000u; int e = (int)((f >> 23) & 0xffu); uint mant = f & 0x7fffffu;
    if (e == 255) return sgn | 0x7c00u | (mant != 0u ? 0x200u : 0u);
    int ex = e - 112;
    if (ex >= 31) return sgn | 0x7c00u;
    if (ex <= 0) {
        if (ex < -10) return sgn;
        mant |= 0x800000u; uint shift = (uint)(14 - ex);
        uint h = mant >> shift; uint rem = mant & ((1u << shift) - 1u); uint hw = 1u << (shift - 1u);
        if (rem > hw || (rem == hw && (h & 1u) != 0u)) h++;
        return sgn | h;
    }
    uint h = sgn | ((uint)ex << 10) | (mant >> 13); uint rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h++;
    return h;
}

float bx_from_16(uint h, int mode) {
    if (mode == 2) return as_float(h << 16);
    uint sgn = (h & 0x8000u) << 16; uint e = (h >> 10) & 0x1fu; uint mant = h & 0x3ffu;
    if (e == 0u) {
        if (mant == 0u) return as_float(sgn);
        int ex = 1; while ((mant & 0x400u) == 0u) { mant <<= 1; ex--; }
        return as_float(sgn | ((uint)(ex + 112) << 23) | ((mant & 0x3ffu) << 13));
    }
    if (e == 31u) return as_float(sgn | 0x7f800000u | (mant << 13));
    return as_float(sgn | ((e + 112u) << 23) | (mant << 13));
}

__kernel void bx_fft_to_storage(__global const float* source, __global uint* words, int count, int mode) {
    int w = get_global_id(0); if (w >= (count + 1) / 2) return;
    uint lo = bx_to_16(source[2 * w], mode);
    uint hi = 2 * w + 1 < count ? bx_to_16(source[2 * w + 1], mode) : 0u;
    words[w] = lo | (hi << 16);
}

__kernel void bx_fft_from_storage(__global const uint* words, __global float* destination, int count, int mode) {
    int i = get_global_id(0); if (i >= count) return;
    uint word = words[i / 2];
    destination[i] = bx_from_16((i & 1) != 0 ? word >> 16 : word & 0xffffu, mode);
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "bx_full_reduce", "bx_reduce_max_backward", "bx_reduce_variance_backward", "bx_reduce_log_variance",
            "bx_reduce_log_variance_backward", "bx_sincos", "bx_slice_axis", "bx_set_slice_axis",
            "bx_crop_2d_backward", "bx_sparsemax", "bx_mse_loss_rows", "bx_meshgrid2d",
            "bx_fft_chirp", "bx_fft_bluestein_pre", "bx_fft_cmul", "bx_fft_bluestein_post", "bx_fft_to_storage", "bx_fft_from_storage"
        };
    }
}
