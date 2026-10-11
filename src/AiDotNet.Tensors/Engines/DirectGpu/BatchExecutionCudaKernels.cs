// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// CUDA C source for <see cref="IGpuBatchExecution"/>, compiled at runtime by both NVRTC (CUDA) and hiprtc (HIP)
/// (#1142). Related operations share one kernel with a <c>mode</c> argument so each backend language ports the same
/// small set. Semantics are the parity test's: whole-tensor reductions overwrite <c>output[0]</c> and return the
/// norm or the full log-sum-exp; the crop backward writes zero outside the window; sparsemax has no row-length cap;
/// the random kernels use the shared counter hash so a seed gives the same values on every backend.
/// </summary>
internal static class BatchExecutionCudaKernels
{
    public const int FullReduceBlockSize = 256;

    // Mode numbering shared with every backend's launcher.
    public const int FullSumOfSquares = 0, FullNormL2 = 1, FullProduct = 2, FullMaxMagnitude = 3, FullMinMagnitude = 4, FullLogSumExp = 5;
    public const int RowVariance = 0, RowStd = 1, RowProduct = 2, RowNorm = 3, RowLogSumExp = 4, RowLogVariance = 5;
    public const int MapCumSum = 0, MapNormalizeL2 = 1, MapLogSoftmax = 2, MapTaylorSoftmax = 3, MapSphericalSoftmax = 4, MapSparsemax = 5;
    public const int ScalarMinusTensor = 0, DivideByScalar = 1;
    public const int UnaryRsqrt = 0, UnaryReluDerivative = 1, UnarySigmoidDerivative = 2, UnaryTanhDerivative = 3;
    public const int BackwardSum = 0, BackwardMax = 1, BackwardVariance = 2, BackwardLogVariance = 3;
    public const int BroadcastAdd = 0, BroadcastMul = 1;
    public const int GateGlu = 0, GateGeGlu = 1, GateReGlu = 2, GateSwiGlu = 3;

    public static string GetSource() => @"
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

__device__ __forceinline__ unsigned int bx_hash(unsigned int counter, unsigned int key) {
    unsigned long long p = (unsigned long long)counter * 0xCD9E8D57u;
    counter = (unsigned int)(p >> 32) ^ key ^ (unsigned int)p;
    p = (unsigned long long)counter * 0xCD9E8D57u;
    return (unsigned int)(p >> 32) ^ (key + 1u) ^ (unsigned int)p;
}

__device__ __forceinline__ int bx_gid() { return blockIdx.x * blockDim.x + threadIdx.x; }

extern ""C"" __global__ __launch_bounds__(256) void bx_full_reduce(const float* input, float* output, int size, int mode, float maxVal) {
    __shared__ float scratch[256];
    int lid = threadIdx.x;
    float acc = mode == 2 ? 1.0f : (mode == 4 ? INFINITY : 0.0f);
    for (int i = lid; i < size; i += blockDim.x) {
        float v = input[i];
        if (mode <= 1) acc += v * v;
        else if (mode == 2) acc *= v;
        else if (mode == 3) acc = fmaxf(acc, fabsf(v));
        else if (mode == 4) acc = fminf(acc, fabsf(v));
        else acc += expf(v - maxVal);
    }
    scratch[lid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lid < s) {
            float a = scratch[lid]; float b = scratch[lid + s];
            scratch[lid] = mode == 2 ? a * b : (mode == 3 ? fmaxf(a, b) : (mode == 4 ? fminf(a, b) : a + b));
        }
        __syncthreads();
    }
    if (lid == 0) {
        float r = scratch[0];
        output[0] = mode == 1 ? sqrtf(r) : (mode == 5 ? maxVal + logf(r) : r);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void bx_row_reduce(const float* input, float* output, int outerSize, int innerSize, int mode) {
    int o = bx_gid(); if (o >= outerSize) return;
    const float* row = input + (long long)o * innerSize;
    float r;
    if (mode == 2) { r = 1.0f; for (int j = 0; j < innerSize; j++) r *= row[j]; }
    else if (mode == 3) { r = 0.0f; for (int j = 0; j < innerSize; j++) r += row[j] * row[j]; r = sqrtf(r); }
    else if (mode == 4) {
        float m = -INFINITY; for (int j = 0; j < innerSize; j++) m = fmaxf(m, row[j]);
        float s = 0.0f; for (int j = 0; j < innerSize; j++) s += expf(row[j] - m);
        r = m + logf(s);
    } else {
        float sum = 0.0f; for (int j = 0; j < innerSize; j++) sum += row[j];
        float mean = sum / (float)innerSize;
        float sq = 0.0f; for (int j = 0; j < innerSize; j++) { float d = row[j] - mean; sq += d * d; }
        float var = sq / (float)innerSize;
        r = mode == 0 ? var : (mode == 1 ? sqrtf(var) : logf(var + 1e-8f));
    }
    output[o] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_row_map(const float* input, float* output, int outerSize, int innerSize, int mode) {
    int o = bx_gid(); if (o >= outerSize) return;
    const float* row = input + (long long)o * innerSize;
    float* out = output + (long long)o * innerSize;
    if (mode == 0) {
        float acc = 0.0f; for (int j = 0; j < innerSize; j++) { acc += row[j]; out[j] = acc; }
    } else if (mode == 1) {
        float sq = 0.0f; for (int j = 0; j < innerSize; j++) sq += row[j] * row[j];
        float inv = 1.0f / (sqrtf(sq) + 1e-12f);
        for (int j = 0; j < innerSize; j++) out[j] = row[j] * inv;
    } else if (mode == 2) {
        float m = -INFINITY; for (int j = 0; j < innerSize; j++) m = fmaxf(m, row[j]);
        float s = 0.0f; for (int j = 0; j < innerSize; j++) s += expf(row[j] - m);
        float l = logf(s);
        for (int j = 0; j < innerSize; j++) out[j] = row[j] - m - l;
    } else if (mode == 3) {
        float s = 0.0f;
        for (int j = 0; j < innerSize; j++) { float x = row[j]; float t = fmaxf(1.0f + x + 0.5f * x * x, 1e-10f); out[j] = t; s += t; }
        float inv = 1.0f / s;
        for (int j = 0; j < innerSize; j++) out[j] *= inv;
    } else if (mode == 4) {
        float sq = 0.0f; for (int j = 0; j < innerSize; j++) sq += row[j] * row[j];
        float n = sqrtf(sq + 1e-12f);
        float m = -INFINITY; for (int j = 0; j < innerSize; j++) m = fmaxf(m, row[j] / n);
        float s = 0.0f; for (int j = 0; j < innerSize; j++) { float e = expf(row[j] / n - m); out[j] = e; s += e; }
        float inv = 1.0f / (s + 1e-10f);
        for (int j = 0; j < innerSize; j++) out[j] *= inv;
    } else {
        // Sparsemax without a sort or a cap: z_j is in the support when 1 + k_j*z_j > s_j, where k_j and s_j count and
        // sum the entries >= z_j. Ties share k_j, which matches the sorted formulation.
        float supportSum = 0.0f; int supportCount = 0;
        for (int j = 0; j < innerSize; j++) {
            float z = row[j]; int k = 0; float s = 0.0f;
            for (int i = 0; i < innerSize; i++) { float v = row[i]; if (v >= z) { k++; s += v; } }
            if (1.0f + (float)k * z > s) { supportSum += z; supportCount++; }
        }
        float tau = (supportSum - 1.0f) / (float)supportCount;
        for (int j = 0; j < innerSize; j++) out[j] = fmaxf(row[j] - tau, 0.0f);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void bx_scalar(const float* input, float* output, float scalar, int size, int mode) {
    int i = bx_gid(); if (i >= size) return;
    output[i] = mode == 0 ? scalar - input[i] : input[i] / scalar;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_unary(const float* input, float* output, int size, int mode) {
    int i = bx_gid(); if (i >= size) return;
    float v = input[i];
    output[i] = mode == 0 ? rsqrtf(v + 1e-12f) : (mode == 1 ? (v > 0.0f ? 1.0f : 0.0f) : (mode == 2 ? v * (1.0f - v) : 1.0f - v * v));
}

extern ""C"" __global__ __launch_bounds__(256) void bx_sincos(const float* input, float* sinOutput, float* cosOutput, int size) {
    int i = bx_gid(); if (i >= size) return;
    float v = input[i];
    sinOutput[i] = sinf(v);
    cosOutput[i] = cosf(v);
}

extern ""C"" __global__ __launch_bounds__(256) void bx_reduce_backward(const float* gradOutput, const float* input, const float* stat,
    const float* stat2, float* gradInput, int outerSize, int reduceSize, int mode) {
    int i = bx_gid(); if (i >= outerSize * reduceSize) return;
    int o = i / reduceSize;
    float g = gradOutput[o];
    if (mode == 0) gradInput[i] = g;
    else if (mode == 1) gradInput[i] = input[i] == stat[o] ? g : 0.0f;
    else if (mode == 2) gradInput[i] = g * 2.0f * (input[i] - stat[o]) / (float)reduceSize;
    else gradInput[i] = g / (stat2[o] + 1e-8f) * 2.0f * (input[i] - stat[o]) / (float)reduceSize;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_broadcast_first(const float* a, const float* b, float* output, int outerSize, int innerSize, int mode) {
    int i = bx_gid(); if (i >= outerSize * innerSize) return;
    float x = a[i / innerSize];
    output[i] = mode == 0 ? x + b[i] : x * b[i];
}

extern ""C"" __global__ __launch_bounds__(256) void bx_gated_backward(const float* gradOutput, const float* input, float* gradInput,
    int outerSize, int halfDim, int mode) {
    int i = bx_gid(); if (i >= outerSize * halfDim) return;
    int o = i / halfDim; int d = i % halfDim; int full = halfDim * 2;
    float v = input[o * full + d]; float g = input[o * full + halfDim + d]; float go = gradOutput[i];
    float act, dact;
    if (mode == 0) { float s = 1.0f / (1.0f + expf(-g)); act = s; dact = s * (1.0f - s); }
    else if (mode == 1) {
        float t = tanhf(0.7978845608f * (g + 0.044715f * g * g * g));
        act = 0.5f * g * (1.0f + t);
        dact = 0.5f * (1.0f + t) + 0.5f * g * (1.0f - t * t) * 0.7978845608f * (1.0f + 0.134145f * g * g);
    }
    else if (mode == 2) { act = fmaxf(g, 0.0f); dact = g > 0.0f ? 1.0f : 0.0f; }
    else { float s = 1.0f / (1.0f + expf(-g)); act = g * s; dact = s + g * s * (1.0f - s); }
    gradInput[o * full + d] = go * act;
    gradInput[o * full + halfDim + d] = go * v * dact;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_concat_axis(const float* a, const float* b, float* output, int outerSize, int aInner, int bInner) {
    int i = bx_gid(); int total = aInner + bInner; if (i >= outerSize * total) return;
    int o = i / total; int j = i % total;
    output[i] = j < aInner ? a[o * aInner + j] : b[o * bInner + j - aInner];
}

extern ""C"" __global__ __launch_bounds__(256) void bx_slice_last_axis(const float* input, float* output, int outerSize, int inputInner, int start, int sliceSize) {
    int i = bx_gid(); if (i >= outerSize * sliceSize) return;
    output[i] = input[(i / sliceSize) * inputInner + start + i % sliceSize];
}

extern ""C"" __global__ __launch_bounds__(256) void bx_set_slice_last_axis(float* output, const float* values, int outerSize, int outputInner, int start, int sliceSize) {
    int i = bx_gid(); if (i >= outerSize * sliceSize) return;
    output[(i / sliceSize) * outputInner + start + i % sliceSize] = values[i];
}

extern ""C"" __global__ __launch_bounds__(256) void bx_slice_axis(const float* input, float* output, int outerSize, int axisSize, int stride, int index) {
    int i = bx_gid(); if (i >= outerSize * stride) return;
    output[i] = input[((i / stride) * axisSize + index) * stride + i % stride];
}

extern ""C"" __global__ __launch_bounds__(256) void bx_set_slice_axis(float* output, const float* values, int outerSize, int axisSize, int stride, int index) {
    int i = bx_gid(); if (i >= outerSize * stride) return;
    output[((i / stride) * axisSize + index) * stride + i % stride] = values[i];
}

extern ""C"" __global__ __launch_bounds__(256) void bx_stack2(const float* a, const float* b, float* output, int size) {
    int i = bx_gid(); if (i >= size) return;
    output[2 * i] = a[i];
    output[2 * i + 1] = b[i];
}

// mode 0: Pad2D forward (output is the padded tensor). mode 1: Pad2D backward (output is the unpadded gradient).
// mode 2: Crop2D forward. mode 3: Crop2D backward, zero outside the window. (h0, w0) is padTop/padLeft or the offset.
extern ""C"" __global__ __launch_bounds__(256) void bx_window_2d(const float* input, float* output, int batch, int channels,
    int inH, int inW, int outH, int outW, int h0, int w0, float padValue, int mode) {
    int i = bx_gid();
    if (mode == 0) {
        if (i >= batch * channels * outH * outW) return;
        int w = i % outW - w0; int h = i / outW % outH - h0; int bc = i / (outW * outH);
        output[i] = (h >= 0 && h < inH && w >= 0 && w < inW) ? input[(bc * inH + h) * inW + w] : padValue;
    } else if (mode == 1) {
        if (i >= batch * channels * inH * inW) return;
        int w = i % inW; int h = i / inW % inH; int bc = i / (inW * inH);
        output[i] = input[(bc * outH + h + h0) * outW + w + w0];
    } else if (mode == 2) {
        if (i >= batch * channels * outH * outW) return;
        int w = i % outW; int h = i / outW % outH; int bc = i / (outW * outH);
        output[i] = input[(bc * inH + h + h0) * inW + w + w0];
    } else {
        if (i >= batch * channels * inH * inW) return;
        int w = i % inW - w0; int h = i / inW % inH - h0; int bc = i / (inW * inH);
        output[i] = (h >= 0 && h < outH && w >= 0 && w < outW) ? input[(bc * outH + h) * outW + w] : 0.0f;
    }
}

extern ""C"" __global__ __launch_bounds__(256) void bx_tile_last_axis(const float* input, float* output, int outerSize, int innerSize, int repeats) {
    int i = bx_gid(); int tiled = innerSize * repeats; if (i >= outerSize * tiled) return;
    output[i] = input[(i / tiled) * innerSize + (i % tiled) % innerSize];
}

extern ""C"" __global__ __launch_bounds__(256) void bx_repeat_elements(const float* input, float* output, int outerSize, int innerSize, int repeats) {
    int i = bx_gid(); int tiled = innerSize * repeats; if (i >= outerSize * tiled) return;
    output[i] = input[(i / tiled) * innerSize + (i % tiled) / repeats];
}

extern ""C"" __global__ __launch_bounds__(256) void bx_linspace(float* output, float start, float step, int size) {
    int i = bx_gid(); if (i >= size) return;
    output[i] = start + step * (float)i;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_diag(const float* input, float* output, int n) {
    int i = bx_gid(); if (i >= n * n) return;
    output[i] = (i / n == i % n) ? input[i / n] : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_extract_diag(const float* input, float* output, int n, int cols) {
    int i = bx_gid(); if (i >= n || i >= cols) return;
    output[i] = input[i * cols + i];
}

extern ""C"" __global__ __launch_bounds__(256) void bx_triangular_mask(float* output, int rows, int cols, int diagonal, float maskValue) {
    int i = bx_gid(); if (i >= rows * cols) return;
    output[i] = (i % cols > i / cols + diagonal) ? maskValue : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_index_select(const float* input, const float* indices, float* output, int numIndices, int innerSize) {
    int i = bx_gid(); if (i >= numIndices * innerSize) return;
    output[i] = input[(int)indices[i / innerSize] * innerSize + i % innerSize];
}

// mode 0: softmax cross-entropy from logits against dense targets. mode 1: mean squared error per row.
extern ""C"" __global__ __launch_bounds__(256) void bx_row_loss(const float* predictions, const float* targets, float* loss, int batchSize, int numClasses, int mode) {
    int b = bx_gid(); if (b >= batchSize) return;
    const float* p = predictions + (long long)b * numClasses;
    const float* t = targets + (long long)b * numClasses;
    float r = 0.0f;
    if (mode == 0) {
        float m = -INFINITY; for (int c = 0; c < numClasses; c++) m = fmaxf(m, p[c]);
        float s = 0.0f; for (int c = 0; c < numClasses; c++) s += expf(p[c] - m);
        float lse = m + logf(s);
        for (int c = 0; c < numClasses; c++) if (t[c] > 0.0f) r -= t[c] * (p[c] - lse);
    } else {
        for (int c = 0; c < numClasses; c++) { float d = p[c] - t[c]; r += d * d; }
        r /= (float)numClasses;
    }
    loss[b] = r;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_dropout_mask(float* mask, int size, float keepProb, unsigned long long seed) {
    int i = bx_gid(); if (i >= size) return;
    unsigned int key = (unsigned int)(seed ^ (seed >> 32));
    float u = (float)(bx_hash((unsigned int)i, key) >> 8) * (1.0f / 16777216.0f);
    float invKeep = keepProb > 1e-7f ? 1.0f / keepProb : 0.0f;
    mask[i] = u < keepProb ? invKeep : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_gaussian_noise(float* output, int size, float mean, float stdDev, unsigned long long seed) {
    int i = bx_gid(); if (i >= size) return;
    unsigned int key = (unsigned int)(seed ^ (seed >> 32));
    float u1 = (float)(bx_hash((unsigned int)(2 * i), key) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
    float u2 = (float)(bx_hash((unsigned int)(2 * i + 1), key + 2u) >> 8) * (1.0f / 16777216.0f);
    output[i] = mean + stdDev * sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

extern ""C"" __global__ __launch_bounds__(256) void bx_gumbel_softmax(const float* logits, float* output, int outerSize, int innerSize, float temperature, unsigned long long seed) {
    int o = bx_gid(); if (o >= outerSize) return;
    unsigned int key = (unsigned int)(seed ^ (seed >> 32));
    float t = fmaxf(temperature, 1e-7f);
    const float* row = logits + (long long)o * innerSize;
    float* out = output + (long long)o * innerSize;
    float m = -INFINITY;
    for (int j = 0; j < innerSize; j++) {
        unsigned int c = (unsigned int)(o * innerSize + j);
        float u = (float)(bx_hash(c, key) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
        float v = (row[j] - logf(-logf(u))) / t;
        out[j] = v; m = fmaxf(m, v);
    }
    float s = 0.0f;
    for (int j = 0; j < innerSize; j++) { float e = expf(out[j] - m); out[j] = e; s += e; }
    float inv = 1.0f / (s + 1e-10f);
    for (int j = 0; j < innerSize; j++) out[j] *= inv;
}

extern ""C"" __global__ __launch_bounds__(256) void bx_batch_outer_product(const float* a, const float* b, float* output, int batchSize, int M, int N) {
    int i = bx_gid(); if (i >= batchSize * M * N) return;
    int mn = M * N; int bt = i / mn; int r = i % mn;
    output[i] = a[bt * M + r / N] * b[bt * N + r % N];
}

// Two-vector meshgrid: 'ij' gives [n0, n1] grids (row from source0, column from source1); 'xy' gives [n1, n0] grids
// with source0 along the columns and source1 along the rows.
extern ""C"" __global__ __launch_bounds__(256) void bx_meshgrid2d(const float* source0, const float* source1, float* output0, float* output1,
    int n0, int n1, int xy) {
    int i = bx_gid(); if (i >= n0 * n1) return;
    int cols = xy ? n0 : n1; int r = i / cols; int c = i % cols;
    output0[i] = source0[xy ? c : r];
    output1[i] = source1[xy ? r : c];
}

";

    public static string[] GetKernelNames() => new[]
    {
        "bx_full_reduce", "bx_row_reduce", "bx_row_map", "bx_scalar", "bx_unary", "bx_sincos", "bx_reduce_backward",
        "bx_broadcast_first", "bx_gated_backward", "bx_concat_axis", "bx_slice_last_axis", "bx_set_slice_last_axis",
        "bx_slice_axis", "bx_set_slice_axis", "bx_stack2", "bx_window_2d", "bx_tile_last_axis", "bx_repeat_elements",
        "bx_linspace", "bx_diag", "bx_extract_diag", "bx_triangular_mask", "bx_index_select", "bx_row_loss",
        "bx_dropout_mask", "bx_gaussian_noise", "bx_gumbel_softmax", "bx_batch_outer_product", "bx_meshgrid2d"
    };
}
