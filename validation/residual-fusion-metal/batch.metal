#include <metal_stdlib>
using namespace metal;

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

inline unsigned int bx_hash(unsigned int counter, unsigned int key) {
    ulong p = (ulong)counter * 0xCD9E8D57u;
    counter = (unsigned int)(p >> 32) ^ key ^ (unsigned int)p;
    p = (ulong)counter * 0xCD9E8D57u;
    return (unsigned int)(p >> 32) ^ (key + 1u) ^ (unsigned int)p;
}


kernel void bx_full_reduce(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& size [[buffer(2)]], constant int& mode [[buffer(3)]], constant float& maxVal [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    threadgroup float scratch[256];
    int lid = (int)bx_lid;
    float acc = mode == 2 ? 1.0f : (mode == 4 ? INFINITY : 0.0f);
    for (int i = lid; i < size; i += (int)bx_bdim) {
        float v = input[i];
        if (mode <= 1) acc += v * v;
        else if (mode == 2) acc *= v;
        else if (mode == 3) acc = fmax(acc, fabs(v));
        else if (mode == 4) acc = fmin(acc, fabs(v));
        else acc += exp(v - maxVal);
    }
    scratch[lid] = acc;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (int s = (int)bx_bdim / 2; s > 0; s >>= 1) {
        if (lid < s) {
            float a = scratch[lid]; float b = scratch[lid + s];
            scratch[lid] = mode == 2 ? a * b : (mode == 3 ? fmax(a, b) : (mode == 4 ? fmin(a, b) : a + b));
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        float r = scratch[0];
        output[0] = mode == 1 ? sqrt(r) : (mode == 5 ? maxVal + log(r) : r);
    }
}

kernel void bx_row_reduce(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& outerSize [[buffer(2)]], constant int& innerSize [[buffer(3)]], constant int& mode [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int o = (int)bx_tid; if (o >= outerSize) return;
    device const float* row = input + (long)o * innerSize;
    float r;
    if (mode == 2) { r = 1.0f; for (int j = 0; j < innerSize; j++) r *= row[j]; }
    else if (mode == 3) { r = 0.0f; for (int j = 0; j < innerSize; j++) r += row[j] * row[j]; r = sqrt(r); }
    else if (mode == 4) {
        float m = -INFINITY; for (int j = 0; j < innerSize; j++) m = fmax(m, row[j]);
        float s = 0.0f; for (int j = 0; j < innerSize; j++) s += exp(row[j] - m);
        r = m + log(s);
    } else {
        float sum = 0.0f; for (int j = 0; j < innerSize; j++) sum += row[j];
        float mean = sum / (float)innerSize;
        float sq = 0.0f; for (int j = 0; j < innerSize; j++) { float d = row[j] - mean; sq += d * d; }
        float var = sq / (float)innerSize;
        r = mode == 0 ? var : (mode == 1 ? sqrt(var) : log(var + 1e-8f));
    }
    output[o] = r;
}

kernel void bx_row_map(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& outerSize [[buffer(2)]], constant int& innerSize [[buffer(3)]], constant int& mode [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int o = (int)bx_tid; if (o >= outerSize) return;
    device const float* row = input + (long)o * innerSize;
    device float* out = output + (long)o * innerSize;
    if (mode == 0) {
        float acc = 0.0f; for (int j = 0; j < innerSize; j++) { acc += row[j]; out[j] = acc; }
    } else if (mode == 1) {
        float sq = 0.0f; for (int j = 0; j < innerSize; j++) sq += row[j] * row[j];
        float inv = 1.0f / (sqrt(sq) + 1e-12f);
        for (int j = 0; j < innerSize; j++) out[j] = row[j] * inv;
    } else if (mode == 2) {
        float m = -INFINITY; for (int j = 0; j < innerSize; j++) m = fmax(m, row[j]);
        float s = 0.0f; for (int j = 0; j < innerSize; j++) s += exp(row[j] - m);
        float l = log(s);
        for (int j = 0; j < innerSize; j++) out[j] = row[j] - m - l;
    } else if (mode == 3) {
        float s = 0.0f;
        for (int j = 0; j < innerSize; j++) { float x = row[j]; float t = fmax(1.0f + x + 0.5f * x * x, 1e-10f); out[j] = t; s += t; }
        float inv = 1.0f / s;
        for (int j = 0; j < innerSize; j++) out[j] *= inv;
    } else if (mode == 4) {
        float sq = 0.0f; for (int j = 0; j < innerSize; j++) sq += row[j] * row[j];
        float n = sqrt(sq + 1e-12f);
        float m = -INFINITY; for (int j = 0; j < innerSize; j++) m = fmax(m, row[j] / n);
        float s = 0.0f; for (int j = 0; j < innerSize; j++) { float e = exp(row[j] / n - m); out[j] = e; s += e; }
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
        for (int j = 0; j < innerSize; j++) out[j] = fmax(row[j] - tau, 0.0f);
    }
}

kernel void bx_scalar(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant float& scalar [[buffer(2)]], constant int& size [[buffer(3)]], constant int& mode [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= size) return;
    output[i] = mode == 0 ? scalar - input[i] : input[i] / scalar;
}

kernel void bx_unary(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& size [[buffer(2)]], constant int& mode [[buffer(3)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= size) return;
    float v = input[i];
    output[i] = mode == 0 ? rsqrt(v + 1e-12f) : (mode == 1 ? (v > 0.0f ? 1.0f : 0.0f) : (mode == 2 ? v * (1.0f - v) : 1.0f - v * v));
}

kernel void bx_sincos(device const float* input [[buffer(0)]], device float* sinOutput [[buffer(1)]], device float* cosOutput [[buffer(2)]], constant int& size [[buffer(3)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= size) return;
    float v = input[i];
    sinOutput[i] = sin(v);
    cosOutput[i] = cos(v);
}

kernel void bx_reduce_backward(device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device const float* stat [[buffer(2)]], device const float* stat2 [[buffer(3)]], device float* gradInput [[buffer(4)]], constant int& outerSize [[buffer(5)]], constant int& reduceSize [[buffer(6)]], constant int& mode [[buffer(7)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= outerSize * reduceSize) return;
    int o = i / reduceSize;
    float g = gradOutput[o];
    if (mode == 0) gradInput[i] = g;
    else if (mode == 1) gradInput[i] = input[i] == stat[o] ? g : 0.0f;
    else if (mode == 2) gradInput[i] = g * 2.0f * (input[i] - stat[o]) / (float)reduceSize;
    else gradInput[i] = g / (stat2[o] + 1e-8f) * 2.0f * (input[i] - stat[o]) / (float)reduceSize;
}

kernel void bx_broadcast_first(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* output [[buffer(2)]], constant int& outerSize [[buffer(3)]], constant int& innerSize [[buffer(4)]], constant int& mode [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= outerSize * innerSize) return;
    float x = a[i / innerSize];
    output[i] = mode == 0 ? x + b[i] : x * b[i];
}

kernel void bx_gated_backward(device const float* gradOutput [[buffer(0)]], device const float* input [[buffer(1)]], device float* gradInput [[buffer(2)]], constant int& outerSize [[buffer(3)]], constant int& halfDim [[buffer(4)]], constant int& mode [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= outerSize * halfDim) return;
    int o = i / halfDim; int d = i % halfDim; int full = halfDim * 2;
    float v = input[o * full + d]; float g = input[o * full + halfDim + d]; float go = gradOutput[i];
    float act, dact;
    if (mode == 0) { float s = 1.0f / (1.0f + exp(-g)); act = s; dact = s * (1.0f - s); }
    else if (mode == 1) {
        float t = tanh(0.7978845608f * (g + 0.044715f * g * g * g));
        act = 0.5f * g * (1.0f + t);
        dact = 0.5f * (1.0f + t) + 0.5f * g * (1.0f - t * t) * 0.7978845608f * (1.0f + 0.134145f * g * g);
    }
    else if (mode == 2) { act = fmax(g, 0.0f); dact = g > 0.0f ? 1.0f : 0.0f; }
    else { float s = 1.0f / (1.0f + exp(-g)); act = g * s; dact = s + g * s * (1.0f - s); }
    gradInput[o * full + d] = go * act;
    gradInput[o * full + halfDim + d] = go * v * dact;
}

kernel void bx_concat_axis(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* output [[buffer(2)]], constant int& outerSize [[buffer(3)]], constant int& aInner [[buffer(4)]], constant int& bInner [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; int total = aInner + bInner; if (i >= outerSize * total) return;
    int o = i / total; int j = i % total;
    output[i] = j < aInner ? a[o * aInner + j] : b[o * bInner + j - aInner];
}

kernel void bx_slice_last_axis(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& outerSize [[buffer(2)]], constant int& inputInner [[buffer(3)]], constant int& start [[buffer(4)]], constant int& sliceSize [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= outerSize * sliceSize) return;
    output[i] = input[(i / sliceSize) * inputInner + start + i % sliceSize];
}

kernel void bx_set_slice_last_axis(device float* output [[buffer(0)]], device const float* values [[buffer(1)]], constant int& outerSize [[buffer(2)]], constant int& outputInner [[buffer(3)]], constant int& start [[buffer(4)]], constant int& sliceSize [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= outerSize * sliceSize) return;
    output[(i / sliceSize) * outputInner + start + i % sliceSize] = values[i];
}

kernel void bx_slice_axis(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& outerSize [[buffer(2)]], constant int& axisSize [[buffer(3)]], constant int& stride [[buffer(4)]], constant int& index [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= outerSize * stride) return;
    output[i] = input[((i / stride) * axisSize + index) * stride + i % stride];
}

kernel void bx_set_slice_axis(device float* output [[buffer(0)]], device const float* values [[buffer(1)]], constant int& outerSize [[buffer(2)]], constant int& axisSize [[buffer(3)]], constant int& stride [[buffer(4)]], constant int& index [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= outerSize * stride) return;
    output[((i / stride) * axisSize + index) * stride + i % stride] = values[i];
}

kernel void bx_stack2(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* output [[buffer(2)]], constant int& size [[buffer(3)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= size) return;
    output[2 * i] = a[i];
    output[2 * i + 1] = b[i];
}

// mode 0: Pad2D forward (output is the padded tensor). mode 1: Pad2D backward (output is the unpadded gradient).
// mode 2: Crop2D forward. mode 3: Crop2D backward, zero outside the window. (h0, w0) is padTop/padLeft or the offset.
kernel void bx_window_2d(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& batch [[buffer(2)]], constant int& channels [[buffer(3)]], constant int& inH [[buffer(4)]], constant int& inW [[buffer(5)]], constant int& outH [[buffer(6)]], constant int& outW [[buffer(7)]], constant int& h0 [[buffer(8)]], constant int& w0 [[buffer(9)]], constant float& padValue [[buffer(10)]], constant int& mode [[buffer(11)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid;
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

kernel void bx_tile_last_axis(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& outerSize [[buffer(2)]], constant int& innerSize [[buffer(3)]], constant int& repeats [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; int tiled = innerSize * repeats; if (i >= outerSize * tiled) return;
    output[i] = input[(i / tiled) * innerSize + (i % tiled) % innerSize];
}

kernel void bx_repeat_elements(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& outerSize [[buffer(2)]], constant int& innerSize [[buffer(3)]], constant int& repeats [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; int tiled = innerSize * repeats; if (i >= outerSize * tiled) return;
    output[i] = input[(i / tiled) * innerSize + (i % tiled) / repeats];
}

kernel void bx_linspace(device float* output [[buffer(0)]], constant float& start [[buffer(1)]], constant float& step [[buffer(2)]], constant int& size [[buffer(3)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= size) return;
    output[i] = start + step * (float)i;
}

kernel void bx_diag(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& n [[buffer(2)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= n * n) return;
    output[i] = (i / n == i % n) ? input[i / n] : 0.0f;
}

kernel void bx_extract_diag(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant int& n [[buffer(2)]], constant int& cols [[buffer(3)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= n || i >= cols) return;
    output[i] = input[i * cols + i];
}

kernel void bx_triangular_mask(device float* output [[buffer(0)]], constant int& rows [[buffer(1)]], constant int& cols [[buffer(2)]], constant int& diagonal [[buffer(3)]], constant float& maskValue [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= rows * cols) return;
    output[i] = (i % cols > i / cols + diagonal) ? maskValue : 0.0f;
}

kernel void bx_index_select(device const float* input [[buffer(0)]], device const float* indices [[buffer(1)]], device float* output [[buffer(2)]], constant int& numIndices [[buffer(3)]], constant int& innerSize [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= numIndices * innerSize) return;
    output[i] = input[(int)indices[i / innerSize] * innerSize + i % innerSize];
}

// mode 0: softmax cross-entropy from logits against dense targets. mode 1: mean squared error per row.
kernel void bx_row_loss(device const float* predictions [[buffer(0)]], device const float* targets [[buffer(1)]], device float* loss [[buffer(2)]], constant int& batchSize [[buffer(3)]], constant int& numClasses [[buffer(4)]], constant int& mode [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batchSize) return;
    device const float* p = predictions + (long)b * numClasses;
    device const float* t = targets + (long)b * numClasses;
    float r = 0.0f;
    if (mode == 0) {
        float m = -INFINITY; for (int c = 0; c < numClasses; c++) m = fmax(m, p[c]);
        float s = 0.0f; for (int c = 0; c < numClasses; c++) s += exp(p[c] - m);
        float lse = m + log(s);
        for (int c = 0; c < numClasses; c++) if (t[c] > 0.0f) r -= t[c] * (p[c] - lse);
    } else {
        for (int c = 0; c < numClasses; c++) { float d = p[c] - t[c]; r += d * d; }
        r /= (float)numClasses;
    }
    loss[b] = r;
}

kernel void bx_dropout_mask(device float* mask [[buffer(0)]], constant int& size [[buffer(1)]], constant float& keepProb [[buffer(2)]], constant ulong& seed [[buffer(3)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= size) return;
    unsigned int key = (unsigned int)(seed ^ (seed >> 32));
    float u = (float)(bx_hash((unsigned int)i, key) >> 8) * (1.0f / 16777216.0f);
    float invKeep = keepProb > 1e-7f ? 1.0f / keepProb : 0.0f;
    mask[i] = u < keepProb ? invKeep : 0.0f;
}

kernel void bx_gaussian_noise(device float* output [[buffer(0)]], constant int& size [[buffer(1)]], constant float& mean [[buffer(2)]], constant float& stdDev [[buffer(3)]], constant ulong& seed [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= size) return;
    unsigned int key = (unsigned int)(seed ^ (seed >> 32));
    float u1 = (float)(bx_hash((unsigned int)(2 * i), key) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
    float u2 = (float)(bx_hash((unsigned int)(2 * i + 1), key + 2u) >> 8) * (1.0f / 16777216.0f);
    output[i] = mean + stdDev * sqrt(-2.0f * log(u1)) * cos(6.283185307179586f * u2);
}

kernel void bx_gumbel_softmax(device const float* logits [[buffer(0)]], device float* output [[buffer(1)]], constant int& outerSize [[buffer(2)]], constant int& innerSize [[buffer(3)]], constant float& temperature [[buffer(4)]], constant ulong& seed [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int o = (int)bx_tid; if (o >= outerSize) return;
    unsigned int key = (unsigned int)(seed ^ (seed >> 32));
    float t = fmax(temperature, 1e-7f);
    device const float* row = logits + (long)o * innerSize;
    device float* out = output + (long)o * innerSize;
    float m = -INFINITY;
    for (int j = 0; j < innerSize; j++) {
        unsigned int c = (unsigned int)(o * innerSize + j);
        float u = (float)(bx_hash(c, key) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
        float v = (row[j] - log(-log(u))) / t;
        out[j] = v; m = fmax(m, v);
    }
    float s = 0.0f;
    for (int j = 0; j < innerSize; j++) { float e = exp(out[j] - m); out[j] = e; s += e; }
    float inv = 1.0f / (s + 1e-10f);
    for (int j = 0; j < innerSize; j++) out[j] *= inv;
}

kernel void bx_batch_outer_product(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* output [[buffer(2)]], constant int& batchSize [[buffer(3)]], constant int& M [[buffer(4)]], constant int& N [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= batchSize * M * N) return;
    int mn = M * N; int bt = i / mn; int r = i % mn;
    output[i] = a[bt * M + r / N] * b[bt * N + r % N];
}

// Two-vector meshgrid: 'ij' gives [n0, n1] grids (row from source0, column from source1); 'xy' gives [n1, n0] grids
// with source0 along the columns and source1 along the rows.
kernel void bx_meshgrid2d(device const float* source0 [[buffer(0)]], device const float* source1 [[buffer(1)]], device float* output0 [[buffer(2)]], device float* output1 [[buffer(3)]], constant int& n0 [[buffer(4)]], constant int& n1 [[buffer(5)]], constant int& xy [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= n0 * n1) return;
    int cols = xy ? n0 : n1; int r = i / cols; int c = i % cols;
    output0[i] = source0[xy ? c : r];
    output1[i] = source1[xy ? r : c];
}


// ---- Generic FFT (#962): Bluestein for any length, and 16-bit storage packed two values per 32-bit word. ----
// (a * b) mod m without 64-bit integers, which GLSL and WGSL lack; a, b < m <= 2^30.
inline unsigned int bx_mulmod(unsigned int a, unsigned int b, unsigned int m) {
    unsigned int r = 0u;
    while (b != 0u) { if (b & 1u) { r += a; if (r >= m) r -= m; } a += a; if (a >= m) a -= m; b >>= 1; }
    return r;
}

// chirp[k] = exp(sign * i * pi * k^2 / n) for k < n (sign -1 forward, +1 inverse); filter is its conjugate laid out
// circularly over m: filter[k] = conj(chirp[k]) for k < n, conj(chirp[m - k]) for k > m - n, else 0.
kernel void bx_fft_chirp(device float* chirpRe [[buffer(0)]], device float* chirpIm [[buffer(1)]], device float* filterRe [[buffer(2)]], device float* filterIm [[buffer(3)]], constant int& n [[buffer(4)]], constant int& m [[buffer(5)]], constant int& inverse [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= m) return;
    unsigned int twoN = 2u * (unsigned int)n;
    float sign = inverse ? 1.0f : -1.0f;
    if (k < n) {
        float a = 3.14159265358979323846f * (float)bx_mulmod((unsigned int)k, (unsigned int)k, twoN) / (float)n;
        chirpRe[k] = cos(a); chirpIm[k] = sign * sin(a);
    }
    int t = k < n ? k : (k > m - n ? m - k : -1);
    if (t < 0) { filterRe[k] = 0.0f; filterIm[k] = 0.0f; return; }
    float a = 3.14159265358979323846f * (float)bx_mulmod((unsigned int)t, (unsigned int)t, twoN) / (float)n;
    filterRe[k] = cos(a); filterIm[k] = -sign * sin(a);
}

kernel void bx_fft_bluestein_pre(device const float* re [[buffer(0)]], device const float* im [[buffer(1)]], device const float* chirpRe [[buffer(2)]], device const float* chirpIm [[buffer(3)]], device float* padRe [[buffer(4)]], device float* padIm [[buffer(5)]], constant int& batch [[buffer(6)]], constant int& n [[buffer(7)]], constant int& m [[buffer(8)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= batch * m) return;
    int b = i / m; int k = i % m;
    if (k >= n) { padRe[i] = 0.0f; padIm[i] = 0.0f; return; }
    float xr = re[b * n + k]; float xi = im[b * n + k]; float cr = chirpRe[k]; float ci = chirpIm[k];
    padRe[i] = xr * cr - xi * ci;
    padIm[i] = xr * ci + xi * cr;
}

kernel void bx_fft_cmul(device float* aRe [[buffer(0)]], device float* aIm [[buffer(1)]], device const float* bRe [[buffer(2)]], device const float* bIm [[buffer(3)]], constant int& batch [[buffer(4)]], constant int& m [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= batch * m) return;
    int k = i % m;
    float xr = aRe[i]; float xi = aIm[i]; float yr = bRe[k]; float yi = bIm[k];
    aRe[i] = xr * yr - xi * yi;
    aIm[i] = xr * yi + xi * yr;
}

kernel void bx_fft_bluestein_post(device const float* padRe [[buffer(0)]], device const float* padIm [[buffer(1)]], device const float* chirpRe [[buffer(2)]], device const float* chirpIm [[buffer(3)]], device float* re [[buffer(4)]], device float* im [[buffer(5)]], constant int& batch [[buffer(6)]], constant int& n [[buffer(7)]], constant int& m [[buffer(8)]], constant float& scale [[buffer(9)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= batch * n) return;
    int b = i / n; int k = i % n;
    float xr = padRe[b * m + k]; float xi = padIm[b * m + k]; float cr = chirpRe[k]; float ci = chirpIm[k];
    re[i] = (xr * cr - xi * ci) * scale;
    im[i] = (xr * ci + xi * cr) * scale;
}

// Batched radix-2 FFT for a power-of-two n (#965), for a backend whose own BatchedFFT is not one: a bit-reversal permute
// that also applies the inverse's 1/n (the butterflies are linear), then log2(n) in-place butterfly stages of span
// half = 1, 2, 4, ... n / 2. The twiddle angle is -/+ pi * pos / half for a forward / inverse transform.
kernel void bx_fft_bit_reverse(device const float* inRe [[buffer(0)]], device const float* inIm [[buffer(1)]], device float* outRe [[buffer(2)]], device float* outIm [[buffer(3)]], constant int& batch [[buffer(4)]], constant int& n [[buffer(5)]], constant int& log2n [[buffer(6)]], constant float& scale [[buffer(7)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= batch * n) return;
    int b = i / n; unsigned int k = (unsigned int)(i % n); unsigned int r = 0u;
    for (int bit = 0; bit < log2n; bit++) { r = (r << 1) | (k & 1u); k = k >> 1; }
    outRe[b * n + (int)r] = inRe[i] * scale;
    outIm[b * n + (int)r] = inIm[i] * scale;
}

kernel void bx_fft_butterfly(device float* re [[buffer(0)]], device float* im [[buffer(1)]], constant int& batch [[buffer(2)]], constant int& n [[buffer(3)]], constant int& span [[buffer(4)]], constant int& inverse [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; int pairs = n / 2; if (t >= batch * pairs) return;
    int b = t / pairs; int j = t % pairs; int pos = j % span;
    int i0 = b * n + (j / span) * 2 * span + pos; int i1 = i0 + span;
    float a = (inverse ? 3.14159265358979323846f : -3.14159265358979323846f) * (float)pos / (float)span;
    float wr = cos(a); float wi = sin(a);
    float xr = re[i1]; float xi = im[i1];
    float tr = wr * xr - wi * xi; float ti = wr * xi + wi * xr;
    float ur = re[i0]; float ui = im[i0];
    re[i0] = ur + tr; im[i0] = ui + ti;
    re[i1] = ur - tr; im[i1] = ui - ti;
}
// mode 1: IEEE binary16, mode 2: bfloat16; both round to nearest even.
inline unsigned int bx_to_16(float x, int mode) {
    unsigned int f = as_type<uint>(x);
    if (mode == 2) {
        if ((f & 0x7fffffffu) > 0x7f800000u) return (f >> 16) | 0x40u;
        return (f + 0x7fffu + ((f >> 16) & 1u)) >> 16;
    }
    unsigned int sign = (f >> 16) & 0x8000u; int e = (int)((f >> 23) & 0xffu); unsigned int mant = f & 0x7fffffu;
    if (e == 255) return sign | 0x7c00u | (mant != 0u ? 0x200u : 0u);
    int ex = e - 112;
    if (ex >= 31) return sign | 0x7c00u;
    if (ex <= 0) {
        if (ex < -10) return sign;
        mant |= 0x800000u; unsigned int shift = (unsigned int)(14 - ex);
        unsigned int h = mant >> shift; unsigned int rem = mant & ((1u << shift) - 1u); unsigned int halfUlp = 1u << (shift - 1u);
        if (rem > halfUlp || (rem == halfUlp && (h & 1u) != 0u)) h++;
        return sign | h;
    }
    unsigned int h = sign | ((unsigned int)ex << 10) | (mant >> 13); unsigned int rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h++;
    return h;
}

inline float bx_from_16(unsigned int h, int mode) {
    if (mode == 2) return as_type<float>(h << 16);
    unsigned int sign = (h & 0x8000u) << 16; unsigned int e = (h >> 10) & 0x1fu; unsigned int mant = h & 0x3ffu;
    if (e == 0u) {
        if (mant == 0u) return as_type<float>(sign);
        int ex = 1; while ((mant & 0x400u) == 0u) { mant <<= 1; ex--; }
        return as_type<float>(sign | ((unsigned int)(ex + 112) << 23) | ((mant & 0x3ffu) << 13));
    }
    if (e == 31u) return as_type<float>(sign | 0x7f800000u | (mant << 13));
    return as_type<float>(sign | ((e + 112u) << 23) | (mant << 13));
}

// One thread per 32-bit word: element 2w in the low half, 2w + 1 in the high half (zero past the end).
kernel void bx_fft_to_storage(device const float* source [[buffer(0)]], device uint* words [[buffer(1)]], constant int& count [[buffer(2)]], constant int& mode [[buffer(3)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int w = (int)bx_tid; if (w >= (count + 1) / 2) return;
    unsigned int lo = bx_to_16(source[2 * w], mode);
    unsigned int hi = 2 * w + 1 < count ? bx_to_16(source[2 * w + 1], mode) : 0u;
    words[w] = lo | (hi << 16);
}

kernel void bx_fft_from_storage(device const uint* words [[buffer(0)]], device float* destination [[buffer(1)]], constant int& count [[buffer(2)]], constant int& mode [[buffer(3)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= count) return;
    unsigned int word = words[i / 2];
    destination[i] = bx_from_16((i & 1) != 0 ? word >> 16 : word & 0xffffu, mode);
}

#define LA_MAXN 8

// out = the symmetric matrix read from one triangle of a (upper != 0: the upper triangle).
kernel void la_symmetrize(device const float* a [[buffer(0)]], device float* outp [[buffer(1)]], constant int& batch [[buffer(2)]], constant int& n [[buffer(3)]], constant int& upper [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= batch * n * n) return;
    int o = (i / (n * n)) * n * n; int r = (i / n) % n; int c = i % n;
    bool keep = upper != 0 ? (r <= c) : (r >= c);
    outp[i] = keep ? a[i] : a[o + c * n + r];
}

// LuDecomposition.Solve: apply the getrf row swaps, then unit-lower forward and upper backward substitution.
kernel void la_lu_solve(device const float* lu [[buffer(0)]], device const int* piv [[buffer(1)]], device const float* rhs [[buffer(2)]], device float* x [[buffer(3)]], constant int& batch [[buffer(4)]], constant int& n [[buffer(5)]], constant int& nrhs [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batch) return;
    int ao = b * n * n; int po = b * n; int xo = b * n * nrhs;
    for (int i = 0; i < n * nrhs; i++) x[xo + i] = rhs[xo + i];
    for (int i = 0; i < n; i++) {
        int p = piv[po + i];
        if (p != i && p >= 0 && p < n) {
            for (int c = 0; c < nrhs; c++) {
                float t = x[xo + i * nrhs + c]; x[xo + i * nrhs + c] = x[xo + p * nrhs + c]; x[xo + p * nrhs + c] = t;
            }
        }
    }
    for (int c = 0; c < nrhs; c++) {
        for (int i = 1; i < n; i++) {
            float s = x[xo + i * nrhs + c];
            for (int j = 0; j < i; j++) s -= lu[ao + i * n + j] * x[xo + j * nrhs + c];
            x[xo + i * nrhs + c] = s;
        }
        for (int i = n - 1; i >= 0; i--) {
            float s = x[xo + i * nrhs + c];
            for (int j = i + 1; j < n; j++) s -= lu[ao + i * n + j] * x[xo + j * nrhs + c];
            x[xo + i * nrhs + c] = s / lu[ao + i * n + i];
        }
    }
}

// LdlDecomposition.Factor: symmetric diagonal pivoting on the largest remaining diagonal, factored in place in ld.
kernel void la_ldl_factor(device const float* a [[buffer(0)]], device float* ld [[buffer(1)]], device int* piv [[buffer(2)]], constant int& batch [[buffer(3)]], constant int& n [[buffer(4)]], constant int& upper [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batch) return;
    int o = b * n * n; int po = b * n;
    for (int i = 0; i < n * n; i++) ld[o + i] = a[o + i];
    for (int j = 0; j < n; j++) {
        int p = j; float best = fabs(ld[o + j * n + j]);
        for (int i = j + 1; i < n; i++) {
            float v = fabs(ld[o + i * n + i]);
            if (v > best) { best = v; p = i; }
        }
        piv[po + j] = p;
        if (p != j) {
            for (int k = 0; k < n; k++) { float t = ld[o + j * n + k]; ld[o + j * n + k] = ld[o + p * n + k]; ld[o + p * n + k] = t; }
            for (int k = 0; k < n; k++) { float t = ld[o + k * n + j]; ld[o + k * n + j] = ld[o + k * n + p]; ld[o + k * n + p] = t; }
        }
        float d = ld[o + j * n + j];
        if (d == 0.0f) continue;
        for (int i = j + 1; i < n; i++) {
            float lij = ld[o + i * n + j] / d;
            ld[o + i * n + j] = lij;
            for (int k = j + 1; k < n; k++) ld[o + i * n + k] -= lij * ld[o + j * n + k];
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) ld[o + i * n + j] = 0.0f;
    if (upper != 0) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++) { ld[o + j * n + i] = ld[o + i * n + j]; ld[o + i * n + j] = 0.0f; }
    }
}

// LdlDecomposition.Solve: permute, unit-L forward, divide by D (NaN on a zero pivot), L^T backward, unpermute.
kernel void la_ldl_solve(device const float* ld [[buffer(0)]], device const int* piv [[buffer(1)]], device const float* rhs [[buffer(2)]], device float* x [[buffer(3)]], constant int& batch [[buffer(4)]], constant int& n [[buffer(5)]], constant int& nrhs [[buffer(6)]], constant int& upper [[buffer(7)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batch) return;
    int o = b * n * n; int po = b * n; int xo = b * n * nrhs;
    for (int i = 0; i < n * nrhs; i++) x[xo + i] = rhs[xo + i];
    for (int c = 0; c < nrhs; c++) {
        for (int j = 0; j < n; j++) {
            int p = piv[po + j];
            if (p != j && p >= 0 && p < n) { float t = x[xo + j * nrhs + c]; x[xo + j * nrhs + c] = x[xo + p * nrhs + c]; x[xo + p * nrhs + c] = t; }
        }
        for (int i = 0; i < n; i++) {
            float s = x[xo + i * nrhs + c];
            for (int k = 0; k < i; k++) s -= (upper != 0 ? ld[o + k * n + i] : ld[o + i * n + k]) * x[xo + k * nrhs + c];
            x[xo + i * nrhs + c] = s;
        }
        for (int i = 0; i < n; i++) {
            float d = ld[o + i * n + i];
            x[xo + i * nrhs + c] = d == 0.0f ? as_type<float>(0x7fc00000u) : x[xo + i * nrhs + c] / d;
        }
        for (int i = n - 1; i >= 0; i--) {
            float s = x[xo + i * nrhs + c];
            for (int k = i + 1; k < n; k++) s -= (upper != 0 ? ld[o + i * n + k] : ld[o + k * n + i]) * x[xo + k * nrhs + c];
            x[xo + i * nrhs + c] = s;
        }
        for (int j = n - 1; j >= 0; j--) {
            int p = piv[po + j];
            if (p != j && p >= 0 && p < n) { float t = x[xo + j * nrhs + c]; x[xo + j * nrhs + c] = x[xo + p * nrhs + c]; x[xo + p * nrhs + c] = t; }
        }
    }
}

// LinearSolvers.SolveEx for one vector: partial-pivot LU in registers, info = 1 + the first zero pivot (0: none).
kernel void la_solve_vector(device const float* a [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* x [[buffer(2)]], device int* info [[buffer(3)]], constant int& batch [[buffer(4)]], constant int& n [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batch) return;
    float w[LA_MAXN * LA_MAXN]; int pv[LA_MAXN];
    int o = b * n * n; int xo = b * n;
    for (int i = 0; i < n * n; i++) w[i] = a[o + i];
    for (int j = 0; j < n; j++) {
        int p = j; float best = fabs(w[j * n + j]);
        for (int i = j + 1; i < n; i++) { float v = fabs(w[i * n + j]); if (v > best) { best = v; p = i; } }
        pv[j] = p;
        if (p != j) for (int c = 0; c < n; c++) { float t = w[p * n + c]; w[p * n + c] = w[j * n + c]; w[j * n + c] = t; }
        float d = w[j * n + j];
        if (d == 0.0f) continue;
        for (int i = j + 1; i < n; i++) {
            float f = w[i * n + j] / d;
            w[i * n + j] = f;
            for (int c = j + 1; c < n; c++) w[i * n + c] -= f * w[j * n + c];
        }
    }
    int status = 0;
    for (int k = 0; k < n; k++) if (w[k * n + k] == 0.0f) { status = k + 1; break; }
    info[b] = status;
    for (int i = 0; i < n; i++) x[xo + i] = rhs[xo + i];
    for (int i = 0; i < n; i++) { int p = pv[i]; if (p != i) { float t = x[xo + i]; x[xo + i] = x[xo + p]; x[xo + p] = t; } }
    for (int i = 1; i < n; i++) { float s = x[xo + i]; for (int j = 0; j < i; j++) s -= w[i * n + j] * x[xo + j]; x[xo + i] = s; }
    for (int i = n - 1; i >= 0; i--) {
        float s = x[xo + i];
        for (int j = i + 1; j < n; j++) s -= w[i * n + j] * x[xo + j];
        x[xo + i] = s / w[i * n + i];
    }
}

// LinearSolvers.SolveTriangular for one vector.
kernel void la_tri_solve_vector(device const float* a [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* x [[buffer(2)]], constant int& batch [[buffer(3)]], constant int& n [[buffer(4)]], constant int& upper [[buffer(5)]], constant int& unitDiagonal [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batch) return;
    int o = b * n * n; int xo = b * n;
    for (int i = 0; i < n; i++) x[xo + i] = rhs[xo + i];
    for (int st = 0; st < n; st++) {
        int i = upper != 0 ? n - 1 - st : st;
        float s = x[xo + i];
        if (upper != 0) { for (int j = i + 1; j < n; j++) s -= a[o + i * n + j] * x[xo + j]; }
        else { for (int j = 0; j < i; j++) s -= a[o + i * n + j] * x[xo + j]; }
        x[xo + i] = unitDiagonal != 0 ? s : s / a[o + i * n + i];
    }
}

// LinalgBackward.CholeskyBackward (lower): gradA = 1/2 L^-T (Phi + Phi^T) L^-1, Phi = tril(L^T gradL), diagonal halved.
kernel void la_cholesky_backward_lower(device const float* l [[buffer(0)]], device const float* g [[buffer(1)]], device float* ga [[buffer(2)]], constant int& batch [[buffer(3)]], constant int& n [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batch) return;
    float t[LA_MAXN * LA_MAXN]; float inv[LA_MAXN * LA_MAXN]; float row[LA_MAXN];
    int o = b * n * n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += l[o + k * n + i] * g[o + k * n + j];
            t[i * n + j] = s;
        }
    // Phi zeroes the upper triangle and halves the diagonal, so Phi + Phi^T mirrors the lower triangle.
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) t[i * n + j] = t[j * n + i];
    for (int c = 0; c < n; c++)
        for (int i = 0; i < n; i++) {
            float s = i == c ? 1.0f : 0.0f;
            for (int k = 0; k < i; k++) s -= l[o + i * n + k] * inv[k * n + c];
            inv[i * n + c] = s / l[o + i * n + i];
        }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float s = 0.0f;
            for (int r = 0; r < n; r++) s += inv[r * n + i] * t[r * n + k];
            row[k] = s;
        }
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += row[k] * inv[k * n + j];
            ga[o + i * n + j] = 0.5f * s;
        }
    }
}

// LinalgBackward.SolveBackward for a vector rhs: gradB = A^-T gradX, gradA = -gradB x^T.
kernel void la_solve_backward_vector(device const float* a [[buffer(0)]], device const float* xs [[buffer(1)]], device const float* g [[buffer(2)]], device float* ga [[buffer(3)]], device float* gb [[buffer(4)]], constant int& batch [[buffer(5)]], constant int& n [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batch) return;
    float w[LA_MAXN * LA_MAXN]; int pv[LA_MAXN]; float y[LA_MAXN];
    int o = b * n * n; int vo = b * n;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) w[i * n + j] = a[o + j * n + i];
    for (int j = 0; j < n; j++) {
        int p = j; float best = fabs(w[j * n + j]);
        for (int i = j + 1; i < n; i++) { float v = fabs(w[i * n + j]); if (v > best) { best = v; p = i; } }
        pv[j] = p;
        if (p != j) for (int c = 0; c < n; c++) { float t = w[p * n + c]; w[p * n + c] = w[j * n + c]; w[j * n + c] = t; }
        float d = w[j * n + j];
        if (d == 0.0f) continue;
        for (int i = j + 1; i < n; i++) {
            float f = w[i * n + j] / d;
            w[i * n + j] = f;
            for (int c = j + 1; c < n; c++) w[i * n + c] -= f * w[j * n + c];
        }
    }
    for (int i = 0; i < n; i++) y[i] = g[vo + i];
    for (int i = 0; i < n; i++) { int p = pv[i]; if (p != i) { float t = y[i]; y[i] = y[p]; y[p] = t; } }
    for (int i = 1; i < n; i++) { float s = y[i]; for (int j = 0; j < i; j++) s -= w[i * n + j] * y[j]; y[i] = s; }
    for (int i = n - 1; i >= 0; i--) { float s = y[i]; for (int j = i + 1; j < n; j++) s -= w[i * n + j] * y[j]; y[i] = s / w[i * n + i]; }
    for (int i = 0; i < n; i++) {
        gb[vo + i] = y[i];
        for (int j = 0; j < n; j++) ga[o + i * n + j] = -y[i] * xs[vo + j];
    }
}

// QrDecomposition (reduced): Householder with alpha = -sign(x0) |x|, as LAPACK geqrf, so R's diagonal carries the
// same signs as the CPU and PyTorch. work holds, per matrix, the m x n working copy (reflector j in column j from
// row j down) followed by k betas and k diagonal entries of R.
kernel void la_qr_reduced(device const float* a [[buffer(0)]], device float* q [[buffer(1)]], device float* r [[buffer(2)]], device float* work [[buffer(3)]], constant int& batch [[buffer(4)]], constant int& m [[buffer(5)]], constant int& n [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batch) return;
    int k = m < n ? m : n;
    int ao = b * m * n; int wo = b * (m * n + 2 * k); int bo = wo + m * n; int dg = bo + k;
    int qo = b * m * k; int ro = b * k * n;
    for (int i = 0; i < m * n; i++) work[wo + i] = a[ao + i];
    for (int j = 0; j < k; j++) {
        float norm2 = 0.0f;
        for (int i = j; i < m; i++) norm2 += work[wo + i * n + j] * work[wo + i * n + j];
        float norm = sqrt(norm2);
        float x0 = work[wo + j * n + j];
        float alpha = x0 > 0.0f ? -norm : (x0 < 0.0f ? norm : 0.0f);
        if (x0 == 0.0f && alpha == 0.0f) alpha = norm;
        work[wo + j * n + j] = x0 - alpha;
        float vnorm2 = 0.0f;
        for (int i = j; i < m; i++) vnorm2 += work[wo + i * n + j] * work[wo + i * n + j];
        float beta = vnorm2 == 0.0f ? 0.0f : 2.0f / vnorm2;
        work[bo + j] = beta;
        work[dg + j] = alpha;
        for (int c = j + 1; c < n; c++) {
            float dot = 0.0f;
            for (int i = j; i < m; i++) dot += work[wo + i * n + j] * work[wo + i * n + c];
            float sc = beta * dot;
            for (int i = j; i < m; i++) work[wo + i * n + c] -= sc * work[wo + i * n + j];
        }
    }
    for (int i = 0; i < k; i++)
        for (int c = 0; c < n; c++)
            r[ro + i * n + c] = c < i ? 0.0f : (c == i ? work[dg + i] : work[wo + i * n + c]);
    for (int i = 0; i < m; i++)
        for (int c = 0; c < k; c++) q[qo + i * k + c] = i == c ? 1.0f : 0.0f;
    for (int j = k - 1; j >= 0; j--) {
        float beta = work[bo + j];
        if (beta == 0.0f) continue;
        for (int c = 0; c < k; c++) {
            float dot = 0.0f;
            for (int i = j; i < m; i++) dot += work[wo + i * n + j] * q[qo + i * k + c];
            float sc = beta * dot;
            for (int i = j; i < m; i++) q[qo + i * k + c] -= sc * work[wo + i * n + j];
        }
    }
}

// SvdWrapper.Full (reduced): Jacobi eigendecomposition of the smaller Gram matrix, singular values descending.
kernel void la_svd_reduced(device const float* a [[buffer(0)]], device float* u [[buffer(1)]], device float* s [[buffer(2)]], device float* vh [[buffer(3)]], constant int& batch [[buffer(4)]], constant int& m [[buffer(5)]], constant int& n [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int b = (int)bx_tid; if (b >= batch) return;
    float g[LA_MAXN * LA_MAXN]; float v[LA_MAXN * LA_MAXN]; float w[LA_MAXN]; int ord[LA_MAXN];
    bool tall = m >= n; int k = m < n ? m : n; int d = tall ? n : m;
    int ao = b * m * n; int uo = b * m * k; int so = b * k; int vo = b * k * n;
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            float acc = 0.0f;
            if (tall) { for (int r = 0; r < m; r++) acc += a[ao + r * n + i] * a[ao + r * n + j]; }
            else { for (int r = 0; r < n; r++) acc += a[ao + i * n + r] * a[ao + j * n + r]; }
            g[i * d + j] = acc;
            v[i * d + j] = i == j ? 1.0f : 0.0f;
        }
    float scale = 0.0f;
    for (int i = 0; i < d * d; i++) scale += g[i] * g[i];
    scale = sqrt(scale);
    for (int sweep = 0; sweep < 100; sweep++) {
        float off = 0.0f;
        for (int p = 0; p < d - 1; p++) for (int q = p + 1; q < d; q++) off += g[p * d + q] * g[p * d + q];
        off = sqrt(off);
        if (off < 1e-12f || off <= 1e-7f * scale) break;
        for (int p = 0; p < d - 1; p++)
            for (int q = p + 1; q < d; q++) {
                float apq = g[p * d + q];
                if (fabs(apq) < 1e-12f) continue;
                float theta = (g[q * d + q] - g[p * d + p]) / (2.0f * apq);
                float sg = theta > 0.0f ? 1.0f : (theta < 0.0f ? -1.0f : 0.0f);
                float t = theta == 0.0f ? 1.0f : sg / (fabs(theta) + sqrt(1.0f + theta * theta));
                float c = 1.0f / sqrt(1.0f + t * t);
                float sn = t * c;
                for (int i = 0; i < d; i++) {
                    float aip = g[i * d + p]; float aiq = g[i * d + q];
                    g[i * d + p] = c * aip - sn * aiq; g[i * d + q] = sn * aip + c * aiq;
                }
                for (int i = 0; i < d; i++) {
                    float api = g[p * d + i]; float aqi = g[q * d + i];
                    g[p * d + i] = c * api - sn * aqi; g[q * d + i] = sn * api + c * aqi;
                }
                for (int i = 0; i < d; i++) {
                    float vip = v[i * d + p]; float viq = v[i * d + q];
                    v[i * d + p] = c * vip - sn * viq; v[i * d + q] = sn * vip + c * viq;
                }
            }
    }
    for (int i = 0; i < d; i++) { w[i] = g[i * d + i]; ord[i] = i; }
    for (int i = 1; i < d; i++) {
        int cur = ord[i]; int j = i - 1;
        while (j >= 0 && w[ord[j]] < w[cur]) { ord[j + 1] = ord[j]; j--; }
        ord[j + 1] = cur;
    }
    for (int j = 0; j < k; j++) {
        float sigma = sqrt(fmax(0.0f, w[ord[j]]));
        s[so + j] = sigma;
        int e = ord[j];
        if (tall) {
            for (int c = 0; c < n; c++) vh[vo + j * n + c] = v[c * d + e];
            for (int r = 0; r < m; r++) {
                float acc = 0.0f;
                for (int c = 0; c < n; c++) acc += a[ao + r * n + c] * v[c * d + e];
                u[uo + r * k + j] = sigma > 1e-14f ? acc / sigma : 0.0f;
            }
        } else {
            for (int r = 0; r < m; r++) u[uo + r * k + j] = v[r * d + e];
            for (int c = 0; c < n; c++) {
                float acc = 0.0f;
                for (int r = 0; r < m; r++) acc += v[r * d + e] * a[ao + r * n + c];
                vh[vo + j * n + c] = sigma > 1e-14f ? acc / sigma : 0.0f;
            }
        }
    }
}

#define FA_META int B = fam[0]; int H = fam[1]; int Sq = fam[2]; int Sk = fam[3]; int D = fam[4]; int qb = fam[5]; int qh = fam[6]; int qr = fam[7]; int kb = fam[8]; int kh = fam[9]; int kr = fam[10]; int vb = fam[11]; int vh = fam[12]; int vr = fam[13]; int ob = fam[14]; int oh = fam[15]; int orw = fam[16]; int mb = fam[17]; int mh = fam[18]; int mi = fam[19]; int mj = fam[20]; int causal = fam[21]; int coff = fam[22]; int hasMask = fam[23]; float scale = as_type<float>(fam[24]); float cap = as_type<float>(fam[25]);
#define FA_ALLOWED(I, J) (!(causal != 0 && (J) > (I) + coff) && !(hasMask != 0 && mask[b * mb + h * mh + (I) * mi + (J) * mj] == 0))
#define FA_SCORE(I, J, S) { S = 0.0f; for (int fa_d = 0; fa_d < D; fa_d++) S += (q[b * qb + h * qh + (I) * qr + fa_d] * scale) * k[b * kb + h * kh + (J) * kr + fa_d]; }
#define FA_CAP(S, SL) { SL = 1.0f; if (cap > 0.0f) { float fa_t = tanh(S / cap); SL = 1.0f - fa_t * fa_t; S = cap * fa_t; } }
#define FA_DP(I, J, DP) { DP = 0.0f; for (int fa_e = 0; fa_e < D; fa_e++) DP += dout[b * ob + h * oh + (I) * orw + fa_e] * v[b * vb + h * vh + (J) * vr + fa_e]; }

kernel void fa_p_lse(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const int* fam [[buffer(2)]], device const int* mask [[buffer(3)]], device float* lse [[buffer(4)]], constant int& total [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; if (t >= total) return;
    FA_META
    int i = t % Sq; int bh = t / Sq; int b = bh / H; int h = bh % H;
    float m = 0.0f; int seen = 0;
    for (int j = 0; j < Sk; j++) {
        if (!(FA_ALLOWED(i, j))) continue;
        float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
        if (seen == 0 || s > m) m = s;
        seen = 1;
    }
    if (seen == 0) { lse[t] = as_type<float>(0xff800000u); return; }
    float l = 0.0f;
    for (int j = 0; j < Sk; j++) {
        if (!(FA_ALLOWED(i, j))) continue;
        float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
        l += exp(s - m);
    }
    lse[t] = m + log(l);
}

kernel void fa_p_out(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]], device const int* fam [[buffer(3)]], device const int* mask [[buffer(4)]], device const float* lse [[buffer(5)]], device float* o [[buffer(6)]], constant int& total [[buffer(7)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; if (t >= total) return;
    FA_META
    int e = t % D; int r = t / D; int i = r % Sq; int bh = r / Sq; int b = bh / H; int h = bh % H;
    float lr = lse[r]; float acc = 0.0f;
    if (lr > -3.0e38f) {
        for (int j = 0; j < Sk; j++) {
            if (!(FA_ALLOWED(i, j))) continue;
            float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
            acc += exp(s - lr) * v[b * vb + h * vh + j * vr + e];
        }
    }
    o[b * ob + h * oh + i * orw + e] = acc;
}

kernel void fa_p_delta(device const float* o [[buffer(0)]], device const float* dout [[buffer(1)]], device const int* fam [[buffer(2)]], device float* st [[buffer(3)]], constant int& total [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; if (t >= total) return;
    FA_META
    int i = t % Sq; int bh = t / Sq; int b = bh / H; int h = bh % H;
    float acc = 0.0f;
    for (int e = 0; e < D; e++) acc += dout[b * ob + h * oh + i * orw + e] * o[b * ob + h * oh + i * orw + e];
    st[total + t] = acc;
}

kernel void fa_p_dq(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]], device const float* dout [[buffer(3)]], device const float* st [[buffer(4)]], device const int* fam [[buffer(5)]], device const int* mask [[buffer(6)]], device float* dq [[buffer(7)]], constant int& total [[buffer(8)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; if (t >= total) return;
    FA_META
    int rows = B * H * Sq;
    int e = t % D; int r = t / D; int i = r % Sq; int bh = r / Sq; int b = bh / H; int h = bh % H;
    float lr = st[r]; float acc = 0.0f;
    if (lr > -3.0e38f) {
        float dl = st[rows + r];
        for (int j = 0; j < Sk; j++) {
            if (!(FA_ALLOWED(i, j))) continue;
            float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
            float dp; FA_DP(i, j, dp)
            acc += exp(s - lr) * (dp - dl) * sl * k[b * kb + h * kh + j * kr + e];
        }
    }
    dq[b * qb + h * qh + i * qr + e] = acc * scale;
}

kernel void fa_p_dk(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]], device const float* dout [[buffer(3)]], device const float* st [[buffer(4)]], device const int* fam [[buffer(5)]], device const int* mask [[buffer(6)]], device float* dk [[buffer(7)]], constant int& total [[buffer(8)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; if (t >= total) return;
    FA_META
    int rows = B * H * Sq;
    int e = t % D; int r = t / D; int j = r % Sk; int bh = r / Sk; int b = bh / H; int h = bh % H;
    float ak = 0.0f;
    for (int i = 0; i < Sq; i++) {
        float lr = st[bh * Sq + i];
        if (!(lr > -3.0e38f)) continue;
        if (!(FA_ALLOWED(i, j))) continue;
        float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
        float dp; FA_DP(i, j, dp)
        ak += exp(s - lr) * (dp - st[rows + bh * Sq + i]) * sl * q[b * qb + h * qh + i * qr + e] * scale;
    }
    dk[b * kb + h * kh + j * kr + e] = ak;
}

kernel void fa_p_dv(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* dout [[buffer(2)]], device const float* st [[buffer(3)]], device const int* fam [[buffer(4)]], device const int* mask [[buffer(5)]], device float* dv [[buffer(6)]], constant int& total [[buffer(7)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; if (t >= total) return;
    FA_META
    int e = t % D; int r = t / D; int j = r % Sk; int bh = r / Sk; int b = bh / H; int h = bh % H;
    float av = 0.0f;
    for (int i = 0; i < Sq; i++) {
        float lr = st[bh * Sq + i];
        if (!(lr > -3.0e38f)) continue;
        if (!(FA_ALLOWED(i, j))) continue;
        float s; float sl; FA_SCORE(i, j, s) FA_CAP(s, sl)
        av += exp(s - lr) * dout[b * ob + h * oh + i * orw + e];
    }
    dv[b * vb + h * vh + j * vr + e] = av;
}

kernel void bx_strided_view_copy(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], device const int* layout [[buffer(2)]], constant int& rank [[buffer(3)]], constant int& offset [[buffer(4)]], constant int& count [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= count) return;
    int remaining = k; int source = offset;
    for (int d = rank - 1; d >= 0; d--) {
        source += (remaining % layout[d]) * layout[rank + d];
        remaining /= layout[d];
    }
    output[k] = input[source];
}
kernel void lstm_p_gates(device const float* x [[buffer(0)]], device const float* wih [[buffer(1)]], device const float* whh [[buffer(2)]], device const float* bih [[buffer(3)]], device const float* bhh [[buffer(4)]], device const float* allH [[buffer(5)]], device float* gates [[buffer(6)]], constant int& t [[buffer(7)]], constant int& B [[buffer(8)]], constant int& T [[buffer(9)]], constant int& In [[buffer(10)]], constant int& H [[buffer(11)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * 4 * H) return;
    int b = k / (4 * H); int row = k - b * 4 * H;
    float s = bih[row] + bhh[row];
    for (int i = 0; i < In; i++) s += wih[row * In + i] * x[(b * T + t) * In + i];
    for (int j = 0; j < H; j++) s += whh[row * H + j] * allH[(t * B + b) * H + j];
    int g = row / H;
    gates[(t * B + b) * 4 * H + row] = g == 2 ? tanh(s) : 1.0f / (1.0f + exp(-s));
}

kernel void lstm_p_cell(device const float* gates [[buffer(0)]], device float* allC [[buffer(1)]], device float* allH [[buffer(2)]], device float* y [[buffer(3)]], constant int& t [[buffer(4)]], constant int& B [[buffer(5)]], constant int& T [[buffer(6)]], constant int& H [[buffer(7)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int gb = (t * B + b) * 4 * H;
    float ig = gates[gb + h]; float fg = gates[gb + H + h]; float gg = gates[gb + 2 * H + h]; float og = gates[gb + 3 * H + h];
    float c = fg * allC[(t * B + b) * H + h] + ig * gg;
    float hv = og * tanh(c);
    allC[((t + 1) * B + b) * H + h] = c;
    allH[((t + 1) * B + b) * H + h] = hv;
    y[(b * T + t) * H + h] = hv;
}

kernel void lstm_p_cell_bwd(device const float* gy [[buffer(0)]], device const float* gates [[buffer(1)]], device const float* allC [[buffer(2)]], device const float* dh [[buffer(3)]], device float* dc [[buffer(4)]], device float* dpre [[buffer(5)]], constant int& t [[buffer(6)]], constant int& B [[buffer(7)]], constant int& T [[buffer(8)]], constant int& H [[buffer(9)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int gb = (t * B + b) * 4 * H;
    float ig = gates[gb + h]; float fg = gates[gb + H + h]; float gg = gates[gb + 2 * H + h]; float og = gates[gb + 3 * H + h];
    float c = allC[((t + 1) * B + b) * H + h]; float tc = tanh(c);
    float dhv = gy[(b * T + t) * H + h] + dh[k];
    float dcv = dhv * og * (1.0f - tc * tc) + dc[k];
    dpre[gb + h] = dcv * gg * ig * (1.0f - ig);
    dpre[gb + H + h] = dcv * allC[(t * B + b) * H + h] * fg * (1.0f - fg);
    dpre[gb + 2 * H + h] = dcv * ig * (1.0f - gg * gg);
    dpre[gb + 3 * H + h] = dhv * tc * og * (1.0f - og);
    dc[k] = dcv * fg;
}

kernel void lstm_p_dh(device const float* dpre [[buffer(0)]], device const float* whh [[buffer(1)]], device float* dh [[buffer(2)]], constant int& t [[buffer(3)]], constant int& B [[buffer(4)]], constant int& H [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int j = k - b * H;
    int gb = (t * B + b) * 4 * H;
    float s = 0.0f;
    for (int row = 0; row < 4 * H; row++) s += dpre[gb + row] * whh[row * H + j];
    dh[k] = s;
}

kernel void lstm_p_gin(device const float* dpre [[buffer(0)]], device const float* wih [[buffer(1)]], device float* gin [[buffer(2)]], constant int& B [[buffer(3)]], constant int& T [[buffer(4)]], constant int& In [[buffer(5)]], constant int& H [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * T * In) return;
    int i = k % In; int bt = k / In; int t = bt % T; int b = bt / T;
    int gb = (t * B + b) * 4 * H;
    float s = 0.0f;
    for (int row = 0; row < 4 * H; row++) s += dpre[gb + row] * wih[row * In + i];
    gin[k] = s;
}

kernel void lstm_p_gwih(device const float* dpre [[buffer(0)]], device const float* x [[buffer(1)]], device float* gwih [[buffer(2)]], constant int& B [[buffer(3)]], constant int& T [[buffer(4)]], constant int& In [[buffer(5)]], constant int& H [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= 4 * H * In) return;
    int row = k / In; int i = k - row * In;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dpre[(t * B + b) * 4 * H + row] * x[(b * T + t) * In + i];
    gwih[k] = s;
}

kernel void lstm_p_gwhh(device const float* dpre [[buffer(0)]], device const float* allH [[buffer(1)]], device float* gwhh [[buffer(2)]], constant int& B [[buffer(3)]], constant int& T [[buffer(4)]], constant int& H [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= 4 * H * H) return;
    int row = k / H; int j = k - row * H;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dpre[(t * B + b) * 4 * H + row] * allH[(t * B + b) * H + j];
    gwhh[k] = s;
}

kernel void lstm_p_gbias(device const float* dpre [[buffer(0)]], device float* gbih [[buffer(1)]], device float* gbhh [[buffer(2)]], constant int& B [[buffer(3)]], constant int& T [[buffer(4)]], constant int& H [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int row = (int)bx_tid; if (row >= 4 * H) return;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dpre[(t * B + b) * 4 * H + row];
    gbih[row] = s;
    gbhh[row] = s;
}

kernel void lstm_p_gate_cell(device const float* pre [[buffer(0)]], device const float* old [[buffer(1)]], device float* y [[buffer(2)]], constant int& N [[buffer(3)]], constant int& H [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= N) return;
    int g = (k / H) * 4 * H + k % H;
    float ig = 1.0f / (1.0f + exp(-pre[g]));
    float fg = 1.0f / (1.0f + exp(-pre[g + H]));
    float z = tanh(pre[g + 2 * H]);
    float og = 1.0f / (1.0f + exp(-pre[g + 3 * H]));
    float c = fg * old[k] + ig * z;
    y[k] = og * tanh(c); y[N + k] = c;
}
kernel void lstm_p_gate_cell_bwd(device const float* pre [[buffer(0)]], device const float* old [[buffer(1)]], device const float* y [[buffer(2)]], device const float* dy [[buffer(3)]], device float* dg [[buffer(4)]], device float* dc [[buffer(5)]], constant int& N [[buffer(6)]], constant int& H [[buffer(7)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= N) return;
    int g = (k / H) * 4 * H + k % H;
    float ig = 1.0f / (1.0f + exp(-pre[g]));
    float fg = 1.0f / (1.0f + exp(-pre[g + H]));
    float z = tanh(pre[g + 2 * H]);
    float og = 1.0f / (1.0f + exp(-pre[g + 3 * H]));
    float tc = tanh(y[N + k]); float dh = dy[k];
    float d = dy[N + k] + dh * og * (1.0f - tc * tc);
    dg[g] = d * z * ig * (1.0f - ig);
    dg[g + H] = d * old[k] * fg * (1.0f - fg);
    dg[g + 2 * H] = d * ig * (1.0f - z * z);
    dg[g + 3 * H] = dh * tc * og * (1.0f - og);
    dc[k] = d * fg;
}

kernel void xlstm_train_forward(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]], device const float* V [[buffer(2)]], device const float* I [[buffer(3)]], device const float* F [[buffer(4)]], device const float* O [[buffer(5)]], device float* Y [[buffer(6)]], device float* S [[buffer(7)]], constant int& B [[buffer(8)]], constant int& T [[buffer(9)]], constant int& M [[buffer(10)]], constant int& H [[buffer(11)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int bh = (int)bx_tid; if (bh >= B * H) return;
    int D = M / H; int W = D * D + D; int z = bh * (T + 1) * W;
    float scale = 1.0f / sqrt((float)D);
    for (int j = 0; j < W; j++) S[z + j] = 0.0f;
    for (int t = 0; t < T; t++) {
        int x = (bh / H * T + t) * M + (bh % H) * D;
        int g = (bh / H * T + t) * H + bh % H;
        int a = z + t * W; int c = a + W;
        float iv = (I[g] > 4.85e8f ? 4.85e8f : I[g]); float f = F[g];
        for (int j = 0; j < D; j++) {
            S[c + D * D + j] = f * S[a + D * D + j] + iv * K[x + j] * scale;
            for (int r = 0; r < D; r++) S[c + r * D + j] = f * S[a + r * D + j] + iv * V[x + r] * K[x + j] * scale;
        }
        float nq = 0.0f;
        for (int j = 0; j < D; j++) nq += S[c + D * D + j] * Q[x + j];
        float nf = fmax(fabs(nq), 1.0f);
        for (int r = 0; r < D; r++) {
            float v = 0.0f;
            for (int j = 0; j < D; j++) v += S[c + r * D + j] * Q[x + j];
            Y[x + r] = O[g] * v / nf;
        }
    }
}
kernel void xlstm_train_backward(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]], device const float* V [[buffer(2)]], device const float* I [[buffer(3)]], device const float* F [[buffer(4)]], device const float* O [[buffer(5)]], device const float* DY [[buffer(6)]], device const float* S [[buffer(7)]], device float* A [[buffer(8)]], device float* G [[buffer(9)]], constant int& B [[buffer(10)]], constant int& T [[buffer(11)]], constant int& M [[buffer(12)]], constant int& H [[buffer(13)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int bh = (int)bx_tid; if (bh >= B * H) return;
    int D = M / H; int W = D * D + D; int z = bh * (T + 1) * W; int az = bh * W;
    int N = B * T * M; int NG = B * T * H;
    float scale = 1.0f / sqrt((float)D);
    for (int j = 0; j < W; j++) A[az + j] = 0.0f;
    for (int t = T - 1; t >= 0; t--) {
        int x = (bh / H * T + t) * M + (bh % H) * D;
        int g = (bh / H * T + t) * H + bh % H;
        int a = z + t * W; int c = a + W;
        float iv = (I[g] > 4.85e8f ? 4.85e8f : I[g]); float f = F[g]; float o = O[g];
        float nq = 0.0f;
        for (int j = 0; j < D; j++) { nq += S[c + D * D + j] * Q[x + j]; G[x + j] = 0.0f; }
        float nf = fmax(fabs(nq), 1.0f); float dnf = 0.0f; float doGate = 0.0f;
        for (int r = 0; r < D; r++) {
            float v = 0.0f;
            for (int j = 0; j < D; j++) v += S[c + r * D + j] * Q[x + j];
            doGate += DY[x + r] * v / nf;
            dnf -= DY[x + r] * o * v / (nf * nf);
            float dn = DY[x + r] * o / nf;
            for (int j = 0; j < D; j++) {
                A[az + r * D + j] += dn * Q[x + j];
                G[x + j] += S[c + r * D + j] * dn;
            }
        }
        float dnq = fabs(nq) > 1.0f ? dnf * (nq > 0.0f ? 1.0f : -1.0f) : 0.0f;
        float di = 0.0f; float df = 0.0f;
        for (int j = 0; j < D; j++) {
            A[az + D * D + j] += dnq * Q[x + j];
            G[x + j] += dnq * S[c + D * D + j];
            di += A[az + D * D + j] * K[x + j] * scale;
            df += A[az + D * D + j] * S[a + D * D + j];
            float dk = A[az + D * D + j] * iv;
            for (int r = 0; r < D; r++) dk += A[az + r * D + j] * iv * V[x + r];
            G[N + x + j] = dk * scale;
        }
        for (int r = 0; r < D; r++) {
            float dv = 0.0f;
            for (int j = 0; j < D; j++) {
                float dc = A[az + r * D + j];
                dv += dc * iv * K[x + j] * scale;
                di += dc * V[x + r] * K[x + j] * scale;
                df += dc * S[a + r * D + j];
            }
            G[2 * N + x + r] = dv;
        }
        G[3 * N + g] = I[g] > 4.85e8f ? 0.0f : di;
        G[3 * N + NG + g] = df; G[3 * N + 2 * NG + g] = doGate;
        for (int j = 0; j < W; j++) A[az + j] *= f;
    }
}

kernel void gru_p_proj(device const float* x [[buffer(0)]], device const float* wih [[buffer(1)]], device const float* whh [[buffer(2)]], device const float* bih [[buffer(3)]], device const float* bhh [[buffer(4)]], device const float* allH [[buffer(5)]], device float* pre [[buffer(6)]], constant int& t [[buffer(7)]], constant int& B [[buffer(8)]], constant int& T [[buffer(9)]], constant int& In [[buffer(10)]], constant int& H [[buffer(11)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * 3 * H) return;
    int b = k / (3 * H); int row = k - b * 3 * H;
    float si = bih[row];
    for (int i = 0; i < In; i++) si += wih[row * In + i] * x[(b * T + t) * In + i];
    float sh = bhh[row];
    for (int j = 0; j < H; j++) sh += whh[row * H + j] * allH[(t * B + b) * H + j];
    pre[b * 6 * H + row] = si;
    pre[b * 6 * H + 3 * H + row] = sh;
}

kernel void gru_p_cell(device const float* pre [[buffer(0)]], device float* gates [[buffer(1)]], device float* ghn [[buffer(2)]], device float* allH [[buffer(3)]], device float* y [[buffer(4)]], constant int& t [[buffer(5)]], constant int& B [[buffer(6)]], constant int& T [[buffer(7)]], constant int& H [[buffer(8)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int p = b * 6 * H;
    float r = 1.0f / (1.0f + exp(-(pre[p + h] + pre[p + 3 * H + h])));
    float z = 1.0f / (1.0f + exp(-(pre[p + H + h] + pre[p + 4 * H + h])));
    float hn = pre[p + 5 * H + h];
    float n = tanh(pre[p + 2 * H + h] + r * hn);
    float hp = allH[(t * B + b) * H + h];
    float hv = (1.0f - z) * n + z * hp;
    int gb = (t * B + b) * 3 * H;
    gates[gb + h] = r; gates[gb + H + h] = z; gates[gb + 2 * H + h] = n;
    ghn[(t * B + b) * H + h] = hn;
    allH[((t + 1) * B + b) * H + h] = hv;
    y[(b * T + t) * H + h] = hv;
}

kernel void gru_p_cell_bwd(device const float* gy [[buffer(0)]], device const float* gates [[buffer(1)]], device const float* ghn [[buffer(2)]], device const float* allH [[buffer(3)]], device const float* dh [[buffer(4)]], device float* dgi [[buffer(5)]], device float* dgh [[buffer(6)]], constant int& t [[buffer(7)]], constant int& B [[buffer(8)]], constant int& T [[buffer(9)]], constant int& H [[buffer(10)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float r = gates[gb + h]; float z = gates[gb + H + h]; float n = gates[gb + 2 * H + h];
    float hn = ghn[(t * B + b) * H + h];
    float hp = allH[(t * B + b) * H + h];
    float dhv = gy[(b * T + t) * H + h] + dh[k];
    float dnp = dhv * (1.0f - z) * (1.0f - n * n);
    float drp = dnp * hn * r * (1.0f - r);
    float dzp = dhv * (hp - n) * z * (1.0f - z);
    dgi[gb + h] = drp; dgi[gb + H + h] = dzp; dgi[gb + 2 * H + h] = dnp;
    dgh[gb + h] = drp; dgh[gb + H + h] = dzp; dgh[gb + 2 * H + h] = dnp * r;
}

kernel void gru_p_dh(device const float* gy [[buffer(0)]], device const float* gates [[buffer(1)]], device const float* dgh [[buffer(2)]], device const float* whh [[buffer(3)]], device float* dh [[buffer(4)]], constant int& t [[buffer(5)]], constant int& B [[buffer(6)]], constant int& T [[buffer(7)]], constant int& H [[buffer(8)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int j = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float s = (gy[(b * T + t) * H + j] + dh[k]) * gates[gb + H + j];
    for (int row = 0; row < 3 * H; row++) s += dgh[gb + row] * whh[row * H + j];
    dh[k] = s;
}

kernel void gru_p_gin(device const float* dgi [[buffer(0)]], device const float* wih [[buffer(1)]], device float* gin [[buffer(2)]], constant int& B [[buffer(3)]], constant int& T [[buffer(4)]], constant int& In [[buffer(5)]], constant int& H [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * T * In) return;
    int i = k % In; int bt = k / In; int t = bt % T; int b = bt / T;
    int gb = (t * B + b) * 3 * H;
    float s = 0.0f;
    for (int row = 0; row < 3 * H; row++) s += dgi[gb + row] * wih[row * In + i];
    gin[k] = s;
}

kernel void gru_p_gwih(device const float* dgi [[buffer(0)]], device const float* x [[buffer(1)]], device float* gwih [[buffer(2)]], constant int& B [[buffer(3)]], constant int& T [[buffer(4)]], constant int& In [[buffer(5)]], constant int& H [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= 3 * H * In) return;
    int row = k / In; int i = k - row * In;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dgi[(t * B + b) * 3 * H + row] * x[(b * T + t) * In + i];
    gwih[k] = s;
}

kernel void gru_p_gwhh(device const float* dgh [[buffer(0)]], device const float* allH [[buffer(1)]], device float* gwhh [[buffer(2)]], constant int& B [[buffer(3)]], constant int& T [[buffer(4)]], constant int& H [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= 3 * H * H) return;
    int row = k / H; int j = k - row * H;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) s += dgh[(t * B + b) * 3 * H + row] * allH[(t * B + b) * H + j];
    gwhh[k] = s;
}

kernel void gru_p_gbias(device const float* dgi [[buffer(0)]], device const float* dgh [[buffer(1)]], device float* gbih [[buffer(2)]], device float* gbhh [[buffer(3)]], constant int& B [[buffer(4)]], constant int& T [[buffer(5)]], constant int& H [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int row = (int)bx_tid; if (row >= 3 * H) return;
    float si = 0.0f; float sh = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++) { si += dgi[(t * B + b) * 3 * H + row]; sh += dgh[(t * B + b) * 3 * H + row]; }
    gbih[row] = si;
    gbhh[row] = sh;
}
kernel void gru_rb_rh(device const float* pre [[buffer(0)]], device const float* allH [[buffer(1)]], device float* gates [[buffer(2)]], device float* rh [[buffer(3)]], constant int& t [[buffer(4)]], constant int& B [[buffer(5)]], constant int& H [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int p = b * 6 * H;
    float r = 1.0f / (1.0f + exp(-(pre[p + h] + pre[p + 3 * H + h])));
    gates[(t * B + b) * 3 * H + h] = r;
    rh[(t * B + b) * H + h] = r * allH[(t * B + b) * H + h];
}

kernel void gru_rb_projn(device const float* whh [[buffer(0)]], device const float* bhh [[buffer(1)]], device const float* rh [[buffer(2)]], device float* pre [[buffer(3)]], constant int& t [[buffer(4)]], constant int& B [[buffer(5)]], constant int& H [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int j = k - b * H;
    float s = bhh[2 * H + j];
    for (int q = 0; q < H; q++) s += whh[(2 * H + j) * H + q] * rh[(t * B + b) * H + q];
    pre[b * 6 * H + 5 * H + j] = s;
}

kernel void gru_rb_cell(device const float* pre [[buffer(0)]], device float* gates [[buffer(1)]], device float* ghn [[buffer(2)]], device float* allH [[buffer(3)]], device float* y [[buffer(4)]], constant int& t [[buffer(5)]], constant int& B [[buffer(6)]], constant int& T [[buffer(7)]], constant int& H [[buffer(8)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int p = b * 6 * H;
    float z = 1.0f / (1.0f + exp(-(pre[p + H + h] + pre[p + 4 * H + h])));
    float hn = pre[p + 5 * H + h];
    float n = tanh(pre[p + 2 * H + h] + hn);
    float hp = allH[(t * B + b) * H + h];
    float hv = (1.0f - z) * n + z * hp;
    int gb = (t * B + b) * 3 * H;
    gates[gb + H + h] = z; gates[gb + 2 * H + h] = n;
    ghn[(t * B + b) * H + h] = hn;
    allH[((t + 1) * B + b) * H + h] = hv;
    y[(b * T + t) * H + h] = hv;
}

kernel void gru_rb_cell_bwd(device const float* gy [[buffer(0)]], device const float* gates [[buffer(1)]], device const float* allH [[buffer(2)]], device const float* dh [[buffer(3)]], device float* dgi [[buffer(4)]], device float* dgh [[buffer(5)]], constant int& t [[buffer(6)]], constant int& B [[buffer(7)]], constant int& T [[buffer(8)]], constant int& H [[buffer(9)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int h = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float z = gates[gb + H + h]; float n = gates[gb + 2 * H + h];
    float hp = allH[(t * B + b) * H + h];
    float dhv = gy[(b * T + t) * H + h] + dh[k];
    float dnp = dhv * (1.0f - z) * (1.0f - n * n);
    float dzp = dhv * (hp - n) * z * (1.0f - z);
    dgi[gb + H + h] = dzp; dgh[gb + H + h] = dzp;
    dgi[gb + 2 * H + h] = dnp; dgh[gb + 2 * H + h] = dnp;
}

kernel void gru_rb_drh(device const float* gates [[buffer(0)]], device const float* allH [[buffer(1)]], device const float* whh [[buffer(2)]], device float* dgi [[buffer(3)]], device float* dgh [[buffer(4)]], device float* drr [[buffer(5)]], constant int& t [[buffer(6)]], constant int& B [[buffer(7)]], constant int& H [[buffer(8)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int q = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float drh = 0.0f;
    for (int j = 0; j < H; j++) drh += dgh[gb + 2 * H + j] * whh[(2 * H + j) * H + q];
    float r = gates[gb + q];
    float drp = drh * allH[(t * B + b) * H + q] * r * (1.0f - r);
    dgi[gb + q] = drp; dgh[gb + q] = drp;
    drr[k] = drh * r;
}

kernel void gru_rb_dh(device const float* gy [[buffer(0)]], device const float* gates [[buffer(1)]], device const float* dgh [[buffer(2)]], device const float* whh [[buffer(3)]], device const float* drr [[buffer(4)]], device float* dh [[buffer(5)]], constant int& t [[buffer(6)]], constant int& B [[buffer(7)]], constant int& T [[buffer(8)]], constant int& H [[buffer(9)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= B * H) return;
    int b = k / H; int j = k - b * H;
    int gb = (t * B + b) * 3 * H;
    float s = (gy[(b * T + t) * H + j] + dh[k]) * gates[gb + H + j] + drr[k];
    for (int row = 0; row < 2 * H; row++) s += dgh[gb + row] * whh[row * H + j];
    dh[k] = s;
}

kernel void gru_rb_gwhh(device const float* dgh [[buffer(0)]], device const float* allH [[buffer(1)]], device const float* rh [[buffer(2)]], device float* gwhh [[buffer(3)]], constant int& B [[buffer(4)]], constant int& T [[buffer(5)]], constant int& H [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int k = (int)bx_tid; if (k >= 3 * H * H) return;
    int row = k / H; int j = k - row * H;
    float s = 0.0f;
    for (int t = 0; t < T; t++)
        for (int b = 0; b < B; b++)
            s += dgh[(t * B + b) * 3 * H + row] * (row < 2 * H ? allH[(t * B + b) * H + j] : rh[(t * B + b) * H + j]);
    gwhh[k] = s;
}

#define DL_GELU_STORE(Y, S) { float dl_u = 0.7978845608028654f * ((S) + 0.044715f * (S) * (S) * (S)); dl_u = dl_u < -15.0f ? -15.0f : (dl_u > 15.0f ? 15.0f : dl_u); Y = 0.5f * (S) * (1.0f + tanh(dl_u)); }
#define DL_HALF(V, W, I) { int dl_b = (W[(I) >> 1] >> (((I) & 1) * 16)) & 65535; int dl_e = (dl_b >> 10) & 31; int dl_m = dl_b & 1023; if (dl_e == 0) { float dl_f = dl_m; V = dl_f * 5.9604644775390625e-8f; } else { V = as_type<float>(((dl_e == 31 ? 255 : dl_e + 112) << 23) | (dl_m << 13)); } if ((dl_b & 32768) != 0) { V = -V; } }
kernel void dl_p_linear_gelu(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device const float* bias [[buffer(2)]], device float* y [[buffer(3)]], constant int& M [[buffer(4)]], constant int& K [[buffer(5)]], constant int& N [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; if (t >= M * N) return;
    int m = t / N; int n = t - m * N;
    float s = 0.0f;
    for (int k = 0; k < K; k++) s += x[m * K + k] * w[n * K + k];
    s += bias[n];
    DL_GELU_STORE(y[t], s)
}

kernel void dl_p_linear_gelu_fp16(device const int* xh [[buffer(0)]], device const int* wh [[buffer(1)]], device const float* bias [[buffer(2)]], device float* y [[buffer(3)]], constant int& M [[buffer(4)]], constant int& K [[buffer(5)]], constant int& N [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; if (t >= M * N) return;
    int m = t / N; int n = t - m * N;
    float s = 0.0f;
    for (int k = 0; k < K; k++) {
        float xv; float wv;
        DL_HALF(xv, xh, m * K + k)
        DL_HALF(wv, wh, n * K + k)
        s += xv * wv;
    }
    s += bias[n];
    DL_GELU_STORE(y[t], s)
}

kernel void dl_p_w8a8_linear_gelu(device const int* x8 [[buffer(0)]], device const int* w8 [[buffer(1)]], device const float* actScale [[buffer(2)]], device const float* wScales [[buffer(3)]], device const float* bias [[buffer(4)]], device float* y [[buffer(5)]], constant int& K [[buffer(6)]], constant int& N [[buffer(7)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int n = (int)bx_tid; if (n >= N) return;
    int acc = 0;
    for (int k = 0; k < K; k++) {
        int xi = (x8[k >> 2] >> ((k & 3) * 8)) & 255; if (xi > 127) xi -= 256;
        int j = n * K + k;
        int wi = (w8[j >> 2] >> ((j & 3) * 8)) & 255; if (wi > 127) wi -= 256;
        acc += xi * wi;
    }
    float af = acc;
    float s = af * actScale[0] * wScales[n] + bias[n];
    DL_GELU_STORE(y[n], s)
}

kernel void dl_p_w8a8_epilogue(device const int* acc [[buffer(0)]], device const float* actScale [[buffer(1)]], device const float* wScales [[buffer(2)]], device const float* bias [[buffer(3)]], device float* y [[buffer(4)]], constant int& N [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int n = (int)bx_tid; if (n >= N) return;
    float af = acc[n];
    float s = af * actScale[0] * wScales[n] + bias[n];
    DL_GELU_STORE(y[n], s)
}

#define DC_ROUND_HALF(V, BITS) { int dc_u = BITS; int dc_mag = dc_u & 2147483647; int dc_e = dc_mag >> 23; if (dc_e == 255) { V = as_type<float>(dc_mag); } else { if (dc_e < 113) { int dc_q = 0; int dc_s = 126 - dc_e; if (dc_e != 0 && dc_s <= 24) { int dc_full = (dc_mag & 8388607) | 8388608; dc_q = dc_full >> dc_s; int dc_rem = dc_full - (dc_q << dc_s); int dc_tie = 1 << (dc_s - 1); if (dc_rem > dc_tie || (dc_rem == dc_tie && (dc_q & 1) != 0)) { dc_q = dc_q + 1; } } float dc_f = dc_q; V = dc_f * 5.9604644775390625e-8f; } else { dc_mag = dc_mag + 4095 + ((dc_mag >> 13) & 1); dc_mag = (dc_mag >> 13) << 13; if (dc_mag >= 1199570944) { dc_mag = 2139095040; } V = as_type<float>(dc_mag); } } if (dc_u < 0) { V = -V; } }
kernel void dc_p_conv2d_direct_fp16hw(device const int* inBits [[buffer(0)]], device const int* wh [[buffer(1)]], device float* y [[buffer(2)]], constant int& batch [[buffer(3)]], constant int& inC [[buffer(4)]], constant int& inH [[buffer(5)]], constant int& inW [[buffer(6)]], constant int& outC [[buffer(7)]], constant int& outH [[buffer(8)]], constant int& outW [[buffer(9)]], constant int& kH [[buffer(10)]], constant int& kW [[buffer(11)]], constant int& sH [[buffer(12)]], constant int& sW [[buffer(13)]], constant int& pH [[buffer(14)]], constant int& pW [[buffer(15)]], constant int& dH [[buffer(16)]], constant int& dW [[buffer(17)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int t = (int)bx_tid; if (t >= batch * outC * outH * outW) return;
    int r1 = t / outW; int ow = t - r1 * outW;
    int r2 = r1 / outH; int oh = r1 - r2 * outH;
    int b = r2 / outC; int oc = r2 - b * outC;
    float acc = 0.0f;
    for (int ic = 0; ic < inC; ic++) {
        for (int kh = 0; kh < kH; kh++) {
            int ih = oh * sH - pH + kh * dH;
            if (ih >= 0 && ih < inH) {
                for (int kw = 0; kw < kW; kw++) {
                    int iw = ow * sW - pW + kw * dW;
                    if (iw >= 0 && iw < inW) {
                        float xv; float wv;
                        int wi = ((oc * inC + ic) * kH + kh) * kW + kw;
                        DC_ROUND_HALF(xv, inBits[((b * inC + ic) * inH + ih) * inW + iw])
                        DL_HALF(wv, wh, wi)
                        acc += xv * wv;
                    }
                }
            }
        }
    }
    y[t] = acc;
}

kernel void bx_weight_norm(device const float* direction [[buffer(0)]], device const float* gain [[buffer(1)]], device float* output [[buffer(2)]], constant int& rows [[buffer(3)]], constant int& columns [[buffer(4)]], constant float& epsilon [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int row = (int)((int)bx_tid / (int)bx_bdim); int lane = (int)(int)bx_lid;
    if (row >= rows) return;
    threadgroup float normPart[256];
    float sum = 0.0f;
    for (int j = lane; j < columns; j += (int)(int)bx_bdim) { float v = direction[row * columns + j]; sum += v * v; }
    normPart[lane] = sum; threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (int s = (int)(int)bx_bdim / 2; s > 0; s /= 2) {
        if (lane < s) normPart[lane] += normPart[lane + s];
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    float scale = gain[row] * (1.0f / sqrt(normPart[0] + epsilon));
    for (int j = lane; j < columns; j += (int)(int)bx_bdim) output[row * columns + j] = direction[row * columns + j] * scale;
}
kernel void bx_weight_norm_stats(device const float* gradOutput [[buffer(0)]], device const float* direction [[buffer(1)]], device float* stats [[buffer(2)]], constant int& rows [[buffer(3)]], constant int& columns [[buffer(4)]], constant float& epsilon [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int row = (int)((int)bx_tid / (int)bx_bdim); int lane = (int)(int)bx_lid;
    if (row >= rows) return;
    threadgroup float normPart[256]; threadgroup float dotPart[256];
    float sum = 0.0f; float dot = 0.0f;
    for (int j = lane; j < columns; j += (int)(int)bx_bdim) {
        float v = direction[row * columns + j]; sum += v * v; dot += gradOutput[row * columns + j] * v;
    }
    normPart[lane] = sum; dotPart[lane] = dot; threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (int s = (int)(int)bx_bdim / 2; s > 0; s /= 2) {
        if (lane < s) { normPart[lane] += normPart[lane + s]; dotPart[lane] += dotPart[lane + s]; }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    if (lane == 0) { stats[2 * row] = 1.0f / sqrt(normPart[0] + epsilon); stats[2 * row + 1] = dotPart[0]; }
}
kernel void bx_weight_norm_dv(device const float* gradOutput [[buffer(0)]], device const float* direction [[buffer(1)]], device const float* gain [[buffer(2)]], device const float* stats [[buffer(3)]], device float* gradDirection [[buffer(4)]], constant int& rows [[buffer(5)]], constant int& columns [[buffer(6)]], constant int& accumulate [[buffer(7)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= rows * columns) return;
    int row = i / columns; float inv = stats[2 * row];
    float value = (gain[row] * inv) * (gradOutput[i] - direction[i] * (stats[2 * row + 1] * inv * inv));
    gradDirection[i] = accumulate != 0 ? gradDirection[i] + value : value;
}
kernel void bx_weight_norm_dg(device const float* stats [[buffer(0)]], device float* gradGain [[buffer(1)]], constant int& rows [[buffer(2)]], constant int& accumulate [[buffer(3)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int row = (int)bx_tid; if (row >= rows) return;
    float value = stats[2 * row + 1] * stats[2 * row];
    gradGain[row] = accumulate != 0 ? gradGain[row] + value : value;
}

kernel void bx_rfft_magnitude(device const float* spectrum [[buffer(0)]], device float* output [[buffer(1)]], constant int& rows [[buffer(2)]], constant int& n [[buffer(3)]], constant int& spectrumStride [[buffer(4)]], constant float& epsilon [[buffer(5)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; int bins = n / 2 + 1; if (i >= rows * bins) return;
    int source = 2 * ((i / bins) * spectrumStride + i % bins);
    float re = spectrum[source]; float im = spectrum[source + 1];
    output[i] = sqrt((re * re + im * im) + epsilon);
}
kernel void bx_rfft_magnitude_seed(device const float* spectrum [[buffer(0)]], device const float* upstream [[buffer(1)]], device float* seed [[buffer(2)]], constant int& rows [[buffer(3)]], constant int& n [[buffer(4)]], constant int& spectrumStride [[buffer(5)]], constant float& epsilon [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= rows * n) return;
    int k = i % n; int bins = n / 2 + 1;
    float re = 0.0f; float im = 0.0f;
    if (k < bins) {
        int source = 2 * ((i / n) * spectrumStride + k);
        re = spectrum[source]; im = spectrum[source + 1];
        float magnitude = sqrt((re * re + im * im) + epsilon);
        float scale = upstream[(i / n) * bins + k] / magnitude;
        re = re * scale; im = im * scale;
    }
    seed[2 * i] = re; seed[2 * i + 1] = im;
}
kernel void bx_rfft_real_gradient(device const float* complex [[buffer(0)]], device float* output [[buffer(1)]], constant int& count [[buffer(2)]], constant float& scale [[buffer(3)]], constant int& accumulate [[buffer(4)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= count) return;
    float value = complex[2 * i] * scale;
    output[i] = accumulate != 0 ? output[i] + value : value;
}

kernel void bx_channel_bias_leaky(device const float* input [[buffer(0)]], device const float* bias [[buffer(1)]], device float* output [[buffer(2)]], constant int& batch [[buffer(3)]], constant int& channels [[buffer(4)]], constant int& spatial [[buffer(5)]], constant float& alpha [[buffer(6)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= batch * channels * spatial) return;
    int channel = (i / spatial) % channels;
#ifdef __CUDA_ARCH__
    float rounded = __fadd_rn(input[i], bias[channel]);
    output[i] = rounded > 0.0f ? rounded : __fmul_rn(alpha, rounded);
#else
    float rounded = input[i] + bias[channel];
    output[i] = rounded > 0.0f ? rounded : alpha * rounded;
#endif
}

kernel void bx_grouped_conv2d(device const float* first [[buffer(0)]], device const float* second [[buffer(1)]], device float* resultValues [[buffer(2)]], constant int& batch [[buffer(3)]], constant int& inChannels [[buffer(4)]], constant int& inHeight [[buffer(5)]], constant int& inWidth [[buffer(6)]], constant int& outChannels [[buffer(7)]], constant int& outHeight [[buffer(8)]], constant int& outWidth [[buffer(9)]], constant int& kernelH [[buffer(10)]], constant int& kernelW [[buffer(11)]], constant int& strideH [[buffer(12)]], constant int& strideW [[buffer(13)]], constant int& padH [[buffer(14)]], constant int& padW [[buffer(15)]], constant int& dilationH [[buffer(16)]], constant int& dilationW [[buffer(17)]], constant int& groups [[buffer(18)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {

int i = (int)bx_tid;
if (i >= batch * outChannels * outHeight * outWidth) return;
int ow = i % outWidth;
int oh = (i / outWidth) % outHeight;
int oc = (i / (outWidth * outHeight)) % outChannels;
int bn = i / (outWidth * outHeight * outChannels);
int cg = inChannels / groups;
int kg = outChannels / groups;
int group = oc / kg;
float sum = 0.0f;
for (int ci = 0; ci < cg; ci += 1) {
    for (int kh = 0; kh < kernelH; kh += 1) {
        int ih = oh * strideH - padH + kh * dilationH;
        if (ih < 0 || ih >= inHeight) continue;
        for (int kw = 0; kw < kernelW; kw += 1) {
            int iw = ow * strideW - padW + kw * dilationW;
            if (iw < 0 || iw >= inWidth) continue;
            int x = ((bn * inChannels + group * cg + ci) * inHeight + ih) * inWidth + iw;
            int w = ((oc * cg + ci) * kernelH + kh) * kernelW + kw;
            sum += first[x] * second[w];
        }
    }
}
resultValues[i] = sum;
}

kernel void bx_grouped_conv2d_dx(device const float* first [[buffer(0)]], device const float* second [[buffer(1)]], device float* resultValues [[buffer(2)]], constant int& batch [[buffer(3)]], constant int& inChannels [[buffer(4)]], constant int& inHeight [[buffer(5)]], constant int& inWidth [[buffer(6)]], constant int& outChannels [[buffer(7)]], constant int& outHeight [[buffer(8)]], constant int& outWidth [[buffer(9)]], constant int& kernelH [[buffer(10)]], constant int& kernelW [[buffer(11)]], constant int& strideH [[buffer(12)]], constant int& strideW [[buffer(13)]], constant int& padH [[buffer(14)]], constant int& padW [[buffer(15)]], constant int& dilationH [[buffer(16)]], constant int& dilationW [[buffer(17)]], constant int& groups [[buffer(18)]], constant int& accumulate [[buffer(19)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {

int i = (int)bx_tid;
if (i >= batch * inChannels * inHeight * inWidth) return;
int iw = i % inWidth;
int ih = (i / inWidth) % inHeight;
int ic = (i / (inWidth * inHeight)) % inChannels;
int bn = i / (inWidth * inHeight * inChannels);
int cg = inChannels / groups;
int kg = outChannels / groups;
int group = ic / cg;
int ci = ic % cg;
float sum = 0.0f;
for (int ko = 0; ko < kg; ko += 1) {
    int oc = group * kg + ko;
    for (int kh = 0; kh < kernelH; kh += 1) {
        int ny = ih + padH - kh * dilationH;
        if (ny < 0 || ny % strideH != 0) continue;
        int oh = ny / strideH;
        if (oh >= outHeight) continue;
        for (int kw = 0; kw < kernelW; kw += 1) {
            int nx = iw + padW - kw * dilationW;
            if (nx < 0 || nx % strideW != 0) continue;
            int ow = nx / strideW;
            if (ow >= outWidth) continue;
            int dy = ((bn * outChannels + oc) * outHeight + oh) * outWidth + ow;
            int w = ((oc * cg + ci) * kernelH + kh) * kernelW + kw;
            sum += first[dy] * second[w];
        }
    }
}
if (accumulate != 0) { resultValues[i] += sum; }
else { resultValues[i] = sum; }
}

kernel void bx_grouped_conv2d_dw(device const float* first [[buffer(0)]], device const float* second [[buffer(1)]], device float* resultValues [[buffer(2)]], constant int& batch [[buffer(3)]], constant int& inChannels [[buffer(4)]], constant int& inHeight [[buffer(5)]], constant int& inWidth [[buffer(6)]], constant int& outChannels [[buffer(7)]], constant int& outHeight [[buffer(8)]], constant int& outWidth [[buffer(9)]], constant int& kernelH [[buffer(10)]], constant int& kernelW [[buffer(11)]], constant int& strideH [[buffer(12)]], constant int& strideW [[buffer(13)]], constant int& padH [[buffer(14)]], constant int& padW [[buffer(15)]], constant int& dilationH [[buffer(16)]], constant int& dilationW [[buffer(17)]], constant int& groups [[buffer(18)]], constant int& accumulate [[buffer(19)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {

int i = (int)bx_tid;
int cg = inChannels / groups;
if (i >= outChannels * cg * kernelH * kernelW) return;
int kw = i % kernelW;
int kh = (i / kernelW) % kernelH;
int ci = (i / (kernelW * kernelH)) % cg;
int oc = i / (kernelW * kernelH * cg);
int group = oc / (outChannels / groups);
int ic = group * cg + ci;
float sum = 0.0f;
for (int bn = 0; bn < batch; bn += 1) {
    for (int oh = 0; oh < outHeight; oh += 1) {
        int ih = oh * strideH - padH + kh * dilationH;
        if (ih < 0 || ih >= inHeight) continue;
        for (int ow = 0; ow < outWidth; ow += 1) {
            int iw = ow * strideW - padW + kw * dilationW;
            if (iw < 0 || iw >= inWidth) continue;
            int x = ((bn * inChannels + ic) * inHeight + ih) * inWidth + iw;
            int dy = ((bn * outChannels + oc) * outHeight + oh) * outWidth + ow;
            sum += first[x] * second[dy];
        }
    }
}
if (accumulate != 0) { resultValues[i] += sum; }
else { resultValues[i] = sum; }
}

kernel void bx_channel_bias_residual(device const float* input [[buffer(0)]], device const float* bias [[buffer(1)]], device const float* residual [[buffer(2)]], device float* output [[buffer(3)]], device float* activated [[buffer(4)]], constant int& batch [[buffer(5)]], constant int& channels [[buffer(6)]], constant int& spatial [[buffer(7)]], constant float& alpha [[buffer(8)]], constant int& writeActivation [[buffer(9)]], uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], uint bx_bdim [[threads_per_threadgroup]]) {
    int i = (int)bx_tid; if (i >= batch * channels * spatial) return;
    int channel = (i / spatial) % channels;
#ifdef __CUDA_ARCH__
    float biased = __fadd_rn(input[i], bias[channel]);
    float value = __fadd_rn(biased, residual[i]);
    output[i] = value;
    if (writeActivation) activated[i] = value > 0.0f ? value : __fmul_rn(alpha, value);
#else
    float biased = input[i] + bias[channel];
    float value = biased + residual[i];
    output[i] = value;
    if (writeActivation) activated[i] = value > 0.0f ? value : alpha * value;
#endif
}
