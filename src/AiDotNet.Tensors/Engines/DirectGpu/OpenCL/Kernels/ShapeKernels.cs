namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL shape/layout kernels: concat, slice, pad, tile, pixel shuffle, utility ops.
/// </summary>
public static class ShapeKernels
{
    public static string GetSource()
    {
        return @"
__kernel void concat_axis(__global const float* a, __global const float* b, __global float* output, int outerSize, int aInnerSize, int bInnerSize) {
    int idx = get_global_id(0); int totalInner = aInnerSize + bInnerSize; if (idx >= outerSize * totalInner) return;
    int outer = idx / totalInner; int inner = idx % totalInner;
    output[idx] = (inner < aInnerSize) ? a[outer * aInnerSize + inner] : b[outer * bInnerSize + (inner - aInnerSize)];
}
__kernel void slice_last_axis(__global const float* input, __global float* output, int outerSize, int inputInnerSize, int start, int sliceSize) {
    int idx = get_global_id(0); if (idx >= outerSize * sliceSize) return;
    output[idx] = input[(idx / sliceSize) * inputInnerSize + start + (idx % sliceSize)];
}
__kernel void set_slice_last_axis(__global float* output, __global const float* values, int outerSize, int outputInnerSize, int start, int sliceSize) {
    int idx = get_global_id(0); if (idx >= outerSize * sliceSize) return;
    output[(idx / sliceSize) * outputInnerSize + start + (idx % sliceSize)] = values[idx];
}
__kernel void stack_2(__global const float* a, __global const float* b, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[2 * idx] = a[idx]; output[2 * idx + 1] = b[idx];
}
__kernel void pad_2d(__global const float* input, __global float* output, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft, float padValue) {
    int idx = get_global_id(0); if (idx >= batch * channels * outH * outW) return;
    int w = idx % outW; int temp = idx / outW; int h = temp % outH; temp /= outH; int c = temp % channels; int b2 = temp / channels;
    int srcH = h - padTop; int srcW = w - padLeft;
    output[idx] = (srcH >= 0 && srcH < inH && srcW >= 0 && srcW < inW) ? input[((b2 * channels + c) * inH + srcH) * inW + srcW] : padValue;
}
__kernel void pad_2d_backward(__global const float* grad_output, __global float* grad_input, int batch, int channels, int inH, int inW, int outH, int outW, int padTop, int padLeft) {
    int idx = get_global_id(0); if (idx >= batch * channels * inH * inW) return;
    int w = idx % inW; int temp = idx / inW; int h = temp % inH; temp /= inH; int c = temp % channels; int b2 = temp / channels;
    grad_input[idx] = grad_output[((b2 * channels + c) * outH + (h + padTop)) * outW + (w + padLeft)];
}
__kernel void tile_last_axis(__global const float* input, __global float* output, int outerSize, int innerSize, int repeats) {
    int idx = get_global_id(0); int tiledInner = innerSize * repeats; if (idx >= outerSize * tiledInner) return;
    output[idx] = input[(idx / tiledInner) * innerSize + ((idx % tiledInner) % innerSize)];
}
__kernel void repeat_elements(__global const float* input, __global float* output, int outerSize, int innerSize, int repeats) {
    int idx = get_global_id(0); if (idx >= outerSize * innerSize * repeats) return;
    int outer = idx / (innerSize * repeats); int inner = (idx % (innerSize * repeats)) / repeats;
    output[idx] = input[outer * innerSize + inner];
}
__kernel void pixel_shuffle(__global const float* input, __global float* output, int batch, int channels, int inH, int inW, int scale) {
    int idx = get_global_id(0); int outH = inH * scale; int outW = inW * scale;
    if (idx >= batch * channels * outH * outW) return;
    int ow = idx % outW; int temp = idx / outW; int oh = temp % outH; temp /= outH; int oc = temp % channels; int b2 = temp / channels;
    int srcC = oc * scale * scale + (oh % scale) * scale + (ow % scale);
    output[idx] = input[((b2 * channels * scale * scale + srcC) * inH + oh / scale) * inW + ow / scale];
}
__kernel void pixel_shuffle_backward(__global const float* grad_output, __global float* grad_input, int batch, int channels, int inH, int inW, int scale) {
    int idx = get_global_id(0); if (idx >= batch * channels * scale * scale * inH * inW) return;
    int iw = idx % inW; int temp = idx / inW; int ih = temp % inH; temp /= inH;
    int ic = temp % (channels * scale * scale); int b2 = temp / (channels * scale * scale);
    int oc = ic / (scale * scale); int subIdx = ic % (scale * scale);
    int outH = inH * scale; int outW = inW * scale;
    grad_input[idx] = grad_output[((b2 * channels + oc) * outH + ih * scale + subIdx / scale) * outW + iw * scale + subIdx % scale];
}
__kernel void crop_2d(__global const float* input, __global float* output, int batch, int channels, int inH, int inW, int outH, int outW, int offsetH, int offsetW) {
    int idx = get_global_id(0); if (idx >= batch * channels * outH * outW) return;
    int w = idx % outW; int temp = idx / outW; int h = temp % outH; temp /= outH; int c = temp % channels; int b2 = temp / channels;
    output[idx] = input[((b2 * channels + c) * inH + (h + offsetH)) * inW + (w + offsetW)];
}
__kernel void eye_kernel(__global float* output, int n) {
    int idx = get_global_id(0); if (idx >= n * n) return;
    output[idx] = ((idx / n) == (idx % n)) ? 1.0f : 0.0f;
}
__kernel void linspace_kernel(__global float* output, float start, float step, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = start + step * (float)idx;
}
__kernel void one_hot_kernel(__global const float* indices, __global float* output, int batchSize, int numClasses) {
    int idx = get_global_id(0); if (idx >= batchSize * numClasses) return;
    output[idx] = ((int)indices[idx / numClasses] == (idx % numClasses)) ? 1.0f : 0.0f;
}
__kernel void diag_kernel(__global const float* input, __global float* output, int n) {
    int idx = get_global_id(0); if (idx >= n * n) return;
    output[idx] = ((idx / n) == (idx % n)) ? input[idx / n] : 0.0f;
}
__kernel void extract_diag_kernel(__global const float* input, __global float* output, int n, int cols) {
    int idx = get_global_id(0); if (idx >= n) return;
    output[idx] = input[idx * cols + idx];
}
__kernel void triangular_mask(__global float* output, int rows, int cols, int diagonal, float maskValue) {
    int idx = get_global_id(0); if (idx >= rows * cols) return;
    output[idx] = ((idx % cols) > (idx / cols) + diagonal) ? maskValue : 0.0f;
}
__kernel void masked_fill_kernel(__global const float* input, __global const float* mask, __global float* output, float fillValue, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = (mask[idx] != 0.0f) ? fillValue : input[idx];
}
__kernel void index_select(__global const float* input, __global const float* indices, __global float* output, int numIndices, int innerSize, int inputRows) {
    int idx = get_global_id(0); if (idx >= numIndices * innerSize) return;
    int selectedRow = (int)indices[idx / innerSize];
    if (selectedRow >= 0 && selectedRow < inputRows) {
        output[idx] = input[selectedRow * innerSize + (idx % innerSize)];
    } else {
        output[idx] = 0.0f;
    }
}
// Gather along one axis (take_along_dim/gather): input viewed [outer, axisIn, inner], indices/output
// viewed [outer, axisOut, inner]; output[o,j,i] = input[o, indices[o,j,i], i].
__kernel void take_along_dim(__global const float* input, __global const int* indices, __global float* output,
    int outerSize, int axisOut, int innerSize, int axisIn) {
    int idx = get_global_id(0); int total = outerSize * axisOut * innerSize; if (idx >= total) return;
    int inner = idx % innerSize;
    int outer = (idx / innerSize) / axisOut;
    int srcJ = indices[idx];
    if (srcJ < 0 || srcJ >= axisIn) { output[idx] = 0.0f; return; }
    output[idx] = input[(outer * axisIn + srcJ) * innerSize + inner];
}
// 3-vector cross product along an axis of size 3 (viewed [outer, 3, inner]).
__kernel void cross3(__global const float* a, __global const float* b, __global float* output, int outerSize, int innerSize) {
    int idx = get_global_id(0); if (idx >= outerSize * innerSize) return;
    int inner = idx % innerSize; int outer = idx / innerSize;
    int p = outer * 3 * innerSize + inner;
    float a0 = a[p], a1 = a[p + innerSize], a2 = a[p + 2 * innerSize];
    float b0 = b[p], b1 = b[p + innerSize], b2 = b[p + 2 * innerSize];
    output[p]                  = a1 * b2 - a2 * b1;
    output[p + innerSize]      = a2 * b0 - a0 * b2;
    output[p + 2 * innerSize]  = a0 * b1 - a1 * b0;
}
// Ldexp: output[i] = input[i] * 2^exponents[i] (ldexp is exact; no fast-math hazard).
__kernel void ldexp_kernel(__global const float* input, __global const int* exponents, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    output[idx] = ldexp(input[idx], exponents[idx]);
}
// 2-D Kronecker product: a[am,an] (X) b[bp,bq] -> out[am*bp, an*bq].
__kernel void kron2d(__global const float* a, __global const float* b, __global float* output, int am, int an, int bp, int bq) {
    int idx = get_global_id(0);
    int outCols = an * bq; int total = (am * bp) * outCols; if (idx >= total) return;
    int oc = idx % outCols; int orow = idx / outCols;
    int i = orow / bp; int k = orow % bp;
    int j = oc / bq;   int l = oc % bq;
    output[idx] = a[i * an + j] * b[k * bq + l];
}
// searchsorted/bucketize: binary search per value into the 1-D sorted sequence. right=0 lower_bound,
// right=1 upper_bound. Index written as float (exact in the small index range).
__kernel void search_sorted(__global const float* seq, __global const float* values, __global float* output, int seqLen, int numValues, int right) {
    int idx = get_global_id(0); if (idx >= numValues) return;
    float v = values[idx];
    int lo = 0, hi = seqLen;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int cond = (right != 0) ? (seq[mid] <= v) : (seq[mid] < v);
        if (cond) lo = mid + 1; else hi = mid;
    }
    output[idx] = (float)lo;
}
// index_copy/index_fill scatter-write. output pre-seeded with the original tensor; overwrite selected
// slices. mode 0 = copy from source[outer,j,inner]; mode 1 = write fillValue. indices is 1-D [idxAxis].
__kernel void index_write(__global float* output, __global const int* indices, __global const float* source,
    float fillValue, int mode, int outerSize, int idxAxis, int innerSize, int dstAxis) {
    int idx = get_global_id(0); int total = outerSize * idxAxis * innerSize; if (idx >= total) return;
    int inner = idx % innerSize;
    int j = (idx / innerSize) % idxAxis;
    int outer = (idx / innerSize) / idxAxis;
    int dstJ = indices[j];
    if (dstJ < 0 || dstJ >= dstAxis) return;
    float v = (mode == 0) ? source[idx] : fillValue;
    output[(outer * dstAxis + dstJ) * innerSize + inner] = v;
}
// Cross pairwise p-norm distance: x1[m,d], x2[n,d] -> out[m,n], out[i,j]=||x1[i]-x2[j]||_p.
__kernel void cdist(__global const float* x1, __global const float* x2, __global float* output, int m, int n, int d, float p) {
    int idx = get_global_id(0); if (idx >= m * n) return;
    int j = idx % n; int i = idx / n;
    float sum = 0.0f;
    for (int k = 0; k < d; k++) {
        float diff = fabs(x1[i * d + k] - x2[j * d + k]);
        if (p == 1.0f) sum += diff;
        else if (p == 2.0f) sum += diff * diff;
        else sum += pow(diff, p);
    }
    output[idx] = (p == 1.0f) ? sum : (p == 2.0f) ? sqrt(sum) : pow(sum, 1.0f / p);
}
// One bitonic compare-exchange step. values/indices are numRows rows of length rowLen (a power of 2).
// k = current bitonic sequence size, j = compare distance. NaN is treated as +inf (torch.sort order).
__kernel void bitonic_step(__global float* values, __global float* indices, int rowLen, int k, int j, int numRows, int descending) {
    int gid = get_global_id(0); if (gid >= numRows * rowLen) return;
    int i = gid % rowLen; int ixj = i ^ j;
    if (ixj <= i) return;
    int base = (gid / rowLen) * rowLen;
    float a = values[base + i], b = values[base + ixj];
    uint ua = as_uint(a), ub = as_uint(b);
    float ka = (((ua >> 23) & 0xFFu) == 0xFFu && (ua & 0x7FFFFFu) != 0u) ? INFINITY : a;
    float kb = (((ub >> 23) & 0xFFu) == 0xFFu && (ub & 0x7FFFFFu) != 0u) ? INFINITY : b;
    int up = ((i & k) == 0); if (descending != 0) up = !up;
    int doSwap = up ? (ka > kb) : (ka < kb);
    if (doSwap) {
        values[base + i] = b; values[base + ixj] = a;
        float t = indices[base + i]; indices[base + i] = indices[base + ixj]; indices[base + ixj] = t;
    }
}
// Copy the first copyLen elements of each row (src stride srcRowLen -> dst stride dstRowLen).
// Used both to pad (L->P, into a pre-filled buffer) and to un-pad (P->L) bitonic rows.
__kernel void copy_rows(__global const float* src, __global float* dst, int srcRowLen, int dstRowLen, int numRows, int copyLen) {
    int gid = get_global_id(0); if (gid >= numRows * copyLen) return;
    int i = gid % copyLen; int r = gid / copyLen;
    dst[r * dstRowLen + i] = src[r * srcRowLen + i];
}
// Initialize padded index rows (stored as float): idx[r*P + i] = (i < L) ? i : -1.
__kernel void iota_pad(__global float* idx, int L, int P, int numRows) {
    int gid = get_global_id(0); if (gid >= numRows * P) return;
    int i = gid % P; idx[gid] = (i < L) ? (float)i : -1.0f;
}
// Histogram into `bins` equal-width bins over [mn,mx]; out-of-range dropped, mx in last bin.
// hist must be pre-zeroed. Float atomic add via int cmpxchg loop (core int32 atomics).
inline void atomicAddF(volatile __global float* addr, float val) {
    union { int i; float f; } prev, next;
    do { prev.f = *addr; next.f = prev.f + val; }
    while (atomic_cmpxchg((volatile __global int*)addr, prev.i, next.i) != prev.i);
}
__kernel void histc(__global const float* input, volatile __global float* hist, int n, int bins, float mn, float mx) {
    int idx = get_global_id(0); if (idx >= n) return;
    float x = input[idx];
    if (x < mn || x > mx) return;
    float bw = (mx - mn) / (float)bins;
    int b = (int)((x - mn) / bw);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    atomicAddF(&hist[b], 1.0f);
}
// Condensed pairwise p-norm over rows of input[n,d]; output 1-D upper-triangle (i<j) order.
__kernel void pdist(__global const float* input, __global float* output, int n, int d, float p) {
    int flat = get_global_id(0); if (flat >= n * n) return;
    int j = flat % n; int i = flat / n;
    if (i >= j) return;
    float sum = 0.0f;
    for (int k = 0; k < d; k++) {
        float diff = fabs(input[i * d + k] - input[j * d + k]);
        if (p == 1.0f) sum += diff;
        else if (p == 2.0f) sum += diff * diff;
        else sum += pow(diff, p);
    }
    float dist = (p == 1.0f) ? sum : (p == 2.0f) ? sqrt(sum) : pow(sum, 1.0f / p);
    int outIdx = i * n - (i * (i + 1)) / 2 + (j - i - 1);
    output[outIdx] = dist;
}
// IEEE nextafter via direct bit manipulation; NaN detected by bit pattern (fast-math safe).
__kernel void next_after(__global const float* a, __global const float* b, __global float* output, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float av = a[idx], bv = b[idx];
    uint ua = as_uint(av), ub = as_uint(bv);
    int aNan = (((ua >> 23) & 0xFFu) == 0xFFu) && ((ua & 0x7FFFFFu) != 0u);
    int bNan = (((ub >> 23) & 0xFFu) == 0xFFu) && ((ub & 0x7FFFFFu) != 0u);
    if (aNan || bNan) { output[idx] = as_float(0x7FC00000u); return; }
    if (av == bv) { output[idx] = bv; return; }
    if (av == 0.0f) { output[idx] = as_float(bv > 0.0f ? 0x00000001u : 0x80000001u); return; }
    uint r = ua;
    if (bv > av) r = (av > 0.0f) ? (r + 1u) : (r - 1u);
    else         r = (av > 0.0f) ? (r - 1u) : (r + 1u);
    output[idx] = as_float(r);
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "concat_axis", "slice_last_axis", "set_slice_last_axis", "stack_2",
            "pad_2d", "pad_2d_backward", "tile_last_axis", "repeat_elements",
            "pixel_shuffle", "pixel_shuffle_backward", "crop_2d",
            "eye_kernel", "linspace_kernel", "one_hot_kernel",
            "diag_kernel", "extract_diag_kernel", "triangular_mask",
            "masked_fill_kernel", "index_select", "take_along_dim",
            "cross3", "ldexp_kernel", "kron2d", "search_sorted", "next_after", "index_write", "cdist", "pdist",
            "histc", "bitonic_step", "copy_rows", "iota_pad"
        };
    }
}
