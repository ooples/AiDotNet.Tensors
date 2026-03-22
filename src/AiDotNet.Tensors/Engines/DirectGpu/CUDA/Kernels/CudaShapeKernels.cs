namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Fused CUDA kernels for tensor shape manipulation operations:
/// Concat, Slice, Stack, Pad, Tile, Repeat, PixelShuffle, Upsample.
/// All operations use single-pass index remapping for maximum throughput.
/// </summary>
public static class CudaShapeKernels
{
    public static string GetSource()
    {
        return @"
// ============================================================================
// Concat along axis (2 tensors)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void concat_axis(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output,
    int outerSize, int aInnerSize, int bInnerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalInner = aInnerSize + bInnerSize;
    int total = outerSize * totalInner;
    if (idx >= total) return;

    int outer = idx / totalInner;
    int inner = idx % totalInner;

    if (inner < aInnerSize)
        output[idx] = a[outer * aInnerSize + inner];
    else
        output[idx] = b[outer * bInnerSize + (inner - aInnerSize)];
}

// ============================================================================
// Slice along last axis: output = input[..., start:start+sliceSize]
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void slice_last_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int inputInnerSize, int start, int sliceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * sliceSize;
    if (idx >= total) return;

    int outer = idx / sliceSize;
    int inner = idx % sliceSize;
    output[idx] = input[outer * inputInnerSize + start + inner];
}

// Set slice: input[..., start:start+sliceSize] = values
extern ""C"" __global__ __launch_bounds__(256) void set_slice_last_axis(
    float* __restrict__ output,
    const float* __restrict__ values,
    int outerSize, int outputInnerSize, int start, int sliceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * sliceSize;
    if (idx >= total) return;

    int outer = idx / sliceSize;
    int inner = idx % sliceSize;
    output[outer * outputInnerSize + start + inner] = values[idx];
}

// ============================================================================
// Stack: interleave N same-shape tensors along a new axis
// For 2 tensors: output[2*i] = a[i], output[2*i+1] = b[i]
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void stack_2(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[2 * idx] = a[idx];
    output[2 * idx + 1] = b[idx];
}

// ============================================================================
// Pad with constant value
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void pad_2d(
    const float* __restrict__ input, float* __restrict__ output,
    int batch, int channels, int inH, int inW,
    int outH, int outW, int padTop, int padLeft, float padValue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * outH * outW;
    if (idx >= total) return;

    int w = idx % outW;
    int temp = idx / outW;
    int h = temp % outH;
    temp = temp / outH;
    int c = temp % channels;
    int b = temp / channels;

    int srcH = h - padTop;
    int srcW = w - padLeft;

    if (srcH >= 0 && srcH < inH && srcW >= 0 && srcW < inW)
        output[idx] = input[((b * channels + c) * inH + srcH) * inW + srcW];
    else
        output[idx] = padValue;
}

extern ""C"" __global__ __launch_bounds__(256) void pad_2d_backward(
    const float* __restrict__ grad_output, float* __restrict__ grad_input,
    int batch, int channels, int inH, int inW,
    int outH, int outW, int padTop, int padLeft)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * inH * inW;
    if (idx >= total) return;

    int w = idx % inW;
    int temp = idx / inW;
    int h = temp % inH;
    temp = temp / inH;
    int c = temp % channels;
    int b = temp / channels;

    grad_input[idx] = grad_output[((b * channels + c) * outH + (h + padTop)) * outW + (w + padLeft)];
}

// ============================================================================
// Tile/Repeat: repeat tensor along dimensions
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void tile_last_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int innerSize, int repeats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tiledInner = innerSize * repeats;
    int total = outerSize * tiledInner;
    if (idx >= total) return;

    int outer = idx / tiledInner;
    int inner = idx % tiledInner;
    output[idx] = input[outer * innerSize + (inner % innerSize)];
}

extern ""C"" __global__ __launch_bounds__(256) void repeat_elements(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int innerSize, int repeats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize * repeats;
    if (idx >= total) return;

    int outer = idx / (innerSize * repeats);
    int remaining = idx % (innerSize * repeats);
    int inner = remaining / repeats;
    output[idx] = input[outer * innerSize + inner];
}

// ============================================================================
// Pixel Shuffle: (B, C*r^2, H, W) → (B, C, H*r, W*r)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void pixel_shuffle(
    const float* __restrict__ input, float* __restrict__ output,
    int batch, int channels, int inH, int inW, int scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outH = inH * scale;
    int outW = inW * scale;
    int total = batch * channels * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int temp = idx / outW;
    int oh = temp % outH;
    temp = temp / outH;
    int oc = temp % channels;
    int b = temp / channels;

    int srcH = oh / scale;
    int srcW = ow / scale;
    int subH = oh % scale;
    int subW = ow % scale;
    int srcC = oc * scale * scale + subH * scale + subW;

    output[idx] = input[((b * channels * scale * scale + srcC) * inH + srcH) * inW + srcW];
}

extern ""C"" __global__ __launch_bounds__(256) void pixel_shuffle_backward(
    const float* __restrict__ grad_output, float* __restrict__ grad_input,
    int batch, int channels, int inH, int inW, int scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalIn = batch * channels * scale * scale * inH * inW;
    if (idx >= totalIn) return;

    int iw = idx % inW;
    int temp = idx / inW;
    int ih = temp % inH;
    temp = temp / inH;
    int ic = temp % (channels * scale * scale);
    int b = temp / (channels * scale * scale);

    int oc = ic / (scale * scale);
    int subIdx = ic % (scale * scale);
    int subH = subIdx / scale;
    int subW = subIdx % scale;

    int outH = inH * scale;
    int outW = inW * scale;
    int oh = ih * scale + subH;
    int ow = iw * scale + subW;

    grad_input[idx] = grad_output[((b * channels + oc) * outH + oh) * outW + ow];
}

// ============================================================================
// Crop and its backward
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void crop_2d(
    const float* __restrict__ input, float* __restrict__ output,
    int batch, int channels, int inH, int inW,
    int outH, int outW, int offsetH, int offsetW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * outH * outW;
    if (idx >= total) return;

    int w = idx % outW;
    int temp = idx / outW;
    int h = temp % outH;
    temp = temp / outH;
    int c = temp % channels;
    int b = temp / channels;

    output[idx] = input[((b * channels + c) * inH + (h + offsetH)) * inW + (w + offsetW)];
}

extern ""C"" __global__ __launch_bounds__(256) void crop_2d_backward(
    const float* __restrict__ grad_output, float* __restrict__ grad_input,
    int batch, int channels, int inH, int inW,
    int outH, int outW, int offsetH, int offsetW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * outH * outW;
    if (idx >= total) return;

    int w = idx % outW;
    int temp = idx / outW;
    int h = temp % outH;
    temp = temp / outH;
    int c = temp % channels;
    int b = temp / channels;

    atomicAdd(&grad_input[((b * channels + c) * inH + (h + offsetH)) * inW + (w + offsetW)], grad_output[idx]);
}

// ============================================================================
// Utility: eye, linspace, one_hot, fill, diag, triangular_mask
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void eye_kernel(
    float* __restrict__ output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int row = idx / n;
    int col = idx % n;
    output[idx] = (row == col) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void linspace_kernel(
    float* __restrict__ output, float start, float step, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = start + step * (float)idx;
}

extern ""C"" __global__ __launch_bounds__(256) void one_hot_kernel(
    const float* __restrict__ indices, float* __restrict__ output,
    int batchSize, int numClasses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * numClasses;
    if (idx >= total) return;

    int b = idx / numClasses;
    int c = idx % numClasses;
    output[idx] = ((int)indices[b] == c) ? 1.0f : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void diag_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int row = idx / n;
    int col = idx % n;
    output[idx] = (row == col) ? input[row] : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void extract_diag_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int n, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = input[idx * cols + idx];
}

extern ""C"" __global__ __launch_bounds__(256) void triangular_mask(
    float* __restrict__ output, int rows, int cols, int diagonal, float maskValue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    int col = idx % cols;
    // Upper triangular mask: mask positions where col > row + diagonal
    output[idx] = (col > row + diagonal) ? maskValue : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void masked_fill_kernel(
    const float* __restrict__ input, const float* __restrict__ mask,
    float* __restrict__ output, float fillValue, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = (mask[idx] != 0.0f) ? fillValue : input[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void where_kernel(
    const float* __restrict__ condition,
    const float* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = (condition[idx] != 0.0f) ? x[idx] : y[idx];
}

// Index select: output[i] = input[indices[i]]
extern ""C"" __global__ __launch_bounds__(256) void index_select(
    const float* __restrict__ input, const float* __restrict__ indices,
    float* __restrict__ output, int numIndices, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numIndices * innerSize;
    if (idx >= total) return;

    int i = idx / innerSize;
    int j = idx % innerSize;
    int srcIdx = (int)indices[i];
    output[idx] = input[srcIdx * innerSize + j];
}

// Squeeze is a no-op (just removes size-1 dims) — handled by shape metadata
// ExpandDims is a no-op — handled by shape metadata
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "concat_axis",
            "slice_last_axis",
            "set_slice_last_axis",
            "stack_2",
            "pad_2d",
            "pad_2d_backward",
            "tile_last_axis",
            "repeat_elements",
            "pixel_shuffle",
            "pixel_shuffle_backward",
            "crop_2d",
            "crop_2d_backward",
            "eye_kernel",
            "linspace_kernel",
            "one_hot_kernel",
            "diag_kernel",
            "extract_diag_kernel",
            "triangular_mask",
            "masked_fill_kernel",
            "where_kernel",
            "index_select"
        ];
    }
}
