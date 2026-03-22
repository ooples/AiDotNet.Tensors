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
__kernel void index_select(__global const float* input, __global const float* indices, __global float* output, int numIndices, int innerSize) {
    int idx = get_global_id(0); if (idx >= numIndices * innerSize) return;
    output[idx] = input[(int)indices[idx / innerSize] * innerSize + (idx % innerSize)];
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
            "masked_fill_kernel", "index_select"
        };
    }
}
