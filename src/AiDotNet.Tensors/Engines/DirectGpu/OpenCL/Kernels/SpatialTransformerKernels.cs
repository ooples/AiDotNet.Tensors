// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for spatial transformer operations including TopK selection.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// OpenCL kernels for spatial transformer and selection operations.
    /// </summary>
    internal static class SpatialTransformerKernels
    {
        public static string GetSource()
        {
            return @"
// ===========================================================================
// TOP-K SELECTION KERNEL
// ===========================================================================

// TopK selection - each work group handles one row
__kernel void topk(
    __global const float* input,
    __global float* values,
    __global int* indices,
    const int outerSize,
    const int reduceSize,
    const int k,
    const int sorted,
    __local float* localTop,
    __local int* localIdx)
{
    int row = get_group_id(0);
    if (row >= outerSize) return;

    int lid = get_local_id(0);
    int localSize = get_local_size(0);

    __global const float* rowData = input + row * reduceSize;

    // Initialize local top-k (only first k threads)
    if (lid < k) {
        localTop[lid] = -INFINITY;
        localIdx[lid] = -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each thread scans its portion and maintains local top-k
    float myTop[8];
    int myIdx[8];
    for (int i = 0; i < k && i < 8; i++) {
        myTop[i] = -INFINITY;
        myIdx[i] = -1;
    }

    // Scan through the row
    for (int i = lid; i < reduceSize; i += localSize) {
        float val = rowData[i];

        // Insert into thread-local top-k
        for (int j = 0; j < k && j < 8; j++) {
            if (val > myTop[j]) {
                // Shift down
                for (int m = min(k, 8) - 1; m > j; m--) {
                    myTop[m] = myTop[m - 1];
                    myIdx[m] = myIdx[m - 1];
                }
                myTop[j] = val;
                myIdx[j] = i;
                break;
            }
        }
    }

    // Write thread-local results to local memory for reduction
    barrier(CLK_LOCAL_MEM_FENCE);

    // Thread 0 merges all results
    if (lid == 0) {
        float finalTop[8];
        int finalIdx[8];
        for (int i = 0; i < k && i < 8; i++) {
            finalTop[i] = -INFINITY;
            finalIdx[i] = -1;
        }

        // Simple merge - collect all candidates from all threads
        // (In practice, would need shared memory staging)
        for (int i = 0; i < reduceSize; i++) {
            float val = rowData[i];
            for (int j = 0; j < k && j < 8; j++) {
                if (val > finalTop[j]) {
                    for (int m = min(k, 8) - 1; m > j; m--) {
                        finalTop[m] = finalTop[m - 1];
                        finalIdx[m] = finalIdx[m - 1];
                    }
                    finalTop[j] = val;
                    finalIdx[j] = i;
                    break;
                }
            }
        }

        // Write output
        __global float* outValues = values + row * k;
        __global int* outIndices = indices + row * k;
        for (int i = 0; i < k && i < 8; i++) {
            outValues[i] = finalTop[i];
            outIndices[i] = finalIdx[i];
        }
    }
}

// ===========================================================================
// AFFINE GRID GENERATION KERNEL
// ===========================================================================

// Generate affine sampling grid
// theta: [batch, 2, 3] affine transformation matrices
// grid: [batch, outH, outW, 2] output sampling coordinates
__kernel void affine_grid(
    __global const float* theta,
    __global float* grid,
    const int batch,
    const int outHeight,
    const int outWidth)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int b = get_global_id(2);

    if (x >= outWidth || y >= outHeight || b >= batch) return;

    // Normalized coordinates in [-1, 1]
    float nx = outWidth > 1 ? 2.0f * x / (outWidth - 1) - 1.0f : 0.0f;
    float ny = outHeight > 1 ? 2.0f * y / (outHeight - 1) - 1.0f : 0.0f;

    // Affine transformation: [x', y'] = theta * [x, y, 1]^T
    __global const float* t = theta + b * 6;
    float outX = t[0] * nx + t[1] * ny + t[2];
    float outY = t[3] * nx + t[4] * ny + t[5];

    // Output grid location
    int gridIdx = ((b * outHeight + y) * outWidth + x) * 2;
    grid[gridIdx] = outX;
    grid[gridIdx + 1] = outY;
}

// ===========================================================================
// GRID SAMPLE KERNEL (BILINEAR INTERPOLATION)
// ===========================================================================

// Sample from input using grid with bilinear interpolation
// input: [batch, channels, inH, inW] NCHW format
// grid: [batch, outH, outW, 2] sampling coordinates in [-1, 1]
// output: [batch, channels, outH, outW]
__kernel void grid_sample(
    __global const float* input,
    __global const float* grid,
    __global float* output,
    const int batch,
    const int channels,
    const int inHeight,
    const int inWidth,
    const int outHeight,
    const int outWidth,
    const int paddingMode,
    const int alignCorners)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int bc = get_global_id(2);
    int b = bc / channels;
    int c = bc % channels;

    if (x >= outWidth || y >= outHeight || b >= batch) return;

    // Get sampling location from grid
    int gridIdx = ((b * outHeight + y) * outWidth + x) * 2;
    float gx = grid[gridIdx];
    float gy = grid[gridIdx + 1];

    // Convert from [-1, 1] to pixel coordinates
    float ix, iy;
    if (alignCorners) {
        ix = (gx + 1.0f) * 0.5f * (inWidth - 1);
        iy = (gy + 1.0f) * 0.5f * (inHeight - 1);
    } else {
        ix = ((gx + 1.0f) * inWidth - 1.0f) * 0.5f;
        iy = ((gy + 1.0f) * inHeight - 1.0f) * 0.5f;
    }

    // Get the four nearest pixel coordinates
    int ix0 = (int)floor(ix);
    int iy0 = (int)floor(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    // Interpolation weights
    float wx1 = ix - ix0;
    float wy1 = iy - iy0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    // Get pixel values with padding
    float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;

    if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight)
        v00 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix0];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v00 = input[((b * channels + c) * inHeight + clamp(iy0, 0, inHeight-1)) * inWidth + clamp(ix0, 0, inWidth-1)];

    if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight)
        v01 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix1];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v01 = input[((b * channels + c) * inHeight + clamp(iy0, 0, inHeight-1)) * inWidth + clamp(ix1, 0, inWidth-1)];

    if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight)
        v10 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix0];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v10 = input[((b * channels + c) * inHeight + clamp(iy1, 0, inHeight-1)) * inWidth + clamp(ix0, 0, inWidth-1)];

    if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight)
        v11 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix1];
    else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
        v11 = input[((b * channels + c) * inHeight + clamp(iy1, 0, inHeight-1)) * inWidth + clamp(ix1, 0, inWidth-1)];

    // Bilinear interpolation
    float result = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);

    // Write output
    int outIdx = ((b * channels + c) * outHeight + y) * outWidth + x;
    output[outIdx] = result;
}

// ===========================================================================
// ATOMIC FLOAT ADD HELPER (for OpenCL without native float atomics)
// ===========================================================================

// Emulate atomic float add using CAS loop
// This is necessary because OpenCL 1.x doesn't have atomic_add for floats
inline void atomicAddFloat(__global float* addr, float val) {
    union { float f; unsigned int i; } old_val, new_val;
    do {
        old_val.f = *addr;
        new_val.f = old_val.f + val;
    } while (atomic_cmpxchg((__global unsigned int*)addr, old_val.i, new_val.i) != old_val.i);
}

// ===========================================================================
// GRID SAMPLE BACKWARD KERNEL
// ===========================================================================

__kernel void grid_sample_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* grid,
    __global float* gradInput,
    __global float* gradGrid,
    const int batch,
    const int channels,
    const int inHeight,
    const int inWidth,
    const int outHeight,
    const int outWidth,
    const int paddingMode,
    const int alignCorners)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int b = get_global_id(2);

    if (x >= outWidth || y >= outHeight || b >= batch) return;

    // Get sampling location from grid
    int gridIdx = ((b * outHeight + y) * outWidth + x) * 2;
    float gx = grid[gridIdx];
    float gy = grid[gridIdx + 1];

    // Convert from [-1, 1] to pixel coordinates
    float ix, iy;
    float gradMultX, gradMultY;
    if (alignCorners) {
        ix = (gx + 1.0f) * 0.5f * (inWidth - 1);
        iy = (gy + 1.0f) * 0.5f * (inHeight - 1);
        gradMultX = 0.5f * (inWidth - 1);
        gradMultY = 0.5f * (inHeight - 1);
    } else {
        ix = ((gx + 1.0f) * inWidth - 1.0f) * 0.5f;
        iy = ((gy + 1.0f) * inHeight - 1.0f) * 0.5f;
        gradMultX = 0.5f * inWidth;
        gradMultY = 0.5f * inHeight;
    }

    int ix0 = (int)floor(ix);
    int iy0 = (int)floor(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    float wx1 = ix - ix0;
    float wy1 = iy - iy0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    float gradGridX = 0.0f;
    float gradGridY = 0.0f;

    // Accumulate gradients for each channel
    for (int c = 0; c < channels; c++) {
        int outIdx = ((b * channels + c) * outHeight + y) * outWidth + x;
        float go = gradOutput[outIdx];

        // Get input values
        float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;
        if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight)
            v00 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix0];
        if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight)
            v01 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix1];
        if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight)
            v10 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix0];
        if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight)
            v11 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix1];

        // Gradient with respect to grid
        gradGridX += go * (wy0 * (v01 - v00) + wy1 * (v11 - v10)) * gradMultX;
        gradGridY += go * (wx0 * (v10 - v00) + wx1 * (v11 - v01)) * gradMultY;

        // Gradient with respect to input (thread-safe atomic add for overlapping writes)
        // NON-DETERMINISTIC (issue #382); see grid_sample_backward_grad_grid_deterministic
        // and grid_sample_backward_grad_input_deterministic below.
        if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy0) * inWidth + ix0;
            atomicAddFloat(&gradInput[idx], go * wy0 * wx0);
        }
        if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy0) * inWidth + ix1;
            atomicAddFloat(&gradInput[idx], go * wy0 * wx1);
        }
        if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy1) * inWidth + ix0;
            atomicAddFloat(&gradInput[idx], go * wy1 * wx0);
        }
        if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight) {
            int idx = ((b * channels + c) * inHeight + iy1) * inWidth + ix1;
            atomicAddFloat(&gradInput[idx], go * wy1 * wx1);
        }
    }

    // Write grid gradient
    gradGrid[gridIdx] = gradGridX;
    gradGrid[gridIdx + 1] = gradGridY;
}

// grid_sample_backward — bit-deterministic split (issue #382). See CUDA equivalent
// in CudaSpatialTransformerKernels for the full algorithm rationale.
__kernel void grid_sample_backward_grad_grid_deterministic(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* grid,
    __global float* gradGrid,
    const int batch, const int channels,
    const int inHeight, const int inWidth,
    const int outHeight, const int outWidth,
    const int paddingMode, const int alignCorners)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int b = get_global_id(2);
    if (x >= outWidth || y >= outHeight || b >= batch) return;

    int gridIdx = ((b * outHeight + y) * outWidth + x) * 2;
    float gx = grid[gridIdx];
    float gy = grid[gridIdx + 1];

    float ix, iy, gradMultX, gradMultY;
    if (alignCorners) {
        ix = (gx + 1.0f) * 0.5f * (inWidth - 1);
        iy = (gy + 1.0f) * 0.5f * (inHeight - 1);
        gradMultX = 0.5f * (inWidth - 1);
        gradMultY = 0.5f * (inHeight - 1);
    } else {
        ix = ((gx + 1.0f) * inWidth - 1.0f) * 0.5f;
        iy = ((gy + 1.0f) * inHeight - 1.0f) * 0.5f;
        gradMultX = 0.5f * inWidth;
        gradMultY = 0.5f * inHeight;
    }
    int ix0 = (int)floor(ix);
    int iy0 = (int)floor(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;
    float wx1 = ix - ix0;
    float wy1 = iy - iy0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    float gradGridX = 0.0f;
    float gradGridY = 0.0f;

    for (int c = 0; c < channels; c++) {
        int outIdx = ((b * channels + c) * outHeight + y) * outWidth + x;
        float go = gradOutput[outIdx];

        // Border padding (paddingMode == 1): out-of-range corners read the
        // clamped boundary value; zero padding (paddingMode == 0): they read 0.
        float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;
        if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight)
            v00 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix0];
        else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
            v00 = input[((b * channels + c) * inHeight + clamp(iy0, 0, inHeight - 1)) * inWidth + clamp(ix0, 0, inWidth - 1)];

        if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight)
            v01 = input[((b * channels + c) * inHeight + iy0) * inWidth + ix1];
        else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
            v01 = input[((b * channels + c) * inHeight + clamp(iy0, 0, inHeight - 1)) * inWidth + clamp(ix1, 0, inWidth - 1)];

        if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight)
            v10 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix0];
        else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
            v10 = input[((b * channels + c) * inHeight + clamp(iy1, 0, inHeight - 1)) * inWidth + clamp(ix0, 0, inWidth - 1)];

        if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight)
            v11 = input[((b * channels + c) * inHeight + iy1) * inWidth + ix1];
        else if (paddingMode == 1 && inWidth > 0 && inHeight > 0)
            v11 = input[((b * channels + c) * inHeight + clamp(iy1, 0, inHeight - 1)) * inWidth + clamp(ix1, 0, inWidth - 1)];

        gradGridX += go * (wy0 * (v01 - v00) + wy1 * (v11 - v10)) * gradMultX;
        gradGridY += go * (wx0 * (v10 - v00) + wx1 * (v11 - v01)) * gradMultY;
    }
    gradGrid[gridIdx] = gradGridX;
    gradGrid[gridIdx + 1] = gradGridY;
}

__kernel void grid_sample_backward_grad_input_deterministic(
    __global const float* gradOutput,
    __global const float* grid,
    __global float* gradInput,
    const int batch, const int channels,
    const int inHeight, const int inWidth,
    const int outHeight, const int outWidth,
    const int paddingMode, const int alignCorners)
{
    int w_in = get_global_id(0);
    int h_in = get_global_id(1);
    int bc = get_global_id(2);
    if (w_in >= inWidth || h_in >= inHeight || bc >= batch * channels) return;

    int b = bc / channels;
    int c = bc % channels;

    float sum = 0.0f;
    for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
            int gridIdx = ((b * outHeight + y) * outWidth + x) * 2;
            float gx = grid[gridIdx];
            float gy = grid[gridIdx + 1];
            float ix, iy;
            if (alignCorners) {
                ix = (gx + 1.0f) * 0.5f * (inWidth - 1);
                iy = (gy + 1.0f) * 0.5f * (inHeight - 1);
            } else {
                ix = ((gx + 1.0f) * inWidth - 1.0f) * 0.5f;
                iy = ((gy + 1.0f) * inHeight - 1.0f) * 0.5f;
            }
            int ix0 = (int)floor(ix);
            int iy0 = (int)floor(iy);
            int ix1 = ix0 + 1;
            int iy1 = iy0 + 1;
            float wx1 = ix - ix0;
            float wy1 = iy - iy0;
            float wx0 = 1.0f - wx1;
            float wy0 = 1.0f - wy1;

            // Border padding clamps out-of-range corner coords to the boundary
            // so border samples in the forward pass contribute back here.
            if (paddingMode == 1) {
                ix0 = clamp(ix0, 0, inWidth - 1);
                ix1 = clamp(ix1, 0, inWidth - 1);
                iy0 = clamp(iy0, 0, inHeight - 1);
                iy1 = clamp(iy1, 0, inHeight - 1);
            }

            int outIdx = ((b * channels + c) * outHeight + y) * outWidth + x;
            float go = gradOutput[outIdx];

            // Explicit in-bounds check on every corner: under zero padding
            // (paddingMode == 0) out-of-range corners must not contribute;
            // under border padding the clamps above guarantee the test passes.
            if (ix0 >= 0 && ix0 < inWidth && iy0 >= 0 && iy0 < inHeight && w_in == ix0 && h_in == iy0) sum += go * wy0 * wx0;
            if (ix1 >= 0 && ix1 < inWidth && iy0 >= 0 && iy0 < inHeight && w_in == ix1 && h_in == iy0) sum += go * wy0 * wx1;
            if (ix0 >= 0 && ix0 < inWidth && iy1 >= 0 && iy1 < inHeight && w_in == ix0 && h_in == iy1) sum += go * wy1 * wx0;
            if (ix1 >= 0 && ix1 < inWidth && iy1 >= 0 && iy1 < inHeight && w_in == ix1 && h_in == iy1) sum += go * wy1 * wx1;
        }
    }
    gradInput[((b * channels + c) * inHeight + h_in) * inWidth + w_in] += sum;
}

// #775: 3D Gaussian-splat covariance. rotations [N,4] quaternion (w,x,y,z), scales [N,3] ->
// covariances [N,6] upper triangular (c00,c01,c02,c11,c12,c22) of Sigma = R * S^2 * R^T. Gather over
// gaussians; mirrors GaussianSplattingOperations.ComputeGaussianCovariance exactly.
__kernel void gaussian_covariance(
    __global const float* rotations,
    __global const float* scales,
    __global float* covariances,
    const int numGaussians)
{
    int i = get_global_id(0);
    if (i >= numGaussians) return;

    float qw = rotations[i * 4], qx = rotations[i * 4 + 1], qy = rotations[i * 4 + 2], qz = rotations[i * 4 + 3];
    float qNorm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qNorm > 0.0f) { float inv = 1.0f / qNorm; qw *= inv; qx *= inv; qy *= inv; qz *= inv; }

    float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    float r01 = 2.0f * (qx * qy - qw * qz);
    float r02 = 2.0f * (qx * qz + qw * qy);
    float r10 = 2.0f * (qx * qy + qw * qz);
    float r11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    float r12 = 2.0f * (qy * qz - qw * qx);
    float r20 = 2.0f * (qx * qz - qw * qy);
    float r21 = 2.0f * (qy * qz + qw * qx);
    float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);

    float sx = fmax(1e-6f, fabs(scales[i * 3])); float sx2 = sx * sx;
    float sy = fmax(1e-6f, fabs(scales[i * 3 + 1])); float sy2 = sy * sy;
    float sz = fmax(1e-6f, fabs(scales[i * 3 + 2])); float sz2 = sz * sz;

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
}
";
        }
    }
}
