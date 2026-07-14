// Copyright (c) AiDotNet. All rights reserved.
// Neural network utility kernels for gradient computation, loss functions, and optimizers.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for neural network operations including activation gradients,
    /// loss functions, optimizers, and utility operations.
    /// </summary>
    internal static class NeuralNetKernels
    {
        /// <summary>
        /// Gets all neural network kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// ATOMIC OPERATIONS FOR OPENCL 1.x COMPATIBILITY
// ===========================================================================

// Atomic float add using compare-and-swap loop (OpenCL 1.x compatible)
// Uses as_int/as_float for robust type reinterpretation instead of union
// volatile qualifier ensures memory visibility across work-items
inline void atomic_add_float(volatile __global float *ptr, float val) {
    int oldVal, newVal;
    do {
        oldVal = *(volatile __global int *)ptr;
        newVal = as_int(as_float(oldVal) + val);
    } while (atomic_cmpxchg((volatile __global int *)ptr, oldVal, newVal) != oldVal);
}

// ===========================================================================
// ACTIVATION GRADIENT KERNELS
// ===========================================================================

// ReLU backward: grad * (input > 0)
__kernel void relu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    gradInput[idx] = input[idx] > 0.0f ? gradOutput[idx] : 0.0f;
}

// Sigmoid backward: grad * output * (1 - output)
__kernel void sigmoid_backward(
    __global const float* gradOutput,
    __global const float* output,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float o = output[idx];
    gradInput[idx] = gradOutput[idx] * o * (1.0f - o);
}

// Tanh backward: grad * (1 - output^2)
__kernel void tanh_backward(
    __global const float* gradOutput,
    __global const float* output,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float o = output[idx];
    gradInput[idx] = gradOutput[idx] * (1.0f - o * o);
}

// GELU backward
__kernel void gelu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    const float SQRT_2_OVER_PI = 0.7978845608f;
    const float COEFF = 0.044715f;

    float x = input[idx];
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    float tanhInner = tanh(inner);
    float sech2 = 1.0f - tanhInner * tanhInner;
    float dInner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);

    gradInput[idx] = gradOutput[idx] * (0.5f * (1.0f + tanhInner) + 0.5f * x * sech2 * dInner);
}

// Softmax backward — workgroup-parallel with local memory reduction
__kernel void softmax_backward(
    __global const float* gradOutput,
    __global const float* output,
    __global float* gradInput,
    const int batchSize,
    const int features,
    __local float* localBuf)
{
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (b >= batchSize) return;

    __global const float* rowGrad = gradOutput + b * features;
    __global const float* rowOut = output + b * features;
    __global float* rowGradIn = gradInput + b * features;

    // Phase 1: Parallel dot product reduction
    float threadDot = 0.0f;
    for (int f = lid; f < features; f += localSize) {
        threadDot += rowGrad[f] * rowOut[f];
    }
    localBuf[lid] = threadDot;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) localBuf[lid] += localBuf[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float dotProd = localBuf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute gradInput
    for (int f = lid; f < features; f += localSize) {
        rowGradIn[f] = rowOut[f] * (rowGrad[f] - dotProd);
    }
}

// Leaky ReLU forward
__kernel void leaky_relu_forward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x > 0.0f ? x : alpha * x;
}

// Leaky ReLU backward
__kernel void leaky_relu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    gradInput[idx] = input[idx] > 0.0f ? gradOutput[idx] : alpha * gradOutput[idx];
}

// ELU forward: x > 0 ? x : alpha * (exp(x) - 1)
__kernel void elu_forward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x > 0.0f ? x : alpha * (exp(x) - 1.0f);
}

// ELU backward
__kernel void elu_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* output,
    __global float* gradInput,
    const float alpha,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    gradInput[idx] = x > 0.0f ? gradOutput[idx] : gradOutput[idx] * (output[idx] + alpha);
}

// Swish forward: x * sigmoid(x)
__kernel void swish_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float sig = 1.0f / (1.0f + exp(-x));
    output[idx] = x * sig;
}

// Swish backward
__kernel void swish_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float sig = 1.0f / (1.0f + exp(-x));
    float swishVal = x * sig;
    gradInput[idx] = gradOutput[idx] * (swishVal + sig * (1.0f - swishVal));
}

// SiLU (Sigmoid Linear Unit) - same as Swish with beta=1
__kernel void silu_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x / (1.0f + exp(-x));
}

// Mish: x * tanh(softplus(x))
__kernel void mish_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    float sp = log(1.0f + exp(x));
    output[idx] = x * tanh(sp);
}

// Softplus: log(1 + exp(x))
__kernel void softplus_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    // Use stable version for large x
    output[idx] = x > 20.0f ? x : log(1.0f + exp(x));
}

// Hardswish: x * min(max(x + 3, 0), 6) / 6
__kernel void hardswish_forward(
    __global const float* input,
    __global float* output,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = x * fmin(fmax(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

// ===========================================================================
// LOSS FUNCTION KERNELS
// ===========================================================================

// Cross-entropy loss (with softmax input)
// Returns per-sample loss, needs reduction afterwards
__kernel void cross_entropy_loss(
    __global const float* predictions,
    __global const float* targets,
    __global float* losses,
    const int batchSize,
    const int numClasses)
{
    const int b = get_global_id(0);
    if (b >= batchSize) return;

    // Find max for numerical stability
    float maxVal = -INFINITY;
    for (int c = 0; c < numClasses; c++) {
        maxVal = fmax(maxVal, predictions[b * numClasses + c]);
    }

    // Compute log-sum-exp
    float sumExp = 0.0f;
    for (int c = 0; c < numClasses; c++) {
        sumExp += exp(predictions[b * numClasses + c] - maxVal);
    }
    float logSumExp = maxVal + log(sumExp);

    // Compute cross-entropy: -sum(target * log(softmax(pred)))
    float loss = 0.0f;
    for (int c = 0; c < numClasses; c++) {
        int idx = b * numClasses + c;
        float logProb = predictions[idx] - logSumExp;
        loss -= targets[idx] * logProb;
    }
    losses[b] = loss;
}

// Cross-entropy backward (combined softmax + cross-entropy gradient)
__kernel void cross_entropy_backward(
    __global const float* predictions,
    __global const float* targets,
    __global float* gradInput,
    const int batchSize,
    const int numClasses)
{
    const int idx = get_global_id(0);
    const int c = idx % numClasses;
    const int b = idx / numClasses;

    if (b >= batchSize) return;

    // Find max for numerical stability
    float maxVal = -INFINITY;
    for (int i = 0; i < numClasses; i++) {
        maxVal = fmax(maxVal, predictions[b * numClasses + i]);
    }

    // Compute softmax
    float sumExp = 0.0f;
    for (int i = 0; i < numClasses; i++) {
        sumExp += exp(predictions[b * numClasses + i] - maxVal);
    }
    float softmax = exp(predictions[idx] - maxVal) / sumExp;

    // Gradient is (softmax - target) / batchSize
    gradInput[idx] = (softmax - targets[idx]) / (float)batchSize;
}

// Binary cross-entropy loss
__kernel void bce_loss(
    __global const float* predictions,
    __global const float* targets,
    __global float* losses,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float p = predictions[idx];
    float t = targets[idx];
    // Clamp to avoid log(0)
    p = fmax(fmin(p, 1.0f - 1e-7f), 1e-7f);
    losses[idx] = -(t * log(p) + (1.0f - t) * log(1.0f - p));
}

// Binary cross-entropy backward
__kernel void bce_backward(
    __global const float* predictions,
    __global const float* targets,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float p = predictions[idx];
    float t = targets[idx];
    // Clamp to avoid division by zero
    p = fmax(fmin(p, 1.0f - 1e-7f), 1e-7f);
    gradInput[idx] = (p - t) / (p * (1.0f - p)) / (float)size;
}

// MSE loss
__kernel void mse_loss(
    __global const float* predictions,
    __global const float* targets,
    __global float* losses,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float diff = predictions[idx] - targets[idx];
    losses[idx] = diff * diff;
}

// MSE backward
__kernel void mse_backward(
    __global const float* predictions,
    __global const float* targets,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    gradInput[idx] = 2.0f * (predictions[idx] - targets[idx]) / (float)size;
}

// Smooth L1 (Huber) loss
__kernel void smooth_l1_loss(
    __global const float* predictions,
    __global const float* targets,
    __global float* losses,
    const int size,
    const float beta)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float diff = fabs(predictions[idx] - targets[idx]);
    losses[idx] = diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta;
}

// Smooth L1 backward
__kernel void smooth_l1_backward(
    __global const float* predictions,
    __global const float* targets,
    __global float* gradInput,
    const int size,
    const float beta)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    float diff = predictions[idx] - targets[idx];
    float absDiff = fabs(diff);
    float grad = absDiff < beta ? diff / beta : (diff > 0.0f ? 1.0f : -1.0f);
    gradInput[idx] = grad / (float)size;
}


// ===========================================================================
// UTILITY KERNELS
// ===========================================================================

// Clamp values between min and max
__kernel void clamp_values(
    __global const float* input,
    __global float* output,
    const float minVal,
    const float maxVal,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = fmin(fmax(input[idx], minVal), maxVal);
}

// Clip by value (symmetric around 0)
__kernel void clip_by_value(
    __global const float* input,
    __global float* output,
    const float clipValue,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    output[idx] = fmin(fmax(input[idx], -clipValue), clipValue);
}

// Transpose 2D matrix
__kernel void transpose2d(
    __global const float* input,
    __global float* output,
    const int rows,
    const int cols)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= rows || col >= cols) return;

    output[col * rows + row] = input[row * cols + col];
}

// Batched transpose
__kernel void batched_transpose(
    __global const float* input,
    __global float* output,
    const int batch,
    const int rows,
    const int cols)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int b = get_global_id(2);

    if (row >= rows || col >= cols || b >= batch) return;

    int inIdx = (b * rows + row) * cols + col;
    int outIdx = (b * cols + col) * rows + row;
    output[outIdx] = input[inIdx];
}

// Fill buffer with constant value
__kernel void fill_buffer(
    __global float* buffer,
    const float value,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    buffer[idx] = value;
}

// Copy buffer
__kernel void copy_buffer(
    __global const float* src,
    __global float* dst,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    dst[idx] = src[idx];
}

// Comparison: greater than
__kernel void greater_than(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] > B[idx] ? 1.0f : 0.0f;
}

// Comparison: less than
__kernel void less_than(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] < B[idx] ? 1.0f : 0.0f;
}

// Comparison: equal
__kernel void equal_values(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = A[idx] == B[idx] ? 1.0f : 0.0f;
}

// Where (conditional select)
__kernel void where_select(
    __global const float* condition,
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    C[idx] = condition[idx] > 0.0f ? A[idx] : B[idx];
}

// Mean along axis
__kernel void mean_axis(
    __global const float* input,
    __global float* output,
    const int outerSize,
    const int reduceSize)
{
    const int outer = get_global_id(0);
    if (outer >= outerSize) return;

    float sum = 0.0f;
    for (int i = 0; i < reduceSize; i++) {
        sum += input[outer * reduceSize + i];
    }
    output[outer] = sum / (float)reduceSize;
}

// Variance along axis
__kernel void var_axis(
    __global const float* input,
    __global const float* mean,
    __global float* variance,
    const int outerSize,
    const int reduceSize)
{
    const int outer = get_global_id(0);
    if (outer >= outerSize) return;

    float m = mean[outer];
    float sum = 0.0f;
    for (int i = 0; i < reduceSize; i++) {
        float diff = input[outer * reduceSize + i] - m;
        sum += diff * diff;
    }
    variance[outer] = sum / (float)reduceSize;
}

// ArgMax along axis
__kernel void argmax_axis(
    __global const float* input,
    __global float* indices,
    const int outerSize,
    const int reduceSize)
{
    const int outer = get_global_id(0);
    if (outer >= outerSize) return;

    float maxVal = -INFINITY;
    int maxIdx = 0;
    for (int i = 0; i < reduceSize; i++) {
        float val = input[outer * reduceSize + i];
        if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
        }
    }
    indices[outer] = (float)maxIdx;
}

// ArgMin along axis
__kernel void argmin_axis(
    __global const float* input,
    __global float* indices,
    const int outerSize,
    const int reduceSize)
{
    const int outer = get_global_id(0);
    if (outer >= outerSize) return;

    float minVal = INFINITY;
    int minIdx = 0;
    for (int i = 0; i < reduceSize; i++) {
        float val = input[outer * reduceSize + i];
        if (val < minVal) {
            minVal = val;
            minIdx = i;
        }
    }
    indices[outer] = (float)minIdx;
}

// Dropout forward
__kernel void dropout_forward(
    __global const float* input,
    __global float* output,
    __global float* mask,
    const int size,
    const float dropoutRate,
    const ulong seed,
    const int training)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    if (!training) {
        output[idx] = input[idx];
        mask[idx] = 1.0f;
        return;
    }

    // Simple LCG random number generator
    ulong state = seed + (ulong)idx * 6364136223846793005UL;
    state = state * 6364136223846793005UL + 1442695040888963407UL;
    float rand = (float)(state >> 33) / (float)(1UL << 31);

    float scale = 1.0f / (1.0f - dropoutRate);
    if (rand < dropoutRate) {
        output[idx] = 0.0f;
        mask[idx] = 0.0f;
    } else {
        output[idx] = input[idx] * scale;
        mask[idx] = scale;
    }
}

// Dropout backward
__kernel void dropout_backward(
    __global const float* gradOutput,
    __global const float* mask,
    __global float* gradInput,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    gradInput[idx] = gradOutput[idx] * mask[idx];
}

// Embedding lookup
__kernel void embedding_lookup(
    __global const int* indices,
    __global const float* embeddingTable,
    __global float* output,
    const int numIndices,
    const int embeddingDim)
{
    const int d = get_global_id(0);
    const int idx = get_global_id(1);
    if (idx >= numIndices || d >= embeddingDim) return;

    // OpenClBackend.AllocateIntBuffer(int[]) stores raw int bits in a float
    // buffer. Reinterpret those bits here; a numeric cast truncates the
    // denormal float representation of small positive indices to zero.
    int index = as_int(indices[idx]);

    output[idx * embeddingDim + d] = embeddingTable[index * embeddingDim + d];
}

// Embedding backward (scatter add gradients)
// Uses atomic operations for thread safety when indices collide.
// Order of atomic_add_float is scheduler-dependent and FP addition is not associative,
// so this kernel is NOT bit-deterministic across runs. See issue #382.
// When AiDotNetEngine.DeterministicMode is set, dispatch routes to
// embedding_backward_deterministic instead.
__kernel void embedding_backward(
    __global const float* gradOutput,
    __global const float* indices,
    __global float* gradEmbedding,
    const int numIndices,
    const int embeddingDim,
    const int vocabSize)
{
    const int d = get_global_id(0);
    const int idx = get_global_id(1);
    if (idx >= numIndices || d >= embeddingDim) return;

    // OpenCL AllocateIntBuffer bit-packs int32 indices into the float buffer via
    // BitConverter (see OpenClBackend.AllocateIntBuffer(int[])). Use as_int() to
    // bit-reinterpret back to int; a numeric `(int)` cast would truncate the
    // float interpretation of the bit pattern (e.g. int 1 stored as denormal
    // ~1.4e-45 would truncate to 0).
    int index = as_int(indices[idx]);
    if (index < 0 || index >= vocabSize) return;

    // Use atomic add for thread safety when multiple indices point to same embedding.
    atomic_add_float(&gradEmbedding[index * embeddingDim + d], gradOutput[idx * embeddingDim + d]);
}

// Embedding backward — bit-deterministic variant (issue #382).
// Transposes the work: one thread per (v, d) output cell scans all numIndices entries
// and accumulates contributions where indices[i] == v. No atomics; accumulation order
// for a given output cell is fixed (ascending i), so the result is bit-identical
// across runs at the same input.
//
// Cost: O(vocabSize * embeddingDim * numIndices) vs the atomic kernel's
// O(numIndices * embeddingDim). Acceptable for the typical reproducibility-mode
// use case (small fixtures, debugging, paper experiments) — that is the documented
// trade-off of the DeterministicMode contract.
__kernel void embedding_backward_deterministic(
    __global const float* gradOutput,
    __global const float* indices,
    __global float* gradEmbedding,
    const int numIndices,
    const int embeddingDim,
    const int vocabSize)
{
    const int v = get_global_id(0);
    const int d = get_global_id(1);
    if (v >= vocabSize || d >= embeddingDim) return;

    float sum = 0.0f;
    for (int i = 0; i < numIndices; i++) {
        // as_int() bit-reinterprets the float storage back to int (see comment in
        // the atomic variant above).
        if (as_int(indices[i]) == v) {
            sum += gradOutput[i * embeddingDim + d];
        }
    }
    // Plain assignment: this launch owns every (v, d) cell, so overwriting keeps
    // the kernel self-contained on reused buffers (no requirement that the caller
    // zero gradEmbedding first). Matches the CUDA/HIP deterministic variants.
    gradEmbedding[v * embeddingDim + d] = sum;
}

// Fused multiply-add: D = A * B + C
__kernel void fma_kernel(
    __global const float* A,
    __global const float* B,
    __global const float* C,
    __global float* D,
    const int size)
{
    const int idx = get_global_id(0);
    if (idx >= size) return;

    D[idx] = fma(A[idx], B[idx], C[idx]);
}

// Gather operation
__kernel void gather_kernel(
    __global const float* source,
    __global const int* indices,
    __global float* output,
    const int numIndices,
    const int featureSize)
{
    const int idx = get_global_id(0);
    if (idx >= numIndices) return;

    int index = indices[idx];
    for (int f = 0; f < featureSize; f++) {
        output[idx * featureSize + f] = source[index * featureSize + f];
    }
}

// ScatterAdd operation
// Uses atomic operations for thread safety when indices collide.
// Non-deterministic across runs (issue #382). DeterministicMode dispatches to
// scatter_add_kernel_deterministic below.
__kernel void scatter_add_kernel(
    __global const float* source,
    __global const int* indices,
    __global float* destination,
    const int sourceSize,
    const int featureSize)
{
    const int idx = get_global_id(0);
    if (idx >= sourceSize) return;

    int destIdx = indices[idx];
    for (int f = 0; f < featureSize; f++) {
        // Use atomic add for thread safety when multiple sources scatter to same destination
        atomic_add_float(&destination[destIdx * featureSize + f], source[idx * featureSize + f]);
    }
}

// ScatterAdd — bit-deterministic variant (issue #382).
// One work-item per (dstRow, f) cell; scans source rows in fixed ascending order.
__kernel void scatter_add_kernel_deterministic(
    __global const float* source,
    __global const int* indices,
    __global float* destination,
    const int sourceSize,
    const int destSize,
    const int featureSize)
{
    const int dstRow = get_global_id(0);
    const int f = get_global_id(1);
    if (dstRow >= destSize || f >= featureSize) return;

    float sum = 0.0f;
    for (int i = 0; i < sourceSize; i++) {
        if (indices[i] == dstRow) {
            sum += source[i * featureSize + f];
        }
    }
    destination[dstRow * featureSize + f] += sum;
}

// ===========================================================================
// LSTM KERNELS
// ===========================================================================

__kernel void lstm_cell_forward(
    __global const float* gates,
    __global const float* cellPrev,
    __global float* cellNext,
    __global float* hiddenNext,
    __global float* gateActivations,
    const int batchSize,
    const int hiddenSize)
{
    const int idx = get_global_id(0);
    const int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    const int b = idx / hiddenSize;
    const int h = idx % hiddenSize;
    const int gateOffset = b * 4 * hiddenSize;

    float gi = gates[gateOffset + h];
    float gf = gates[gateOffset + hiddenSize + h];
    float gg = gates[gateOffset + 2 * hiddenSize + h];
    float go = gates[gateOffset + 3 * hiddenSize + h];

    float i = 1.0f / (1.0f + exp(-gi));
    float f = 1.0f / (1.0f + exp(-gf));
    float g = tanh(gg);
    float o = 1.0f / (1.0f + exp(-go));

    float cPrev = cellPrev[idx];
    float c = f * cPrev + i * g;
    float tanhC = tanh(c);
    float hNew = o * tanhC;

    cellNext[idx] = c;
    hiddenNext[idx] = hNew;

    gateActivations[gateOffset + h] = i;
    gateActivations[gateOffset + hiddenSize + h] = f;
    gateActivations[gateOffset + 2 * hiddenSize + h] = g;
    gateActivations[gateOffset + 3 * hiddenSize + h] = o;
}

__kernel void lstm_cell_backward(
    __global const float* gradHidden,
    __global const float* gradCellNext,
    __global const float* gateActivations,
    __global const float* cellPrev,
    __global const float* cellNext,
    __global float* gradGates,
    __global float* gradCellPrev,
    const int batchSize,
    const int hiddenSize)
{
    const int idx = get_global_id(0);
    const int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    const int b = idx / hiddenSize;
    const int h = idx % hiddenSize;
    const int gateOffset = b * 4 * hiddenSize;

    float i = gateActivations[gateOffset + h];
    float f = gateActivations[gateOffset + hiddenSize + h];
    float g = gateActivations[gateOffset + 2 * hiddenSize + h];
    float o = gateActivations[gateOffset + 3 * hiddenSize + h];

    float cPrev = cellPrev[idx];
    float c = cellNext[idx];
    float tanhC = tanh(c);

    float dH = gradHidden[idx];
    float dO = dH * tanhC;
    float dTanhC = dH * o;
    float dC = dTanhC * (1.0f - tanhC * tanhC);
    dC += gradCellNext[idx];

    float dF = dC * cPrev;
    float dI = dC * g;
    float dG = dC * i;
    float dCPrev = dC * f;

    float gradGi = dI * i * (1.0f - i);
    float gradGf = dF * f * (1.0f - f);
    float gradGg = dG * (1.0f - g * g);
    float gradGo = dO * o * (1.0f - o);

    gradGates[gateOffset + h] = gradGi;
    gradGates[gateOffset + hiddenSize + h] = gradGf;
    gradGates[gateOffset + 2 * hiddenSize + h] = gradGg;
    gradGates[gateOffset + 3 * hiddenSize + h] = gradGo;
    gradCellPrev[idx] = dCPrev;
}

__kernel void lstm_gates_precompute(
    __global const float* input,
    __global const float* hiddenPrev,
    __global const float* weightsIH,
    __global const float* weightsHH,
    __global const float* bias,
    __global float* gates,
    const int batchSize,
    const int inputSize,
    const int hiddenSize)
{
    const int idx = get_global_id(0);
    const int totalSize = batchSize * 4 * hiddenSize;
    if (idx >= totalSize) return;

    const int b = idx / (4 * hiddenSize);
    const int g = idx % (4 * hiddenSize);

    float sum = bias[g];

    for (int i = 0; i < inputSize; i++) {
        sum += weightsIH[g * inputSize + i] * input[b * inputSize + i];
    }

    for (int h = 0; h < hiddenSize; h++) {
        sum += weightsHH[g * hiddenSize + h] * hiddenPrev[b * hiddenSize + h];
    }

    gates[idx] = sum;
}

// ===========================================================================
// GRU KERNELS
// ===========================================================================

__kernel void gru_cell_forward(
    __global const float* gatesRZ,
    __global const float* gateN_input,
    __global const float* gateN_hidden,
    __global const float* hiddenPrev,
    __global float* hiddenNext,
    __global float* gateActivations,
    const int batchSize,
    const int hiddenSize)
{
    const int idx = get_global_id(0);
    const int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    const int b = idx / hiddenSize;
    const int h = idx % hiddenSize;

    float gr = gatesRZ[b * 2 * hiddenSize + h];
    float gz = gatesRZ[b * 2 * hiddenSize + hiddenSize + h];

    float r = 1.0f / (1.0f + exp(-gr));
    float z = 1.0f / (1.0f + exp(-gz));

    float nInput = gateN_input[idx];
    float nHidden = gateN_hidden[idx];
    float nPre = nInput + r * nHidden;
    float n = tanh(nPre);

    float hPrev = hiddenPrev[idx];
    float hNew = (1.0f - z) * n + z * hPrev;

    hiddenNext[idx] = hNew;

    const int actOffset = b * 3 * hiddenSize;
    gateActivations[actOffset + h] = r;
    gateActivations[actOffset + hiddenSize + h] = z;
    gateActivations[actOffset + 2 * hiddenSize + h] = n;
}

__kernel void gru_cell_backward(
    __global const float* gradHidden,
    __global const float* gateActivations,
    __global const float* hiddenPrev,
    __global const float* gateN_hidden,
    __global float* gradGatesRZ,
    __global float* gradGateN,
    __global float* gradHiddenPrev,
    const int batchSize,
    const int hiddenSize)
{
    const int idx = get_global_id(0);
    const int totalSize = batchSize * hiddenSize;
    if (idx >= totalSize) return;

    const int b = idx / hiddenSize;
    const int h = idx % hiddenSize;

    const int actOffset = b * 3 * hiddenSize;
    float r = gateActivations[actOffset + h];
    float z = gateActivations[actOffset + hiddenSize + h];
    float n = gateActivations[actOffset + 2 * hiddenSize + h];

    float hPrev = hiddenPrev[idx];
    float dH = gradHidden[idx];

    float dZ = dH * (hPrev - n);
    float dN = dH * (1.0f - z);
    float dHPrev = dH * z;

    float dNPre = dN * (1.0f - n * n);

    float nHidden = gateN_hidden[idx];
    float dR = dNPre * nHidden;
    dHPrev += dNPre * r;

    float gradGr = dR * r * (1.0f - r);
    float gradGz = dZ * z * (1.0f - z);

    gradGatesRZ[b * 2 * hiddenSize + h] = gradGr;
    gradGatesRZ[b * 2 * hiddenSize + hiddenSize + h] = gradGz;
    gradGateN[idx] = dNPre;
    gradHiddenPrev[idx] = dHPrev;
}

// ===========================================================================
// ADDITIONAL SCATTER OPERATIONS FOR GNNs
// ===========================================================================

__kernel void scatter_add_batched(
    __global const float* src,
    __global const int* indices,
    __global float* dst,
    const int numElements,
    const int featureSize)
{
    const int idx = get_global_id(0);
    if (idx >= numElements * featureSize) return;

    const int elemIdx = idx / featureSize;
    const int featIdx = idx % featureSize;
    const int dstIdx = indices[elemIdx];

    // Use atomic float add for correctness when multiple elements map to same destination
    atomic_add_float(&dst[dstIdx * featureSize + featIdx], src[idx]);
}

// scatter_add_batched — bit-deterministic variant (issue #382).
// CodeRabbit (#390): self-contained-write pattern — one thread owns its
// (dstIdx, featIdx) cell; final `=` not `+=`. Matches CUDA + HIP twins.
__kernel void scatter_add_batched_deterministic(
    __global const float* src,
    __global const int* indices,
    __global float* dst,
    const int numElements,
    const int numNodes,
    const int featureSize)
{
    const int dstIdx = get_global_id(0);
    const int featIdx = get_global_id(1);
    if (dstIdx >= numNodes || featIdx >= featureSize) return;

    float sum = 0.0f;
    for (int i = 0; i < numElements; i++) {
        if (indices[i] == dstIdx) {
            sum += src[i * featureSize + featIdx];
        }
    }
    dst[dstIdx * featureSize + featIdx] = sum;
}

__kernel void scatter_mean_accumulate(
    __global const float* src,
    __global const int* indices,
    __global float* dst,
    __global int* counts,
    const int numElements,
    const int featureSize)
{
    const int idx = get_global_id(0);
    if (idx >= numElements * featureSize) return;

    const int elemIdx = idx / featureSize;
    const int featIdx = idx % featureSize;
    const int dstIdx = indices[elemIdx];

    // Use atomic float add for correctness
    atomic_add_float(&dst[dstIdx * featureSize + featIdx], src[idx]);

    if (featIdx == 0) {
        // Use atomic increment for counts
        atomic_inc(&counts[dstIdx]);
    }
}

// scatter_mean_accumulate — bit-deterministic variant (issue #382).
__kernel void scatter_mean_accumulate_deterministic(
    __global const float* src,
    __global const int* indices,
    __global float* dst,
    __global int* counts,
    const int numElements,
    const int numNodes,
    const int featureSize)
{
    const int dstIdx = get_global_id(0);
    const int featIdx = get_global_id(1);
    if (dstIdx >= numNodes || featIdx >= featureSize) return;

    float sum = 0.0f;
    int cnt = 0;
    for (int i = 0; i < numElements; i++) {
        if (indices[i] == dstIdx) {
            sum += src[i * featureSize + featIdx];
            if (featIdx == 0) cnt++;
        }
    }
    dst[dstIdx * featureSize + featIdx] += sum;
    if (featIdx == 0) counts[dstIdx] += cnt;
}

__kernel void scatter_mean_normalize(
    __global float* dst,
    __global const int* counts,
    const int numNodes,
    const int featureSize)
{
    const int idx = get_global_id(0);
    if (idx >= numNodes * featureSize) return;

    const int nodeIdx = idx / featureSize;
    const int count = counts[nodeIdx];
    if (count > 0) {
        dst[idx] /= (float)count;
    }
}

// ===========================================================================
// ADDITIONAL NORMALIZATION BACKWARD KERNELS
// ===========================================================================

// Group normalization backward - Pass 1: Compute group-wise sums
// CRITICAL: Host MUST validate that C % G == 0 before launching this kernel.
// Launching with C not divisible by G will produce incorrect results.
// Non-deterministic across runs (issue #382); see groupnorm_backward_sums_per_channel
// + groupnorm_backward_sums_per_group below for the deterministic variants.
__kernel void groupnorm_backward_sums(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* mean,
    __global const float* invStd,
    __global const float* gamma,
    __global float* sumDy,
    __global float* sumDyXhat,
    __global float* gradGamma,
    __global float* gradBeta,
    const int N, const int C, const int H, const int W, const int G)
{
    const int idx = get_global_id(0);
    const int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    const int c = (idx / (W * H)) % C;
    const int n = idx / (W * H * C);

    const int channelsPerGroup = C / G;
    const int g = c / channelsPerGroup;

    float dy = gradOutput[idx];
    float x = input[idx];
    float m = mean[n * G + g];
    float s = invStd[n * G + g];
    float gam = gamma[c];

    float xHat = (x - m) * s;
    float dyGam = dy * gam;

    // Use atomic operations for correct accumulation
    atomic_add_float(&gradGamma[c], dy * xHat);
    atomic_add_float(&gradBeta[c], dy);

    int groupIdx = n * G + g;
    atomic_add_float(&sumDy[groupIdx], dyGam);
    atomic_add_float(&sumDyXhat[groupIdx], dyGam * xHat);
}

// GroupNorm backward sums — bit-deterministic variants (issue #382).
// Split into two kernels because the four atomic outputs reduce over different shapes:
//   gradGamma/gradBeta:  one cell per channel C, sum over (N, H, W)
//   sumDy/sumDyXhat:     one cell per (N, G), sum over (channelsPerGroup, H, W)
// Each work-item owns one output cell and scans the contributing input range in a fixed
// order — accumulation is bit-identical across runs.
__kernel void groupnorm_backward_sums_per_channel_deterministic(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* mean,
    __global const float* invStd,
    __global float* gradGamma,
    __global float* gradBeta,
    const int N, const int C, const int H, const int W, const int G)
{
    const int c = get_global_id(0);
    if (c >= C) return;

    const int channelsPerGroup = C / G;
    const int g = c / channelsPerGroup;
    const int HW = H * W;

    float accGamma = 0.0f;
    float accBeta = 0.0f;
    for (int n = 0; n < N; n++) {
        float m = mean[n * G + g];
        float s = invStd[n * G + g];
        int baseNC = (n * C + c) * HW;
        for (int hw = 0; hw < HW; hw++) {
            float dy = gradOutput[baseNC + hw];
            float x = input[baseNC + hw];
            float xHat = (x - m) * s;
            accGamma += dy * xHat;
            accBeta += dy;
        }
    }
    // Plain assignment: each thread owns one (c) slot and performs the full
    // reduction locally. Overwriting keeps the kernel self-contained on reused
    // scratch buffers.
    gradGamma[c] = accGamma;
    gradBeta[c] = accBeta;
}

__kernel void groupnorm_backward_sums_per_group_deterministic(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* mean,
    __global const float* invStd,
    __global const float* gamma,
    __global float* sumDy,
    __global float* sumDyXhat,
    const int N, const int C, const int H, const int W, const int G)
{
    const int ng = get_global_id(0);
    if (ng >= N * G) return;

    const int n = ng / G;
    const int g = ng % G;
    const int channelsPerGroup = C / G;
    const int cStart = g * channelsPerGroup;
    const int HW = H * W;

    float m = mean[n * G + g];
    float s = invStd[n * G + g];

    float accDy = 0.0f;
    float accDyXhat = 0.0f;
    for (int cOff = 0; cOff < channelsPerGroup; cOff++) {
        int c = cStart + cOff;
        float gam = gamma[c];
        int baseNC = (n * C + c) * HW;
        for (int hw = 0; hw < HW; hw++) {
            float dy = gradOutput[baseNC + hw];
            float x = input[baseNC + hw];
            float xHat = (x - m) * s;
            float dyGam = dy * gam;
            accDy += dyGam;
            accDyXhat += dyGam * xHat;
        }
    }
    // Plain assignment: one thread owns each (n*G+g) slot; full reduction is local.
    sumDy[ng] = accDy;
    sumDyXhat[ng] = accDyXhat;
}

// Group normalization backward - Pass 2: Compute final input gradients
__kernel void groupnorm_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* mean,
    __global const float* invStd,
    __global const float* gamma,
    __global const float* sumDy,
    __global const float* sumDyXhat,
    __global float* gradInput,
    const int N, const int C, const int H, const int W, const int G)
{
    const int idx = get_global_id(0);
    const int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    const int c = (idx / (W * H)) % C;
    const int n = idx / (W * H * C);

    const int channelsPerGroup = C / G;
    const int g = c / channelsPerGroup;
    const int groupSize = channelsPerGroup * H * W;
    const float invN = 1.0f / (float)groupSize;

    float dy = gradOutput[idx];
    float x = input[idx];
    float m = mean[n * G + g];
    float s = invStd[n * G + g];
    float gam = gamma[c];

    float xHat = (x - m) * s;
    float dyGam = dy * gam;

    int groupIdx = n * G + g;
    float sDy = sumDy[groupIdx];
    float sDyXhat = sumDyXhat[groupIdx];

    // Full group normalization backward formula
    gradInput[idx] = s * (dyGam - invN * (sDy + xHat * sDyXhat));
}

// Instance normalization backward - Pass 1: Compute instance-wise sums
// Non-deterministic across runs (issue #382). DeterministicMode dispatches to the
// per-channel + per-instance kernels below.
__kernel void instancenorm_backward_sums(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* mean,
    __global const float* invStd,
    __global const float* gamma,
    __global float* sumDy,
    __global float* sumDyXhat,
    __global float* gradGamma,
    __global float* gradBeta,
    const int N, const int C, const int H, const int W)
{
    const int idx = get_global_id(0);
    const int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    const int c = (idx / (W * H)) % C;
    const int n = idx / (W * H * C);

    float dy = gradOutput[idx];
    float x = input[idx];
    float m = mean[n * C + c];
    float s = invStd[n * C + c];
    float gam = gamma[c];

    float xHat = (x - m) * s;
    float dyGam = dy * gam;

    atomic_add_float(&gradGamma[c], dy * xHat);
    atomic_add_float(&gradBeta[c], dy);

    int instanceIdx = n * C + c;
    atomic_add_float(&sumDy[instanceIdx], dyGam);
    atomic_add_float(&sumDyXhat[instanceIdx], dyGam * xHat);
}

// InstanceNorm backward sums — bit-deterministic variants (issue #382).
// Same two-axis split as the GroupNorm variants above:
//   gradGamma/gradBeta:  one cell per channel C, sum over (N, H, W)
//   sumDy/sumDyXhat:     one cell per (N, C) instance, sum over (H, W)
__kernel void instancenorm_backward_sums_per_channel_deterministic(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* mean,
    __global const float* invStd,
    __global float* gradGamma,
    __global float* gradBeta,
    const int N, const int C, const int H, const int W)
{
    const int c = get_global_id(0);
    if (c >= C) return;

    const int HW = H * W;

    float accGamma = 0.0f;
    float accBeta = 0.0f;
    for (int n = 0; n < N; n++) {
        float m = mean[n * C + c];
        float s = invStd[n * C + c];
        int baseNC = (n * C + c) * HW;
        for (int hw = 0; hw < HW; hw++) {
            float dy = gradOutput[baseNC + hw];
            float x = input[baseNC + hw];
            float xHat = (x - m) * s;
            accGamma += dy * xHat;
            accBeta += dy;
        }
    }
    // Plain assignment: see groupnorm_backward_sums_per_channel_deterministic for rationale.
    gradGamma[c] = accGamma;
    gradBeta[c] = accBeta;
}

__kernel void instancenorm_backward_sums_per_instance_deterministic(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* mean,
    __global const float* invStd,
    __global const float* gamma,
    __global float* sumDy,
    __global float* sumDyXhat,
    const int N, const int C, const int H, const int W)
{
    const int nc = get_global_id(0);
    if (nc >= N * C) return;

    const int n = nc / C;
    const int c = nc % C;
    const int HW = H * W;

    float m = mean[nc];
    float s = invStd[nc];
    float gam = gamma[c];

    float accDy = 0.0f;
    float accDyXhat = 0.0f;
    int baseNC = nc * HW;
    for (int hw = 0; hw < HW; hw++) {
        float dy = gradOutput[baseNC + hw];
        float x = input[baseNC + hw];
        float xHat = (x - m) * s;
        float dyGam = dy * gam;
        accDy += dyGam;
        accDyXhat += dyGam * xHat;
    }
    // Plain assignment: one owner per (n*C+c), full local reduction.
    sumDy[nc] = accDy;
    sumDyXhat[nc] = accDyXhat;
}

// Instance normalization backward - Pass 2: Compute final input gradients
__kernel void instancenorm_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* mean,
    __global const float* invStd,
    __global const float* gamma,
    __global const float* sumDy,
    __global const float* sumDyXhat,
    __global float* gradInput,
    const int N, const int C, const int H, const int W)
{
    const int idx = get_global_id(0);
    const int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    const int c = (idx / (W * H)) % C;
    const int n = idx / (W * H * C);

    const int instanceSize = H * W;
    const float invN = 1.0f / (float)instanceSize;

    float dy = gradOutput[idx];
    float x = input[idx];
    float m = mean[n * C + c];
    float s = invStd[n * C + c];
    float gam = gamma[c];

    float xHat = (x - m) * s;
    float dyGam = dy * gam;

    int instanceIdx = n * C + c;
    float sDy = sumDy[instanceIdx];
    float sDyXhat = sumDyXhat[instanceIdx];

    // Full instance normalization backward formula
    gradInput[idx] = s * (dyGam - invN * (sDy + xHat * sDyXhat));
}

// ===========================================================================
// CONV3D BACKWARD KERNELS
// ===========================================================================

__kernel void conv3d_backward_input(
    __global const float* gradOutput,
    __global const float* weights,
    __global float* gradInput,
    const int N, const int inC, const int D, const int H, const int W,
    const int outC, const int outD, const int outH, const int outW,
    const int kD, const int kH, const int kW,
    const int strideD, const int strideH, const int strideW,
    const int padD, const int padH, const int padW)
{
    const int idx = get_global_id(0);
    const int totalSize = N * inC * D * H * W;
    if (idx >= totalSize) return;

    const int w = idx % W;
    const int h = (idx / W) % H;
    const int d = (idx / (W * H)) % D;
    const int ic = (idx / (W * H * D)) % inC;
    const int n = idx / (W * H * D * inC);

    float sum = 0.0f;

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
}

__kernel void conv3d_backward_weights(
    __global const float* gradOutput,
    __global const float* input,
    __global float* gradKernel,
    const int N, const int inC, const int D, const int H, const int W,
    const int outC, const int outD, const int outH, const int outW,
    const int kD, const int kH, const int kW,
    const int strideD, const int strideH, const int strideW,
    const int padD, const int padH, const int padW)
{
    const int idx = get_global_id(0);
    const int totalKernelSize = outC * inC * kD * kH * kW;
    if (idx >= totalKernelSize) return;

    const int kw = idx % kW;
    const int kh = (idx / kW) % kH;
    const int kd = (idx / (kW * kH)) % kD;
    const int ic = (idx / (kW * kH * kD)) % inC;
    const int oc = idx / (kW * kH * kD * inC);

    float sum = 0.0f;

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
                        sum += gradOutput[gradOutIdx] * input[inputIdx];
                    }
                }
            }
        }
    }

    gradKernel[idx] = sum;
}

// ===========================================================================
// GLOBAL POOLING BACKWARD KERNELS
// ===========================================================================

__kernel void global_avgpool_backward(
    __global const float* gradOutput,
    __global float* gradInput,
    const int N, const int C, const int H, const int W)
{
    const int idx = get_global_id(0);
    const int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    const int c = (idx / (W * H)) % C;
    const int n = idx / (W * H * C);

    float scale = 1.0f / (float)(H * W);
    gradInput[idx] = gradOutput[n * C + c] * scale;
}

__kernel void global_maxpool_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const int* maxIndices,
    __global float* gradInput,
    const int N, const int C, const int H, const int W)
{
    const int idx = get_global_id(0);
    const int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    const int w = idx % W;
    const int h = (idx / W) % H;
    const int c = (idx / (W * H)) % C;
    const int n = idx / (W * H * C);

    int spatialIdx = h * W + w;
    int maxIdx = maxIndices[n * C + c];

    gradInput[idx] = (spatialIdx == maxIdx) ? gradOutput[n * C + c] : 0.0f;
}

__kernel void adaptive_avgpool_backward(
    __global const float* gradOutput,
    __global float* gradInput,
    const int N, const int C, const int H, const int W, const int outH, const int outW)
{
    const int idx = get_global_id(0);
    const int totalSize = N * C * H * W;
    if (idx >= totalSize) return;

    const int w = idx % W;
    const int h = (idx / W) % H;
    const int c = (idx / (W * H)) % C;
    const int n = idx / (W * H * C);

    float sum = 0.0f;

    for (int oh = 0; oh < outH; oh++) {
        int hStart = (oh * H) / outH;
        int hEnd = ((oh + 1) * H) / outH;
        if (h < hStart || h >= hEnd) continue;

        for (int ow = 0; ow < outW; ow++) {
            int wStart = (ow * W) / outW;
            int wEnd = ((ow + 1) * W) / outW;
            if (w < wStart || w >= wEnd) continue;

            int poolSize = (hEnd - hStart) * (wEnd - wStart);
            int gradOutIdx = ((n * C + c) * outH + oh) * outW + ow;
            sum += gradOutput[gradOutIdx] / (float)poolSize;
        }
    }

    gradInput[idx] = sum;
}

// #775: GNN scatter-add (index_add) along dim 0. output[m, inner] = sum over source rows d whose
// index == m of source[d, inner]. GATHER over the output (no atomics); ascending d matches the
// CpuEngine accumulation order bit-for-bit. Uses the first srcDimSize flattened index values
// (indices[d]), mirroring CpuEngine.ScatterAdd's indicesData[d].
__kernel void scatter_add_rows(
    __global const float* source,   // [srcDimSize, innerSize]
    __global const int* indices,    // first srcDimSize values are the per-row targets
    __global float* output,         // [outDimSize, innerSize]
    const int srcDimSize, const int innerSize, const int outDimSize)
{
    int idx = get_global_id(0);
    if (idx >= outDimSize * innerSize) return;
    int inner = idx % innerSize;
    int m = idx / innerSize;
    float sum = 0.0f;
    for (int d = 0; d < srcDimSize; d++) {
        if (indices[d] == m) sum += source[d * innerSize + inner];
    }
    output[idx] = sum;
}

// #775: GNN scatter-mean along dim 0 = scatter-add / per-output-row count. Multiply by 1/count
// (matching CpuEngine's invDivisor) rather than divide, and leave 0 where the count is 0.
__kernel void scatter_mean_rows(
    __global const float* source,
    __global const int* indices,
    __global float* output,
    const int srcDimSize, const int innerSize, const int outDimSize)
{
    int idx = get_global_id(0);
    if (idx >= outDimSize * innerSize) return;
    int inner = idx % innerSize;
    int m = idx / innerSize;
    float sum = 0.0f;
    int count = 0;
    for (int d = 0; d < srcDimSize; d++) {
        if (indices[d] == m) { sum += source[d * innerSize + inner]; count++; }
    }
    output[idx] = count > 0 ? sum * (1.0f / (float)count) : 0.0f;
}
";
        }

        /// <summary>
        /// Gets kernel names for compilation.
        /// </summary>
        public static string[] GetKernelNames()
        {
            return new string[]
            {
                // Activation gradients
                "relu_backward", "sigmoid_backward", "tanh_backward",
                "gelu_backward", "softmax_backward",
                "leaky_relu_forward", "leaky_relu_backward",
                "elu_forward", "elu_backward",
                "swish_forward", "swish_backward",
                "silu_forward", "mish_forward", "softplus_forward", "hardswish_forward",
                // Loss functions
                "cross_entropy_loss", "cross_entropy_backward",
                "bce_loss", "bce_backward",
                "mse_loss", "mse_backward",
                "smooth_l1_loss", "smooth_l1_backward",
                // Optimizers
                // Utilities
                "clamp_values", "clip_by_value",
                "transpose2d", "batched_transpose",
                "fill_buffer", "copy_buffer",
                "greater_than", "less_than", "equal_values", "where_select",
                "mean_axis", "var_axis", "argmax_axis", "argmin_axis",
                "dropout_forward", "dropout_backward",
                "embedding_lookup", "embedding_backward", "embedding_backward_deterministic",
                "fma_kernel", "gather_kernel", "scatter_add_kernel", "scatter_add_kernel_deterministic", "scatter_add_rows", "scatter_mean_rows",
                // LSTM kernels
                "lstm_cell_forward", "lstm_cell_backward", "lstm_gates_precompute",
                // GRU kernels
                "gru_cell_forward", "gru_cell_backward",
                // Additional scatter operations
                "scatter_add_batched", "scatter_add_batched_deterministic",
                "scatter_mean_accumulate", "scatter_mean_accumulate_deterministic",
                "scatter_mean_normalize",
                // Additional normalization backward (issue #382: deterministic variants split
                // per-channel and per-group/instance to avoid atomic accumulation)
                "groupnorm_backward_sums",
                "groupnorm_backward_sums_per_channel_deterministic",
                "groupnorm_backward_sums_per_group_deterministic",
                "groupnorm_backward",
                "instancenorm_backward_sums",
                "instancenorm_backward_sums_per_channel_deterministic",
                "instancenorm_backward_sums_per_instance_deterministic",
                "instancenorm_backward",
                // Conv3D backward
                "conv3d_backward_input", "conv3d_backward_weights",
                // Global pooling backward
                "global_avgpool_backward", "global_maxpool_backward", "adaptive_avgpool_backward"
            };
        }
    }
}
