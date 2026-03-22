namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Fused CUDA reduction kernels: Sum, Mean, Variance, StdDev, Norm, LogSumExp,
/// Min/Max with magnitude, Product, CumSum, ScalarMinus, and axis-based variants.
/// All kernels use warp-level primitives (__shfl_down_sync) for maximum throughput
/// and shared memory tiling to minimize global memory traffic.
/// </summary>
public static class CudaReductionKernels
{
    public static string GetSource()
    {
        return @"
// ============================================================================
// Warp-level reduction primitives (32 threads, zero shared memory)
// ============================================================================

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ float warp_reduce_min(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ float warp_reduce_prod(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val *= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// Block-level reduction using shared memory + warp primitives
// ============================================================================

__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // One slot per warp
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ float block_reduce_max(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

__device__ float block_reduce_min(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_min(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : INFINITY;
    if (wid == 0) val = warp_reduce_min(val);
    return val;
}

// ============================================================================
// Full tensor reductions (single output scalar)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void reduce_mean(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        sum += input[i];
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) atomicAdd(output, sum / (float)size);
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_product(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    float prod = 1.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        prod *= input[i];
    prod = warp_reduce_prod(prod);
    // atomicMul doesn't exist — use atomicCAS-based multiply
    if ((threadIdx.x & 31) == 0) {
        float* addr = output;
        float old = *addr, assumed;
        do {
            assumed = old;
            old = __int_as_float(atomicCAS((int*)addr, __float_as_int(assumed), __float_as_int(assumed * prod)));
        } while (assumed != old);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_norm_l2(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    float sum_sq = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float v = input[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    if (threadIdx.x == 0) atomicAdd(output, sum_sq);
    // Caller takes sqrt of output
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_sum_of_squares(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    float sum_sq = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        float v = input[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);
    if (threadIdx.x == 0) atomicAdd(output, sum_sq);
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_max_magnitude(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    float max_abs = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        max_abs = fmaxf(max_abs, fabsf(input[i]));
    max_abs = block_reduce_max(max_abs);
    if (threadIdx.x == 0) {
        // atomicMax for float via CAS
        float* addr = output;
        float old = *addr, assumed;
        do {
            assumed = old;
            if (assumed >= max_abs) break;
            old = __int_as_float(atomicCAS((int*)addr, __float_as_int(assumed), __float_as_int(max_abs)));
        } while (assumed != old);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_min_magnitude(
    const float* __restrict__ input, float* __restrict__ output, int size)
{
    float min_abs = INFINITY;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        min_abs = fminf(min_abs, fabsf(input[i]));
    min_abs = block_reduce_min(min_abs);
    if (threadIdx.x == 0) {
        float* addr = output;
        float old = *addr, assumed;
        do {
            assumed = old;
            if (assumed <= min_abs) break;
            old = __int_as_float(atomicCAS((int*)addr, __float_as_int(assumed), __float_as_int(min_abs)));
        } while (assumed != old);
    }
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_logsumexp(
    const float* __restrict__ input, float* __restrict__ output,
    float maxVal, int size)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
        sum += expf(input[i] - maxVal);
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) atomicAdd(output, sum);
    // Caller computes: maxVal + log(output)
}

// ============================================================================
// Axis-based reductions: operate on innermost dimension
// Each thread handles one outer position
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void mean_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outerSize) return;

    float sum = 0.0f;
    const float* row = input + idx * reduceSize;
    for (int j = 0; j < reduceSize; j++)
        sum += row[j];
    output[idx] = sum / (float)reduceSize;
}

extern ""C"" __global__ __launch_bounds__(256) void variance_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outerSize) return;

    const float* row = input + idx * reduceSize;
    float sum = 0.0f;
    for (int j = 0; j < reduceSize; j++)
        sum += row[j];
    float mean = sum / (float)reduceSize;

    float var_sum = 0.0f;
    for (int j = 0; j < reduceSize; j++) {
        float diff = row[j] - mean;
        var_sum += diff * diff;
    }
    output[idx] = var_sum / (float)reduceSize;
}

extern ""C"" __global__ __launch_bounds__(256) void std_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outerSize) return;

    const float* row = input + idx * reduceSize;
    float sum = 0.0f;
    for (int j = 0; j < reduceSize; j++)
        sum += row[j];
    float mean = sum / (float)reduceSize;

    float var_sum = 0.0f;
    for (int j = 0; j < reduceSize; j++) {
        float diff = row[j] - mean;
        var_sum += diff * diff;
    }
    output[idx] = sqrtf(var_sum / (float)reduceSize);
}

extern ""C"" __global__ __launch_bounds__(256) void product_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outerSize) return;

    float prod = 1.0f;
    const float* row = input + idx * reduceSize;
    for (int j = 0; j < reduceSize; j++)
        prod *= row[j];
    output[idx] = prod;
}

extern ""C"" __global__ __launch_bounds__(256) void norm_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outerSize) return;

    float sum_sq = 0.0f;
    const float* row = input + idx * reduceSize;
    for (int j = 0; j < reduceSize; j++) {
        float v = row[j];
        sum_sq += v * v;
    }
    output[idx] = sqrtf(sum_sq);
}

extern ""C"" __global__ __launch_bounds__(256) void logsumexp_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outerSize) return;

    const float* row = input + idx * reduceSize;

    // Numerically stable: max then log(sum(exp(x - max)))
    float max_val = -INFINITY;
    for (int j = 0; j < reduceSize; j++)
        max_val = fmaxf(max_val, row[j]);

    float sum_exp = 0.0f;
    for (int j = 0; j < reduceSize; j++)
        sum_exp += expf(row[j] - max_val);

    output[idx] = max_val + logf(sum_exp);
}

// ============================================================================
// Cumulative sum (inclusive prefix sum along last axis)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void cumsum_axis(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int innerSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outerSize) return;

    const float* in_row = input + idx * innerSize;
    float* out_row = output + idx * innerSize;

    float running = 0.0f;
    for (int j = 0; j < innerSize; j++) {
        running += in_row[j];
        out_row[j] = running;
    }
}

// ============================================================================
// Element-wise: scalar minus tensor, normalize, broadcast divide
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void scalar_minus_tensor(
    const float* __restrict__ input, float* __restrict__ output,
    float scalar, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = scalar - input[idx];
}

extern ""C"" __global__ __launch_bounds__(256) void normalize_l2(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int innerSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outerSize) return;

    const float* in_row = input + row * innerSize;
    float* out_row = output + row * innerSize;

    float sum_sq = 0.0f;
    for (int j = 0; j < innerSize; j++) {
        float v = in_row[j];
        sum_sq += v * v;
    }
    float inv_norm = 1.0f / (sqrtf(sum_sq) + 1e-12f);

    for (int j = 0; j < innerSize; j++)
        out_row[j] = in_row[j] * inv_norm;
}

// Reduce backward: dL/dx_i = dL/dsum for ReduceSum, dL/dmean * (1/n) for ReduceMean
extern ""C"" __global__ __launch_bounds__(256) void reduce_sum_backward(
    const float* __restrict__ grad_output, float* __restrict__ grad_input,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * reduceSize;
    if (idx >= total) return;

    int outer = idx / reduceSize;
    grad_input[idx] = grad_output[outer];
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_mean_backward(
    const float* __restrict__ grad_output, float* __restrict__ grad_input,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * reduceSize;
    if (idx >= total) return;

    int outer = idx / reduceSize;
    grad_input[idx] = grad_output[outer] / (float)reduceSize;
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_max_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ max_values,
    float* __restrict__ grad_input,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * reduceSize;
    if (idx >= total) return;

    int outer = idx / reduceSize;
    // Gradient flows only to the max element
    grad_input[idx] = (input[idx] == max_values[outer]) ? grad_output[outer] : 0.0f;
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_variance_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ means,
    float* __restrict__ grad_input,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * reduceSize;
    if (idx >= total) return;

    int outer = idx / reduceSize;
    // dVar/dx_i = 2*(x_i - mean) / N
    float diff = input[idx] - means[outer];
    grad_input[idx] = grad_output[outer] * 2.0f * diff / (float)reduceSize;
}

// ============================================================================
// Log-variance reduction: log(variance) = log(E[(x-mu)^2])
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void reduce_log_variance(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outerSize) return;

    const float* row = input + idx * reduceSize;
    float sum = 0.0f;
    for (int j = 0; j < reduceSize; j++)
        sum += row[j];
    float mean = sum / (float)reduceSize;

    float var_sum = 0.0f;
    for (int j = 0; j < reduceSize; j++) {
        float diff = row[j] - mean;
        var_sum += diff * diff;
    }
    output[idx] = logf(var_sum / (float)reduceSize + 1e-8f);
}

extern ""C"" __global__ __launch_bounds__(256) void reduce_log_variance_backward(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ means,
    const float* __restrict__ variances,
    float* __restrict__ grad_input,
    int outerSize, int reduceSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * reduceSize;
    if (idx >= total) return;

    int outer = idx / reduceSize;
    // d(log(var))/dx_i = (1/var) * d(var)/dx_i = (1/var) * 2*(x_i - mean)/N
    float diff = input[idx] - means[outer];
    float inv_var = 1.0f / (variances[outer] + 1e-8f);
    grad_input[idx] = grad_output[outer] * inv_var * 2.0f * diff / (float)reduceSize;
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "reduce_mean",
            "reduce_product",
            "reduce_norm_l2",
            "reduce_sum_of_squares",
            "reduce_max_magnitude",
            "reduce_min_magnitude",
            "reduce_logsumexp",
            "mean_axis",
            "variance_axis",
            "std_axis",
            "product_axis",
            "norm_axis",
            "logsumexp_axis",
            "cumsum_axis",
            "scalar_minus_tensor",
            "normalize_l2",
            "reduce_sum_backward",
            "reduce_mean_backward",
            "reduce_max_backward",
            "reduce_variance_backward",
            "reduce_log_variance",
            "reduce_log_variance_backward"
        ];
    }
}
