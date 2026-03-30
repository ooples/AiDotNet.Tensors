namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Fused CUDA kernels for loss function forward passes: CrossEntropy, MSE, BCE.
/// Each kernel computes per-sample loss in a single pass.
/// </summary>
public static class CudaLossForwardKernels
{
    public static string GetSource()
    {
        return @"
// ============================================================================
// Cross-Entropy Loss: -sum(target * log(pred)) per sample
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void cross_entropy_loss(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ loss,
    int batchSize, int numClasses)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    float sample_loss = 0.0f;
    for (int c = 0; c < numClasses; c++) {
        float pred = predictions[b * numClasses + c];
        float target = targets[b * numClasses + c];
        if (target > 0.0f)
            sample_loss -= target * logf(fmaxf(pred, 1e-7f));
    }
    loss[b] = sample_loss;
}

// ============================================================================
// MSE Loss: mean((pred - target)^2) per sample
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void mse_loss(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ loss,
    int batchSize, int numFeatures)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    float sum_sq = 0.0f;
    for (int f = 0; f < numFeatures; f++) {
        float diff = predictions[b * numFeatures + f] - targets[b * numFeatures + f];
        sum_sq += diff * diff;
    }
    loss[b] = sum_sq / (float)numFeatures;
}

// ============================================================================
// Binary Cross-Entropy: -[t*log(p) + (1-t)*log(1-p)] per element
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void bce_loss(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ loss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float p = fminf(fmaxf(predictions[idx], 1e-7f), 1.0f - 1e-7f);
    float t = targets[idx];
    loss[idx] = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
}

// ============================================================================
// Dropout mask generation using Philox PRNG
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void dropout_mask(
    float* __restrict__ mask, int size, float keepProb, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Philox PRNG
    unsigned int counter = (unsigned int)idx;
    unsigned int key = (unsigned int)(seed ^ (seed >> 32));
    unsigned long long product = (unsigned long long)counter * 0xCD9E8D57u;
    counter = (unsigned int)(product >> 32) ^ key ^ (unsigned int)product;
    product = (unsigned long long)counter * 0xCD9E8D57u;
    counter = (unsigned int)(product >> 32) ^ (key + 1u) ^ (unsigned int)product;

    float u = (float)(counter >> 8) * (1.0f / 16777216.0f);
    // Inverted dropout: scale by 1/keepProb so test-time doesn't need scaling
    // Guard against keepProb==0 (full dropout) to avoid division by zero
    float invKeep = (keepProb > 1e-7f) ? (1.0f / keepProb) : 0.0f;
    mask[idx] = (u < keepProb) ? invKeep : 0.0f;
}

// ============================================================================
// Gaussian noise generation
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void gaussian_noise(
    float* __restrict__ output, int size, float mean, float stdDev, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned int key = (unsigned int)(seed ^ (seed >> 32));
    unsigned int c1 = (unsigned int)(2 * idx);
    unsigned long long p1 = (unsigned long long)c1 * 0xCD9E8D57u;
    c1 = (unsigned int)(p1 >> 32) ^ key ^ (unsigned int)p1;
    p1 = (unsigned long long)c1 * 0xCD9E8D57u;
    c1 = (unsigned int)(p1 >> 32) ^ (key + 1u) ^ (unsigned int)p1;

    unsigned int c2 = (unsigned int)(2 * idx + 1);
    unsigned long long p2 = (unsigned long long)c2 * 0xCD9E8D57u;
    c2 = (unsigned int)(p2 >> 32) ^ (key + 2u) ^ (unsigned int)p2;
    p2 = (unsigned long long)c2 * 0xCD9E8D57u;
    c2 = (unsigned int)(p2 >> 32) ^ (key + 3u) ^ (unsigned int)p2;

    float u1 = (float)(c1 >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
    float u2 = (float)(c2 >> 8) * (1.0f / 16777216.0f);

    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    output[idx] = mean + stdDev * z;
}

// ============================================================================
// L1 Loss: mean(|pred - target|) per sample
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void l1_loss(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ loss,
    int batchSize, int numFeatures)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    float sum_abs = 0.0f;
    for (int f = 0; f < numFeatures; f++) {
        float diff = predictions[b * numFeatures + f] - targets[b * numFeatures + f];
        sum_abs += fabsf(diff);
    }
    loss[b] = sum_abs / (float)numFeatures;
}

// ============================================================================
// Huber Loss: smooth L1 that transitions from L2 to L1 at delta
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void huber_loss(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ loss,
    int batchSize, int numFeatures, float delta)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    float sum = 0.0f;
    for (int f = 0; f < numFeatures; f++) {
        float diff = predictions[b * numFeatures + f] - targets[b * numFeatures + f];
        float abs_diff = fabsf(diff);
        sum += (abs_diff <= delta) ? (0.5f * diff * diff) : (delta * (abs_diff - 0.5f * delta));
    }
    loss[b] = sum / (float)numFeatures;
}

// ============================================================================
// BCE with Logits Loss: sigmoid cross-entropy (numerically stable)
// max(x,0) - x*t + log(1+exp(-|x|))
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void bce_with_logits_loss(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ loss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = logits[idx];
    float t = targets[idx];
    float abs_x = fabsf(x);
    loss[idx] = fmaxf(x, 0.0f) - x * t + log1pf(expf(-abs_x));
}

// ============================================================================
// NLL Loss: -log_probs[target_class] per sample (sparse targets)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void nll_loss(
    const float* __restrict__ logProbs,
    const int* __restrict__ targets,
    float* __restrict__ loss,
    int batchSize, int numClasses)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    int targetClass = targets[b];
    loss[b] = (targetClass >= 0 && targetClass < numClasses)
        ? -logProbs[b * numClasses + targetClass] : 0.0f;
}

// ============================================================================
// KL Divergence Loss: target * (log(target) - input) per element
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void kl_div_loss(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ loss, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float t = target[idx];
    loss[idx] = (t > 0.0f) ? t * (logf(t) - input[idx]) : 0.0f;
}

// ============================================================================
// Cosine Similarity: dot(a,b) / (||a|| * ||b||) per batch
// Uses warp-level reduction for high-performance dot products
// ============================================================================

extern ""C"" __global__ void cosine_similarity_batch(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int batchSize, int dim)
{
    int batch = blockIdx.x;
    if (batch >= batchSize) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int base_idx = batch * dim;

    float local_dot = 0.0f, local_normA = 0.0f, local_normB = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float va = a[base_idx + i];
        float vb = b[base_idx + i];
        local_dot += va * vb;
        local_normA += va * va;
        local_normB += vb * vb;
    }

    // Reduce within block using shared memory
    smem[tid] = local_dot;
    smem[tid + blockDim.x] = local_normA;
    smem[tid + 2 * blockDim.x] = local_normB;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem[tid + blockDim.x] += smem[tid + blockDim.x + s];
            smem[tid + 2 * blockDim.x] += smem[tid + 2 * blockDim.x + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float dot = smem[0];
        float norm = sqrtf(smem[blockDim.x]) * sqrtf(smem[2 * blockDim.x]) + 1e-8f;
        output[batch] = dot / norm;
    }
}

// ============================================================================
// MSE Loss backward: 2*(pred - target) / N
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void mse_loss_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ gradInput,
    int size, float inv_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    gradInput[idx] = gradOutput[0] * 2.0f * (predictions[idx] - targets[idx]) * inv_n;
}

// ============================================================================
// L1 Loss backward: sign(pred - target) / N
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void l1_loss_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ gradInput,
    int size, float inv_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = predictions[idx] - targets[idx];
    float s = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
    gradInput[idx] = gradOutput[0] * s * inv_n;
}

// ============================================================================
// Huber Loss backward: d/dpred = (|diff| <= delta) ? diff/N : delta*sign(diff)/N
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void huber_loss_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ gradInput,
    int size, float inv_n, float delta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float diff = predictions[idx] - targets[idx];
    float abs_diff = fabsf(diff);
    float grad = (abs_diff <= delta) ? diff : (delta * ((diff > 0.0f) ? 1.0f : -1.0f));
    gradInput[idx] = gradOutput[0] * grad * inv_n;
}

// ============================================================================
// BCE with Logits backward: sigmoid(logits) - targets
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void bce_with_logits_backward(
    const float* __restrict__ gradOutput,
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ gradInput,
    int size, float inv_n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float sig = 1.0f / (1.0f + expf(-logits[idx]));
    gradInput[idx] = gradOutput[0] * (sig - targets[idx]) * inv_n;
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "cross_entropy_loss",
            "mse_loss",
            "bce_loss",
            "dropout_mask",
            "gaussian_noise",
            // New loss kernels
            "l1_loss",
            "huber_loss",
            "bce_with_logits_loss",
            "nll_loss",
            "kl_div_loss",
            "cosine_similarity_batch",
            // Loss backward kernels
            "mse_loss_backward",
            "l1_loss_backward",
            "huber_loss_backward",
            "bce_with_logits_backward"
        ];
    }
}
