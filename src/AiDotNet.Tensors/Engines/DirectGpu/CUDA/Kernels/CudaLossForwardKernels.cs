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
            "gaussian_noise"
        ];
    }
}
