namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// Fused CUDA kernels for softmax variants: LogSoftmax, GumbelSoftmax,
/// Sparsemax, TaylorSoftmax, SphericalSoftmax.
/// Each is a single-pass fused kernel (no intermediate allocations).
/// </summary>
public static class CudaSoftmaxVariantKernels
{
    public static string GetSource()
    {
        return @"
// ============================================================================
// LogSoftmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
// Numerically stable, single-pass per row
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void log_softmax(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int innerSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outerSize) return;

    const float* in_row = input + row * innerSize;
    float* out_row = output + row * innerSize;

    // Pass 1: find max
    float max_val = -INFINITY;
    for (int j = 0; j < innerSize; j++)
        max_val = fmaxf(max_val, in_row[j]);

    // Pass 2: compute log(sum(exp))
    float sum_exp = 0.0f;
    for (int j = 0; j < innerSize; j++)
        sum_exp += expf(in_row[j] - max_val);
    float log_sum = logf(sum_exp);

    // Pass 3: write output
    for (int j = 0; j < innerSize; j++)
        out_row[j] = in_row[j] - max_val - log_sum;
}

// ============================================================================
// Gumbel-Softmax: softmax((log_probs + gumbel_noise) / temperature)
// For differentiable discrete sampling (Jang et al. 2017)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void gumbel_softmax(
    const float* __restrict__ logits, float* __restrict__ output,
    int outerSize, int innerSize, float temperature, unsigned long long seed)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outerSize) return;

    const float* in_row = logits + row * innerSize;
    float* out_row = output + row * innerSize;

    // Generate Gumbel noise and add to logits, divide by temperature
    float max_val = -INFINITY;
    for (int j = 0; j < innerSize; j++) {
        // Philox PRNG for this element
        unsigned int counter = (unsigned int)(row * innerSize + j);
        unsigned int key = (unsigned int)(seed ^ (seed >> 32));
        unsigned long long prod = (unsigned long long)counter * 0xCD9E8D57u;
        counter = (unsigned int)(prod >> 32) ^ key ^ (unsigned int)prod;
        prod = (unsigned long long)counter * 0xCD9E8D57u;
        counter = (unsigned int)(prod >> 32) ^ (key + 1u) ^ (unsigned int)prod;

        float u = (float)(counter >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
        float gumbel = -logf(-logf(u));

        float safe_temp = fmaxf(temperature, 1e-7f);
        float perturbed = (in_row[j] + gumbel) / safe_temp;
        out_row[j] = perturbed;
        max_val = fmaxf(max_val, perturbed);
    }

    // Softmax over perturbed logits
    float sum_exp = 0.0f;
    for (int j = 0; j < innerSize; j++) {
        out_row[j] = expf(out_row[j] - max_val);
        sum_exp += out_row[j];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (int j = 0; j < innerSize; j++)
        out_row[j] *= inv_sum;
}

// ============================================================================
// Sparsemax: projects onto probability simplex (Martins & Astudillo 2016)
// Produces sparse outputs (exact zeros)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void sparsemax(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int innerSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outerSize) return;

    const float* in_row = input + row * innerSize;
    float* out_row = output + row * innerSize;

    // Sort descending using insertion sort (ok for small innerSize typical in classification)
    // For large innerSize, a parallel sort would be needed.
    // IMPORTANT: Host code MUST validate innerSize <= 1024 before launching this kernel.
    // If innerSize > 1024, only the first 1024 elements are considered for tau computation.
    float sorted[1024];
    int n = innerSize < 1024 ? innerSize : 1024;
    for (int j = 0; j < n; j++) sorted[j] = in_row[j];

    // Insertion sort descending
    for (int i = 1; i < n; i++) {
        float key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] < key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }

    // Find threshold tau
    float cumsum = 0.0f;
    float tau = 0.0f;
    int k_star = 0;
    for (int k = 0; k < n; k++) {
        cumsum += sorted[k];
        float candidate = (cumsum - 1.0f) / (float)(k + 1);
        if (sorted[k] > candidate) {
            tau = candidate;
            k_star = k + 1;
        }
    }

    // Output = max(input - tau, 0)
    for (int j = 0; j < innerSize; j++)
        out_row[j] = fmaxf(in_row[j] - tau, 0.0f);
}

// ============================================================================
// Taylor Softmax: uses Taylor expansion of exp for better numerical stability
// exp(x) ≈ 1 + x + x²/2 (second-order Taylor)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void taylor_softmax(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int innerSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outerSize) return;

    const float* in_row = input + row * innerSize;
    float* out_row = output + row * innerSize;

    float sum = 0.0f;
    for (int j = 0; j < innerSize; j++) {
        float x = in_row[j];
        float taylor = 1.0f + x + 0.5f * x * x;
        taylor = fmaxf(taylor, 1e-10f); // Ensure positive
        out_row[j] = taylor;
        sum += taylor;
    }

    float inv_sum = 1.0f / sum;
    for (int j = 0; j < innerSize; j++)
        out_row[j] *= inv_sum;
}

// ============================================================================
// Spherical Softmax: normalizes to unit sphere then applies softmax
// output = softmax(||x|| * (x / ||x||))
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void spherical_softmax(
    const float* __restrict__ input, float* __restrict__ output,
    int outerSize, int innerSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outerSize) return;

    const float* in_row = input + row * innerSize;
    float* out_row = output + row * innerSize;

    // Compute L2 norm
    float norm_sq = 0.0f;
    for (int j = 0; j < innerSize; j++)
        norm_sq += in_row[j] * in_row[j];
    float norm = sqrtf(norm_sq + 1e-12f);

    // Normalize then softmax
    float max_val = -INFINITY;
    for (int j = 0; j < innerSize; j++) {
        float normalized = in_row[j] / norm;
        out_row[j] = normalized; // Spherical normalization (L2-normalized input to softmax)
        max_val = fmaxf(max_val, out_row[j]);
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < innerSize; j++) {
        out_row[j] = expf(out_row[j] - max_val);
        sum_exp += out_row[j];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (int j = 0; j < innerSize; j++)
        out_row[j] *= inv_sum;
}

// ============================================================================
// Batch dot product: output[b] = sum_i(a[b,i] * b[b,i])
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void batch_dot_product(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int batchSize, int dim)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    float sum = 0.0f;
    for (int i = 0; i < dim; i++)
        sum += a[batch * dim + i] * b[batch * dim + i];
    output[batch] = sum;
}

// ============================================================================
// Outer product: output[i,j] = a[i] * b[j]
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void outer_product(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int i = idx / N;
    int j = idx % N;
    output[idx] = a[i] * b[j];
}

// ============================================================================
// Batch outer product: output[b,i,j] = a[b,i] * b[b,j]
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void batch_outer_product(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int batchSize, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * M * N;
    if (idx >= total) return;

    int mn = M * N;
    int batch = idx / mn;
    int inner = idx % mn;
    int i = inner / N;
    int j = inner % N;

    output[idx] = a[batch * M + i] * b[batch * N + j];
}

// Cosine similarity: dot(a,b) / (||a|| * ||b||)
extern ""C"" __global__ __launch_bounds__(256) void cosine_similarity(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int batchSize, int dim)
{
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        float ai = a[batch * dim + i];
        float bi = b[batch * dim + i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    output[batch] = dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
}

// Pairwise L2 distance: output[i,j] = ||a[i] - b[j]||
extern ""C"" __global__ __launch_bounds__(256) void pairwise_distance(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int M, int N, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int i = idx / N;
    int j = idx % N;

    float dist_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = a[i * dim + d] - b[j * dim + d];
        dist_sq += diff * diff;
    }
    output[idx] = sqrtf(dist_sq);
}

extern ""C"" __global__ __launch_bounds__(256) void pairwise_distance_squared(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ output, int M, int N, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int i = idx / N;
    int j = idx % N;

    float dist_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = a[i * dim + d] - b[j * dim + d];
        dist_sq += diff * diff;
    }
    output[idx] = dist_sq;
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "log_softmax",
            "gumbel_softmax",
            "sparsemax",
            "taylor_softmax",
            "spherical_softmax",
            "batch_dot_product",
            "outer_product",
            "batch_outer_product",
            "cosine_similarity",
            "pairwise_distance",
            "pairwise_distance_squared"
        ];
    }
}
