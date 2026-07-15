namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for Spiking Neural Networks (SNN), Radial Basis Functions (RBF),
/// random number generation, and miscellaneous utility operations.
/// </summary>
internal static class CudaSnnKernels
{
    public static string GetSource()
    {
        return @"
// ============================================================================
// STDP (Spike-Timing-Dependent Plasticity) learning rule
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void stdp_update(
    float* __restrict__ weights,
    const float* __restrict__ preTrace,
    const float* __restrict__ postTrace,
    const float* __restrict__ preSpike,
    const float* __restrict__ postSpike,
    float ltpRate, float ltdRate, float homeostasisRate,
    float minWeight, float maxWeight,
    int numPre, int numPost)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numPre * numPost;
    if (idx >= total) return;

    int i = idx / numPost;  // pre-synaptic neuron
    int j = idx % numPost;  // post-synaptic neuron

    float w = weights[idx];
    float dw = 0.0f;

    // LTP: pre fires before post (causal, strengthen)
    if (preSpike[i] > 0.5f && postTrace[j] > 0.0f)
        dw += ltpRate * postTrace[j];

    // LTD: post fires before pre (anti-causal, weaken)
    if (postSpike[j] > 0.5f && preTrace[i] > 0.0f)
        dw -= ltdRate * preTrace[i];

    // Homeostasis: pull toward center of [minWeight, maxWeight]
    float center = (minWeight + maxWeight) * 0.5f;
    dw += homeostasisRate * (center - w);

    w += dw;
    weights[idx] = fminf(fmaxf(w, minWeight), maxWeight);
}

// ============================================================================
// Trace update for spiking neural networks
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void update_traces(
    float* __restrict__ traces,
    float* __restrict__ spikes,
    const float* __restrict__ input,
    float decay, float threshold, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Exponential trace decay
    float trace = decay * traces[idx];

    // Fire spike if input exceeds threshold
    if (input[idx] > threshold) {
        spikes[idx] = 1.0f;
        trace += 1.0f;
    } else {
        spikes[idx] = 0.0f;
    }

    traces[idx] = trace;
}

// ============================================================================
// RBF (Radial Basis Function) forward pass
// output[b * numCenters + c] = exp(-epsilon[c] * ||input[b] - center[c]||^2)
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void rbf_forward(
    const float* __restrict__ input,
    const float* __restrict__ centers,
    const float* __restrict__ epsilons,
    float* __restrict__ output,
    int batchSize, int numCenters, int inputDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * numCenters;
    if (idx >= total) return;

    int b = idx / numCenters;
    int c = idx % numCenters;

    float distSq = 0.0f;
    for (int d = 0; d < inputDim; d++) {
        float diff = input[b * inputDim + d] - centers[c * inputDim + d];
        distSq += diff * diff;
    }

    output[idx] = expf(-epsilons[c] * distSq);
}

// ============================================================================
// Philox-based PRNG for GPU random number generation
// Produces statistically independent uniform random floats
// ============================================================================

__device__ unsigned int philox_round(unsigned int ctr, unsigned int key) {
    unsigned long long product = (unsigned long long)ctr * 0xCD9E8D57u;
    return (unsigned int)(product >> 32) ^ key ^ (unsigned int)product;
}

__device__ float uint_to_float01(unsigned int x) {
    // Map [0, 2^32-1] to (0, 1]
    return (float)(x >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
}

extern ""C"" __global__ __launch_bounds__(256) void generate_random_uniform(
    float* __restrict__ output,
    int size, float minVal, float maxVal, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned int seed32 = (unsigned int)seed ^ (unsigned int)(seed >> 32);
    unsigned int state = (unsigned int)idx * 747796405u + seed32 + 2891336453u;
    unsigned int word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    unsigned int sample = (word >> 22) ^ word;
    float uniform = (float)(sample >> 8) * (1.0f / 16777216.0f);
    output[idx] = fmaf(uniform, maxVal - minVal, minVal);
}

extern ""C"" __global__ __launch_bounds__(256) void generate_random_normal(
    float* __restrict__ output,
    int size, float mean, float stdDev, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Two independent Philox streams for Box-Muller, 10 rounds each
    unsigned int key = (unsigned int)(seed ^ (seed >> 32));

    unsigned int c1 = (unsigned int)(2 * (unsigned int)idx);
    for (int r = 0; r < 10; r++) c1 = philox_round(c1, key + (unsigned int)r);
    unsigned int c2 = (unsigned int)(2 * (unsigned int)idx + 1u);
    for (int r = 0; r < 10; r++) c2 = philox_round(c2, key + 10u + (unsigned int)r);

    float u1 = uint_to_float01(c1);
    float u2 = uint_to_float01(c2);

    // Box-Muller transform
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    output[idx] = mean + stdDev * z;
}

// ============================================================================
// 2:4 Structured Sparsity (Ampere+)
// Enforce 2:4 pattern: in each group of 4, keep 2 largest magnitude values
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void enforce_2x4_sparsity(
    const float* __restrict__ denseInput,
    float* __restrict__ sparseValues,
    unsigned char* __restrict__ sparseIndices,
    int M, int K)
{
    // Each thread handles one group of 4 elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalGroups = M * (K / 4);
    if (idx >= totalGroups) return;

    int row = idx / (K / 4);
    int group = idx % (K / 4);
    int baseCol = group * 4;

    // Load 4 elements
    float vals[4];
    float absVals[4];
    for (int i = 0; i < 4; i++) {
        vals[i] = denseInput[row * K + baseCol + i];
        absVals[i] = fabsf(vals[i]);
    }

    // Find the 2 largest by magnitude using sorting network
    int indices[4] = {0, 1, 2, 3};
    // Bubble sort by abs value descending (only need top 2)
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < 4; j++) {
            if (absVals[indices[j]] > absVals[indices[i]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    // Output: 2 values and one packed byte per group.
    int outIdx = idx * 2;
    // Sort the kept indices ascending for consistent ordering
    int k0 = indices[0], k1 = indices[1];
    if (k0 > k1) { int tmp = k0; k0 = k1; k1 = tmp; }

    sparseValues[outIdx] = vals[k0];
    sparseValues[outIdx + 1] = vals[k1];
    sparseIndices[idx] = (unsigned char)((k1 << 2) | k0);
}

extern ""C"" __global__ __launch_bounds__(256) void decompress_2x4_sparse(
    const float* __restrict__ sparseValues,
    const unsigned char* __restrict__ sparseIndices,
    float* __restrict__ denseOutput,
    int M, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int groupsPerRow = K / 4;
    int totalGroups = M * groupsPerRow;
    if (idx >= totalGroups) return;
    int row = idx / groupsPerRow;
    int group = idx % groupsPerRow;
    int denseBase = row * K + group * 4;
    int sparseBase = idx * 2;
    unsigned char packed = sparseIndices[idx];
    int k0 = packed & 3;
    int k1 = (packed >> 2) & 3;
    denseOutput[denseBase] = 0.0f;
    denseOutput[denseBase + 1] = 0.0f;
    denseOutput[denseBase + 2] = 0.0f;
    denseOutput[denseBase + 3] = 0.0f;
    denseOutput[denseBase + k0] = sparseValues[sparseBase];
    denseOutput[denseBase + k1] = sparseValues[sparseBase + 1];
}

// ============================================================================
// Sparse GEMM: C = sparse(A) * B where A is in 2:4 format
// Uses the sparse structure to skip zero multiplications
// ============================================================================

extern ""C"" __global__ __launch_bounds__(256) void sparse_gemm_2x4(
    const float* __restrict__ sparseAValues,
    const unsigned char* __restrict__ sparseAIndices,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int numGroups = K / 4;

    for (int g = 0; g < numGroups; g++) {
        int sparseBase = (row * numGroups + g) * 2;
        unsigned char packed = sparseAIndices[row * numGroups + g];
        int first = packed & 3;
        int second = (packed >> 2) & 3;
        sum += sparseAValues[sparseBase] * B[(g * 4 + first) * N + col];
        sum += sparseAValues[sparseBase + 1] * B[(g * 4 + second) * N + col];
    }

    int cIdx = row * N + col;
    C[cIdx] = alpha * sum + beta * C[cIdx];
}

extern ""C"" __global__ __launch_bounds__(256) void sparse_gemm_bias_relu(
    const float* __restrict__ sparseAValues,
    const unsigned char* __restrict__ sparseAIndices,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int numGroups = K / 4;

    for (int g = 0; g < numGroups; g++) {
        int sparseBase = (row * numGroups + g) * 2;
        unsigned char packed = sparseAIndices[row * numGroups + g];
        int first = packed & 3;
        int second = (packed >> 2) & 3;
        sum += sparseAValues[sparseBase] * B[(g * 4 + first) * N + col];
        sum += sparseAValues[sparseBase + 1] * B[(g * 4 + second) * N + col];
    }

    // Add bias and ReLU
    sum += bias[col];
    C[row * N + col] = fmaxf(sum, 0.0f);
}
";
    }

    public static string[] GetKernelNames()
    {
        return
        [
            "stdp_update",
            "update_traces",
            "rbf_forward",
            "generate_random_uniform",
            "generate_random_normal",
            "enforce_2x4_sparsity",
            "decompress_2x4_sparse",
            "sparse_gemm_2x4",
            "sparse_gemm_bias_relu"
        ];
    }
}
