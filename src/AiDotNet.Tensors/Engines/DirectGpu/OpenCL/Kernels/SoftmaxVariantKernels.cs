namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL softmax variants, GEMM extensions, distance/similarity kernels.
/// </summary>
public static class SoftmaxVariantKernels
{
    public static string GetSource()
    {
        return @"
__kernel void log_softmax(__global const float* input, __global float* output, int outerSize, int innerSize) {
    int row = get_global_id(0); if (row >= outerSize) return;
    int base_idx = row * innerSize;
    float max_val = -INFINITY;
    for (int j = 0; j < innerSize; j++) max_val = fmax(max_val, input[base_idx + j]);
    float sum_exp = 0.0f;
    for (int j = 0; j < innerSize; j++) sum_exp += exp(input[base_idx + j] - max_val);
    float log_sum = log(sum_exp);
    for (int j = 0; j < innerSize; j++) output[base_idx + j] = input[base_idx + j] - max_val - log_sum;
}
__kernel void gumbel_softmax(__global const float* logits, __global float* output, int outerSize, int innerSize, float temperature, ulong seed) {
    int row = get_global_id(0); if (row >= outerSize) return;
    int base_idx = row * innerSize;
    float max_val = -INFINITY;
    for (int j = 0; j < innerSize; j++) {
        uint counter = (uint)(row * innerSize + j); uint key = (uint)(seed ^ (seed >> 32));
        ulong prod = (ulong)counter * 0xCD9E8D57u; counter = (uint)(prod >> 32) ^ key ^ (uint)prod;
        prod = (ulong)counter * 0xCD9E8D57u; counter = (uint)(prod >> 32) ^ (key + 1u) ^ (uint)prod;
        float u = (float)(counter >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
        float perturbed = (logits[base_idx + j] + (-log(-log(u)))) / temperature;
        output[base_idx + j] = perturbed; max_val = fmax(max_val, perturbed);
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < innerSize; j++) { output[base_idx + j] = exp(output[base_idx + j] - max_val); sum_exp += output[base_idx + j]; }
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (int j = 0; j < innerSize; j++) output[base_idx + j] *= inv_sum;
}
__kernel void taylor_softmax(__global const float* input, __global float* output, int outerSize, int innerSize) {
    int row = get_global_id(0); if (row >= outerSize) return;
    int base_idx = row * innerSize;
    float sum = 0.0f;
    for (int j = 0; j < innerSize; j++) {
        float x = input[base_idx + j]; float t = fmax(1.0f + x + 0.5f * x * x, 1e-10f);
        output[base_idx + j] = t; sum += t;
    }
    float inv = 1.0f / sum;
    for (int j = 0; j < innerSize; j++) output[base_idx + j] *= inv;
}
__kernel void spherical_softmax(__global const float* input, __global float* output, int outerSize, int innerSize) {
    int row = get_global_id(0); if (row >= outerSize) return;
    int base_idx = row * innerSize;
    float norm_sq = 0.0f;
    for (int j = 0; j < innerSize; j++) norm_sq += input[base_idx + j] * input[base_idx + j];
    float norm = sqrt(norm_sq + 1e-12f);
    float max_val = -INFINITY;
    for (int j = 0; j < innerSize; j++) { float v = input[base_idx + j] / norm; output[base_idx + j] = v; max_val = fmax(max_val, v); }
    float sum_exp = 0.0f;
    for (int j = 0; j < innerSize; j++) { output[base_idx + j] = exp(output[base_idx + j] - max_val); sum_exp += output[base_idx + j]; }
    float inv = 1.0f / (sum_exp + 1e-10f);
    for (int j = 0; j < innerSize; j++) output[base_idx + j] *= inv;
}
__kernel void batch_dot_product(__global const float* a, __global const float* b, __global float* output, int batchSize, int dim) {
    int batch = get_global_id(0); if (batch >= batchSize) return;
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += a[batch * dim + i] * b[batch * dim + i];
    output[batch] = sum;
}
__kernel void outer_product(__global const float* a, __global const float* b, __global float* output, int M, int N) {
    int idx = get_global_id(0); if (idx >= M * N) return;
    output[idx] = a[idx / N] * b[idx % N];
}
__kernel void batch_outer_product(__global const float* a, __global const float* b, __global float* output, int batchSize, int M, int N) {
    int idx = get_global_id(0); if (idx >= batchSize * M * N) return;
    int mn = M * N; int batch = idx / mn; int inner = idx % mn;
    output[idx] = a[batch * M + inner / N] * b[batch * N + inner % N];
}
__kernel void cosine_similarity(__global const float* a, __global const float* b, __global float* output, int batchSize, int dim) {
    int batch = get_global_id(0); if (batch >= batchSize) return;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; i++) { float ai = a[batch*dim+i]; float bi = b[batch*dim+i]; dot += ai*bi; na += ai*ai; nb += bi*bi; }
    output[batch] = dot / (sqrt(na) * sqrt(nb) + 1e-8f);
}
__kernel void pairwise_distance(__global const float* a, __global const float* b, __global float* output, int M, int N, int dim) {
    int idx = get_global_id(0); if (idx >= M * N) return;
    int i = idx / N; int j = idx % N;
    float d = 0.0f;
    for (int k = 0; k < dim; k++) { float diff = a[i*dim+k] - b[j*dim+k]; d += diff*diff; }
    output[idx] = sqrt(d);
}
__kernel void pairwise_distance_squared(__global const float* a, __global const float* b, __global float* output, int M, int N, int dim) {
    int idx = get_global_id(0); if (idx >= M * N) return;
    int i = idx / N; int j = idx % N;
    float d = 0.0f;
    for (int k = 0; k < dim; k++) { float diff = a[i*dim+k] - b[j*dim+k]; d += diff*diff; }
    output[idx] = d;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[]
        {
            "log_softmax", "gumbel_softmax", "taylor_softmax", "spherical_softmax",
            "batch_dot_product", "outer_product", "batch_outer_product",
            "cosine_similarity", "pairwise_distance", "pairwise_distance_squared"
        };
    }
}
