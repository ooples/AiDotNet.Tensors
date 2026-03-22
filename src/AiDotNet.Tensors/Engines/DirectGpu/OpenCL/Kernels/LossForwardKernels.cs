namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL loss function forward kernels + dropout/noise generation.
/// </summary>
public static class LossForwardKernels
{
    public static string GetSource()
    {
        return @"
__kernel void cross_entropy_loss(__global const float* predictions, __global const float* targets, __global float* loss, int batchSize, int numClasses) {
    int b = get_global_id(0); if (b >= batchSize) return;
    float sample_loss = 0.0f;
    for (int c = 0; c < numClasses; c++) {
        float target = targets[b * numClasses + c];
        if (target > 0.0f) sample_loss -= target * log(fmax(predictions[b * numClasses + c], 1e-7f));
    }
    loss[b] = sample_loss;
}
__kernel void mse_loss(__global const float* predictions, __global const float* targets, __global float* loss, int batchSize, int numFeatures) {
    int b = get_global_id(0); if (b >= batchSize) return;
    float sum_sq = 0.0f;
    for (int f = 0; f < numFeatures; f++) { float d = predictions[b * numFeatures + f] - targets[b * numFeatures + f]; sum_sq += d * d; }
    loss[b] = (numFeatures > 0) ? (sum_sq / (float)numFeatures) : 0.0f;
}
__kernel void bce_loss(__global const float* predictions, __global const float* targets, __global float* loss, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float p = fmin(fmax(predictions[idx], 1e-7f), 1.0f - 1e-7f); float t = targets[idx];
    loss[idx] = -(t * log(p) + (1.0f - t) * log(1.0f - p));
}
__kernel void dropout_mask(__global float* mask, int size, float keepProb, ulong seed) {
    int idx = get_global_id(0); if (idx >= size) return;
    uint counter = (uint)idx; uint key = (uint)(seed ^ (seed >> 32));
    ulong prod = (ulong)counter * 0xCD9E8D57u;
    counter = (uint)(prod >> 32) ^ key ^ (uint)prod;
    prod = (ulong)counter * 0xCD9E8D57u;
    counter = (uint)(prod >> 32) ^ (key + 1u) ^ (uint)prod;
    float u = (float)(counter >> 8) * (1.0f / 16777216.0f);
    float safe_keep = fmax(keepProb, 1e-7f);
    mask[idx] = (u < keepProb) ? (1.0f / safe_keep) : 0.0f;
}
__kernel void gaussian_noise(__global float* output, int size, float mean, float stdDev, ulong seed) {
    int idx = get_global_id(0); if (idx >= size) return;
    uint key = (uint)(seed ^ (seed >> 32));
    uint c1 = (uint)(2 * idx); ulong p1 = (ulong)c1 * 0xCD9E8D57u;
    c1 = (uint)(p1 >> 32) ^ key ^ (uint)p1; p1 = (ulong)c1 * 0xCD9E8D57u;
    c1 = (uint)(p1 >> 32) ^ (key + 1u) ^ (uint)p1;
    uint c2 = (uint)(2 * idx + 1); ulong p2 = (ulong)c2 * 0xCD9E8D57u;
    c2 = (uint)(p2 >> 32) ^ (key + 2u) ^ (uint)p2; p2 = (ulong)c2 * 0xCD9E8D57u;
    c2 = (uint)(p2 >> 32) ^ (key + 3u) ^ (uint)p2;
    float u1 = (float)(c1 >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);
    float u2 = (float)(c2 >> 8) * (1.0f / 16777216.0f);
    output[idx] = mean + stdDev * sqrt(-2.0f * log(u1)) * cos(2.0f * 3.14159265358979323846f * u2);
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[] { "cross_entropy_loss", "mse_loss", "bce_loss", "dropout_mask", "gaussian_noise" };
    }
}
