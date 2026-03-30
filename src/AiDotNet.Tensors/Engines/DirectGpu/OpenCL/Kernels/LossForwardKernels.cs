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
__kernel void l1_loss(__global const float* predictions, __global const float* targets, __global float* loss, int batchSize, int numFeatures) {
    int b = get_global_id(0); if (b >= batchSize) return;
    float sum_abs = 0.0f;
    for (int f = 0; f < numFeatures; f++) sum_abs += fabs(predictions[b * numFeatures + f] - targets[b * numFeatures + f]);
    loss[b] = sum_abs / (float)numFeatures;
}
__kernel void huber_loss(__global const float* predictions, __global const float* targets, __global float* loss, int batchSize, int numFeatures, float delta) {
    int b = get_global_id(0); if (b >= batchSize) return;
    float sum = 0.0f;
    for (int f = 0; f < numFeatures; f++) {
        float diff = predictions[b * numFeatures + f] - targets[b * numFeatures + f];
        float ad = fabs(diff);
        sum += (ad <= delta) ? (0.5f * diff * diff) : (delta * (ad - 0.5f * delta));
    }
    loss[b] = sum / (float)numFeatures;
}
__kernel void bce_with_logits_loss(__global const float* logits, __global const float* targets, __global float* loss, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float x = logits[idx]; float t = targets[idx]; float ax = fabs(x);
    loss[idx] = fmax(x, 0.0f) - x * t + log1p(exp(-ax));
}
__kernel void nll_loss(__global const float* logProbs, __global const int* targets, __global float* loss, int batchSize, int numClasses) {
    int b = get_global_id(0); if (b >= batchSize) return;
    int tc = targets[b];
    loss[b] = (tc >= 0 && tc < numClasses) ? -logProbs[b * numClasses + tc] : 0.0f;
}
__kernel void kl_div_loss(__global const float* input, __global const float* target, __global float* loss, int size) {
    int idx = get_global_id(0); if (idx >= size) return;
    float t = target[idx];
    loss[idx] = (t > 0.0f) ? t * (log(t) - input[idx]) : 0.0f;
}
__kernel void mse_loss_backward(__global const float* gradOutput, __global const float* predictions, __global const float* targets, __global float* gradInput, int size, float inv_n) {
    int idx = get_global_id(0); if (idx >= size) return;
    gradInput[idx] = gradOutput[0] * 2.0f * (predictions[idx] - targets[idx]) * inv_n;
}
__kernel void l1_loss_backward(__global const float* gradOutput, __global const float* predictions, __global const float* targets, __global float* gradInput, int size, float inv_n) {
    int idx = get_global_id(0); if (idx >= size) return;
    float diff = predictions[idx] - targets[idx];
    float s = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
    gradInput[idx] = gradOutput[0] * s * inv_n;
}
__kernel void huber_loss_backward(__global const float* gradOutput, __global const float* predictions, __global const float* targets, __global float* gradInput, int size, float inv_n, float delta) {
    int idx = get_global_id(0); if (idx >= size) return;
    float diff = predictions[idx] - targets[idx]; float ad = fabs(diff);
    float grad = (ad <= delta) ? diff : (delta * ((diff > 0.0f) ? 1.0f : -1.0f));
    gradInput[idx] = gradOutput[0] * grad * inv_n;
}
__kernel void bce_with_logits_backward(__global const float* gradOutput, __global const float* logits, __global const float* targets, __global float* gradInput, int size, float inv_n) {
    int idx = get_global_id(0); if (idx >= size) return;
    float sig = 1.0f / (1.0f + exp(-logits[idx]));
    gradInput[idx] = gradOutput[0] * (sig - targets[idx]) * inv_n;
}
";
    }

    public static string[] GetKernelNames()
    {
        return new[] { "cross_entropy_loss", "mse_loss", "bce_loss", "dropout_mask", "gaussian_noise",
            "l1_loss", "huber_loss", "bce_with_logits_loss", "nll_loss", "kl_div_loss",
            "mse_loss_backward", "l1_loss_backward", "huber_loss_backward", "bce_with_logits_backward" };
    }
}
