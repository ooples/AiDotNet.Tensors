// Copyright (c) AiDotNet. All rights reserved.
// Normalization kernels for neural network layers.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels
{
    /// <summary>
    /// GPU kernels for normalization operations.
    /// </summary>
    internal static class NormalizationKernels
    {
        /// <summary>
        /// Gets all normalization kernel sources.
        /// </summary>
        public static string GetSource()
        {
            return @"
// ===========================================================================
// NORMALIZATION KERNELS
// ===========================================================================

// Batch Normalization forward pass — workgroup-parallel
// 1 workgroup per channel, threads cooperate on batch*spatial reduction
__kernel void batchnorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global float* runningMean,
    __global float* runningVar,
    __global float* saveMean,
    __global float* saveInvVar,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon,
    const float momentum,
    const int training,
    __local float* localBuf)
{
    const int c = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (c >= channels) return;

    int batchSpatial = batch * spatialSize;
    float mean, invVar;

    if (training) {
        // Training: compute mean/var from batch data
        float threadSum = 0.0f;
        for (int i = lid; i < batchSpatial; i += localSize) {
            int b = i / spatialSize;
            int s = i % spatialSize;
            threadSum += input[(b * channels + c) * spatialSize + s];
        }
        localBuf[lid] = threadSum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) localBuf[lid] += localBuf[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        mean = localBuf[0] / (float)batchSpatial;
        barrier(CLK_LOCAL_MEM_FENCE);

        float threadVar = 0.0f;
        for (int i = lid; i < batchSpatial; i += localSize) {
            int b = i / spatialSize;
            int s = i % spatialSize;
            float diff = input[(b * channels + c) * spatialSize + s] - mean;
            threadVar += diff * diff;
        }
        localBuf[lid] = threadVar;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) localBuf[lid] += localBuf[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float var = localBuf[0] / (float)batchSpatial;
        invVar = 1.0f / sqrt(var + epsilon);

        if (lid == 0) {
            saveMean[c] = mean;
            saveInvVar[c] = invVar;
            runningMean[c] = (1.0f - momentum) * runningMean[c] + momentum * mean;
            runningVar[c] = (1.0f - momentum) * runningVar[c] + momentum * var;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
        // Inference: use running statistics
        mean = runningMean[c];
        invVar = 1.0f / sqrt(runningVar[c] + epsilon);
    }

    // Normalize and apply affine transform
    float g = gamma[c];
    float b_val = beta[c];
    for (int i = lid; i < batchSpatial; i += localSize) {
        int b = i / spatialSize;
        int s = i % spatialSize;
        int idx = (b * channels + c) * spatialSize + s;
        float normalized = (input[idx] - mean) * invVar;
        output[idx] = g * normalized + b_val;
    }
}

// Batch Normalization backward pass — workgroup-parallel
__kernel void batchnorm_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* gamma,
    __global const float* saveMean,
    __global const float* saveInvVar,
    __global float* gradInput,
    __global float* gradGamma,
    __global float* gradBeta,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon,
    __local float* localBuf,
    __local float* localBuf2)
{
    const int c = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (c >= channels) return;

    int batchSpatial = batch * spatialSize;
    float mean = saveMean[c];
    float invVar = saveInvVar[c];
    float g = gamma[c];

    // Phase 1: Parallel reduction for gradGamma, gradBeta, and gamma-scaled sums
    float tDGamma = 0.0f;
    float tDBeta = 0.0f;
    float tSumDxhat = 0.0f;
    float tSumDxhatXhat = 0.0f;
    for (int i = lid; i < batchSpatial; i += localSize) {
        int b = i / spatialSize;
        int s = i % spatialSize;
        int idx = (b * channels + c) * spatialSize + s;
        float xhat = (input[idx] - mean) * invVar;
        float dxhat = gradOutput[idx] * g;
        tDGamma += gradOutput[idx] * xhat;
        tDBeta += gradOutput[idx];
        tSumDxhat += dxhat;
        tSumDxhatXhat += dxhat * xhat;
    }
    localBuf[lid] = tDGamma;
    localBuf2[lid] = tDBeta;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            localBuf[lid] += localBuf[lid + stride];
            localBuf2[lid] += localBuf2[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        gradGamma[c] = localBuf[0];
        gradBeta[c] = localBuf2[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Reduce sum(dxhat) and sum(dxhat * xhat)
    localBuf[lid] = tSumDxhat;
    localBuf2[lid] = tSumDxhatXhat;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            localBuf[lid] += localBuf[lid + stride];
            localBuf2[lid] += localBuf2[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sumDxhat = localBuf[0];
    float sumDxhatXhat = localBuf2[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: Compute gradInput
    float invN = 1.0f / (float)batchSpatial;
    for (int i = lid; i < batchSpatial; i += localSize) {
        int b = i / spatialSize;
        int s = i % spatialSize;
        int idx = (b * channels + c) * spatialSize + s;
        float xhat = (input[idx] - mean) * invVar;
        float dxhat = gradOutput[idx] * g;
        gradInput[idx] = invVar * (dxhat - invN * (sumDxhat + xhat * sumDxhatXhat));
    }
}

// Layer Normalization forward pass — workgroup-parallel
// 1 workgroup per batch element, threads cooperate on mean/var reduction
__kernel void layernorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global float* saveMean,
    __global float* saveInvVar,
    const int batchSize,
    const int normalizedSize,
    const float epsilon,
    __local float* localBuf)
{
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (b >= batchSize) return;

    __global const float* rowIn = input + b * normalizedSize;

    // Phase 1: Parallel sum for mean
    float threadSum = 0.0f;
    for (int i = lid; i < normalizedSize; i += localSize) {
        threadSum += rowIn[i];
    }
    localBuf[lid] = threadSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) localBuf[lid] += localBuf[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean = localBuf[0] / (float)normalizedSize;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Parallel sum for variance
    float threadVar = 0.0f;
    for (int i = lid; i < normalizedSize; i += localSize) {
        float diff = rowIn[i] - mean;
        threadVar += diff * diff;
    }
    localBuf[lid] = threadVar;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) localBuf[lid] += localBuf[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float var = localBuf[0] / (float)normalizedSize;
    float invVar = 1.0f / sqrt(var + epsilon);

    // Save stats (only thread 0)
    if (lid == 0) {
        saveMean[b] = mean;
        saveInvVar[b] = invVar;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: Normalize and apply affine transform
    __global float* rowOut = output + b * normalizedSize;
    for (int i = lid; i < normalizedSize; i += localSize) {
        float normalized = (rowIn[i] - mean) * invVar;
        rowOut[i] = gamma[i] * normalized + beta[i];
    }
}

// Layer Normalization backward pass — workgroup-parallel
__kernel void layernorm_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* gamma,
    __global const float* saveMean,
    __global const float* saveInvVar,
    __global float* gradInput,
    __global float* gradGamma,
    __global float* gradBeta,
    const int batchSize,
    const int normalizedSize,
    const float epsilon,
    __local float* localBuf,
    __local float* localBuf2)
{
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (b >= batchSize) return;

    float mean = saveMean[b];
    float invVar = saveInvVar[b];
    __global const float* rowIn = input + b * normalizedSize;
    __global const float* rowGrad = gradOutput + b * normalizedSize;

    // Phase 1: Parallel reduction for sumDy and sumDyXmu
    float tSumDy = 0.0f;
    float tSumDyXmu = 0.0f;
    for (int i = lid; i < normalizedSize; i += localSize) {
        float dy = rowGrad[i] * gamma[i];
        tSumDy += dy;
        tSumDyXmu += dy * (rowIn[i] - mean);
    }
    localBuf[lid] = tSumDy;
    localBuf2[lid] = tSumDyXmu;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            localBuf[lid] += localBuf[lid + stride];
            localBuf2[lid] += localBuf2[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sumDy = localBuf[0];
    float sumDyXmu = localBuf2[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute gradInput
    __global float* rowGradIn = gradInput + b * normalizedSize;
    for (int i = lid; i < normalizedSize; i += localSize) {
        float xmu = rowIn[i] - mean;
        float dxhat = rowGrad[i] * gamma[i];
        rowGradIn[i] = invVar * (dxhat - (sumDy + xmu * invVar * invVar * sumDyXmu) / (float)normalizedSize);
    }
}

// Layer Normalization gradient accumulation for gamma and beta
__kernel void layernorm_grad_params(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* saveMean,
    __global const float* saveInvVar,
    __global float* gradGamma,
    __global float* gradBeta,
    const int batchSize,
    const int normalizedSize)
{
    const int i = get_global_id(0);
    if (i >= normalizedSize) return;

    float dGamma = 0.0f;
    float dBeta = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        int idx = b * normalizedSize + i;
        float mean = saveMean[b];
        float invVar = saveInvVar[b];
        float normalized = (input[idx] - mean) * invVar;
        dGamma += gradOutput[idx] * normalized;
        dBeta += gradOutput[idx];
    }
    gradGamma[i] = dGamma;
    gradBeta[i] = dBeta;
}

// Group Normalization forward pass
__kernel void groupnorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global float* saveMean,
    __global float* saveInvVar,
    const int batch,
    const int numGroups,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int idx = get_global_id(0);
    const int g = idx % numGroups;
    const int b = idx / numGroups;

    if (b >= batch) return;

    int channelsPerGroup = channels / numGroups;
    int groupSize = channelsPerGroup * spatialSize;

    // Compute mean for this group
    float mean = 0.0f;
    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++) {
        for (int s = 0; s < spatialSize; s++) {
            mean += input[(b * channels + c) * spatialSize + s];
        }
    }
    mean /= (float)groupSize;

    // Compute variance
    float var = 0.0f;
    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++) {
        for (int s = 0; s < spatialSize; s++) {
            float diff = input[(b * channels + c) * spatialSize + s] - mean;
            var += diff * diff;
        }
    }
    var /= (float)groupSize;

    float invVar = 1.0f / sqrt(var + epsilon);

    int saveIdx = b * numGroups + g;
    saveMean[saveIdx] = mean;
    saveInvVar[saveIdx] = invVar;

    // Normalize and apply affine transform
    for (int c = g * channelsPerGroup; c < (g + 1) * channelsPerGroup; c++) {
        for (int s = 0; s < spatialSize; s++) {
            int inputIdx = (b * channels + c) * spatialSize + s;
            float normalized = (input[inputIdx] - mean) * invVar;
            output[inputIdx] = gamma[c] * normalized + beta[c];
        }
    }
}

// Instance Normalization forward pass
__kernel void instancenorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global const float* beta,
    __global float* saveMean,
    __global float* saveInvVar,
    const int batch,
    const int channels,
    const int spatialSize,
    const float epsilon)
{
    const int idx = get_global_id(0);
    const int c = idx % channels;
    const int b = idx / channels;

    if (b >= batch) return;

    // Compute mean for this instance
    float mean = 0.0f;
    for (int s = 0; s < spatialSize; s++) {
        mean += input[(b * channels + c) * spatialSize + s];
    }
    mean /= (float)spatialSize;

    // Compute variance
    float var = 0.0f;
    for (int s = 0; s < spatialSize; s++) {
        float diff = input[(b * channels + c) * spatialSize + s] - mean;
        var += diff * diff;
    }
    var /= (float)spatialSize;

    float invVar = 1.0f / sqrt(var + epsilon);

    int saveIdx = b * channels + c;
    saveMean[saveIdx] = mean;
    saveInvVar[saveIdx] = invVar;

    // Normalize and apply affine transform
    float g = gamma[c];
    float bt = beta[c];
    for (int s = 0; s < spatialSize; s++) {
        int inputIdx = (b * channels + c) * spatialSize + s;
        float normalized = (input[inputIdx] - mean) * invVar;
        output[inputIdx] = g * normalized + bt;
    }
}

// RMS Normalization forward pass — workgroup-parallel
// 1 workgroup per batch element, threads cooperate on sum-of-squares reduction
__kernel void rmsnorm_forward(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    __global float* saveRms,
    const int batchSize,
    const int normalizedSize,
    const float epsilon,
    __local float* localBuf)
{
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (b >= batchSize) return;

    __global const float* rowIn = input + b * normalizedSize;

    // Phase 1: Parallel sum of squares
    float threadSumSq = 0.0f;
    for (int i = lid; i < normalizedSize; i += localSize) {
        float x = rowIn[i];
        threadSumSq += x * x;
    }
    localBuf[lid] = threadSumSq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) localBuf[lid] += localBuf[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float meanSq = localBuf[0] / (float)normalizedSize;
    float rms = sqrt(meanSq + epsilon);
    float invRms = 1.0f / rms;

    if (lid == 0) {
        saveRms[b] = rms;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Normalize and scale
    __global float* rowOut = output + b * normalizedSize;
    for (int i = lid; i < normalizedSize; i += localSize) {
        rowOut[i] = rowIn[i] * invRms * gamma[i];
    }
}

// RMS Normalization backward pass — workgroup-parallel
__kernel void rmsnorm_backward(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* gamma,
    __global const float* saveRms,
    __global float* gradInput,
    __global float* gradGamma,
    const int batchSize,
    const int normalizedSize,
    const float epsilon,
    __local float* localBuf)
{
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    const int localSize = get_local_size(0);

    if (b >= batchSize) return;

    float rms = saveRms[b];
    float invRms = 1.0f / rms;
    float invRms3 = invRms * invRms * invRms;

    __global const float* rowIn = input + b * normalizedSize;
    __global const float* rowGrad = gradOutput + b * normalizedSize;

    // Phase 1: Parallel reduction for sumGradGammaX
    float tSum = 0.0f;
    for (int i = lid; i < normalizedSize; i += localSize) {
        tSum += rowGrad[i] * gamma[i] * rowIn[i];
    }
    localBuf[lid] = tSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = localSize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) localBuf[lid] += localBuf[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sumGradGammaX = localBuf[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute gradInput
    __global float* rowGradIn = gradInput + b * normalizedSize;
    for (int i = lid; i < normalizedSize; i += localSize) {
        float x = rowIn[i];
        float dy = rowGrad[i];
        float g = gamma[i];
        rowGradIn[i] = g * dy * invRms - x * sumGradGammaX * invRms3 / (float)normalizedSize;
    }
}

// RMS Normalization gradient accumulation for gamma
__kernel void rmsnorm_grad_gamma(
    __global const float* gradOutput,
    __global const float* input,
    __global const float* saveRms,
    __global float* gradGamma,
    const int batchSize,
    const int normalizedSize)
{
    const int i = get_global_id(0);
    if (i >= normalizedSize) return;

    float dGamma = 0.0f;
    for (int b = 0; b < batchSize; b++) {
        int idx = b * normalizedSize + i;
        float rms = saveRms[b];
        float invRms = 1.0f / rms;
        dGamma += gradOutput[idx] * input[idx] * invRms;
    }
    gradGamma[i] = dGamma;
}

// Scatter-Add backward pass (essentially a gather)
// gradSource[i] = gradDestination[indices[i]]
__kernel void scatter_add_backward(
    __global const float* gradDestination,
    __global const int* indices,
    __global float* gradSource,
    const int numIndices,
    const int featureSize)
{
    const int i = get_global_id(0);
    if (i >= numIndices) return;

    int srcIdx = indices[i];
    for (int f = 0; f < featureSize; f++) {
        gradSource[i * featureSize + f] = gradDestination[srcIdx * featureSize + f];
    }
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
                "batchnorm_forward",
                "batchnorm_backward",
                "layernorm_forward",
                "layernorm_backward",
                "layernorm_grad_params",
                "groupnorm_forward",
                "instancenorm_forward",
                "rmsnorm_forward",
                "rmsnorm_backward",
                "rmsnorm_grad_gamma",
                "scatter_add_backward"
            };
        }
    }
}
