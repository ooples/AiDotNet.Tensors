namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for OpenClBackend lookups that no OpenCL program defined, so each op threw on the device and ran on
/// the CPU (see KernelLookupIntegrityTests). Each follows its CUDA counterpart (CudaMissingLookupKernels,
/// CudaSnnKernels) with the OpenCL launch geometry of its call site: the CSR reductions are Execute2D(N, M), so j is
/// dimension 0 and the row dimension 1; rbf_forward is Execute2D(batch, centers); stdp_update is Execute2D(pre, post).
/// </summary>
internal static class OpenClMissingLookupKernels
{
    public static string GetSource()
    {
        return @"
// Column/row-pointer buffers are float-typed and hold integral values, as in the other CSR kernels. An empty row
// writes 0 for every reduction.
__kernel void csr_segmented_max(__global const float* colIndices, __global const float* rowPointers,
    __global const float* input, __global float* output, int M, int K, int N)
{
    int j = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || j >= N) return;
    int start = (int)rowPointers[row];
    int end = (int)rowPointers[row + 1];
    int idx = row * N + j;
    if (start >= end) { output[idx] = 0.0f; return; }
    float maxVal = -3.402823e+38f;
    for (int i = start; i < end; i++) maxVal = fmax(maxVal, input[(int)colIndices[i] * N + j]);
    output[idx] = maxVal;
}

__kernel void csr_segmented_min(__global const float* colIndices, __global const float* rowPointers,
    __global const float* input, __global float* output, int M, int K, int N)
{
    int j = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || j >= N) return;
    int start = (int)rowPointers[row];
    int end = (int)rowPointers[row + 1];
    int idx = row * N + j;
    if (start >= end) { output[idx] = 0.0f; return; }
    float minVal = 3.402823e+38f;
    for (int i = start; i < end; i++) minVal = fmin(minVal, input[(int)colIndices[i] * N + j]);
    output[idx] = minVal;
}

// Two-pass mean then variance; sqrt(variance + epsilon).
__kernel void csr_segmented_stddev(__global const float* colIndices, __global const float* rowPointers,
    __global const float* input, __global float* output, int M, int K, int N, float epsilon)
{
    int j = get_global_id(0);
    int row = get_global_id(1);
    if (row >= M || j >= N) return;
    int start = (int)rowPointers[row];
    int end = (int)rowPointers[row + 1];
    int idx = row * N + j;
    int count = end - start;
    if (count <= 0) { output[idx] = 0.0f; return; }
    float mean = 0.0f;
    for (int i = start; i < end; i++) mean += input[(int)colIndices[i] * N + j];
    mean /= (float)count;
    float varAcc = 0.0f;
    for (int i = start; i < end; i++)
    {
        float d = input[(int)colIndices[i] * N + j] - mean;
        varAcc += d * d;
    }
    output[idx] = sqrt(varAcc / (float)count + epsilon);
}

// output[b, c] = exp(-epsilons[c] * ||input[b] - centers[c]||^2)
__kernel void rbf_forward(__global const float* input, __global const float* centers, __global const float* epsilons,
    __global float* output, int batchSize, int numCenters, int inputDim)
{
    int b = get_global_id(0);
    int c = get_global_id(1);
    if (b >= batchSize || c >= numCenters) return;
    float distSq = 0.0f;
    for (int d = 0; d < inputDim; d++)
    {
        float diff = input[b * inputDim + d] - centers[c * inputDim + d];
        distSq += diff * diff;
    }
    output[b * numCenters + c] = exp(-epsilons[c] * distSq);
}

// Exponential trace decay; a spike (input > threshold) sets spikes to 1 and adds 1 to the trace.
__kernel void update_traces(__global float* traces, __global float* spikes, __global const float* input,
    float decay, float threshold, int size)
{
    int idx = get_global_id(0);
    if (idx >= size) return;
    float trace = decay * traces[idx];
    if (input[idx] > threshold) { spikes[idx] = 1.0f; trace += 1.0f; }
    else { spikes[idx] = 0.0f; }
    traces[idx] = trace;
}

// Pair-based STDP with homeostasis toward the weight range's centre, clamped to [minWeight, maxWeight].
__kernel void stdp_update(__global float* weights, __global const float* preTrace, __global const float* postTrace,
    __global const float* preSpike, __global const float* postSpike,
    float ltpRate, float ltdRate, float homeostasisRate, float minWeight, float maxWeight, int numPre, int numPost)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= numPre || j >= numPost) return;
    int idx = i * numPost + j;
    float w = weights[idx];
    float dw = 0.0f;
    if (preSpike[i] > 0.5f && postTrace[j] > 0.0f) dw += ltpRate * postTrace[j];
    if (postSpike[j] > 0.5f && preTrace[i] > 0.0f) dw -= ltdRate * preTrace[i];
    float center = (minWeight + maxWeight) * 0.5f;
    dw += homeostasisRate * (center - w);
    w += dw;
    weights[idx] = fmin(fmax(w, minWeight), maxWeight);
}
";
    }

    public static string[] GetKernelNames() =>
    [
        "csr_segmented_max", "csr_segmented_min", "csr_segmented_stddev", "rbf_forward", "update_traces", "stdp_update",
    ];
}