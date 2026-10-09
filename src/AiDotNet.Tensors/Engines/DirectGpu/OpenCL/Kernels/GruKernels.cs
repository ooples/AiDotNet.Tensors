// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for GRU (Gated Recurrent Unit) sequence neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT training.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for GRU sequence operations used in recurrent neural networks.
/// Implements full forward and backward passes for GRU cells with 3 gates (reset, update, new).
/// These are sequence-level kernels that process all timesteps efficiently for BPTT.
/// </summary>
internal static class GruKernels
{
    public static string GetSource()
    {
        return @"
#define EPSILON 1e-15f

// ===========================================================================
// ACTIVATION FUNCTIONS
// ===========================================================================

inline float sigmoid_fn(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float sigmoid_derivative(float sigmoid_output) {
    return sigmoid_output * (1.0f - sigmoid_output);
}

inline float tanh_derivative(float tanh_output) {
    return 1.0f - tanh_output * tanh_output;
}

// ===========================================================================
// GRU CELL FORWARD KERNEL (Single Timestep)
// ===========================================================================
// Processes one GRU cell computation for a single timestep.
// Each thread handles one (batch, hidden) element.

__kernel void gru_cell_forward(
    __global const float* input,       // [batch, input_size]
    __global const float* prevH,       // [batch, hidden_size]
    __global const float* weightsIh,   // [3 * hidden_size, input_size] - input to hidden weights
    __global const float* weightsHh,   // [3 * hidden_size, hidden_size] - hidden to hidden weights
    __global const float* biasIh,      // [3 * hidden_size] - input to hidden bias
    __global const float* biasHh,      // [3 * hidden_size] - hidden to hidden bias
    __global float* newH,              // [batch, hidden_size]
    __global float* gateR,             // [batch, hidden_size] - reset gate cache
    __global float* gateZ,             // [batch, hidden_size] - update gate cache
    __global float* gateN,             // [batch, hidden_size] - new gate (candidate) cache
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Compute gate pre-activations
    // Gates order: reset(r), update(z), new(n)
    float sumR = biasIh[h] + biasHh[h];
    float sumZ = biasIh[hiddenSize + h] + biasHh[hiddenSize + h];
    float sumN_input = biasIh[2 * hiddenSize + h];
    float sumN_hidden = biasHh[2 * hiddenSize + h];

    // Input to hidden contribution
    for (int j = 0; j < inputSize; j++) {
        float inVal = input[b * inputSize + j];
        sumR += inVal * weightsIh[h * inputSize + j];
        sumZ += inVal * weightsIh[(hiddenSize + h) * inputSize + j];
        sumN_input += inVal * weightsIh[(2 * hiddenSize + h) * inputSize + j];
    }

    // Hidden to hidden contribution for r and z gates
    for (int j = 0; j < hiddenSize; j++) {
        float hVal = prevH[b * hiddenSize + j];
        sumR += hVal * weightsHh[h * hiddenSize + j];
        sumZ += hVal * weightsHh[(hiddenSize + h) * hiddenSize + j];
    }

    // Apply activations for r and z
    float r = sigmoid_fn(sumR);
    float z = sigmoid_fn(sumZ);

    // Hidden to hidden contribution for n gate (uses reset gate)
    for (int j = 0; j < hiddenSize; j++) {
        float hVal = prevH[b * hiddenSize + j];
        sumN_hidden += (r * hVal) * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
    }

    // New gate activation
    float n = tanh(sumN_input + sumN_hidden);

    // Hidden state update: h_new = (1 - z) * n + z * h_prev
    float prevHVal = prevH[gid];
    float newHVal = (1.0f - z) * n + z * prevHVal;

    // Store results
    newH[gid] = newHVal;
    gateR[gid] = r;
    gateZ[gid] = z;
    gateN[gid] = n;
}

// ===========================================================================
// GRU FORWARD SEQUENCE KERNEL
// ===========================================================================
// Processes the entire sequence in a single kernel launch.

// GRU forward over a whole sequence, PyTorch's formulation (reset applied after the hidden matmul):
//   r = sigmoid(W_ir x + b_ir + W_hr h + b_hr), z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
//   n = tanh(W_in x + b_in + r * (W_hn h + b_hn)), h' = (1 - z) * n + z * h
// One work-group per batch row; its work-items stride over the hidden units, so any hidden size runs and the barrier
// after each step makes the whole new state visible to the next. Writes go to allH[t + 1] while reads come from
// allH[t], so a step never overwrites state it is still reading. cacheGates is [T, B, 3, H]: r, z, W_hn h + b_hn.
__kernel void gru_forward_sequence(
    __global const float* input, __global const float* hInit,
    __global const float* weightsIh, __global const float* weightsHh,
    __global const float* biasIh, __global const float* biasHh,
    __global float* output, __global float* hFinal, __global float* allH, __global float* cacheGates,
    const int seqLen, const int batch, const int inputSize, const int hiddenSize)
{
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    if (b >= batch) return;
    const int H = hiddenSize, I = inputSize;
    const long rowH = (long)b * H;
    for (int j = lid; j < H; j += lsize) allH[rowH + j] = hInit[rowH + j];
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int t = 0; t < seqLen; t++)
    {
        __global const float* hPrev = allH + (long)t * batch * H + rowH;
        __global const float* x = input + ((long)t * batch + b) * I;
        __global float* gates = cacheGates + ((long)t * batch + b) * 3 * H;
        for (int j = lid; j < H; j += lsize)
        {
            float xr = biasIh[j], xz = biasIh[H + j], xn = biasIh[2 * H + j];
            for (int i = 0; i < I; i++)
            {
                float xi = x[i];
                xr += weightsIh[(long)j * I + i] * xi;
                xz += weightsIh[(long)(H + j) * I + i] * xi;
                xn += weightsIh[(long)(2 * H + j) * I + i] * xi;
            }
            float hr = biasHh[j], hz = biasHh[H + j], hn = biasHh[2 * H + j];
            for (int k = 0; k < H; k++)
            {
                float hk = hPrev[k];
                hr += weightsHh[(long)j * H + k] * hk;
                hz += weightsHh[(long)(H + j) * H + k] * hk;
                hn += weightsHh[(long)(2 * H + j) * H + k] * hk;
            }
            float r = 1.0f / (1.0f + exp(-(xr + hr)));
            float z = 1.0f / (1.0f + exp(-(xz + hz)));
            float n = tanh(xn + r * hn);
            float hNew = (1.0f - z) * n + z * hPrev[j];
            gates[j] = r; gates[H + j] = z; gates[2 * H + j] = hn;
            allH[(long)(t + 1) * batch * H + rowH + j] = hNew;
            output[((long)t * batch + b) * H + j] = hNew;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    for (int j = lid; j < H; j += lsize) hFinal[rowH + j] = allH[(long)seqLen * batch * H + rowH + j];
}

// ===========================================================================
// GRU CELL BACKWARD KERNEL (Single Timestep)
// ===========================================================================

__kernel void gru_cell_backward(
    __global const float* gradH,       // [batch, hidden_size] - gradient from next layer
    __global const float* gateR,       // [batch, hidden_size]
    __global const float* gateZ,       // [batch, hidden_size]
    __global const float* gateN,       // [batch, hidden_size]
    __global const float* prevH,       // [batch, hidden_size]
    __global const float* weightsHh,   // [3 * hidden_size, hidden_size] - recurrent weights
    __global float* gradPrevH,         // [batch, hidden_size] - gradient to previous hidden
    __global float* gradGateR,         // [batch, hidden_size] - gradient for reset gate
    __global float* gradGateZ,         // [batch, hidden_size] - gradient for update gate
    __global float* gradGateN,         // [batch, hidden_size] - gradient for new gate
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float r = gateR[gid];
    float z = gateZ[gid];
    float n = gateN[gid];
    float hPrevLocal = prevH[gid];

    float dH = gradH[gid];

    // Gradient through hidden state update: h_new = (1 - z) * h_prev + z * n
    // For variant 1: h_new = (1-z)*h_prev + z*n
    // dL/dz = dH * (n - h_prev) * sigmoid'(z)
    float dZ = dH * (n - hPrevLocal) * sigmoid_derivative(z);

    // dL/dn = dH * z * tanh'(n)
    float dN = dH * z * tanh_derivative(n);

    // Compute Wn_hh @ h_prev for reset gate gradient
    // n = tanh(Wn_ih @ x + r * (Wn_hh @ h_prev) + bias)
    // dR = dN * (Wn_hh @ h_prev) * sigmoid_derivative(r)
    float Wn_h_prev_dot = 0.0f;
    for (int j = 0; j < hiddenSize; j++) {
        float hPrevJ = prevH[b * hiddenSize + j];
        // Wn_hh is at offset 2*hiddenSize in weightsHh
        Wn_h_prev_dot += hPrevJ * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
    }
    float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

    // Store gate gradients first - these are needed by gru_backward_prevh
    gradGateR[gid] = dR;
    gradGateZ[gid] = dZ;
    gradGateN[gid] = dN;

    // Direct path gradient to prev hidden: dL/dh_prev from (1-z) branch = dH * (1-z)
    // NOTE: This is ONLY the direct path. Full BPTT prev hidden gradient requires
    // calling gru_backward_prevh AFTER this kernel to sum contributions from all
    // hidden positions using the gate gradients stored above. A single kernel cannot
    // do both because OpenCL has no global barrier - each thread would need to read
    // gate gradients from ALL other threads, but those aren't written yet.
    float dHPrev = dH * (1.0f - z);

    // Store partial result - caller must add full gate contributions via gru_backward_prevh
    gradPrevH[gid] = dHPrev;
}

// ===========================================================================
// GRU BACKWARD INPUT GRADIENT KERNEL
// ===========================================================================

__kernel void gru_backward_input(
    __global const float* gradGateR,   // [batch, hidden_size]
    __global const float* gradGateZ,   // [batch, hidden_size]
    __global const float* gradGateN,   // [batch, hidden_size]
    __global const float* weightsIh,   // [3 * hidden_size, input_size]
    __global float* gradInput,         // [batch, input_size]
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * inputSize;

    if (gid >= totalElements) return;

    int b = gid / inputSize;
    int j = gid % inputSize;

    float gradSum = 0.0f;

    for (int h = 0; h < hiddenSize; h++) {
        int batchHiddenIdx = b * hiddenSize + h;

        float dR = gradGateR[batchHiddenIdx];
        float dZ = gradGateZ[batchHiddenIdx];
        float dN = gradGateN[batchHiddenIdx];

        gradSum += dR * weightsIh[h * inputSize + j];
        gradSum += dZ * weightsIh[(hiddenSize + h) * inputSize + j];
        gradSum += dN * weightsIh[(2 * hiddenSize + h) * inputSize + j];
    }

    gradInput[gid] = gradSum;
}

// ===========================================================================
// GRU BACKWARD PREVIOUS HIDDEN GRADIENT KERNEL
// ===========================================================================

__kernel void gru_backward_prevh(
    __global const float* gradGateR,   // [batch, hidden_size]
    __global const float* gradGateZ,   // [batch, hidden_size]
    __global const float* gradGateN,   // [batch, hidden_size]
    __global const float* gradH,       // [batch, hidden_size] - gradient from output
    __global const float* gateR,       // [batch, hidden_size]
    __global const float* gateZ,       // [batch, hidden_size]
    __global const float* weightsHh,   // [3 * hidden_size, hidden_size]
    __global float* gradPrevH,         // [batch, hidden_size]
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float z = gateZ[gid];
    float dH = gradH[gid];

    // Gradient through (1-z) path (direct contribution) for variant 1: h_new = (1-z)*h_prev + z*n
    float gradSum = dH * (1.0f - z);

    // Gradient through gates
    for (int h = 0; h < hiddenSize; h++) {
        int batchHiddenIdx = b * hiddenSize + h;

        float dR = gradGateR[batchHiddenIdx];
        float dZ = gradGateZ[batchHiddenIdx];
        float dN = gradGateN[batchHiddenIdx];
        float r = gateR[batchHiddenIdx];

        gradSum += dR * weightsHh[h * hiddenSize + j];
        gradSum += dZ * weightsHh[(hiddenSize + h) * hiddenSize + j];
        gradSum += dN * r * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
    }

    gradPrevH[gid] = gradSum;
}

// ===========================================================================
// GRU BACKWARD SEQUENCE KERNEL
// ===========================================================================
// GRU backward pass for entire sequence with full BPTT
// Uses local memory to store accumulated hidden gradients so all threads
// can access each other's dH values for proper reset-gate gradient computation.

// Float add on global memory through compare-and-swap (OpenCL 1.2 has no float atomic add). Summation order is not
// fixed, as with CUDA's atomicAdd.
inline void gru_atomic_add(volatile __global float* addr, float v)
{
    union { uint u; float f; } prev, next;
    do { prev.f = *addr; next.f = prev.f + v; }
    while (atomic_cmpxchg((volatile __global uint*)addr, prev.u, next.u) != prev.u);
}

// Full BPTT for gru_forward_sequence (same formulation, one work-group per batch row, strided hidden units). s is
// 4 * hiddenSize floats of local memory holding one step's pre-activation gate gradients (r, z, n, r * n). dHBuffer
// ([B, H]) carries dL/dh back from step t + 1; unit j is always handled by the same work-item, so it needs no barrier.
// The weight and bias gradients accumulate across batch rows and steps; the host zeroes them first.
__kernel void gru_backward_sequence(
    __global const float* gradOutput, __global const float* allH, __global const float* cacheGates,
    __global const float* weightsIh, __global const float* weightsHh, __global const float* input,
    __global float* gradInput, __global float* gradHInit, __global float* dHBuffer,
    __global float* gradWeightsIh, __global float* gradWeightsHh, __global float* gradBiasIh, __global float* gradBiasHh,
    const int seqLen, const int batch, const int inputSize, const int hiddenSize,
    __local float* s)
{
    const int H = hiddenSize, I = inputSize;
    __local float* sDr = s;
    __local float* sDz = s + H;
    __local float* sDn = s + 2 * H;
    __local float* sDnr = s + 3 * H;
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    if (b >= batch) return;
    const long rowH = (long)b * H;
    __global float* carry = dHBuffer + rowH;
    for (int j = lid; j < H; j += lsize) carry[j] = 0.0f;
    for (int t = seqLen - 1; t >= 0; t--)
    {
        __global const float* hPrev = allH + (long)t * batch * H + rowH;
        __global const float* hCur = allH + (long)(t + 1) * batch * H + rowH;
        __global const float* x = input + ((long)t * batch + b) * I;
        __global const float* gates = cacheGates + ((long)t * batch + b) * 3 * H;
        for (int j = lid; j < H; j += lsize)
        {
            float r = gates[j], z = gates[H + j], hn = gates[2 * H + j];
            float dh = gradOutput[((long)t * batch + b) * H + j] + carry[j];
            float oneMinusZ = 1.0f - z;
            float nScaled = hCur[j] - z * hPrev[j];                       // (1 - z) * n
            // dh * (1 - z) * (1 - n^2), written through (1 - z) * n; zero once z saturates at 1.
            float dnPre = oneMinusZ > 1e-12f ? dh * (oneMinusZ - nScaled * nScaled / oneMinusZ) : 0.0f;
            sDz[j] = dh * z * (hPrev[j] - hCur[j]);                         // dh * (h - n) * z * (1 - z)
            sDn[j] = dnPre;
            sDnr[j] = dnPre * r;
            sDr[j] = dnPre * hn * r * (1.0f - r);
            carry[j] = dh * z;                                              // the direct path; matrix paths below
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        for (int j = lid; j < H; j += lsize)
        {
            float dr = sDr[j], dz = sDz[j], dn = sDn[j], dnr = sDnr[j];
            float acc = 0.0f;
            for (int m = 0; m < H; m++)
                acc += weightsHh[(long)m * H + j] * sDr[m]
                     + weightsHh[(long)(H + m) * H + j] * sDz[m]
                     + weightsHh[(long)(2 * H + m) * H + j] * sDnr[m];
            carry[j] += acc;
            for (int i = 0; i < I; i++)
            {
                float xi = x[i];
                gru_atomic_add(&gradWeightsIh[(long)j * I + i], dr * xi);
                gru_atomic_add(&gradWeightsIh[(long)(H + j) * I + i], dz * xi);
                gru_atomic_add(&gradWeightsIh[(long)(2 * H + j) * I + i], dn * xi);
            }
            for (int k = 0; k < H; k++)
            {
                float hk = hPrev[k];
                gru_atomic_add(&gradWeightsHh[(long)j * H + k], dr * hk);
                gru_atomic_add(&gradWeightsHh[(long)(H + j) * H + k], dz * hk);
                gru_atomic_add(&gradWeightsHh[(long)(2 * H + j) * H + k], dnr * hk);
            }
            gru_atomic_add(&gradBiasIh[j], dr); gru_atomic_add(&gradBiasIh[H + j], dz); gru_atomic_add(&gradBiasIh[2 * H + j], dn);
            gru_atomic_add(&gradBiasHh[j], dr); gru_atomic_add(&gradBiasHh[H + j], dz); gru_atomic_add(&gradBiasHh[2 * H + j], dnr);
        }
        for (int i = lid; i < I; i += lsize)
        {
            float acc = 0.0f;
            for (int m = 0; m < H; m++)
                acc += weightsIh[(long)m * I + i] * sDr[m]
                     + weightsIh[(long)(H + m) * I + i] * sDz[m]
                     + weightsIh[(long)(2 * H + m) * I + i] * sDn[m];
            gradInput[((long)t * batch + b) * I + i] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);   // the next step overwrites the local gradients
    }
    for (int j = lid; j < H; j += lsize) gradHInit[rowH + j] = carry[j];
}

// ===========================================================================
// GRU WEIGHT GRADIENT ACCUMULATION KERNELS
// ===========================================================================

__kernel void gru_accumulate_weight_gradients_ih(
    __global const float* input,        // [seqLen, batch, input_size]
    __global const float* allH,         // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,   // [seqLen, batch, hidden_size, 3]
    __global const float* gradOutput,   // [seqLen, batch, hidden_size]
    __global const float* weightsHh,    // [3 * hidden_size, hidden_size] - for proper dR computation
    __global float* gradWeightsIh,      // [3 * hidden_size, input_size]
    const int seqLen,
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalWeights = 3 * hiddenSize * inputSize;

    if (gid >= totalWeights) return;

    int gateIdx = gid / (hiddenSize * inputSize);  // Which gate (0-2)
    int remainder = gid % (hiddenSize * inputSize);
    int h = remainder / inputSize;
    int j = remainder % inputSize;

    float gradSum = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        for (int b = 0; b < batch; b++) {
            int batchHiddenIdx = b * hiddenSize + h;

            // Load cached values
            int cacheIdx = (t * batch * hiddenSize + batchHiddenIdx) * 3;
            float r = cacheGates[cacheIdx + 0];
            float z = cacheGates[cacheIdx + 1];
            float n = cacheGates[cacheIdx + 2];

            // Previous hidden state
            int prevStateIdx = t * batch * hiddenSize + batchHiddenIdx;
            float hPrev = allH[prevStateIdx];

            // Get output gradient
            float dH = gradOutput[t * batch * hiddenSize + batchHiddenIdx];

            // Compute gate gradients
            float dZ = dH * (hPrev - n) * sigmoid_derivative(z);
            float dN = dH * (1.0f - z) * tanh_derivative(n);

            // Compute proper dR using Wn_hh @ h_prev dot product
            float Wn_h_prev_dot = 0.0f;
            for (int jj = 0; jj < hiddenSize; jj++) {
                float hPrevJ = allH[t * batch * hiddenSize + b * hiddenSize + jj];
                Wn_h_prev_dot += hPrevJ * weightsHh[(2 * hiddenSize + h) * hiddenSize + jj];
            }
            float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

            // Get input value
            float inputVal = input[t * batch * inputSize + b * inputSize + j];

            // Accumulate based on gate index
            if (gateIdx == 0) {
                gradSum += dR * inputVal;
            } else if (gateIdx == 1) {
                gradSum += dZ * inputVal;
            } else {
                gradSum += dN * inputVal;
            }
        }
    }

    gradWeightsIh[gid] = gradSum;
}

__kernel void gru_accumulate_weight_gradients_hh(
    __global const float* allH,         // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,   // [seqLen, batch, hidden_size, 3]
    __global const float* gradOutput,   // [seqLen, batch, hidden_size]
    __global const float* weightsHh,    // [3 * hidden_size, hidden_size] - for proper dR computation
    __global float* gradWeightsHh,      // [3 * hidden_size, hidden_size]
    const int seqLen,
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalWeights = 3 * hiddenSize * hiddenSize;

    if (gid >= totalWeights) return;

    int gateIdx = gid / (hiddenSize * hiddenSize);  // Which gate (0-2)
    int remainder = gid % (hiddenSize * hiddenSize);
    int h = remainder / hiddenSize;
    int colIdx = remainder % hiddenSize;

    float gradSum = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        for (int b = 0; b < batch; b++) {
            int batchHiddenIdx = b * hiddenSize + h;

            // Load cached values
            int cacheIdx = (t * batch * hiddenSize + batchHiddenIdx) * 3;
            float r = cacheGates[cacheIdx + 0];
            float z = cacheGates[cacheIdx + 1];
            float n = cacheGates[cacheIdx + 2];

            // Previous hidden state
            int prevStateIdx = t * batch * hiddenSize + batchHiddenIdx;
            float hPrev = allH[prevStateIdx];
            float hPrevColIdx = allH[t * batch * hiddenSize + b * hiddenSize + colIdx];

            // Get output gradient
            float dH = gradOutput[t * batch * hiddenSize + batchHiddenIdx];

            // Compute gate gradients
            float dZ = dH * (hPrev - n) * sigmoid_derivative(z);
            float dN = dH * (1.0f - z) * tanh_derivative(n);

            // Compute proper dR using Wn_hh @ h_prev dot product
            float Wn_h_prev_dot = 0.0f;
            for (int k = 0; k < hiddenSize; k++) {
                float hPrevK = allH[t * batch * hiddenSize + b * hiddenSize + k];
                Wn_h_prev_dot += hPrevK * weightsHh[(2 * hiddenSize + h) * hiddenSize + k];
            }
            float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

            // Accumulate based on gate index
            if (gateIdx == 0) {
                gradSum += dR * hPrevColIdx;
            } else if (gateIdx == 1) {
                gradSum += dZ * hPrevColIdx;
            } else {
                // For n gate, the hidden contribution is gated by r
                gradSum += dN * (r * hPrevColIdx);
            }
        }
    }

    gradWeightsHh[gid] = gradSum;
}

__kernel void gru_accumulate_bias_gradients(
    __global const float* allH,         // [seqLen + 1, batch, hidden_size]
    __global const float* cacheGates,   // [seqLen, batch, hidden_size, 3]
    __global const float* gradOutput,   // [seqLen, batch, hidden_size]
    __global const float* weightsHh,    // [3 * hidden_size, hidden_size] - for proper dR computation
    __global float* gradBias,           // [3 * hidden_size]
    const int seqLen,
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalBiases = 3 * hiddenSize;

    if (gid >= totalBiases) return;

    int gateIdx = gid / hiddenSize;  // Which gate (0-2)
    int h = gid % hiddenSize;

    float gradSum = 0.0f;

    for (int t = 0; t < seqLen; t++) {
        for (int b = 0; b < batch; b++) {
            int batchHiddenIdx = b * hiddenSize + h;

            // Load cached values
            int cacheIdx = (t * batch * hiddenSize + batchHiddenIdx) * 3;
            float r = cacheGates[cacheIdx + 0];
            float z = cacheGates[cacheIdx + 1];
            float n = cacheGates[cacheIdx + 2];

            // Previous hidden state
            int prevStateIdx = t * batch * hiddenSize + batchHiddenIdx;
            float hPrev = allH[prevStateIdx];

            // Get output gradient
            float dH = gradOutput[t * batch * hiddenSize + batchHiddenIdx];

            // Compute gate gradients
            float dZ = dH * (hPrev - n) * sigmoid_derivative(z);
            float dN = dH * (1.0f - z) * tanh_derivative(n);

            // Compute proper dR using Wn_hh @ h_prev dot product
            float Wn_h_prev_dot = 0.0f;
            for (int k = 0; k < hiddenSize; k++) {
                float hPrevK = allH[t * batch * hiddenSize + b * hiddenSize + k];
                Wn_h_prev_dot += hPrevK * weightsHh[(2 * hiddenSize + h) * hiddenSize + k];
            }
            float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

            // Accumulate based on gate index
            if (gateIdx == 0) {
                gradSum += dR;
            } else if (gateIdx == 1) {
                gradSum += dZ;
            } else {
                gradSum += dN;
            }
        }
    }

    gradBias[gid] = gradSum;
}
";
    }

    /// <summary>
    /// Gets the list of kernel names provided by this source.
    /// </summary>
    public static string[] GetKernelNames()
    {
        return new[]
        {
            "gru_cell_forward",
            "gru_forward_sequence",
            "gru_cell_backward",
            "gru_backward_input",
            "gru_backward_prevh",
            "gru_backward_sequence",
            "gru_accumulate_weight_gradients_ih",
            "gru_accumulate_weight_gradients_hh",
            "gru_accumulate_bias_gradients"
        };
    }
}
