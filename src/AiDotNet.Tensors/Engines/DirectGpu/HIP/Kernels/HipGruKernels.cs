// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for GRU (Gated Recurrent Unit) neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT on AMD GPUs.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for sequence-level GRU operations on AMD GPUs.
/// Implements full forward and backward passes for GRU layers processing entire sequences.
/// </summary>
internal static class HipGruKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define EPSILON 1e-15f
#define WARP_SIZE 64

// ===========================================================================
// ACTIVATION FUNCTIONS
// ===========================================================================

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_derivative(float sigmoid_output) {
    return sigmoid_output * (1.0f - sigmoid_output);
}

__device__ __forceinline__ float tanh_derivative(float tanh_output) {
    return 1.0f - tanh_output * tanh_output;
}

// ===========================================================================
// GRU CELL FORWARD KERNEL (Single Timestep)
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(256) void gru_cell_forward(
    const float* input,
    const float* prevH,
    const float* Wz, const float* Wr, const float* Wh,
    const float* Uz, const float* Ur, const float* Uh,
    const float* bz, const float* br, const float* bh,
    float* newH,
    float* gateZ,
    float* gateR,
    float* gateHCandidate,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float sumZ = bz[h];
    float sumR = br[h];

    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        sumZ += Wz[h * inputSize + i] * x_val;
        sumR += Wr[h * inputSize + i] * x_val;
    }

    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        sumZ += Uz[h * hiddenSize + j] * h_val;
        sumR += Ur[h * hiddenSize + j] * h_val;
    }

    float z = sigmoid(sumZ);
    float r = sigmoid(sumR);

    float sumH = bh[h];

    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        sumH += Wh[h * inputSize + i] * x_val;
    }

    // Use per-element reset gate r_j for proper GRU computation
    // In standard GRU: candidate = tanh(Wh*x + Uh*(r ⊙ h_prev) + bh)
    // We need to compute r_j for each hidden unit j, not use scalar r
    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        // Compute r_j = sigmoid(br[j] + Wr[j,:] @ x + Ur[j,:] @ h_prev)
        float sumRj = br[j];
        for (int ii = 0; ii < inputSize; ii++) {
            sumRj += Wr[j * inputSize + ii] * input[b * inputSize + ii];
        }
        for (int jj = 0; jj < hiddenSize; jj++) {
            sumRj += Ur[j * hiddenSize + jj] * prevH[b * hiddenSize + jj];
        }
        float rj = sigmoid(sumRj);
        sumH += Uh[h * hiddenSize + j] * rj * h_val;
    }

    float h_candidate = tanhf(sumH);

    float prevHVal = prevH[gid];
    float newHVal = (1.0f - z) * prevHVal + z * h_candidate;

    newH[gid] = newHVal;
    gateZ[gid] = z;
    gateR[gid] = r;
    gateHCandidate[gid] = h_candidate;
}

// ===========================================================================
// GRU SEQUENCE FORWARD KERNEL
// ===========================================================================

// GRU over a whole sequence, PyTorch layout and gate order (r, z, n), one block per batch row and one thread per
// hidden unit (hiddenSize <= 1024, checked by the launch):
//   r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
//   n = tanh(W_in x + b_in + r * (W_hn h + b_hn))  h' = (1 - z) * n + z * h
// input [seqLen, batch, inputSize]; weightsIh [3 * hidden, inputSize]; weightsHh [3 * hidden, hidden]; biases [3 * hidden];
// output [seqLen, batch, hidden]; allH [(seqLen + 1), batch, hidden] with allH[0] = hInit; cacheGates [seqLen, batch, 3, hidden]
// holding r, z and hn = W_hn h + b_hn (the backward has no biases, and n is recoverable from allH: (1 - z) n = h' - z h). These replace kernels written for separate per-gate matrices (Wz, Wr, Wh, Uz, ...) that the launch never
// passed: the backend's packed arguments filled the wrong parameters and the last four were read past the argument array.
extern ""C"" __global__ __launch_bounds__(1024) void gru_forward_sequence(
    const float* input, const float* hInit,
    const float* weightsIh, const float* weightsHh, const float* biasIh, const float* biasHh,
    float* output, float* hFinal, float* allH, float* cacheGates,
    int seqLen, int batch, int inputSize, int hiddenSize)
{
    int b = blockIdx.x;
    int j = threadIdx.x;
    if (b >= batch) return;
    bool active = j < hiddenSize;
    const int H = hiddenSize, I = inputSize;
    long long rowH = (long long)b * H;
    if (active) allH[rowH + j] = hInit[rowH + j];
    __syncthreads();
    for (int t = 0; t < seqLen; t++)
    {
        const float* hPrev = allH + (long long)t * batch * H + rowH;
        float hNew = 0.0f;
        if (active)
        {
            const float* x = input + ((long long)t * batch + b) * I;
            float xr = biasIh[j], xz = biasIh[H + j], xn = biasIh[2 * H + j];
            for (int i = 0; i < I; i++)
            {
                float xi = x[i];
                xr += weightsIh[(long long)j * I + i] * xi;
                xz += weightsIh[(long long)(H + j) * I + i] * xi;
                xn += weightsIh[(long long)(2 * H + j) * I + i] * xi;
            }
            float hr = biasHh[j], hz = biasHh[H + j], hn = biasHh[2 * H + j];
            for (int k = 0; k < H; k++)
            {
                float hk = hPrev[k];
                hr += weightsHh[(long long)j * H + k] * hk;
                hz += weightsHh[(long long)(H + j) * H + k] * hk;
                hn += weightsHh[(long long)(2 * H + j) * H + k] * hk;
            }
            float r = 1.0f / (1.0f + expf(-(xr + hr)));
            float z = 1.0f / (1.0f + expf(-(xz + hz)));
            float n = tanhf(xn + r * hn);
            hNew = (1.0f - z) * n + z * hPrev[j];
            float* gates = cacheGates + ((long long)t * batch + b) * 3 * H;
            gates[j] = r; gates[H + j] = z; gates[2 * H + j] = hn;
        }
        __syncthreads();   // every thread has read hPrev before anyone writes the next state
        if (active)
        {
            allH[(long long)(t + 1) * batch * H + rowH + j] = hNew;
            output[((long long)t * batch + b) * H + j] = hNew;
        }
        __syncthreads();
    }
    if (active) hFinal[rowH + j] = allH[(long long)seqLen * batch * H + rowH + j];
}


// ===========================================================================
// GRU BACKWARD KERNELS
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(256) void gru_backward_input(
    const float* dGates,
    const float* Wz, const float* Wr, const float* Wh,
    float* dInput,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * inputSize;

    if (gid >= totalElements) return;

    int b = gid / inputSize;
    int i = gid % inputSize;

    float grad = 0.0f;

    for (int h = 0; h < hiddenSize; h++) {
        float dZ = dGates[b * 3 * hiddenSize + h];
        float dR = dGates[b * 3 * hiddenSize + hiddenSize + h];
        float dHCand = dGates[b * 3 * hiddenSize + 2 * hiddenSize + h];

        grad += dZ * Wz[h * inputSize + i];
        grad += dR * Wr[h * inputSize + i];
        grad += dHCand * Wh[h * inputSize + i];
    }

    dInput[gid] = grad;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_backward_prevh(
    const float* dH,
    const float* dGates,
    const float* gateZ,
    const float* gateR,
    const float* Uz, const float* Ur, const float* Uh,
    float* dPrevH,
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float z = gateZ[gid];
    float r = gateR[gid];

    float grad = dH[gid] * (1.0f - z);

    for (int h = 0; h < hiddenSize; h++) {
        float dZ = dGates[b * 3 * hiddenSize + h];
        float dR = dGates[b * 3 * hiddenSize + hiddenSize + h];
        float dHCand = dGates[b * 3 * hiddenSize + 2 * hiddenSize + h];

        grad += dZ * Uz[h * hiddenSize + j];
        grad += dR * Ur[h * hiddenSize + j];
        grad += dHCand * Uh[h * hiddenSize + j] * r;
    }

    dPrevH[gid] = grad;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_compute_gate_gradients(
    const float* dH,
    const float* gateZ,
    const float* gateR,
    const float* gateHCand,
    const float* prevH,
    const float* Uh,
    float* dGates,
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float z = gateZ[gid];
    float r = gateR[gid];
    float h_cand = gateHCand[gid];
    float h_prev = prevH[gid];

    float dh = dH[gid];

    float dHCand = dh * z * tanh_derivative(h_cand);
    float dZ = dh * (h_cand - h_prev) * sigmoid_derivative(z);

    // Compute sumUhH with correct index ordering
    // Uh is [hiddenSize, hiddenSize] in row-major, so Uh[row, col] = Uh[row * hiddenSize + col]
    // For the reset gate gradient, we need sum over input j: Uh[j, h] * prevH[j]
    // which is Uh[j * hiddenSize + h] in row-major
    float sumUhH = 0.0f;
    for (int j = 0; j < hiddenSize; j++) {
        sumUhH += Uh[j * hiddenSize + h] * prevH[b * hiddenSize + j];
    }
    float dR = dHCand * sumUhH * sigmoid_derivative(r);

    int gateOffset = b * 3 * hiddenSize;
    dGates[gateOffset + h] = dZ;
    dGates[gateOffset + hiddenSize + h] = dR;
    dGates[gateOffset + 2 * hiddenSize + h] = dHCand;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_accumulate_weight_gradients(
    const float* input,
    const float* prevH,
    const float* gateR,
    const float* dGates,
    float* dWz, float* dWr, float* dWh,
    float* dUz, float* dUr, float* dUh,
    float* dbz, float* dbr, float* dbh,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gateIdx = blockIdx.x;
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (gateIdx >= 3 * hiddenSize) return;

    int gateType = gateIdx / hiddenSize;
    int h = gateIdx % hiddenSize;

    if (colIdx < inputSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 3 * hiddenSize + gateIdx];
            float x_val = input[b * inputSize + colIdx];
            grad += dGate * x_val;
        }

        if (gateType == 0) atomicAdd(&dWz[h * inputSize + colIdx], grad);
        else if (gateType == 1) atomicAdd(&dWr[h * inputSize + colIdx], grad);
        else atomicAdd(&dWh[h * inputSize + colIdx], grad);
    }

    if (colIdx < hiddenSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 3 * hiddenSize + gateIdx];
            float h_val = prevH[b * hiddenSize + colIdx];

            // For Uh (candidate gate), multiply by reset gate for the INPUT hidden unit (colIdx)
            // not the output unit (h), since GRU applies r element-wise to prevH before Uh multiplication
            if (gateType == 2) {
                float r = gateR[b * hiddenSize + colIdx];
                h_val *= r;
            }

            grad += dGate * h_val;
        }

        if (gateType == 0) atomicAdd(&dUz[h * hiddenSize + colIdx], grad);
        else if (gateType == 1) atomicAdd(&dUr[h * hiddenSize + colIdx], grad);
        else atomicAdd(&dUh[h * hiddenSize + colIdx], grad);
    }

    if (colIdx == 0) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            grad += dGates[b * 3 * hiddenSize + gateIdx];
        }

        if (gateType == 0) atomicAdd(&dbz[h], grad);
        else if (gateType == 1) atomicAdd(&dbr[h], grad);
        else atomicAdd(&dbh[h], grad);
    }
}

// gru_accumulate_weight_gradients — bit-deterministic variant (issue #382).
extern ""C"" __global__ __launch_bounds__(256) void gru_accumulate_weight_gradients_deterministic(
    const float* input, const float* prevH, const float* gateR, const float* dGates,
    float* dWz, float* dWr, float* dWh,
    float* dUz, float* dUr, float* dUh,
    float* dbz, float* dbr, float* dbh,
    int batch, int inputSize, int hiddenSize)
{
    int gateIdx = blockIdx.x;
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
    if (gateIdx >= 3 * hiddenSize) return;
    int gateType = gateIdx / hiddenSize;
    int h = gateIdx % hiddenSize;

    if (colIdx < inputSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 3 * hiddenSize + gateIdx];
            float x_val = input[b * inputSize + colIdx];
            grad += dGate * x_val;
        }
        if (gateType == 0) dWz[h * inputSize + colIdx] += grad;
        else if (gateType == 1) dWr[h * inputSize + colIdx] += grad;
        else dWh[h * inputSize + colIdx] += grad;
    }

    if (colIdx < hiddenSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 3 * hiddenSize + gateIdx];
            float h_val = prevH[b * hiddenSize + colIdx];
            if (gateType == 2) {
                float r = gateR[b * hiddenSize + colIdx];
                h_val *= r;
            }
            grad += dGate * h_val;
        }
        if (gateType == 0) dUz[h * hiddenSize + colIdx] += grad;
        else if (gateType == 1) dUr[h * hiddenSize + colIdx] += grad;
        else dUh[h * hiddenSize + colIdx] += grad;
    }

    if (colIdx == 0) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) grad += dGates[b * 3 * hiddenSize + gateIdx];
        if (gateType == 0) dbz[h] += grad;
        else if (gateType == 1) dbr[h] += grad;
        else dbh[h] += grad;
    }
}

// Backpropagation through time for gru_forward_sequence. Weight and bias gradients are summed over batch rows with
// atomicAdd into buffers the launch zeroes; gradInput and gradHInit are written. Dynamic shared memory: 4 * hiddenSize
// floats holding one step's pre-activation gradients (r, z, n and r * n-gate), which every unit needs for dh_prev and dx.
// n is never divided out: every use of it is scaled by (1 - z), and (1 - z) n = h_t - z h_{t-1} comes from allH.
// dHBuffer is kept for the API and unused: the carried hidden gradient stays in a register per unit.
extern ""C"" __global__ __launch_bounds__(1024) void gru_backward_sequence(
    const float* gradOutput, const float* allH, const float* cacheGates,
    const float* weightsIh, const float* weightsHh, const float* input,
    float* gradInput, float* gradHInit, float* dHBuffer,
    float* gradWeightsIh, float* gradWeightsHh, float* gradBiasIh, float* gradBiasHh,
    int seqLen, int batch, int inputSize, int hiddenSize)
{
    extern __shared__ float s[];
    const int H = hiddenSize, I = inputSize;
    float* sDr = s; float* sDz = s + H; float* sDn = s + 2 * H; float* sDnr = s + 3 * H;
    int b = blockIdx.x;
    int j = threadIdx.x;
    if (b >= batch) return;
    bool active = j < H;
    long long rowH = (long long)b * H;
    float dhCarry = 0.0f;
    for (int t = seqLen - 1; t >= 0; t--)
    {
        const float* hPrev = allH + (long long)t * batch * H + rowH;
        const float* hCur = allH + (long long)(t + 1) * batch * H + rowH;
        const float* x = input + ((long long)t * batch + b) * I;
        if (active)
        {
            const float* gates = cacheGates + ((long long)t * batch + b) * 3 * H;
            float r = gates[j], z = gates[H + j], hn = gates[2 * H + j];
            float dh = gradOutput[((long long)t * batch + b) * H + j] + dhCarry;
            float oneMinusZ = 1.0f - z;
            float nScaled = hCur[j] - z * hPrev[j];                 // (1 - z) * n
            // d/d(pre-n) = dh * (1 - z) * (1 - n^2) = dh * ((1 - z) - ((1 - z) n)^2 / (1 - z)); zero once z saturates at 1.
            float dnPre = oneMinusZ > 1e-12f ? dh * (oneMinusZ - nScaled * nScaled / oneMinusZ) : 0.0f;
            sDz[j] = dh * z * (hPrev[j] - hCur[j]);                   // dh * (h - n) * z * (1 - z)
            sDn[j] = dnPre;
            sDnr[j] = dnPre * r;
            sDr[j] = dnPre * hn * r * (1.0f - r);
            dhCarry = dh * z;                                         // direct path; matrix paths added below
        }
        __syncthreads();
        if (active)
        {
            float dr = sDr[j], dz = sDz[j], dn = sDn[j], dnr = sDnr[j];
            float acc = 0.0f;
            for (int m = 0; m < H; m++)
                acc += weightsHh[(long long)m * H + j] * sDr[m]
                     + weightsHh[(long long)(H + m) * H + j] * sDz[m]
                     + weightsHh[(long long)(2 * H + m) * H + j] * sDnr[m];
            dhCarry += acc;
            for (int i = 0; i < I; i++)
            {
                float xi = x[i];
                atomicAdd(&gradWeightsIh[(long long)j * I + i], dr * xi);
                atomicAdd(&gradWeightsIh[(long long)(H + j) * I + i], dz * xi);
                atomicAdd(&gradWeightsIh[(long long)(2 * H + j) * I + i], dn * xi);
            }
            for (int k = 0; k < H; k++)
            {
                float hk = hPrev[k];
                atomicAdd(&gradWeightsHh[(long long)j * H + k], dr * hk);
                atomicAdd(&gradWeightsHh[(long long)(H + j) * H + k], dz * hk);
                atomicAdd(&gradWeightsHh[(long long)(2 * H + j) * H + k], dnr * hk);
            }
            atomicAdd(&gradBiasIh[j], dr); atomicAdd(&gradBiasIh[H + j], dz); atomicAdd(&gradBiasIh[2 * H + j], dn);
            atomicAdd(&gradBiasHh[j], dr); atomicAdd(&gradBiasHh[H + j], dz); atomicAdd(&gradBiasHh[2 * H + j], dnr);
        }
        for (int i = j; i < I; i += blockDim.x)
        {
            float acc = 0.0f;
            for (int m = 0; m < H; m++)
                acc += weightsIh[(long long)m * I + i] * sDr[m]
                     + weightsIh[(long long)(H + m) * I + i] * sDz[m]
                     + weightsIh[(long long)(2 * H + m) * I + i] * sDn[m];
            gradInput[((long long)t * batch + b) * I + i] = acc;
        }
        __syncthreads();   // the next step overwrites the shared gradients
    }
    if (active) gradHInit[rowH + j] = dhCarry;
}


// gru_backward_sequence — bit-deterministic split (issue #382).
// Six-kernel pipeline keyed off dGates_t[T, B, 3*H] scratch buffer.
// Layout per (t, b, h): [dZ, dR, dHCand].
//
// Pass 1 (precompute_gates_deterministic): per (b, h) thread iterates t in reverse.
// Within each t: computes dZ, dHCand directly; computes dR using the dR formula
// (nested sum over k of dHCand_k * Uh[k, h_idx] then * h_prev * sigmoid'(r)).
// Writes dGates_t[t, b, *] scratch. Updates dH_buffer (intra-batch shared) for next
// timestep using cooperative grid sync.
// Remaining intra-kernel atomic: dH propagation across timesteps via Uz/Ur/Uh dot
// products. For fully atomic-free, callers can use the per-timestep external
// pipeline (gru_cell_backward_unified + gru_backward_prevh_unified +
// gru_accumulate_weight_gradients_deterministic + gru_backward_input).
//
// Passes 2-6: per-output-cell accumulators (no atomics):
//   dWz/dWr/dWh: per (h, i) scans (t, b)
//   dUz/dUr:     per (h, j) scans (t, b)
//   dUh:         per (h, j) scans (t, b) with r[j] factor
//   dbz/dbr/dbh: per h scans (t, b)
//   gradInput:   per (b, t, i) reads dGates_t, dot product with Wz/Wr/Wh

// Pass 1: per (b, h_idx) thread iterates t in reverse, computes dZ/dR/dHCand
// for the (b, h_idx, t) cell and writes them to dGates_t[t, b, *]. Across-t
// dH propagation uses atomicAdd into dH_init (heavy-lift to fully eliminate
// — see PR #390 review for the LSTM equivalent at lstm_backward_sequence_*).
extern ""C"" __global__ __launch_bounds__(1024) void gru_backward_sequence_precompute_gates_deterministic(
    const float* gradOutput,   // [B, T, H]
    const float* h_states,     // [T, B, H]
    const float* h_init,       // [B, H]
    const float* gates,        // [T, B, 3*H] — (z, r, n_candidate) cached
    const float* Uz,           // [H, H]
    const float* Ur,           // [H, H]
    const float* Uh,           // [H, H]
    float* dGates_t,           // [T, B, 3*H] — scratch (output)
    float* dH_init,            // [B, H]      — scratch / output
    int batch, int timeSteps, int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;
    bool isValid = gid < totalElements;
    int b = isValid ? (gid / hiddenSize) : 0;
    int h_idx = isValid ? (gid % hiddenSize) : 0;

    if (isValid) dH_init[gid] = 0.0f;
    __syncthreads();

    float dH = 0.0f;

    for (int t = timeSteps - 1; t >= 0; t--) {
        if (isValid && t < timeSteps - 1) {
            dH = dH_init[gid];
            dH_init[gid] = 0.0f;
        }
        __syncthreads();

        if (isValid) {
            dH += gradOutput[(b * timeSteps + t) * hiddenSize + h_idx];

            int gateOffset = t * batch * 3 * hiddenSize + b * 3 * hiddenSize;
            float z = gates[gateOffset + h_idx];
            float r = gates[gateOffset + hiddenSize + h_idx];
            float n = gates[gateOffset + 2 * hiddenSize + h_idx];

            float h_prev = (t == 0)
                ? h_init[b * hiddenSize + h_idx]
                : h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + h_idx];

            // Direct gate derivatives at (b, h_idx, t).
            float dZ = dH * (n - h_prev) * sigmoid_derivative(z);
            float dHCand = dH * z * tanh_derivative(n);

            // dR depends on a full sum over k of dHCand_k * Uh[k, h_idx] times
            // h_prev[h_idx] * sigmoid'(r). Recompute dHCand_k locally for each k.
            // CodeRabbit (#390): for k == h_idx the read from dH_init at line
            // (b * hiddenSize + k) returns 0 because line 740 zeroed it. The
            // correct value is held in the local `dH` variable, which already
            // equals gradOutput[t, h_idx] + (old dH_init[gid] when t < T-1).
            float dR_sum = 0.0f;
            for (int k = 0; k < hiddenSize; k++) {
                float z_k = gates[gateOffset + k];
                float n_k = gates[gateOffset + 2 * hiddenSize + k];
                float dH_k;
                if (k == h_idx) {
                    // Local `dH` holds the (preserved + this-timestep) sum.
                    dH_k = dH;
                } else if (t == timeSteps - 1) {
                    dH_k = gradOutput[(b * timeSteps + t) * hiddenSize + k];
                } else {
                    dH_k = gradOutput[(b * timeSteps + t) * hiddenSize + k]
                         + dH_init[b * hiddenSize + k];
                }
                float dHCand_k = dH_k * z_k * tanh_derivative(n_k);
                dR_sum += dHCand_k * Uh[k * hiddenSize + h_idx];
            }
            float dR = dR_sum * h_prev * sigmoid_derivative(r);

            int scratchBase = t * batch * 3 * hiddenSize + b * 3 * hiddenSize;
            dGates_t[scratchBase + h_idx] = dZ;
            dGates_t[scratchBase + hiddenSize + h_idx] = dR;
            dGates_t[scratchBase + 2 * hiddenSize + h_idx] = dHCand;

            // dH at previous timestep: direct (1 - z) plus contributions through
            // U{z,r,h}. atomicAdd on dH_init[b, j] is the remaining nondeterministic op in
            // this kernel (PR #390 review).
            float dH_self_next = dH * (1.0f - z);
            atomicAdd(&dH_init[b * hiddenSize + h_idx], dH_self_next);
            for (int j = 0; j < hiddenSize; j++) {
                float contrib = dZ * Uz[h_idx * hiddenSize + j];
                contrib += dR * Ur[h_idx * hiddenSize + j];
                contrib += dHCand * Uh[h_idx * hiddenSize + j] * r;
                atomicAdd(&dH_init[b * hiddenSize + j], contrib);
            }
        }
        __syncthreads();
    }
}

extern ""C"" __global__ __launch_bounds__(256) void gru_backward_sequence_dWi_deterministic(
    const float* input, const float* dGates_t,
    float* dWz, float* dWr, float* dWh,
    int batch, int timeSteps, int inputSize, int hiddenSize)
{
    int gateType = blockIdx.x / hiddenSize;   // 0=Z, 1=R, 2=N
    int h = blockIdx.x % hiddenSize;
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
    if (gateType > 2 || h >= hiddenSize || colIdx >= inputSize) return;

    float sum = 0.0f;
    for (int t = 0; t < timeSteps; t++) {
        int scratchBaseT = t * batch * 3 * hiddenSize;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates_t[scratchBaseT + b * 3 * hiddenSize + gateType * hiddenSize + h];
            float x_val = input[(b * timeSteps + t) * inputSize + colIdx];
            sum += dGate * x_val;
        }
    }
    if (gateType == 0) dWz[h * inputSize + colIdx] += sum;
    else if (gateType == 1) dWr[h * inputSize + colIdx] += sum;
    else dWh[h * inputSize + colIdx] += sum;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_backward_sequence_dUi_deterministic(
    const float* h_states, const float* h_init, const float* gates, const float* dGates_t,
    float* dUz, float* dUr, float* dUh,
    int batch, int timeSteps, int hiddenSize)
{
    int gateType = blockIdx.x / hiddenSize;
    int h = blockIdx.x % hiddenSize;
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
    if (gateType > 2 || h >= hiddenSize || colIdx >= hiddenSize) return;

    float sum = 0.0f;
    for (int t = 0; t < timeSteps; t++) {
        int scratchBaseT = t * batch * 3 * hiddenSize;
        int gateBaseT = t * batch * 3 * hiddenSize;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates_t[scratchBaseT + b * 3 * hiddenSize + gateType * hiddenSize + h];
            float hj = (t == 0) ? h_init[b * hiddenSize + colIdx]
                                : h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + colIdx];
            if (gateType == 2) {
                // Uh path: multiply by r[j] at the colIdx position
                float r_j = gates[gateBaseT + b * 3 * hiddenSize + hiddenSize + colIdx];
                hj *= r_j;
            }
            sum += dGate * hj;
        }
    }
    if (gateType == 0) dUz[h * hiddenSize + colIdx] += sum;
    else if (gateType == 1) dUr[h * hiddenSize + colIdx] += sum;
    else dUh[h * hiddenSize + colIdx] += sum;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_backward_sequence_dBias_deterministic(
    const float* dGates_t,
    float* dbz, float* dbr, float* dbh,
    int batch, int timeSteps, int hiddenSize)
{
    int gateType = blockIdx.x / hiddenSize;
    int h = blockIdx.x % hiddenSize;
    if (gateType > 2 || h >= hiddenSize) return;

    float sum = 0.0f;
    for (int t = 0; t < timeSteps; t++) {
        int scratchBaseT = t * batch * 3 * hiddenSize;
        for (int b = 0; b < batch; b++) {
            sum += dGates_t[scratchBaseT + b * 3 * hiddenSize + gateType * hiddenSize + h];
        }
    }
    if (gateType == 0) dbz[h] += sum;
    else if (gateType == 1) dbr[h] += sum;
    else dbh[h] += sum;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_backward_sequence_dInput_deterministic(
    const float* dGates_t,
    const float* Wz, const float* Wr, const float* Wh,
    float* gradInput,
    int batch, int timeSteps, int inputSize, int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * timeSteps * inputSize;
    if (gid >= totalElements) return;
    int i = gid % inputSize;
    int tmp = gid / inputSize;
    int t = tmp % timeSteps;
    int b = tmp / timeSteps;

    int scratchBase = t * batch * 3 * hiddenSize + b * 3 * hiddenSize;
    float grad = 0.0f;
    for (int h = 0; h < hiddenSize; h++) {
        float dZ = dGates_t[scratchBase + h];
        float dR = dGates_t[scratchBase + hiddenSize + h];
        float dHCand = dGates_t[scratchBase + 2 * hiddenSize + h];
        grad += dZ * Wz[h * inputSize + i];
        grad += dR * Wr[h * inputSize + i];
        grad += dHCand * Wh[h * inputSize + i];
    }
    gradInput[(b * timeSteps + t) * inputSize + i] += grad;
}

// ===========================================================================
// UNIFIED GRU CELL BACKWARD KERNEL
// ===========================================================================
// Matches the OpenCL interface for cross-backend compatibility.
// Computes gate gradients and partial prevH (direct path only).
// Must be followed by gru_backward_prevh_unified for full BPTT gradient.

extern ""C"" __global__ __launch_bounds__(256) void gru_cell_backward_unified(
    const float* gradH,       // [batch, hiddenSize] - gradient from next layer
    const float* gateR,       // [batch, hiddenSize] - reset gate values
    const float* gateZ,       // [batch, hiddenSize] - update gate values
    const float* gateN,       // [batch, hiddenSize] - candidate values
    const float* prevH,       // [batch, hiddenSize] - previous hidden state
    const float* weightsHh,   // [3 * hiddenSize, hiddenSize] - recurrent weights (R, Z, N stacked)
    float* gradPrevH,         // [batch, hiddenSize] - output: partial gradient (direct path only)
    float* gradGateR,         // [batch, hiddenSize] - output: reset gate gradient
    float* gradGateZ,         // [batch, hiddenSize] - output: update gate gradient
    float* gradGateN,         // [batch, hiddenSize] - output: candidate gradient
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
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
    // dR = dN * (Wn_hh @ h_prev) * sigmoid'(r)
    float Wn_h_prev_dot = 0.0f;
    for (int j = 0; j < hiddenSize; j++) {
        float hPrevJ = prevH[b * hiddenSize + j];
        // Wn_hh is at offset 2*hiddenSize in weightsHh (layout: R, Z, N)
        Wn_h_prev_dot += hPrevJ * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
    }
    float dR = dN * Wn_h_prev_dot * sigmoid_derivative(r);

    // Store gate gradients
    gradGateR[gid] = dR;
    gradGateZ[gid] = dZ;
    gradGateN[gid] = dN;

    // Direct path gradient to prev hidden: dL/dh_prev from (1-z) branch = dH * (1-z)
    // NOTE: This is ONLY the direct path. Full gradient requires gru_backward_prevh_unified.
    float dHPrev = dH * (1.0f - z);
    gradPrevH[gid] = dHPrev;
}

// ===========================================================================
// UNIFIED GRU BACKWARD PREVH KERNEL
// ===========================================================================
// Computes full gradient to previous hidden state by summing contributions
// from all hidden positions through the gate weight matrices.
// Must be called AFTER gru_cell_backward_unified which computes gate gradients.

extern ""C"" __global__ __launch_bounds__(256) void gru_backward_prevh_unified(
    const float* gradGateR,   // [batch, hiddenSize] - from gru_cell_backward_unified
    const float* gradGateZ,   // [batch, hiddenSize] - from gru_cell_backward_unified
    const float* gradGateN,   // [batch, hiddenSize] - from gru_cell_backward_unified
    const float* gradH,       // [batch, hiddenSize] - gradient from output
    const float* gateR,       // [batch, hiddenSize] - reset gate values
    const float* gateZ,       // [batch, hiddenSize] - update gate values
    const float* weightsHh,   // [3 * hiddenSize, hiddenSize] - recurrent weights (R, Z, N stacked)
    float* gradPrevH,         // [batch, hiddenSize] - output: OVERWRITES with full gradient
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;  // This thread computes gradPrevH[b, j]

    float z = gateZ[gid];
    float dH = gradH[gid];

    // Gradient through (1-z) path (direct contribution) for variant 1: h_new = (1-z)*h_prev + z*n
    float gradSum = dH * (1.0f - z);

    // Gradient through all gates from all hidden positions h
    for (int hh = 0; hh < hiddenSize; hh++) {
        int batchHiddenIdx = b * hiddenSize + hh;

        float dR = gradGateR[batchHiddenIdx];
        float dZ = gradGateZ[batchHiddenIdx];
        float dN = gradGateN[batchHiddenIdx];
        float r = gateR[batchHiddenIdx];

        // weightsHh layout: [R weights, Z weights, N weights] each [hiddenSize, hiddenSize]
        // R weights: weightsHh[hh * hiddenSize + j] for Ur[hh, j]
        // Z weights: weightsHh[(hiddenSize + hh) * hiddenSize + j] for Uz[hh, j]
        // N weights: weightsHh[(2 * hiddenSize + hh) * hiddenSize + j] for Uh[hh, j]
        gradSum += dR * weightsHh[hh * hiddenSize + j];
        gradSum += dZ * weightsHh[(hiddenSize + hh) * hiddenSize + j];
        gradSum += dN * r * weightsHh[(2 * hiddenSize + hh) * hiddenSize + j];
    }

    gradPrevH[gid] = gradSum;
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
            "gru_backward_input",
            "gru_backward_prevh",
            "gru_compute_gate_gradients",
            "gru_accumulate_weight_gradients",
            "gru_accumulate_weight_gradients_deterministic",
            "gru_backward_sequence",
            "gru_backward_sequence_precompute_gates_deterministic",
            "gru_backward_sequence_dWi_deterministic",
            "gru_backward_sequence_dUi_deterministic",
            "gru_backward_sequence_dBias_deterministic",
            "gru_backward_sequence_dInput_deterministic",
            "gru_cell_backward_unified",
            "gru_backward_prevh_unified"
        };
    }
}
