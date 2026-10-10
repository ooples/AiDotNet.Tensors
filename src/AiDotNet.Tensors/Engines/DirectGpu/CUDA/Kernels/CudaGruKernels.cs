// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for GRU (Gated Recurrent Unit) neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for sequence-level GRU operations.
/// Implements full forward and backward passes for GRU layers processing entire sequences.
/// </summary>
/// <remarks>
/// GRU equations:
/// z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)       // update gate
/// r_t = sigmoid(W_r * x_t + U_r * h_{t-1} + b_r)       // reset gate
/// h̃_t = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h)  // candidate hidden
/// h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t               // hidden state
/// </remarks>
internal static class CudaGruKernels
{
    public static string GetSource()
    {
        return @"
#include <math.h>

#define EPSILON 1e-15f
#define WARP_SIZE 32

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

// GRU cell forward pass for a single time step
// input: [batch, inputSize]
// prevH: [batch, hiddenSize]
// Wz, Wr, Wh: [hiddenSize, inputSize]
// Uz, Ur, Uh: [hiddenSize, hiddenSize]
// bz, br, bh: [hiddenSize]
// output newH: [batch, hiddenSize]
// gateZ, gateR, gateH: [batch, hiddenSize] (cached for backward)
extern ""C"" __global__ __launch_bounds__(256) void gru_cell_forward(
    const float* __restrict__ input,
    const float* __restrict__ prevH,
    const float* __restrict__ Wz, const float* __restrict__ Wr, const float* __restrict__ Wh,
    const float* __restrict__ Uz, const float* __restrict__ Ur, const float* __restrict__ Uh,
    const float* __restrict__ bz, const float* __restrict__ br, const float* __restrict__ bh,
    float* __restrict__ newH,
    float* __restrict__ gateZ,
    float* __restrict__ gateR,
    float* __restrict__ gateHCandidate,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;
    int b = gid / hiddenSize;
    int h = gid % hiddenSize;
    int isValid = (gid < totalElements) ? 1 : 0;

    // Phase 1: Compute z and r gates for all threads
    float z = 0.0f;
    float r = 0.0f;

    if (isValid) {
        // Compute update gate z: sigmoid(Wz*x + Uz*h + bz)
        float sumZ = bz[h];
        float sumR = br[h];

        // Input contribution
        for (int i = 0; i < inputSize; i++) {
            float x_val = input[b * inputSize + i];
            sumZ += Wz[h * inputSize + i] * x_val;
            sumR += Wr[h * inputSize + i] * x_val;
        }

        // Hidden contribution for z and r gates
        for (int j = 0; j < hiddenSize; j++) {
            float h_val = prevH[b * hiddenSize + j];
            sumZ += Uz[h * hiddenSize + j] * h_val;
            sumR += Ur[h * hiddenSize + j] * h_val;
        }

        z = sigmoid(sumZ);
        r = sigmoid(sumR);

        // Store r to global buffer so other threads can read it
        gateR[gid] = r;
    }

    // Synchronize to ensure all r values are written before reading
    __syncthreads();

    // Phase 2: Compute candidate using per-element r_j values
    float h_candidate = 0.0f;

    if (isValid) {
        float sumH = bh[h];

        for (int i = 0; i < inputSize; i++) {
            float x_val = input[b * inputSize + i];
            sumH += Wh[h * inputSize + i] * x_val;
        }

        // Use per-element reset gate r_j for proper GRU computation
        // In standard GRU: candidate = tanh(Wh*x + Uh*(r ⊙ h_prev) + bh)
        for (int j = 0; j < hiddenSize; j++) {
            float h_val = prevH[b * hiddenSize + j];
            float r_j = gateR[b * hiddenSize + j];  // Read r for hidden unit j
            sumH += Uh[h * hiddenSize + j] * r_j * h_val;
        }

        h_candidate = tanhf(sumH);

        // Compute new hidden: (1-z)*h_prev + z*h_candidate
        float prevHVal = prevH[gid];
        float newHVal = (1.0f - z) * prevHVal + z * h_candidate;

        // Store outputs
        newH[gid] = newHVal;

        // Store gate values for backward pass
        gateZ[gid] = z;
        // gateR already stored above
        gateHCandidate[gid] = h_candidate;
    }
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
// GRU CELL BACKWARD KERNEL
// ===========================================================================

// Computes gradients for a single GRU cell timestep
// Reset gate gradient is computed per-element: for each input hidden unit j,
// we compute sum over outputs h of dHCand_h * Uh[h, j], then multiply by
// prevH[j] * sigmoid_derivative(r[j])
//
// The reset gate is applied element-wise in forward: candidate uses r[j] * prevH[j]
// So the backward must compute dR[j] as a reduction over all output positions h.
extern ""C"" __global__ __launch_bounds__(256) void gru_cell_backward(
    const float* dH,          // [batch, hidden]
    const float* gateZ,       // [batch, hidden]
    const float* gateR,       // [batch, hidden]
    const float* gateHCand,   // [batch, hidden]
    const float* prevH,       // [batch, hidden]
    const float* input,       // [batch, input]
    const float* Wz, const float* Wr, const float* Wh,
    const float* Uz, const float* Ur, const float* Uh,
    float* dPrevH,            // [batch, hidden]
    float* dInput,            // [batch, input]
    float* dWz, float* dWr, float* dWh,
    float* dUz, float* dUr, float* dUh,
    float* dbz, float* dbr, float* dbh,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Get cached values for this output position h
    float z = gateZ[gid];
    float h_cand = gateHCand[gid];
    float h_prev_h = prevH[gid];

    // Gradient from output
    float dh = dH[gid];

    // Gradient through hidden state update: h_t = (1-z)*h_prev + z*h_cand
    // dh_cand = dh * z * tanh'(h_cand)
    float dHCand = dh * z * tanh_derivative(h_cand);

    // dz = dh * (h_cand - h_prev) * sigmoid'(z)
    float dZ = dh * (h_cand - h_prev_h) * sigmoid_derivative(z);

    // dh_prev from direct path = dh * (1-z)
    float dHPrev_direct = dh * (1.0f - z);

    // Add direct path contribution to dPrevH[h]
    atomicAdd(&dPrevH[gid], dHPrev_direct);

    // Gradient to previous hidden through reset gate path and reset gate gradient
    // Forward: candidate_h = tanh(... + sum_j(Uh[h,j] * r[j] * prevH[j]) + ...)
    // So for each input position j:
    //   d(candidate_h)/d(prevH[j]) = Uh[h,j] * r[j]
    //   d(candidate_h)/d(r[j]) = Uh[h,j] * prevH[j]
    //
    // dPrevH[j] from reset path = sum_h(dHCand_h * Uh[h,j] * r[j])
    // dR[j] (pre-activation) = sum_h(dHCand_h * Uh[h,j]) * prevH[j] * sigmoid'(r[j])
    //
    // This thread handles output h, so we contribute to each j via atomicAdd
    for (int j = 0; j < hiddenSize; j++) {
        float r_j = gateR[b * hiddenSize + j];
        float prevH_j = prevH[b * hiddenSize + j];

        // Contribution from output h to dPrevH[j] through reset path
        float dPrevH_contrib = dHCand * Uh[h * hiddenSize + j] * r_j;
        atomicAdd(&dPrevH[b * hiddenSize + j], dPrevH_contrib);

        // Contribution from output h to dR[j]
        // dR[j] = sum_h(dHCand_h * Uh[h,j]) * prevH[j] * sigmoid'(r[j])
        // This thread contributes: dHCand * Uh[h,j]
        // Then multiply by prevH[j] * sigmoid'(r[j]) to get contribution to pre-activation gradient
        float dR_contrib_h = dHCand * Uh[h * hiddenSize + j];
        float dR_j_contrib = dR_contrib_h * prevH_j * sigmoid_derivative(r_j);

        // Accumulate reset gate weight gradients: dUr[h,j] = dR[h] * prevH[j]
        // But reset weights connect output h to input j, and dR is per input position j
        // The gradient for Ur[h,j] comes from output h's contribution to the reset path
        // dL/dUr[h,j] = dL/d(preR[h]) * d(preR[h])/dUr[h,j] = dR[h] * prevH[j]
        // We need dR[h] which is the gradient at output position h, not input position j

        // Hidden weight gradient for reset gate: dUr[h,j] uses dR[h], not dR[j]
        // We compute dR[h] later for weight updates

        // Accumulate dUh: dUh[h,j] = dHCand * r[j] * prevH[j]
        atomicAdd(&dUh[h * hiddenSize + j], dHCand * r_j * prevH_j);

        // Accumulate bias gradient for reset gate position j
        // dbr[j] = sum_b sum_h dR_j_contrib (already has sigmoid' and prevH)
        atomicAdd(&dbr[j], dR_j_contrib);

        // Accumulate hidden weight gradient for reset: dUr[h,j] from this output h
        // The reset gate computation is: preR[h] = sum_j(Ur[h,j] * prevH[j]) + ...
        // So dUr[h,j] needs dR[h] * prevH[j]
        // We'll compute dR[h] separately below for correct weight gradient
    }

    // Compute dR[h] - the gradient of the reset gate at output position h
    // This is needed for Wr weight gradient: dWr[h,i] = dR[h] * x[i]
    // and Ur weight gradient: dUr[h,j] = dR[h] * prevH[j]
    //
    // dR[h] = sum_j(dL/d(r[j]) contribution from position h)
    // But actually for the reset gate weights, we need the gradient w.r.t. the
    // pre-activation at position h, not j.
    //
    // The forward is: r[h] = sigmoid(sum_i(Wr[h,i]*x[i]) + sum_j(Ur[h,j]*prevH[j]) + br[h])
    // The reset gate r[h] at position h is used to gate prevH[h] in the candidate for all outputs.
    //
    // So dL/dr[h] = sum_over_outputs_k(dHCand_k * Uh[k,h]) * prevH[h]
    // And dL/d(preR[h]) = dL/dr[h] * sigmoid'(r[h])
    float dR_h_sum = 0.0f;
    for (int k = 0; k < hiddenSize; k++) {
        float dHCand_k = dH[b * hiddenSize + k] * gateZ[b * hiddenSize + k] *
                         tanh_derivative(gateHCand[b * hiddenSize + k]);
        dR_h_sum += dHCand_k * Uh[k * hiddenSize + h];
    }
    float r_h = gateR[gid];
    float dR_h = dR_h_sum * prevH[gid] * sigmoid_derivative(r_h);

    // Accumulate input weight gradients
    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        atomicAdd(&dWz[h * inputSize + i], dZ * x_val);
        atomicAdd(&dWr[h * inputSize + i], dR_h * x_val);
        atomicAdd(&dWh[h * inputSize + i], dHCand * x_val);
    }

    // Hidden weight gradients for Z gate and Ur
    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        atomicAdd(&dUz[h * hiddenSize + j], dZ * h_val);
        atomicAdd(&dUr[h * hiddenSize + j], dR_h * h_val);
    }

    // Bias gradients for Z gate and candidate
    atomicAdd(&dbz[h], dZ);
    // Note: dbr was already accumulated per-element in the loop above
    atomicAdd(&dbh[h], dHCand);
}

// gru_cell_backward — bit-deterministic split (issue #382).
// The original single-pass kernel has many overlapping atomic-add patterns
// (dPrevH direct path + reset path, dUh, dbr, dWz/dWr/dWh, dUz/dUr, dbz/dbh).
// Deterministic split:
//   1. compute_gates: per (b, h) writes dZ, dHCand, dR_h, dHPrev_direct to scratch
//      dGates_z, dGates_h, dGates_r [each batch * hidden]; also writes
//      dPrevH[b, h] += dHPrev_direct (own cell, no atomic).
//   2. dPrevH_reset: per (b, j) reads scratch and gates, += reset-path contribution.
//   3. dbr: per j reads scratch, sums over (b, h).
//   4. dUh: per (h, j) reads scratch, sums over b (with r[j] factor).
//   5. dWz/dWr/dWh: per (h, i) reads scratch, sums over b.
//   6. dUz/dUr: per (h, j) reads scratch, sums over b.
//   7. dbz/dbh: per h reads scratch, sums over b.
// Kernels 5-7 are largely subsumed by the existing
// gru_accumulate_weight_gradients_deterministic when called with dGates layout
// [batch, 3*hidden] = [dZ, dR, dHCand] per (b, h).
extern ""C"" __global__ __launch_bounds__(256) void gru_cell_backward_compute_gates_deterministic(
    const float* dH,          // [batch, hidden]
    const float* gateZ, const float* gateR, const float* gateHCand,
    const float* prevH,       // [batch, hidden]
    const float* Uh,          // [hidden, hidden] (reset-gated candidate weight)
    float* dGates,            // [batch, 3*hidden]: [dZ, dR_h, dHCand] per (b, h)
    float* dPrevH,            // [batch, hidden] — gets DIRECT-path contribution here
    int batch, int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;
    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float z = gateZ[gid];
    float h_cand = gateHCand[gid];
    float h_prev_h = prevH[gid];
    float r_h = gateR[gid];
    float dh = dH[gid];

    float dHCand = dh * z * tanh_derivative(h_cand);
    float dZ = dh * (h_cand - h_prev_h) * sigmoid_derivative(z);
    float dHPrev_direct = dh * (1.0f - z);

    // Compute dR_h = sum_k(dHCand_k * Uh[k, h]) * prevH[b, h] * sigmoid'(r[h])
    int bBase = b * hiddenSize;
    float dR_h_sum = 0.0f;
    for (int k = 0; k < hiddenSize; k++) {
        float dHCand_k = dH[bBase + k] * gateZ[bBase + k] * tanh_derivative(gateHCand[bBase + k]);
        dR_h_sum += dHCand_k * Uh[k * hiddenSize + h];
    }
    float dR_h = dR_h_sum * h_prev_h * sigmoid_derivative(r_h);

    int gateBase = b * 3 * hiddenSize;
    dGates[gateBase + h] = dZ;
    dGates[gateBase + hiddenSize + h] = dR_h;
    dGates[gateBase + 2 * hiddenSize + h] = dHCand;

    // Direct-path contribution to dPrevH[b, h] (own cell, no atomic needed)
    dPrevH[gid] += dHPrev_direct;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_cell_backward_dPrevH_reset_deterministic(
    const float* dGates,      // [batch, 3*hidden]
    const float* gateR,       // [batch, hidden]
    const float* Uh,          // [hidden, hidden]
    float* dPrevH,            // [batch, hidden]
    int batch, int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;
    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    int gateBase = b * 3 * hiddenSize;
    float r_j = gateR[b * hiddenSize + j];

    float sum = 0.0f;
    for (int h = 0; h < hiddenSize; h++) {
        float dHCand_h = dGates[gateBase + 2 * hiddenSize + h];
        sum += dHCand_h * Uh[h * hiddenSize + j] * r_j;
    }
    dPrevH[gid] += sum;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_cell_backward_dbr_deterministic(
    const float* dGates,      // [batch, 3*hidden]
    const float* gateR,       // [batch, hidden]
    const float* prevH,       // [batch, hidden]
    const float* Uh,          // [hidden, hidden]
    float* dbr,               // [hidden]
    int batch, int hiddenSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hiddenSize) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        float r_j = gateR[b * hiddenSize + j];
        float prevH_j = prevH[b * hiddenSize + j];
        float sigmoid_r_j = sigmoid_derivative(r_j);
        int gateBase = b * 3 * hiddenSize;
        for (int h = 0; h < hiddenSize; h++) {
            float dHCand_h = dGates[gateBase + 2 * hiddenSize + h];
            float dR_contrib_h = dHCand_h * Uh[h * hiddenSize + j];
            sum += dR_contrib_h * prevH_j * sigmoid_r_j;
        }
    }
    dbr[j] += sum;
}

extern ""C"" __global__ __launch_bounds__(256) void gru_cell_backward_dUh_deterministic(
    const float* dGates,      // [batch, 3*hidden]
    const float* gateR,       // [batch, hidden]
    const float* prevH,       // [batch, hidden]
    float* dUh,               // [hidden, hidden]
    int batch, int hiddenSize)
{
    int h = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hiddenSize || j >= hiddenSize) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        float dHCand_h = dGates[b * 3 * hiddenSize + 2 * hiddenSize + h];
        float r_j = gateR[b * hiddenSize + j];
        float prevH_j = prevH[b * hiddenSize + j];
        sum += dHCand_h * r_j * prevH_j;
    }
    dUh[h * hiddenSize + j] += sum;
}

// ===========================================================================
// GRU BACKWARD INPUT KERNEL
// ===========================================================================

// Computes gradient with respect to input
extern ""C"" __global__ __launch_bounds__(256) void gru_backward_input(
    const float* dGates,      // [batch, 3*hidden] - dZ, dR, dH concatenated
    const float* Wz, const float* Wr, const float* Wh,
    float* dInput,            // [batch, input]
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

    // Accumulate gradients from all 3 gates
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

// ===========================================================================
// GRU BACKWARD PREV HIDDEN KERNEL
// ===========================================================================

// Computes gradient with respect to previous hidden state
extern ""C"" __global__ __launch_bounds__(256) void gru_backward_prevh(
    const float* dH,          // [batch, hidden] - total gradient to h_t
    const float* dGates,      // [batch, 3*hidden]
    const float* gateZ,       // [batch, hidden]
    const float* gateR,       // [batch, hidden]
    const float* Uz, const float* Ur, const float* Uh,
    float* dPrevH,            // [batch, hidden]
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

    // Direct gradient from h_t = (1-z)*h_prev + z*h_cand
    float grad = dH[gid] * (1.0f - z);

    // Gradient through gates
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

// ===========================================================================
// GRU SEQUENCE BACKWARD KERNEL
// ===========================================================================

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


// gru_backward_sequence — bit-deterministic split (issue #382). Mirror of HIP.
// dGates_t[T, B, 3*H] scratch buffer, layout [dZ, dR, dHCand] per (b, h).
// The precompute pass is deferred (full deterministic precompute requires
// host-driven per-timestep external pipeline using
// gru_cell_backward_unified + gru_backward_prevh_unified +
// gru_accumulate_weight_gradients_deterministic + gru_backward_input).
// These four accumulator kernels eliminate weight/bias/input atomic sites
// once dGates_t scratch is populated.

extern ""C"" __global__ __launch_bounds__(256) void gru_backward_sequence_dWi_deterministic(
    const float* input, const float* dGates_t,
    float* dWz, float* dWr, float* dWh,
    int batch, int timeSteps, int inputSize, int hiddenSize)
{
    int gateType = blockIdx.x / hiddenSize;
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
// GRU COMPUTE GATE GRADIENTS KERNEL
// ===========================================================================

// Computes gate gradients from hidden gradient
// dR[h] is computed correctly as: sum_k(dHCand_k * Uh[k,h]) * prevH[h] * sigmoid'(r[h])
// This is because r[h] is used to gate prevH[h] for all output positions k in the candidate computation
extern ""C"" __global__ __launch_bounds__(256) void gru_compute_gate_gradients(
    const float* dH,          // [batch, hidden]
    const float* gateZ,       // [batch, hidden]
    const float* gateR,       // [batch, hidden]
    const float* gateHCand,   // [batch, hidden]
    const float* prevH,       // [batch, hidden]
    const float* Uh,          // [hidden, hidden]
    float* dGates,            // [batch, 3*hidden] - output
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Get cached values
    float z = gateZ[gid];
    float r = gateR[gid];
    float h_cand = gateHCand[gid];
    float h_prev = prevH[gid];

    // Gradient from output
    float dh = dH[gid];

    // Gradient through candidate
    float dHCand = dh * z * tanh_derivative(h_cand);

    // Gradient through update gate
    float dZ = dh * (h_cand - h_prev) * sigmoid_derivative(z);

    // Gradient through reset gate
    // Forward: candidate_k = tanh(... + sum_j(Uh[k,j] * r[j] * prevH[j]) + ...)
    // So r[h] affects all candidate outputs k through the term Uh[k,h] * r[h] * prevH[h]
    // dL/dr[h] = sum_k(dHCand_k * Uh[k,h]) * prevH[h]
    // dL/d(preR[h]) = dL/dr[h] * sigmoid'(r[h])
    float dR_sum = 0.0f;
    for (int k = 0; k < hiddenSize; k++) {
        // Compute dHCand for output k
        float dHCand_k = dH[b * hiddenSize + k] * gateZ[b * hiddenSize + k] *
                         tanh_derivative(gateHCand[b * hiddenSize + k]);
        dR_sum += dHCand_k * Uh[k * hiddenSize + h];
    }
    float dR = dR_sum * h_prev * sigmoid_derivative(r);

    // Store gate gradients
    int gateOffset = b * 3 * hiddenSize;
    dGates[gateOffset + h] = dZ;
    dGates[gateOffset + hiddenSize + h] = dR;
    dGates[gateOffset + 2 * hiddenSize + h] = dHCand;
}

// ===========================================================================
// GRU WEIGHT GRADIENT ACCUMULATION KERNEL
// ===========================================================================

// Accumulates weight gradients across batch dimension.
// NON-DETERMINISTIC across timestep invocations (issue #382): each (gateIdx, colIdx)
// cell is written by exactly one thread per launch (atomic is over-defensive), but
// when the kernel is called per timestep the cross-timestep accumulation ordering
// is scheduler-dependent. See gru_accumulate_weight_gradients_deterministic below.
extern ""C"" __global__ __launch_bounds__(256) void gru_accumulate_weight_gradients(
    const float* input,       // [batch, inputSize]
    const float* prevH,       // [batch, hiddenSize]
    const float* gateR,       // [batch, hiddenSize]
    const float* dGates,      // [batch, 3*hiddenSize]
    float* dWz, float* dWr, float* dWh,
    float* dUz, float* dUr, float* dUh,
    float* dbz, float* dbr, float* dbh,
    int batch,
    int inputSize,
    int hiddenSize)
{
    // Grid: (3*hidden, max(input, hidden))
    int gateIdx = blockIdx.x;  // Which gate row (0-hidden*3)
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (gateIdx >= 3 * hiddenSize) return;

    int gateType = gateIdx / hiddenSize;  // 0=Z, 1=R, 2=H
    int h = gateIdx % hiddenSize;

    // Accumulate input weight gradients
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

    // Accumulate hidden weight gradients
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

    // Accumulate bias gradients (only first column handles this)
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
// Direct += instead of atomicAdd (each cell has exactly one writer per launch).
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
    for (int h = 0; h < hiddenSize; h++) {
        int batchHiddenIdx = b * hiddenSize + h;

        float dR = gradGateR[batchHiddenIdx];
        float dZ = gradGateZ[batchHiddenIdx];
        float dN = gradGateN[batchHiddenIdx];
        float r = gateR[batchHiddenIdx];

        // weightsHh layout: [R weights, Z weights, N weights] each [hiddenSize, hiddenSize]
        // R weights: weightsHh[h * hiddenSize + j] for Ur[h, j]
        // Z weights: weightsHh[(hiddenSize + h) * hiddenSize + j] for Uz[h, j]
        // N weights: weightsHh[(2 * hiddenSize + h) * hiddenSize + j] for Uh[h, j]
        gradSum += dR * weightsHh[h * hiddenSize + j];
        gradSum += dZ * weightsHh[(hiddenSize + h) * hiddenSize + j];
        gradSum += dN * r * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
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
            "gru_cell_backward",
            "gru_cell_backward_compute_gates_deterministic",
            "gru_cell_backward_dPrevH_reset_deterministic",
            "gru_cell_backward_dbr_deterministic",
            "gru_cell_backward_dUh_deterministic",
            "gru_backward_input",
            "gru_backward_prevh",
            "gru_backward_sequence",
            "gru_backward_sequence_dWi_deterministic",
            "gru_backward_sequence_dUi_deterministic",
            "gru_backward_sequence_dBias_deterministic",
            "gru_backward_sequence_dInput_deterministic",
            "gru_compute_gate_gradients",
            "gru_accumulate_weight_gradients",
            "gru_accumulate_weight_gradients_deterministic",
            "gru_cell_backward_unified",
            "gru_backward_prevh_unified"
        };
    }
}
