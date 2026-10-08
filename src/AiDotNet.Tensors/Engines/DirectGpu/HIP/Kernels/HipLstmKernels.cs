// Copyright (c) AiDotNet. All rights reserved.
// HIP kernels for standard LSTM (Long Short-Term Memory) neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT on AMD GPUs.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;

/// <summary>
/// HIP kernels for sequence-level LSTM operations on AMD GPUs.
/// Implements full forward and backward passes for LSTM layers processing entire sequences.
/// </summary>
internal static class HipLstmKernels
{
    public static string GetSource()
    {
        return @"
#include <hip/hip_runtime.h>

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
// LSTM CELL FORWARD KERNEL (Single Timestep)
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(1024) void lstm_cell_forward(
    const float* input,
    const float* prevH,
    const float* prevC,
    const float* Wi,
    const float* Wh,
    const float* bias,
    float* newH,
    float* newC,
    float* gateF,
    float* gateI,
    float* gateC,
    float* gateO,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float sumF = bias[h];
    float sumI = bias[hiddenSize + h];
    float sumC = bias[2 * hiddenSize + h];
    float sumO = bias[3 * hiddenSize + h];

    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        sumF += Wi[h * inputSize + i] * x_val;
        sumI += Wi[(hiddenSize + h) * inputSize + i] * x_val;
        sumC += Wi[(2 * hiddenSize + h) * inputSize + i] * x_val;
        sumO += Wi[(3 * hiddenSize + h) * inputSize + i] * x_val;
    }

    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        sumF += Wh[h * hiddenSize + j] * h_val;
        sumI += Wh[(hiddenSize + h) * hiddenSize + j] * h_val;
        sumC += Wh[(2 * hiddenSize + h) * hiddenSize + j] * h_val;
        sumO += Wh[(3 * hiddenSize + h) * hiddenSize + j] * h_val;
    }

    float f = sigmoid(sumF);
    float i_gate = sigmoid(sumI);
    float c_candidate = tanhf(sumC);
    float o = sigmoid(sumO);

    float prevCVal = prevC[gid];
    float newCVal = f * prevCVal + i_gate * c_candidate;
    float newHVal = o * tanhf(newCVal);

    newC[gid] = newCVal;
    newH[gid] = newHVal;
    gateF[gid] = f;
    gateI[gid] = i_gate;
    gateC[gid] = c_candidate;
    gateO[gid] = o;
}

// ===========================================================================
// LSTM SEQUENCE FORWARD KERNEL
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(1024) void lstm_forward_sequence(
    const float* input,
    const float* h_init,
    const float* c_init,
    const float* Wi,      // [4*hidden, input]
    const float* Wh,      // [4*hidden, hidden]
    const float* biasIh,  // [4*hidden] - input-hidden bias
    const float* biasHh,  // [4*hidden] - hidden-hidden bias
    float* output,
    float* h_states,      // Cache: [timeSteps, batch, hidden]
    float* c_states,      // Cache: [timeSteps, batch, hidden]
    float* gates,         // Cache: [timeSteps, batch, 4*hidden]
    int batch,
    int timeSteps,
    int inputSize,
    int hiddenSize)
{
    // Each thread handles one (batch, hidden) element
    // Outer loop over timesteps is sequential
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    // Use isValid flag instead of early return to avoid __syncthreads deadlock
    bool isValid = gid < totalElements;

    int b = isValid ? (gid / hiddenSize) : 0;
    int h_idx = isValid ? (gid % hiddenSize) : 0;

    // Initialize states (only valid threads)
    float h_val = isValid ? h_init[gid] : 0.0f;
    float c_val = isValid ? c_init[gid] : 0.0f;

    // Process each timestep
    for (int t = 0; t < timeSteps; t++) {
        // Compute gate pre-activations (only valid threads)
        float sumF = 0.0f, sumI = 0.0f, sumC = 0.0f, sumO = 0.0f;
        float f = 0.0f, i_gate = 0.0f, c_candidate = 0.0f, o = 0.0f;
        float prev_c = 0.0f;

        if (isValid) {
            // Gate slice layout is PyTorch/CPU order i,f,g,o: slice 0 = INPUT, slice 1 = FORGET,
            // slice 2 = cell-candidate (g), slice 3 = OUTPUT. This kernel keeps variables named by ROLE
            // (sumF=forget, sumI=input), so forget reads slice 1 and input reads slice 0. (The original
            // code read forget from slice 0 → it multiplied prev_c by the INPUT gate: a gross parity bug.)
            // Bias is sum of input-hidden and hidden-hidden biases
            sumI = biasIh[h_idx] + biasHh[h_idx];                                       // slice 0 = input
            sumF = biasIh[hiddenSize + h_idx] + biasHh[hiddenSize + h_idx];             // slice 1 = forget
            sumC = biasIh[2 * hiddenSize + h_idx] + biasHh[2 * hiddenSize + h_idx];
            sumO = biasIh[3 * hiddenSize + h_idx] + biasHh[3 * hiddenSize + h_idx];

            // Input contribution: Wi * x_t
            int inputOffset = (b * timeSteps + t) * inputSize;
            for (int i = 0; i < inputSize; i++) {
                float x_val = input[inputOffset + i];
                sumI += Wi[h_idx * inputSize + i] * x_val;                               // slice 0 = input
                sumF += Wi[(hiddenSize + h_idx) * inputSize + i] * x_val;                // slice 1 = forget
                sumC += Wi[(2 * hiddenSize + h_idx) * inputSize + i] * x_val;
                sumO += Wi[(3 * hiddenSize + h_idx) * inputSize + i] * x_val;
            }

            // Hidden contribution: Wh * h_prev
            // Need to read h_val from all hidden units - use shared memory for efficiency
            for (int j = 0; j < hiddenSize; j++) {
                // Read from prev timestep's stored h, or from h_val if same element
                float hj;
                if (t == 0) {
                    hj = h_init[b * hiddenSize + j];
                } else {
                    // Read from cached h_states for previous timestep
                    hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
                }
                sumI += Wh[h_idx * hiddenSize + j] * hj;                                 // slice 0 = input
                sumF += Wh[(hiddenSize + h_idx) * hiddenSize + j] * hj;                  // slice 1 = forget
                sumC += Wh[(2 * hiddenSize + h_idx) * hiddenSize + j] * hj;
                sumO += Wh[(3 * hiddenSize + h_idx) * hiddenSize + j] * hj;
            }

            // Apply activations
            f = sigmoid(sumF);
            i_gate = sigmoid(sumI);
            c_candidate = tanhf(sumC);
            o = sigmoid(sumO);

            // Previous cell state
            if (t == 0) {
                prev_c = c_init[gid];
            } else {
                prev_c = c_states[(t - 1) * batch * hiddenSize + gid];
            }

            // Update cell state
            c_val = f * prev_c + i_gate * c_candidate;

            // Update hidden state
            h_val = o * tanhf(c_val);

            // Store states for output and caching
            int stateOffset = t * batch * hiddenSize + gid;
            h_states[stateOffset] = h_val;
            c_states[stateOffset] = c_val;

            // Store output
            output[(b * timeSteps + t) * hiddenSize + h_idx] = h_val;

            // Store gates for backward pass
            int gateOffset = t * batch * 4 * hiddenSize + b * 4 * hiddenSize;
            gates[gateOffset + h_idx] = f;
            gates[gateOffset + hiddenSize + h_idx] = i_gate;
            gates[gateOffset + 2 * hiddenSize + h_idx] = c_candidate;
            gates[gateOffset + 3 * hiddenSize + h_idx] = o;
        }

        // Sync to ensure all threads have written h_states before next iteration
        // All threads (valid and invalid) must reach this barrier
        __syncthreads();
    }
}

// ===========================================================================
// LSTM BACKWARD KERNELS
// ===========================================================================

extern ""C"" __global__ __launch_bounds__(1024) void lstm_backward_input(
    const float* dGates,
    const float* Wi,
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
        float dF = dGates[b * 4 * hiddenSize + h];
        float dI = dGates[b * 4 * hiddenSize + hiddenSize + h];
        float dC = dGates[b * 4 * hiddenSize + 2 * hiddenSize + h];
        float dO = dGates[b * 4 * hiddenSize + 3 * hiddenSize + h];

        grad += dF * Wi[h * inputSize + i];
        grad += dI * Wi[(hiddenSize + h) * inputSize + i];
        grad += dC * Wi[(2 * hiddenSize + h) * inputSize + i];
        grad += dO * Wi[(3 * hiddenSize + h) * inputSize + i];
    }

    dInput[gid] = grad;
}

extern ""C"" __global__ __launch_bounds__(1024) void lstm_backward_prevh(
    const float* dGates,
    const float* Wh,
    float* dPrevH,
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float grad = 0.0f;

    for (int h = 0; h < hiddenSize; h++) {
        float dF = dGates[b * 4 * hiddenSize + h];
        float dI = dGates[b * 4 * hiddenSize + hiddenSize + h];
        float dC = dGates[b * 4 * hiddenSize + 2 * hiddenSize + h];
        float dO = dGates[b * 4 * hiddenSize + 3 * hiddenSize + h];

        grad += dF * Wh[h * hiddenSize + j];
        grad += dI * Wh[(hiddenSize + h) * hiddenSize + j];
        grad += dC * Wh[(2 * hiddenSize + h) * hiddenSize + j];
        grad += dO * Wh[(3 * hiddenSize + h) * hiddenSize + j];
    }

    dPrevH[gid] = grad;
}

extern ""C"" __global__ __launch_bounds__(1024) void lstm_compute_gate_gradients(
    const float* dH,
    const float* dC_next,
    const float* gateF,
    const float* gateI,
    const float* gateC,
    const float* gateO,
    const float* prevC,
    const float* newC,
    float* dGates,
    float* dPrevC,
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    float f = gateF[gid];
    float i_gate = gateI[gid];
    float c_candidate = gateC[gid];
    float o = gateO[gid];
    float prevCVal = prevC[gid];
    float newCVal = newC[gid];

    float dh = dH[gid];
    float tanh_c = tanhf(newCVal);

    float dO = dh * tanh_c * sigmoid_derivative(o);
    float dC_from_H = dh * o * tanh_derivative(tanh_c);
    float dC = dC_next[gid] + dC_from_H;

    float dF = dC * prevCVal * sigmoid_derivative(f);
    float dI = dC * c_candidate * sigmoid_derivative(i_gate);
    float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);

    int gateOffset = b * 4 * hiddenSize;
    dGates[gateOffset + h] = dF;
    dGates[gateOffset + hiddenSize + h] = dI;
    dGates[gateOffset + 2 * hiddenSize + h] = dCCandidate;
    dGates[gateOffset + 3 * hiddenSize + h] = dO;

    dPrevC[gid] = dC * f;
}

extern ""C"" __global__ __launch_bounds__(1024) void lstm_accumulate_weight_gradients(
    const float* input,
    const float* prevH,
    const float* dGates,
    float* dWi,
    float* dWh,
    float* dBias,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gateIdx = blockIdx.x;
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (gateIdx >= 4 * hiddenSize) return;

    if (colIdx < inputSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float x_val = input[b * inputSize + colIdx];
            grad += dGate * x_val;
        }
        atomicAdd(&dWi[gateIdx * inputSize + colIdx], grad);
    }

    if (colIdx < hiddenSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float h_val = prevH[b * hiddenSize + colIdx];
            grad += dGate * h_val;
        }
        atomicAdd(&dWh[gateIdx * hiddenSize + colIdx], grad);
    }

    if (colIdx == 0) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            grad += dGates[b * 4 * hiddenSize + gateIdx];
        }
        atomicAdd(&dBias[gateIdx], grad);
    }
}

// lstm_accumulate_weight_gradients — bit-deterministic variant (issue #382).
// Each (gateIdx, colIdx) cell has exactly one writer per launch; direct +=
// instead of atomicAdd. See CUDA equivalent for full rationale.
extern ""C"" __global__ __launch_bounds__(1024) void lstm_accumulate_weight_gradients_deterministic(
    const float* input, const float* prevH, const float* dGates,
    float* dWi, float* dWh, float* dBias,
    int batch, int inputSize, int hiddenSize)
{
    int gateIdx = blockIdx.x;
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;
    if (gateIdx >= 4 * hiddenSize) return;

    if (colIdx < inputSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float x_val = input[b * inputSize + colIdx];
            grad += dGate * x_val;
        }
        dWi[gateIdx * inputSize + colIdx] += grad;
    }

    if (colIdx < hiddenSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float h_val = prevH[b * hiddenSize + colIdx];
            grad += dGate * h_val;
        }
        dWh[gateIdx * hiddenSize + colIdx] += grad;
    }

    if (colIdx == 0) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            grad += dGates[b * 4 * hiddenSize + gateIdx];
        }
        dBias[gateIdx] += grad;
    }
}

extern ""C"" __global__ __launch_bounds__(1024) void lstm_backward_sequence(
    const float* gradOutput,  // [batch, timeSteps, hidden]
    const float* h_states,    // [timeSteps, batch, hidden]
    const float* c_states,    // [timeSteps, batch, hidden]
    const float* gates,       // [timeSteps, batch, 4*hidden]
    const float* c_init,      // [batch, hidden]
    const float* h_init,      // [batch, hidden]
    const float* input,       // [batch, timeSteps, input]
    const float* Wi,          // [4*hidden, input]
    const float* Wh,          // [4*hidden, hidden]
    float* gradInput,         // [batch, timeSteps, input]
    float* dWi,               // [4*hidden, input]
    float* dWh,               // [4*hidden, hidden]
    float* dBiasIh,           // [4*hidden] - input-hidden bias gradient
    float* dBiasHh,           // [4*hidden] - hidden-hidden bias gradient
    float* dH_init,           // [batch, hidden]
    float* dC_init,           // [batch, hidden]
    int batch,
    int timeSteps,
    int inputSize,
    int hiddenSize)
{
    // Each thread handles one (batch, hidden) element
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    // Use isValid flag instead of early return to avoid __syncthreads deadlock
    bool isValid = gid < totalElements;

    int b = isValid ? (gid / hiddenSize) : 0;
    int h_idx = isValid ? (gid % hiddenSize) : 0;

    // Initialize gradients for recurrence
    float dH = 0.0f;
    float dC = 0.0f;

    // Clear dH_init buffer for use as intermediate storage during BPTT (only valid threads)
    if (isValid) {
        dH_init[gid] = 0.0f;
    }
    __syncthreads();

    // Process timesteps in reverse (BPTT)
    for (int t = timeSteps - 1; t >= 0; t--) {
        // Read accumulated recurrent gradient from previous iteration (if any)
        if (isValid && t < timeSteps - 1) {
            dH = dH_init[gid];
            dH_init[gid] = 0.0f;  // Clear for next accumulation
        }
        __syncthreads();

        if (isValid) {
            // Add gradient from output at this timestep
            dH += gradOutput[(b * timeSteps + t) * hiddenSize + h_idx];

            // Get cached gate values
            int gateOffset = t * batch * 4 * hiddenSize + b * 4 * hiddenSize;
            float f = gates[gateOffset + h_idx];
            float i_gate = gates[gateOffset + hiddenSize + h_idx];
            float c_candidate = gates[gateOffset + 2 * hiddenSize + h_idx];
            float o = gates[gateOffset + 3 * hiddenSize + h_idx];

            // Get cell states
            int stateOffset = t * batch * hiddenSize + gid;
            float c_t = c_states[stateOffset];
            float c_prev;
            if (t == 0) {
                c_prev = c_init[gid];
            } else {
                c_prev = c_states[(t - 1) * batch * hiddenSize + gid];
            }

            // tanh(c_t)
            float tanh_c = tanhf(c_t);

            // Gradient through output gate
            float dO = dH * tanh_c * sigmoid_derivative(o);

            // Gradient to cell state from hidden state
            float dC_from_H = dH * o * tanh_derivative(tanh_c);

            // Total cell state gradient
            dC += dC_from_H;

            // Gradient through cell state equation
            float dF = dC * c_prev * sigmoid_derivative(f);
            float dI = dC * c_candidate * sigmoid_derivative(i_gate);
            float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);

            // Gradient to previous cell state for next iteration
            float dC_prev = dC * f;

            // Map the gate-ROLE derivatives (dF=forget, dI=input, dCCandidate=g, dO=output) onto the
            // WEIGHT-ROW order, which is PyTorch i,f,g,o: row0=input, row1=forget, row2=g, row3=o. So the
            // input weight row (row0) receives dI and the forget weight row (row1) receives dF. (The gate
            // CACHE is [f,i,g,o] and is read correctly above; only the weight-row mapping needs this swap —
            // the original kernel put dF on row0/dI on row1, i.e. gradients on the wrong weight rows.)
            float dRow0 = dI;            // input  gate -> weight row 0
            float dRow1 = dF;            // forget gate -> weight row 1
            float dRow2 = dCCandidate;   // cell candidate g -> weight row 2
            float dRow3 = dO;            // output gate -> weight row 3

            // Get previous hidden state for weight gradients
            float h_prev_val;
            if (t == 0) {
                h_prev_val = h_init[b * hiddenSize + h_idx];
            } else {
                h_prev_val = h_states[(t - 1) * batch * hiddenSize + gid];
            }

            // Accumulate weight gradients (atomic for multi-thread safety)
            int inputOffset = (b * timeSteps + t) * inputSize;
            for (int i = 0; i < inputSize; i++) {
                float x_val = input[inputOffset + i];
                atomicAdd(&dWi[h_idx * inputSize + i], dRow0 * x_val);
                atomicAdd(&dWi[(hiddenSize + h_idx) * inputSize + i], dRow1 * x_val);
                atomicAdd(&dWi[(2 * hiddenSize + h_idx) * inputSize + i], dRow2 * x_val);
                atomicAdd(&dWi[(3 * hiddenSize + h_idx) * inputSize + i], dRow3 * x_val);
            }

            // Hidden weight gradients - need all prev hidden values
            for (int j = 0; j < hiddenSize; j++) {
                float hj;
                if (t == 0) {
                    hj = h_init[b * hiddenSize + j];
                } else {
                    hj = h_states[(t - 1) * batch * hiddenSize + b * hiddenSize + j];
                }
                atomicAdd(&dWh[h_idx * hiddenSize + j], dRow0 * hj);
                atomicAdd(&dWh[(hiddenSize + h_idx) * hiddenSize + j], dRow1 * hj);
                atomicAdd(&dWh[(2 * hiddenSize + h_idx) * hiddenSize + j], dRow2 * hj);
                atomicAdd(&dWh[(3 * hiddenSize + h_idx) * hiddenSize + j], dRow3 * hj);
            }

            // Bias gradients - same gradient flows to both biasIh and biasHh since they're summed
            atomicAdd(&dBiasIh[h_idx], dRow0);
            atomicAdd(&dBiasIh[hiddenSize + h_idx], dRow1);
            atomicAdd(&dBiasIh[2 * hiddenSize + h_idx], dRow2);
            atomicAdd(&dBiasIh[3 * hiddenSize + h_idx], dRow3);
            atomicAdd(&dBiasHh[h_idx], dRow0);
            atomicAdd(&dBiasHh[hiddenSize + h_idx], dRow1);
            atomicAdd(&dBiasHh[2 * hiddenSize + h_idx], dRow2);
            atomicAdd(&dBiasHh[3 * hiddenSize + h_idx], dRow3);

            // Compute gradient to input at this timestep
            int gradInputOffset = (b * timeSteps + t) * inputSize;
            for (int i = 0; i < inputSize; i++) {
                float grad_i = 0.0f;
                grad_i += dRow0 * Wi[h_idx * inputSize + i];
                grad_i += dRow1 * Wi[(hiddenSize + h_idx) * inputSize + i];
                grad_i += dRow2 * Wi[(2 * hiddenSize + h_idx) * inputSize + i];
                grad_i += dRow3 * Wi[(3 * hiddenSize + h_idx) * inputSize + i];
                atomicAdd(&gradInput[gradInputOffset + i], grad_i);
            }

            // Gradient to previous hidden state for BPTT
            // dH_prev[j] = sum_k (dGate[k] * Wh[k, j]) for all four gates
            // This is a matrix-vector product: dH_prev = Wh^T @ dGates
            // Each thread k contributes: dGates[k] * Wh[k, j] for all j
            // Accumulate to dH_init buffer (used as temp storage during loop, final output at t=0)
            for (int j = 0; j < hiddenSize; j++) {
                // Contribution from gate derivatives at position h_idx to hidden unit j
                // Wh layout: [4*hiddenSize, hiddenSize], so Wh[k, j] = Wh[k * hiddenSize + j]
                float contrib = dRow0 * Wh[h_idx * hiddenSize + j];
                contrib += dRow1 * Wh[(hiddenSize + h_idx) * hiddenSize + j];
                contrib += dRow2 * Wh[(2 * hiddenSize + h_idx) * hiddenSize + j];
                contrib += dRow3 * Wh[(3 * hiddenSize + h_idx) * hiddenSize + j];
                atomicAdd(&dH_init[b * hiddenSize + j], contrib);
            }

            // Update dC for next iteration
            dC = dC_prev;
        }

        // All threads must reach this barrier
        __syncthreads();
    }

    // Store initial cell state gradient (only valid threads)
    // dH_init already contains the accumulated gradient for h_init from the t=0 iteration
    if (isValid) {
        dC_init[gid] = dC;
    }
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
            "lstm_cell_forward",
            "lstm_forward_sequence",
            "lstm_backward_input",
            "lstm_backward_prevh",
            "lstm_compute_gate_gradients",
            "lstm_accumulate_weight_gradients",
            "lstm_accumulate_weight_gradients_deterministic",
            "lstm_backward_sequence",
        };
    }
}
