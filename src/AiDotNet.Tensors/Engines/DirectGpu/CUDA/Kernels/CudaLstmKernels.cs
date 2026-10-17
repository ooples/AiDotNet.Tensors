// Copyright (c) AiDotNet. All rights reserved.
// CUDA kernels for standard LSTM (Long Short-Term Memory) neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

/// <summary>
/// CUDA kernels for sequence-level LSTM operations.
/// Implements full forward and backward passes for LSTM layers processing entire sequences.
/// </summary>
/// <remarks>
/// LSTM equations:
/// f_t = sigmoid(W_f * x_t + U_f * h_{t-1} + b_f)  // forget gate
/// i_t = sigmoid(W_i * x_t + U_i * h_{t-1} + b_i)  // input gate
/// c̃_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c)     // candidate cell
/// o_t = sigmoid(W_o * x_t + U_o * h_{t-1} + b_o)  // output gate
/// c_t = f_t * c_{t-1} + i_t * c̃_t                 // cell state
/// h_t = o_t * tanh(c_t)                           // hidden state
/// </remarks>
internal static class CudaLstmKernels
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

// tanhf is accurate only to about 1e-6 here (measured: the CUDA LSTM forward was 25-60x less accurate than an
// fp32 implementation of the same math, all of it from tanhf, with or without --use_fast_math), and BPTT compounds
// it through every timestep. expm1f keeps small arguments exact, and |x| > 9 already rounds to +-1 in fp32.
__device__ __forceinline__ float lstm_tanh(float x) {
    float ax = fabsf(x);
    if (ax > 9.0f) return copysignf(1.0f, x);
    float e = expm1f(2.0f * ax);
    return copysignf(e / (e + 2.0f), x);
}

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

// LSTM cell forward pass for a single time step
// input: [batch, inputSize]
// prevH: [batch, hiddenSize]
// prevC: [batch, hiddenSize]
// Wi: [4 * hiddenSize, inputSize] (Wf, Wi, Wc, Wo stacked)
// Wh: [4 * hiddenSize, hiddenSize] (Uf, Ui, Uc, Uo stacked)
// bias: [4 * hiddenSize]
// output newH, newC: [batch, hiddenSize]
// gateF, gateI, gateC, gateO: [batch, hiddenSize] (cached for backward)
extern ""C"" __global__ __launch_bounds__(1024) void lstm_cell_forward(
    const float* __restrict__ input,
    const float* __restrict__ prevH,
    const float* __restrict__ prevC,
    const float* __restrict__ Wi,      // [4*hidden, input]
    const float* __restrict__ Wh,      // [4*hidden, hidden]
    const float* __restrict__ bias,    // [4*hidden]
    float* __restrict__ newH,
    float* __restrict__ newC,
    float* __restrict__ gateF,
    float* __restrict__ gateI,
    float* __restrict__ gateC,
    float* __restrict__ gateO,
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Compute gate pre-activations via dot products
    float sumF = bias[h];
    float sumI = bias[hiddenSize + h];
    float sumC = bias[2 * hiddenSize + h];
    float sumO = bias[3 * hiddenSize + h];

    // Input contribution: Wi * x
    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        sumF += Wi[h * inputSize + i] * x_val;
        sumI += Wi[(hiddenSize + h) * inputSize + i] * x_val;
        sumC += Wi[(2 * hiddenSize + h) * inputSize + i] * x_val;
        sumO += Wi[(3 * hiddenSize + h) * inputSize + i] * x_val;
    }

    // Hidden contribution: Wh * h_prev
    for (int j = 0; j < hiddenSize; j++) {
        float h_val = prevH[b * hiddenSize + j];
        sumF += Wh[h * hiddenSize + j] * h_val;
        sumI += Wh[(hiddenSize + h) * hiddenSize + j] * h_val;
        sumC += Wh[(2 * hiddenSize + h) * hiddenSize + j] * h_val;
        sumO += Wh[(3 * hiddenSize + h) * hiddenSize + j] * h_val;
    }

    // Apply activations
    float f = sigmoid(sumF);
    float i_gate = sigmoid(sumI);
    float c_candidate = lstm_tanh(sumC);
    float o = sigmoid(sumO);

    // Compute new cell state: newC = f * prevC + i * c_candidate
    float prevCVal = prevC[gid];
    float newCVal = f * prevCVal + i_gate * c_candidate;

    // Compute new hidden state: newH = o * tanh(newC)
    float newHVal = o * lstm_tanh(newCVal);

    // Store outputs
    newC[gid] = newCVal;
    newH[gid] = newHVal;

    // Store gate values for backward pass
    gateF[gid] = f;
    gateI[gid] = i_gate;
    gateC[gid] = c_candidate;
    gateO[gid] = o;
}

// ===========================================================================
// LSTM SEQUENCE FORWARD KERNEL
// ===========================================================================

// LSTM forward pass for entire sequence
// Processes all timesteps sequentially within the kernel
// input: [batch, timeSteps, inputSize]
// h_init: [batch, hiddenSize]
// c_init: [batch, hiddenSize]
// output: [batch, timeSteps, hiddenSize]
// h_states: [timeSteps, batch, hiddenSize] (cached for backward)
// c_states: [timeSteps, batch, hiddenSize] (cached for backward)
// gates: [timeSteps, batch, 4*hiddenSize] (F,I,C,O cached for backward)
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
            c_candidate = lstm_tanh(sumC);
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
            h_val = o * lstm_tanh(c_val);

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
// LSTM CELL BACKWARD KERNEL (Single Timestep)
// ===========================================================================

// Computes gradients for a single LSTM cell timestep
// dH: gradient from output [batch, hiddenSize]
// dC_next: gradient from next cell state [batch, hiddenSize]
// Returns: dInput, dPrevH, dPrevC, and accumulated weight/bias gradients
extern ""C"" __global__ __launch_bounds__(1024) void lstm_cell_backward(
    const float* dH,          // [batch, hidden]
    const float* dC_next,     // [batch, hidden] gradient from next timestep
    const float* gateF,       // [batch, hidden]
    const float* gateI,       // [batch, hidden]
    const float* gateC,       // [batch, hidden]
    const float* gateO,       // [batch, hidden]
    const float* prevC,       // [batch, hidden]
    const float* newC,        // [batch, hidden]
    const float* prevH,       // [batch, hidden]
    const float* input,       // [batch, input]
    const float* Wi,          // [4*hidden, input]
    const float* Wh,          // [4*hidden, hidden]
    float* dPrevH,            // [batch, hidden]
    float* dPrevC,            // [batch, hidden]
    float* dInput,            // [batch, input]
    float* dWi,               // [4*hidden, input] - atomic add
    float* dWh,               // [4*hidden, hidden] - atomic add
    float* dBias,             // [4*hidden] - atomic add
    int batch,
    int inputSize,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Get cached values
    float f = gateF[gid];
    float i_gate = gateI[gid];
    float c_candidate = gateC[gid];
    float o = gateO[gid];
    float prevCVal = prevC[gid];
    float newCVal = newC[gid];

    // Gradient from output hidden state
    float dh = dH[gid];

    // tanh(c_t)
    float tanh_c = lstm_tanh(newCVal);

    // Gradient through output gate
    float dO = dh * tanh_c * sigmoid_derivative(o);

    // Gradient to cell state from hidden state
    float dC_from_H = dh * o * tanh_derivative(tanh_c);

    // Total cell state gradient (from next timestep + from output)
    float dC = dC_next[gid] + dC_from_H;

    // Gradient through cell state equation: c_t = f * c_{t-1} + i * c_candidate
    float dF = dC * prevCVal * sigmoid_derivative(f);
    float dI = dC * c_candidate * sigmoid_derivative(i_gate);
    float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);

    // Gradient to previous cell state
    dPrevC[gid] = dC * f;

    // Gradient to previous hidden state (through all gates)
    // dPrevH[b,j] = sum_h (dF_h * Wh[h,j] + dI_h * Wh[H+h,j] + dCCandidate_h * Wh[2H+h,j] + dO_h * Wh[3H+h,j])
    // This thread handles output position h, so we contribute to each j using atomicAdd
    for (int j = 0; j < hiddenSize; j++) {
        float contrib = dF * Wh[h * hiddenSize + j];
        contrib += dI * Wh[(hiddenSize + h) * hiddenSize + j];
        contrib += dCCandidate * Wh[(2 * hiddenSize + h) * hiddenSize + j];
        contrib += dO * Wh[(3 * hiddenSize + h) * hiddenSize + j];
        atomicAdd(&dPrevH[b * hiddenSize + j], contrib);
    }

    // Accumulate weight gradients (using atomic operations)
    for (int i = 0; i < inputSize; i++) {
        float x_val = input[b * inputSize + i];
        atomicAdd(&dWi[h * inputSize + i], dF * x_val);
        atomicAdd(&dWi[(hiddenSize + h) * inputSize + i], dI * x_val);
        atomicAdd(&dWi[(2 * hiddenSize + h) * inputSize + i], dCCandidate * x_val);
        atomicAdd(&dWi[(3 * hiddenSize + h) * inputSize + i], dO * x_val);
    }

    // Accumulate bias gradients
    atomicAdd(&dBias[h], dF);
    atomicAdd(&dBias[hiddenSize + h], dI);
    atomicAdd(&dBias[2 * hiddenSize + h], dCCandidate);
    atomicAdd(&dBias[3 * hiddenSize + h], dO);
}

// lstm_cell_backward — bit-deterministic split (issue #382).
// The original single-pass kernel has three different accumulation patterns that
// each need atomics (dPrevH: scatter-across-h per j; dWi: scatter-across-b per (gate*H+h, i);
// dBias: scatter-across-b per (gate*H+h)). The deterministic version splits into a
// compute-gates kernel that writes dGates[batch, 4*hidden] scratch + dPrevC, plus
// three reduction kernels that each own one output cell and scan inputs in fixed order.
//
// Caller must allocate dGates scratch of size [batch * 4 * hidden] floats and zero
// dPrevH, dWi, dBias before invocation. Workflow:
//   1. lstm_cell_backward_compute_gates_deterministic   -> writes dGates, dPrevC
//   2. lstm_cell_backward_dPrevH_deterministic           -> reads dGates+Wh, writes dPrevH
//   3. lstm_cell_backward_dWi_deterministic              -> reads dGates+input, writes dWi
//   4. lstm_cell_backward_dBias_deterministic            -> reads dGates, writes dBias

extern ""C"" __global__ __launch_bounds__(1024) void lstm_cell_backward_compute_gates_deterministic(
    const float* dH, const float* dC_next,
    const float* gateF, const float* gateI, const float* gateC, const float* gateO,
    const float* prevC, const float* newC,
    float* dGates,             // [batch, 4*hidden] scratch: [dF, dI, dCCandidate, dO] per (b, h)
    float* dPrevC,             // [batch, hidden]
    int batch, int hiddenSize)
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
    float tanh_c = lstm_tanh(newCVal);
    float dO = dh * tanh_c * sigmoid_derivative(o);
    float dC_from_H = dh * o * tanh_derivative(tanh_c);
    float dC = dC_next[gid] + dC_from_H;
    float dF = dC * prevCVal * sigmoid_derivative(f);
    float dI = dC * c_candidate * sigmoid_derivative(i_gate);
    float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);

    dPrevC[gid] = dC * f;

    // dGates layout: [b, gate=0..3, h] flattened as b * 4 * H + gate * H + h
    int gateBase = b * 4 * hiddenSize;
    dGates[gateBase + h] = dF;
    dGates[gateBase + hiddenSize + h] = dI;
    dGates[gateBase + 2 * hiddenSize + h] = dCCandidate;
    dGates[gateBase + 3 * hiddenSize + h] = dO;
}

extern ""C"" __global__ __launch_bounds__(1024) void lstm_cell_backward_dPrevH_deterministic(
    const float* dGates,       // [batch, 4*hidden]
    const float* Wh,           // [4*hidden, hidden]
    float* dPrevH,             // [batch, hidden]
    int batch, int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;
    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    int gateBase = b * 4 * hiddenSize;
    float sum = 0.0f;
    for (int h = 0; h < hiddenSize; h++) {
        float dF = dGates[gateBase + h];
        float dI = dGates[gateBase + hiddenSize + h];
        float dCCandidate = dGates[gateBase + 2 * hiddenSize + h];
        float dO = dGates[gateBase + 3 * hiddenSize + h];
        sum += dF * Wh[h * hiddenSize + j];
        sum += dI * Wh[(hiddenSize + h) * hiddenSize + j];
        sum += dCCandidate * Wh[(2 * hiddenSize + h) * hiddenSize + j];
        sum += dO * Wh[(3 * hiddenSize + h) * hiddenSize + j];
    }
    dPrevH[gid] += sum;
}

extern ""C"" __global__ __launch_bounds__(1024) void lstm_cell_backward_dWi_deterministic(
    const float* dGates,       // [batch, 4*hidden]
    const float* input,        // [batch, input]
    float* dWi,                // [4*hidden, input]
    int batch, int inputSize, int hiddenSize)
{
    int row = blockIdx.x;      // 0..(4*hidden)-1
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;  // 0..(input)-1
    if (row >= 4 * hiddenSize || colIdx >= inputSize) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        float dGate = dGates[b * 4 * hiddenSize + row];
        float x_val = input[b * inputSize + colIdx];
        sum += dGate * x_val;
    }
    dWi[row * inputSize + colIdx] += sum;
}

extern ""C"" __global__ __launch_bounds__(1024) void lstm_cell_backward_dBias_deterministic(
    const float* dGates,       // [batch, 4*hidden]
    float* dBias,              // [4*hidden]
    int batch, int hiddenSize)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // 0..(4*hidden)-1
    if (row >= 4 * hiddenSize) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; b++) {
        sum += dGates[b * 4 * hiddenSize + row];
    }
    dBias[row] += sum;
}

// ===========================================================================
// LSTM BACKWARD INPUT KERNEL
// ===========================================================================

// Computes gradient with respect to input
extern ""C"" __global__ __launch_bounds__(1024) void lstm_backward_input(
    const float* dGates,      // [batch, 4*hidden] - dF, dI, dC, dO concatenated
    const float* Wi,          // [4*hidden, input]
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

    // Accumulate gradients from all 4 gates
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

// ===========================================================================
// LSTM BACKWARD PREV HIDDEN KERNEL
// ===========================================================================

// Computes gradient with respect to previous hidden state
extern ""C"" __global__ __launch_bounds__(1024) void lstm_backward_prevh(
    const float* dGates,      // [batch, 4*hidden]
    const float* Wh,          // [4*hidden, hidden]
    float* dPrevH,            // [batch, hidden]
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float grad = 0.0f;

    // Accumulate gradients from all 4 gates
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

// ===========================================================================
// LSTM SEQUENCE BACKWARD KERNEL
// ===========================================================================

// LSTM backward pass for entire sequence (BPTT)
// Processes timesteps in reverse order
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
            float tanh_c = lstm_tanh(c_t);

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

// ===========================================================================
// LSTM FORWARD RECURRENCE (input projection precomputed)
// ===========================================================================

// The sequential part of the forward only. The caller computes xProj = X Wi^T for all batch*timeSteps rows as one
// GEMM first, so a step reads its input contribution instead of forming it; h_{t-1} is shared through shared
// memory, and the recurrent weights are read transposed (WhT [hidden, 4*hidden]) so that, for a fixed input j,
// the block's threads read consecutive addresses. Outputs and caches are exactly lstm_forward_sequence's.
//   xProj : [batch*timeSteps, 4*hidden]   row r = b*timeSteps + t, weight-row order i,f,g,o
extern ""C"" __global__ __launch_bounds__(1024) void lstm_forward_recurrence(
    const float* xProj,
    const float* h_init,      // [batch, hidden]
    const float* c_init,      // [batch, hidden]
    const float* WhT,         // [hidden, 4*hidden]
    const float* biasIh,      // [4*hidden]
    const float* biasHh,      // [4*hidden]
    float* output,            // [batch, timeSteps, hidden]
    float* h_states,          // [timeSteps, batch, hidden]
    float* c_states,          // [timeSteps, batch, hidden]
    float* gates,             // [timeSteps, batch, 4*hidden], role order f,i,g,o
    int batch,
    int timeSteps,
    int hiddenSize)
{
    extern __shared__ float hPrev[];   // [hidden]
    int b = blockIdx.x;
    int h_idx = threadIdx.x;
    int G = 4 * hiddenSize;
    int gid = b * hiddenSize + h_idx;

    hPrev[h_idx] = h_init[gid];
    float c = c_init[gid];
    float bI = biasIh[h_idx] + biasHh[h_idx];
    float bF = biasIh[hiddenSize + h_idx] + biasHh[hiddenSize + h_idx];
    float bG = biasIh[2 * hiddenSize + h_idx] + biasHh[2 * hiddenSize + h_idx];
    float bO = biasIh[3 * hiddenSize + h_idx] + biasHh[3 * hiddenSize + h_idx];
    __syncthreads();

    for (int t = 0; t < timeSteps; t++) {
        int r = b * timeSteps + t;
        float sI = xProj[r * G + h_idx] + bI;
        float sF = xProj[r * G + hiddenSize + h_idx] + bF;
        float sG = xProj[r * G + 2 * hiddenSize + h_idx] + bG;
        float sO = xProj[r * G + 3 * hiddenSize + h_idx] + bO;
        for (int j = 0; j < hiddenSize; j++) {
            float hj = hPrev[j];
            const float* row = WhT + j * G;
            sI += row[h_idx] * hj;
            sF += row[hiddenSize + h_idx] * hj;
            sG += row[2 * hiddenSize + h_idx] * hj;
            sO += row[3 * hiddenSize + h_idx] * hj;
        }
        float f = sigmoid(sF);
        float i_gate = sigmoid(sI);
        float g = lstm_tanh(sG);
        float o = sigmoid(sO);
        c = f * c + i_gate * g;
        float h = o * lstm_tanh(c);

        int stateOffset = t * batch * hiddenSize + gid;
        h_states[stateOffset] = h;
        c_states[stateOffset] = c;
        output[r * hiddenSize + h_idx] = h;
        int gateOffset = t * batch * G + b * G;
        gates[gateOffset + h_idx] = f;
        gates[gateOffset + hiddenSize + h_idx] = i_gate;
        gates[gateOffset + 2 * hiddenSize + h_idx] = g;
        gates[gateOffset + 3 * hiddenSize + h_idx] = o;

        __syncthreads();   // every thread has read hPrev for this step
        hPrev[h_idx] = h;
        __syncthreads();   // the next step reads the new hPrev
    }
}

// ===========================================================================
// LSTM BACKWARD RECURRENCE (no weight-gradient atomics)
// ===========================================================================

// The sequential part of BPTT only: per timestep, the gate gradients and dH_prev = Wh^T dGates. One block per
// batch row with one thread per hidden unit; the row's 4H gate gradients are shared through shared memory, so
// each thread forms its own dH_prev[j] as a dot product instead of atomically scattering into every j.
// The weight, bias and input gradients are NOT formed here: the kernel writes dGates and the previous hidden
// states, and the caller forms them as GEMMs over all batch*timeSteps rows afterwards
// (dWi = dGates^T X, dWh = dGates^T Hprev, dX = dGates Wi, dBias = dGates^T 1). The previous kernel
// accumulated those with ~4*(input+hidden) contended atomicAdds per thread per timestep, which made it the
// whole cost of a training step.
//   dGates  : [batch*timeSteps, 4*hidden]   row r = b*timeSteps + t, weight-row order i,f,g,o
//   hPrev   : [batch*timeSteps, hidden]     h_{t-1} (h_init at t = 0)
extern ""C"" __global__ __launch_bounds__(1024) void lstm_backward_recurrence(
    const float* gradOutput,  // [batch, timeSteps, hidden]
    const float* h_states,    // [timeSteps, batch, hidden]
    const float* c_states,    // [timeSteps, batch, hidden]
    const float* gates,       // [timeSteps, batch, 4*hidden], cached in role order f,i,g,o
    const float* c_init,      // [batch, hidden]
    const float* h_init,      // [batch, hidden]
    const float* Wh,          // [4*hidden, hidden]
    float* dGates,
    float* hPrev,
    float* dH_init,           // [batch, hidden]
    float* dC_init,           // [batch, hidden]
    int batch,
    int timeSteps,
    int hiddenSize)
{
    extern __shared__ float dG[];   // [4*hidden]
    int b = blockIdx.x;
    int h_idx = threadIdx.x;
    int G = 4 * hiddenSize;
    int gid = b * hiddenSize + h_idx;

    float dHnext = 0.0f;   // dL/dh_t carried from the later timestep
    float dC = 0.0f;

    for (int t = timeSteps - 1; t >= 0; t--) {
        float dH = dHnext + gradOutput[(b * timeSteps + t) * hiddenSize + h_idx];

        int gateOffset = t * batch * G + b * G;
        float f = gates[gateOffset + h_idx];
        float i_gate = gates[gateOffset + hiddenSize + h_idx];
        float g = gates[gateOffset + 2 * hiddenSize + h_idx];
        float o = gates[gateOffset + 3 * hiddenSize + h_idx];

        float c_t = c_states[t * batch * hiddenSize + gid];
        float c_prev = t == 0 ? c_init[gid] : c_states[(t - 1) * batch * hiddenSize + gid];
        float h_prev = t == 0 ? h_init[gid] : h_states[(t - 1) * batch * hiddenSize + gid];

        float tanh_c = lstm_tanh(c_t);
        float dO = dH * tanh_c * sigmoid_derivative(o);
        dC += dH * o * tanh_derivative(tanh_c);
        float dF = dC * c_prev * sigmoid_derivative(f);
        float dI = dC * g * sigmoid_derivative(i_gate);
        float dGc = dC * i_gate * tanh_derivative(g);
        dC = dC * f;

        // Weight-row order i, f, g, o (PyTorch), as lstm_backward_sequence maps it.
        dG[h_idx] = dI;
        dG[hiddenSize + h_idx] = dF;
        dG[2 * hiddenSize + h_idx] = dGc;
        dG[3 * hiddenSize + h_idx] = dO;

        int r = b * timeSteps + t;
        dGates[r * G + h_idx] = dI;
        dGates[r * G + hiddenSize + h_idx] = dF;
        dGates[r * G + 2 * hiddenSize + h_idx] = dGc;
        dGates[r * G + 3 * hiddenSize + h_idx] = dO;
        hPrev[r * hiddenSize + h_idx] = h_prev;
        __syncthreads();

        // dH_prev[j] = sum_k dG[k] * Wh[k, j]; for a fixed k the block reads one contiguous row of Wh.
        float acc = 0.0f;
        for (int k = 0; k < G; k++) acc += dG[k] * Wh[k * hiddenSize + h_idx];
        dHnext = acc;
        __syncthreads();   // dG is overwritten by the next timestep
    }

    dH_init[gid] = dHnext;
    dC_init[gid] = dC;
}

// ===========================================================================
// LSTM RECURRENCE, REGISTER-RESIDENT RECURRENT WEIGHTS (H <= HP)
// ===========================================================================

// The forward and backward recurrences above give each block one batch row and one thread per hidden unit, so every
// thread re-reads 4*H recurrent weights from global memory (L1/L2) at every timestep, with one FMA per load, and the
// step's critical path is that dependent load chain (measured on the parity LSTM, B 32, T 16, H 64: 39 us forward and
// 63 us backward on an RTX 3080, against ~3 us of arithmetic).
// These kernels give each block BB batch rows and 4*H threads, and each thread keeps its H recurrent weights in
// registers for the whole sequence: a step is then H register FMAs against h_{t-1} (or the gate gradients) broadcast
// from shared memory, for BB rows at once, and the global loads a step needs are issued one step ahead.
//   forward  : thread k owns gate row k of Wh (weight-row order i,f,g,o) and forms that row's preactivation for its
//              BB rows; then thread (bb, j) = (k / H, k % H) runs the cell update of unit j of row bb.
//   backward : thread (q, j) = (k / H, k % H) owns column j of gate block q of Wh, i.e. Wh[q*H + m, j] for m < H, and
//              forms gate block q's share of dH_prev[j]; thread (bb, j) sums the four shares and runs the cell's BPTT.
// Same arguments, outputs and caches as lstm_forward_recurrence / lstm_backward_recurrence, except that the forward
// takes Wh untransposed ([4H, H]) and also writes the final hidden and cell states (hFinal, cFinal [batch, hidden]),
// which the caller otherwise copies out of the state caches. Rows past the batch are computed on zeros and never
// stored.
template <int HP, int BB>
__device__ __forceinline__ void lstm_forward_recurrence_reg_body(
    const float* __restrict__ xProj,   // [batch*timeSteps, 4*hidden], row r = b*timeSteps + t, order i,f,g,o
    const float* __restrict__ h_init, const float* __restrict__ c_init,
    const float* __restrict__ Wh,      // [4*hidden, hidden]
    const float* __restrict__ biasIh, const float* __restrict__ biasHh,
    float* __restrict__ output, float* __restrict__ h_states, float* __restrict__ c_states, float* __restrict__ gates,
    int batch, int timeSteps, int H, float* __restrict__ hFinal, float* __restrict__ cFinal)
{
    __shared__ __align__(16) float hs[BB][HP];
    __shared__ float pre[BB][4 * HP];
    const int G = 4 * H;
    const int k = threadIdx.x;
    const int b0 = blockIdx.x * BB;

    // Thread k's weights are row k of Wh, so reading them directly costs a separate cache line per thread per load
    // (measured ~9 us of fixed cost at H 64); the block stages them through shared memory 32 columns at a time,
    // each thread issuing its CW loads of a chunk back to back (the block has exactly G = 4H threads).
    constexpr int CW = 32;
    __shared__ float stage[4 * HP][CW + 1];
    float w[HP];
#pragma unroll
    for (int c0 = 0; c0 < HP; c0 += CW) {
        float tmp[CW];
#pragma unroll
        for (int n = 0; n < CW; n++) {
            int i = n * G + k, row = i / CW, col = i % CW;
            tmp[n] = c0 + col < H ? Wh[row * H + c0 + col] : 0.0f;
        }
#pragma unroll
        for (int n = 0; n < CW; n++) {
            int i = n * G + k;
            stage[i / CW][i % CW] = tmp[n];
        }
        __syncthreads();
#pragma unroll
        for (int col = 0; col < CW; col++) w[c0 + col] = stage[k][col];
        __syncthreads();
    }
    const float bias = biasIh[k] + biasHh[k];

    const int cb = k / H, cj = k - cb * H;
    const bool cell = cb < BB && b0 + cb < batch;
    const int cgid = (b0 + cb) * H + cj;
    float c = 0.0f;
    for (int i = k; i < BB * HP; i += blockDim.x) (&hs[0][0])[i] = 0.0f;
    __syncthreads();
    if (cell) { c = c_init[cgid]; hs[cb][cj] = h_init[cgid]; }

    float xp[BB];
#pragma unroll
    for (int bb = 0; bb < BB; bb++) xp[bb] = b0 + bb < batch ? xProj[((b0 + bb) * timeSteps) * G + k] : 0.0f;
    __syncthreads();

    for (int t = 0; t < timeSteps; t++) {
        // Independent partial sums keep the FMA chain short when BB is small.
        constexpr int P = BB >= 4 ? 1 : 4 / BB;
        float acc[BB][P];
#pragma unroll
        for (int bb = 0; bb < BB; bb++) {
            acc[bb][0] = xp[bb] + bias;
#pragma unroll
            for (int p = 1; p < P; p++) acc[bb][p] = 0.0f;
        }
        if (t + 1 < timeSteps) {
#pragma unroll
            for (int bb = 0; bb < BB; bb++)
                xp[bb] = b0 + bb < batch ? xProj[((b0 + bb) * timeSteps + t + 1) * G + k] : 0.0f;
        }
#pragma unroll
        for (int m = 0; m < HP; m += 4) {
#pragma unroll
            for (int bb = 0; bb < BB; bb++) {
                float4 hv = *reinterpret_cast<const float4*>(&hs[bb][m]);
                float& a = acc[bb][(m / 4) % P];
                a = fmaf(w[m], hv.x, a);
                a = fmaf(w[m + 1], hv.y, a);
                a = fmaf(w[m + 2], hv.z, a);
                a = fmaf(w[m + 3], hv.w, a);
            }
        }
#pragma unroll
        for (int bb = 0; bb < BB; bb++) {
            float s = acc[bb][0];
#pragma unroll
            for (int p = 1; p < P; p++) s += acc[bb][p];
            pre[bb][k] = s;
        }
        __syncthreads();   // every preactivation is in pre, and every thread has read hs for this step

        if (cell) {
            float f = sigmoid(pre[cb][H + cj]);
            float i_gate = sigmoid(pre[cb][cj]);
            float g = lstm_tanh(pre[cb][2 * H + cj]);
            float o = sigmoid(pre[cb][3 * H + cj]);
            c = f * c + i_gate * g;
            float h = o * lstm_tanh(c);
            hs[cb][cj] = h;

            int b = b0 + cb;
            int stateOffset = t * batch * H + cgid;
            h_states[stateOffset] = h;
            c_states[stateOffset] = c;
            output[(b * timeSteps + t) * H + cj] = h;
            int gateOffset = t * batch * G + b * G;
            gates[gateOffset + cj] = f;
            gates[gateOffset + H + cj] = i_gate;
            gates[gateOffset + 2 * H + cj] = g;
            gates[gateOffset + 3 * H + cj] = o;
            if (t == timeSteps - 1) { hFinal[cgid] = h; cFinal[cgid] = c; }
        }
        __syncthreads();   // the next step reads the new hs, and pre is rewritten
    }
}

template <int HP, int BB>
__device__ __forceinline__ void lstm_backward_recurrence_reg_body(
    const float* __restrict__ gradOutput,  // [batch, timeSteps, hidden]
    const float* __restrict__ h_states, const float* __restrict__ c_states,
    const float* __restrict__ gates,       // [timeSteps, batch, 4*hidden], role order f,i,g,o
    const float* __restrict__ c_init, const float* __restrict__ h_init,
    const float* __restrict__ Wh,          // [4*hidden, hidden]
    float* __restrict__ dGates, float* __restrict__ hPrev,
    float* __restrict__ dH_init, float* __restrict__ dC_init,
    int batch, int timeSteps, int H)
{
    __shared__ __align__(16) float dg[BB][4][HP];   // gate gradients, weight-row order i,f,g,o; padding stays zero
    __shared__ float part[BB][4][HP];
    const int G = 4 * H;
    const int tid = threadIdx.x;
    const int q = tid / H, j = tid - q * H;

    float w[HP];
#pragma unroll
    for (int m = 0; m < HP; m++) w[m] = m < H ? Wh[(q * H + m) * H + j] : 0.0f;
    for (int i = tid; i < BB * 4 * HP; i += blockDim.x) (&dg[0][0][0])[i] = 0.0f;

    const int b0 = blockIdx.x * BB;
    const int cb = q, cj = j;   // the cell role uses the same split of the thread index
    const bool cell = cb < BB && b0 + cb < batch;
    const int b = b0 + cb;
    const int cgid = b * H + cj;
    const int BH = batch * H;

    float dHnext = 0.0f, dC = 0.0f;
    // The cell inputs of step t, loaded during step t+1.
    float go = 0.0f, f = 0.0f, ig = 0.0f, g = 0.0f, o = 0.0f, cT = 0.0f, cPrev = 0.0f, hPrevV = 0.0f;
    int t0 = timeSteps - 1;
    if (cell) {
        go = gradOutput[(b * timeSteps + t0) * H + cj];
        int gateOffset = t0 * batch * G + b * G;
        f = gates[gateOffset + cj];
        ig = gates[gateOffset + H + cj];
        g = gates[gateOffset + 2 * H + cj];
        o = gates[gateOffset + 3 * H + cj];
        cT = c_states[t0 * BH + cgid];
        cPrev = t0 == 0 ? c_init[cgid] : c_states[(t0 - 1) * BH + cgid];
        hPrevV = t0 == 0 ? h_init[cgid] : h_states[(t0 - 1) * BH + cgid];
    }
    __syncthreads();

    for (int t = timeSteps - 1; t >= 0; t--) {
        if (cell) {
            float dH = dHnext + go;
            float tanh_c = lstm_tanh(cT);
            float dO = dH * tanh_c * sigmoid_derivative(o);
            dC += dH * o * tanh_derivative(tanh_c);
            float dF = dC * cPrev * sigmoid_derivative(f);
            float dI = dC * g * sigmoid_derivative(ig);
            float dGc = dC * ig * tanh_derivative(g);
            dC = dC * f;

            dg[cb][0][cj] = dI;
            dg[cb][1][cj] = dF;
            dg[cb][2][cj] = dGc;
            dg[cb][3][cj] = dO;

            int r = b * timeSteps + t;
            dGates[r * G + cj] = dI;
            dGates[r * G + H + cj] = dF;
            dGates[r * G + 2 * H + cj] = dGc;
            dGates[r * G + 3 * H + cj] = dO;
            hPrev[r * H + cj] = hPrevV;

            if (t > 0) {
                int tn = t - 1;
                go = gradOutput[(b * timeSteps + tn) * H + cj];
                int gateOffset = tn * batch * G + b * G;
                f = gates[gateOffset + cj];
                ig = gates[gateOffset + H + cj];
                g = gates[gateOffset + 2 * H + cj];
                o = gates[gateOffset + 3 * H + cj];
                cT = cPrev;
                cPrev = tn == 0 ? c_init[cgid] : c_states[(tn - 1) * BH + cgid];
                hPrevV = tn == 0 ? h_init[cgid] : h_states[(tn - 1) * BH + cgid];
            }
        }
        __syncthreads();   // dg holds this step's gate gradients

        // Gate block q's share of dH_prev[j] = sum_m dG[q*H + m] * Wh[q*H + m, j].
        constexpr int P = BB >= 4 ? 1 : 4 / BB;
#pragma unroll
        for (int bb = 0; bb < BB; bb++) {
            float acc[P];
#pragma unroll
            for (int p = 0; p < P; p++) acc[p] = 0.0f;
#pragma unroll
            for (int m = 0; m < HP; m += 4) {
                float4 dv = *reinterpret_cast<const float4*>(&dg[bb][q][m]);
                float& a = acc[(m / 4) % P];
                a = fmaf(w[m], dv.x, a);
                a = fmaf(w[m + 1], dv.y, a);
                a = fmaf(w[m + 2], dv.z, a);
                a = fmaf(w[m + 3], dv.w, a);
            }
            float s = acc[0];
#pragma unroll
            for (int p = 1; p < P; p++) s += acc[p];
            part[bb][q][j] = s;
        }
        __syncthreads();   // every share is in part, and every thread has read dg

        if (cell) dHnext = part[cb][0][cj] + part[cb][1][cj] + part[cb][2][cj] + part[cb][3][cj];
    }

    if (cell) {
        dH_init[cgid] = dHnext;
        dC_init[cgid] = dC;
    }
}

#define LSTM_REG_KERNELS(HP, BB) \
extern ""C"" __global__ __launch_bounds__(4 * HP) void lstm_forward_recurrence_reg_h##HP##_b##BB( \
    const float* xProj, const float* h_init, const float* c_init, const float* Wh, const float* biasIh, \
    const float* biasHh, float* output, float* h_states, float* c_states, float* gates, \
    int batch, int timeSteps, int hiddenSize, float* hFinal, float* cFinal) \
{ \
    lstm_forward_recurrence_reg_body<HP, BB>(xProj, h_init, c_init, Wh, biasIh, biasHh, output, h_states, c_states, \
        gates, batch, timeSteps, hiddenSize, hFinal, cFinal); \
} \
extern ""C"" __global__ __launch_bounds__(4 * HP) void lstm_backward_recurrence_reg_h##HP##_b##BB( \
    const float* gradOutput, const float* h_states, const float* c_states, const float* gates, const float* c_init, \
    const float* h_init, const float* Wh, float* dGates, float* hPrev, float* dH_init, float* dC_init, \
    int batch, int timeSteps, int hiddenSize) \
{ \
    lstm_backward_recurrence_reg_body<HP, BB>(gradOutput, h_states, c_states, gates, c_init, h_init, Wh, dGates, \
        hPrev, dH_init, dC_init, batch, timeSteps, hiddenSize); \
}

// One batch row per block (BB 1) measured fastest; BB 2 and 4 trade the grid for longer per-step chains.
LSTM_REG_KERNELS(32, 1)
LSTM_REG_KERNELS(64, 1)

// ===========================================================================
// LSTM BACKWARD DENSE PRODUCTS (one launch)
// ===========================================================================

// Every dense product of the LSTM backward over all R = batch*timeSteps rows, after the recurrence has written dGates
// and Hprev:
//   [dWi | dWh | dBias] = dGates^T [X | Hprev | 1]    ([G, In + H + 1], G = 4*hidden)
//   dX                  = dGates Wi                   ([R, In])
// They were four GEMMs (and a copy for the second bias), each a few dozen blocks or fewer at a parity LSTM's sizes
// (measured 9.8 + 9.8 + 14.8 + 2.4 us at B 32, T 16, In 32, H 64 on an RTX 3080). Blocks [0, nW) take 32x32 tiles of
// the weight-gradient matrix, blocks [nW, nW + nX) 32x32 tiles of dX. A block stages 64-deep slices of the reduction
// in shared memory (the next slice's loads in flight during this slice's FMAs); its eight 64-thread groups each take
// an eighth of a slice with a 4x4 register tile per thread, and the eight partial tiles are summed in a fixed order at
// the end (deterministic, no atomics). Every output is written, not accumulated into.
// With splits > 1 each weight-gradient tile takes `splits` blocks, each reducing rowsPerSplit of the R rows into its own
// slot of `partial`, and lstm_backward_gemms_reduce sums the slots in split order: R grows the grid instead of each
// block's loop (one 32x32 tile per block left a long sequence's weight gradient on a few dozen blocks).
#define LBG_KC 64          // reduction depth per staged slice
#define LBG_LD 36          // shared row stride: float4-aligned
#define LBG_GROUPS 8       // 64-thread groups splitting each slice
#define LBG_THREADS (64 * LBG_GROUPS)
extern ""C"" __global__ __launch_bounds__(LBG_THREADS) void lstm_backward_gemms(
    const float* __restrict__ dGates,   // [R, G], weight-row order i,f,g,o
    const float* __restrict__ X,        // [R, In]
    const float* __restrict__ Hprev,    // [R, H]
    const float* __restrict__ Wi,       // [G, In]
    float* __restrict__ dWi,            // [G, In]
    float* __restrict__ dWh,            // [G, H]
    float* __restrict__ dBiasIh,        // [G]
    float* __restrict__ dBiasHh,        // [G]
    float* __restrict__ dX,             // [R, In]
    int R, int In, int H, int wTilesJ, int nW,
    int splits, int rowsPerSplit,
    float* __restrict__ partial)        // [splits][nW][32 * 32] when splits > 1
{
    // sA[k][row], sB[k][col] for one slice; reused for the groups' partial tiles (LBG_GROUPS x 32 x 32) at the end.
    __shared__ __align__(16) float smem[LBG_GROUPS * 32 * 32 > 2 * LBG_KC * LBG_LD ? LBG_GROUPS * 32 * 32 : 2 * LBG_KC * LBG_LD];
    float (*sA)[LBG_LD] = reinterpret_cast<float (*)[LBG_LD]>(smem);
    float (*sB)[LBG_LD] = reinterpret_cast<float (*)[LBG_LD]>(smem + LBG_KC * LBG_LD);
    const int G = 4 * H;
    const int tid = threadIdx.x;
    const int grp = tid >> 6, lane = tid & 63, lr = (lane >> 3) * 4, lc = (lane & 7) * 4;
    const bool weights = (int)blockIdx.x < nW * splits;
    int row0, col0, kBegin, kEnd, tile = 0, split = 0;
    if (weights) {
        tile = (int)blockIdx.x % nW;
        split = (int)blockIdx.x / nW;
        row0 = (tile / wTilesJ) * 32;              // gate row g
        col0 = (tile % wTilesJ) * 32;              // column of [X | Hprev | 1]
        kBegin = split * rowsPerSplit;             // this block's rows of R
        kEnd = min(R, kBegin + rowsPerSplit);
    } else {
        int b = (int)blockIdx.x - nW * splits, tilesI = (In + 31) / 32;
        row0 = (b / tilesI) * 32;                  // row r of dX
        col0 = (b % tilesI) * 32;                  // input feature i
        kBegin = 0;
        kEnd = G;
    }

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    // Each thread stages LBG_KC * 32 / LBG_THREADS elements of each operand per slice.
    constexpr int PER = LBG_KC * 32 / LBG_THREADS;
    float la[PER], lb[PER];
    auto load = [&](int k0) {
#pragma unroll
        for (int u = 0; u < PER; u++) {
            int i = tid + u * LBG_THREADS;
            if (weights) {
                // A: dGates[r, g];  B: [X | Hprev | 1][r, j]   (r = k0 + a, g = row0 + c, j = col0 + c)
                int a = i >> 5, c = i & 31, r = k0 + a, g = row0 + c, j = col0 + c;
                la[u] = r < kEnd && g < G ? dGates[(size_t)r * G + g] : 0.0f;
                float z = 0.0f;
                if (r < kEnd) {
                    if (j < In) z = X[(size_t)r * In + j];
                    else if (j < In + H) z = Hprev[(size_t)r * H + (j - In)];
                    else if (j == In + H) z = 1.0f;
                }
                lb[u] = z;
            } else {
                // A: dGates[r, k] read along k (rr = i / 64, kk = i % 64);  B: Wi[k, i] (kk = i / 32, ii = i % 32)
                int rr = i >> 6, kk = i & 63, r = row0 + rr, k = k0 + kk;
                la[u] = r < R && k < G ? dGates[(size_t)r * G + k] : 0.0f;
                int kb = k0 + (i >> 5), ii = col0 + (i & 31);
                lb[u] = kb < G && ii < In ? Wi[(size_t)kb * In + ii] : 0.0f;
            }
        }
    };
    load(kBegin);
    for (int k0 = kBegin; k0 < kEnd; k0 += LBG_KC) {
#pragma unroll
        for (int u = 0; u < PER; u++) {
            int i = tid + u * LBG_THREADS;
            if (weights) sA[i >> 5][i & 31] = la[u];
            else sA[i & 63][i >> 6] = la[u];
            sB[i >> 5][i & 31] = lb[u];
        }
        __syncthreads();
        if (k0 + LBG_KC < kEnd) load(k0 + LBG_KC);
#pragma unroll
        for (int q = 0; q < LBG_KC / LBG_GROUPS; q++) {
            int kk = grp * (LBG_KC / LBG_GROUPS) + q;
            float4 a = *reinterpret_cast<const float4*>(&sA[kk][lr]);
            float4 b = *reinterpret_cast<const float4*>(&sB[kk][lc]);
            float av[4] = { a.x, a.y, a.z, a.w }, bv[4] = { b.x, b.y, b.z, b.w };
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Sum the groups' partial tiles in a fixed order: group g writes part[g][32][32], then each thread sums its outputs.
    float* part = smem;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) part[(grp * 32 + lr + i) * 32 + lc + j] = acc[i][j];
    __syncthreads();
#pragma unroll
    for (int u = 0; u < 1024 / LBG_THREADS; u++) {
        int e = tid + u * LBG_THREADS, rowL = e >> 5, colL = e & 31;
        float v = 0.0f;
#pragma unroll
        for (int g = 0; g < LBG_GROUPS; g++) v += part[(g * 32 + rowL) * 32 + colL];
        int row = row0 + rowL, col = col0 + colL;
        if (weights && splits > 1) {
            partial[((size_t)split * nW + tile) * 1024 + e] = v;
        } else if (weights) {
            if (row >= G) continue;
            if (col < In) dWi[(size_t)row * In + col] = v;
            else if (col < In + H) dWh[(size_t)row * H + (col - In)] = v;
            else if (col == In + H) { dBiasIh[row] = v; dBiasHh[row] = v; }
        } else if (row < R && col < In) {
            dX[(size_t)row * In + col] = v;
        }
    }
}
#undef LBG_KC
#undef LBG_LD
#undef LBG_GROUPS
#undef LBG_THREADS

// Sums lstm_backward_gemms' split partials (splits > 1) in split order into dWi, dWh and both bias gradients: one thread
// per element of the [G, In + H + 1] weight-gradient matrix.
extern ""C"" __global__ __launch_bounds__(256) void lstm_backward_gemms_reduce(
    const float* __restrict__ partial, float* __restrict__ dWi, float* __restrict__ dWh,
    float* __restrict__ dBiasIh, float* __restrict__ dBiasHh, int splits, int nW, int wTilesJ, int In, int H)
{
    int G = 4 * H, J = In + H + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= G * J) return;
    int row = idx / J, col = idx - row * J;
    int tile = (row >> 5) * wTilesJ + (col >> 5), e = (row & 31) * 32 + (col & 31);
    float v = 0.0f;
    for (int s = 0; s < splits; s++) v += partial[((size_t)s * nW + tile) * 1024 + e];
    if (col < In) dWi[(size_t)row * In + col] = v;
    else if (col < In + H) dWh[(size_t)row * H + (col - In)] = v;
    else { dBiasIh[row] = v; dBiasHh[row] = v; }
}

// ===========================================================================
// LSTM WEIGHT GRADIENT ACCUMULATION KERNEL
// ===========================================================================

// Accumulates weight gradients across batch dimension.
// NON-DETERMINISTIC across calls (issue #382): the atomicAdds at the end are
// over-defensive — each (gateIdx, colIdx) cell is written by exactly one thread per
// launch. But the kernel is invoked once per timestep, and atomicAdd ordering across
// those calls is not bit-stable. See lstm_accumulate_weight_gradients_deterministic.
extern ""C"" __global__ __launch_bounds__(1024) void lstm_accumulate_weight_gradients(
    const float* input,       // [batch, inputSize]
    const float* prevH,       // [batch, hiddenSize]
    const float* dGates,      // [batch, 4*hiddenSize]
    float* dWi,               // [4*hiddenSize, inputSize]
    float* dWh,               // [4*hiddenSize, hiddenSize]
    float* dBias,             // [4*hiddenSize]
    int batch,
    int inputSize,
    int hiddenSize)
{
    // Grid: (4*hidden, max(input, hidden))
    int gateIdx = blockIdx.x;  // Which gate row
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    if (gateIdx >= 4 * hiddenSize) return;

    // Accumulate Wi gradients
    if (colIdx < inputSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float x_val = input[b * inputSize + colIdx];
            grad += dGate * x_val;
        }
        atomicAdd(&dWi[gateIdx * inputSize + colIdx], grad);
    }

    // Accumulate Wh gradients
    if (colIdx < hiddenSize) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            float dGate = dGates[b * 4 * hiddenSize + gateIdx];
            float h_val = prevH[b * hiddenSize + colIdx];
            grad += dGate * h_val;
        }
        atomicAdd(&dWh[gateIdx * hiddenSize + colIdx], grad);
    }

    // Accumulate bias gradients (only first column handles this)
    if (colIdx == 0) {
        float grad = 0.0f;
        for (int b = 0; b < batch; b++) {
            grad += dGates[b * 4 * hiddenSize + gateIdx];
        }
        atomicAdd(&dBias[gateIdx], grad);
    }
}

// lstm_accumulate_weight_gradients — bit-deterministic variant (issue #382).
// Within a single invocation each (gateIdx, colIdx) cell is touched by exactly one
// thread, so direct += is safe. Across multiple timestep invocations the caller
// accumulates by += into the same dWi/dWh/dBias buffers; doing this with direct
// += inside the kernel keeps the cross-timestep accumulation order pinned to the
// host-driven invocation sequence (host iterates timesteps in fixed order).
extern ""C"" __global__ __launch_bounds__(1024) void lstm_accumulate_weight_gradients_deterministic(
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

// ===========================================================================
// LSTM COMPUTE GATE GRADIENTS KERNEL
// ===========================================================================

// Computes gate gradients from hidden and cell gradients
extern ""C"" __global__ __launch_bounds__(1024) void lstm_compute_gate_gradients(
    const float* dH,          // [batch, hidden]
    const float* dC_next,     // [batch, hidden]
    const float* gateF,       // [batch, hidden]
    const float* gateI,       // [batch, hidden]
    const float* gateC,       // [batch, hidden]
    const float* gateO,       // [batch, hidden]
    const float* prevC,       // [batch, hidden]
    const float* newC,        // [batch, hidden]
    float* dGates,            // [batch, 4*hidden] - output
    float* dPrevC,            // [batch, hidden] - output
    int batch,
    int hiddenSize)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int h = gid % hiddenSize;

    // Get cached values
    float f = gateF[gid];
    float i_gate = gateI[gid];
    float c_candidate = gateC[gid];
    float o = gateO[gid];
    float prevCVal = prevC[gid];
    float newCVal = newC[gid];

    // Gradient from output
    float dh = dH[gid];

    // tanh(c_t)
    float tanh_c = lstm_tanh(newCVal);

    // Gradient through output gate
    float dO = dh * tanh_c * sigmoid_derivative(o);

    // Gradient to cell state from hidden state
    float dC_from_H = dh * o * tanh_derivative(tanh_c);

    // Total cell state gradient
    float dC = dC_next[gid] + dC_from_H;

    // Gradient through cell state equation
    float dF = dC * prevCVal * sigmoid_derivative(f);
    float dI = dC * c_candidate * sigmoid_derivative(i_gate);
    float dCCandidate = dC * i_gate * tanh_derivative(c_candidate);

    // Store gate gradients
    int gateOffset = b * 4 * hiddenSize;
    dGates[gateOffset + h] = dF;
    dGates[gateOffset + hiddenSize + h] = dI;
    dGates[gateOffset + 2 * hiddenSize + h] = dCCandidate;
    dGates[gateOffset + 3 * hiddenSize + h] = dO;

    // Gradient to previous cell state
    dPrevC[gid] = dC * f;
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
            "lstm_cell_backward",
            "lstm_cell_backward_compute_gates_deterministic",
            "lstm_cell_backward_dPrevH_deterministic",
            "lstm_cell_backward_dWi_deterministic",
            "lstm_cell_backward_dBias_deterministic",
            "lstm_backward_input",
            "lstm_backward_prevh",
            "lstm_backward_sequence",
            "lstm_backward_recurrence",
            "lstm_forward_recurrence",
            "lstm_backward_gemms",
            "lstm_backward_gemms_reduce",
            "lstm_forward_recurrence_reg_h32_b1",
            "lstm_backward_recurrence_reg_h32_b1",
            "lstm_forward_recurrence_reg_h64_b1",
            "lstm_backward_recurrence_reg_h64_b1",
            "lstm_accumulate_weight_gradients",
            "lstm_accumulate_weight_gradients_deterministic",
            "lstm_compute_gate_gradients"
        };
    }
}
