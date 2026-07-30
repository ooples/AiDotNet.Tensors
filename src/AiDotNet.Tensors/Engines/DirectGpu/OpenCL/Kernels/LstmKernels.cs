// Copyright (c) AiDotNet. All rights reserved.
// OpenCL kernels for LSTM (Long Short-Term Memory) sequence neural network operations.
// Implements sequence-level forward and backward passes for efficient BPTT training.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// OpenCL kernels for LSTM sequence operations used in recurrent neural networks.
/// Implements full forward and backward passes for LSTM cells with 4 gates (forget, input, cell, output).
/// These are sequence-level kernels that process all timesteps efficiently for BPTT.
/// </summary>
internal static class LstmKernels
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
// LSTM CELL FORWARD KERNEL (Single Timestep)
// ===========================================================================
// Processes one LSTM cell computation for a single timestep.
// Each thread handles one (batch, hidden) element.

__kernel void lstm_cell_forward(
    __global const float* input,       // [batch, input_size]
    __global const float* prevH,       // [batch, hidden_size]
    __global const float* prevC,       // [batch, hidden_size]
    __global const float* weightsIh,   // [4 * hidden_size, input_size] - input to hidden weights
    __global const float* weightsHh,   // [4 * hidden_size, hidden_size] - hidden to hidden weights
    __global const float* biasIh,      // [4 * hidden_size] - input to hidden bias
    __global const float* biasHh,      // [4 * hidden_size] - hidden to hidden bias
    __global float* newH,              // [batch, hidden_size]
    __global float* newC,              // [batch, hidden_size]
    __global float* gateF,             // [batch, hidden_size] - forget gate cache
    __global float* gateI,             // [batch, hidden_size] - input gate cache
    __global float* gateG,             // [batch, hidden_size] - cell gate (g) cache
    __global float* gateO,             // [batch, hidden_size] - output gate cache
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
    // Gates order: input(i), forget(f), cell(g), output(o)
    float sumI = biasIh[h] + biasHh[h];
    float sumF = biasIh[hiddenSize + h] + biasHh[hiddenSize + h];
    float sumG = biasIh[2 * hiddenSize + h] + biasHh[2 * hiddenSize + h];
    float sumO = biasIh[3 * hiddenSize + h] + biasHh[3 * hiddenSize + h];

    // Input to hidden contribution
    for (int j = 0; j < inputSize; j++) {
        float inVal = input[b * inputSize + j];
        sumI += inVal * weightsIh[h * inputSize + j];
        sumF += inVal * weightsIh[(hiddenSize + h) * inputSize + j];
        sumG += inVal * weightsIh[(2 * hiddenSize + h) * inputSize + j];
        sumO += inVal * weightsIh[(3 * hiddenSize + h) * inputSize + j];
    }

    // Hidden to hidden contribution
    for (int j = 0; j < hiddenSize; j++) {
        float hVal = prevH[b * hiddenSize + j];
        sumI += hVal * weightsHh[h * hiddenSize + j];
        sumF += hVal * weightsHh[(hiddenSize + h) * hiddenSize + j];
        sumG += hVal * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
        sumO += hVal * weightsHh[(3 * hiddenSize + h) * hiddenSize + j];
    }

    // Apply activations
    float i = sigmoid_fn(sumI);
    float f = sigmoid_fn(sumF);
    float g = tanh(sumG);
    float o = sigmoid_fn(sumO);

    // Cell state update
    float prevCVal = prevC[gid];
    float newCVal = f * prevCVal + i * g;

    // Hidden state update
    float newHVal = o * tanh(newCVal);

    // Store results
    newC[gid] = newCVal;
    newH[gid] = newHVal;
    gateI[gid] = i;
    gateF[gid] = f;
    gateG[gid] = g;
    gateO[gid] = o;
}

// ===========================================================================
// LSTM FORWARD SEQUENCE KERNEL
// ===========================================================================
// Processes the entire sequence in a single kernel launch.
// Outer loop iterates over timesteps, inner parallel threads handle batch * hidden.

__kernel void lstm_forward_sequence(
    __global const float* input,       // [seqLen, batch, input_size]
    __global const float* hInit,       // [batch, hidden_size]
    __global const float* cInit,       // [batch, hidden_size]
    __global const float* weightsIh,   // [4 * hidden_size, input_size]
    __global const float* weightsHh,   // [4 * hidden_size, hidden_size]
    __global const float* biasIh,      // [4 * hidden_size]
    __global const float* biasHh,      // [4 * hidden_size]
    __global float* output,            // [seqLen, batch, hidden_size]
    __global float* hFinal,            // [batch, hidden_size]
    __global float* cFinal,            // [batch, hidden_size]
    __global float* allH,              // [seqLen + 1, batch, hidden_size] - all hidden states
    __global float* allC,              // [seqLen + 1, batch, hidden_size] - all cell states
    __global float* cacheGates,        // [seqLen, batch, hidden_size, 4] - gate values for backward
    const int seqLen,
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;
    int b = gid / hiddenSize;
    int h = gid % hiddenSize;
    int isValid = (gid < totalElements) ? 1 : 0;

    // Initialize from hInit and cInit (only valid threads)
    float hPrev = 0.0f;
    float cPrev = 0.0f;
    if (isValid) {
        hPrev = hInit[gid];
        cPrev = cInit[gid];

        // Store initial states
        allH[gid] = hPrev;
        allC[gid] = cPrev;
    }

    // Barrier to ensure all threads have written initial states
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Process each timestep
    for (int t = 0; t < seqLen; t++) {
        if (isValid) {
            // Compute gate pre-activations
            float sumI = biasIh[h] + biasHh[h];
            float sumF = biasIh[hiddenSize + h] + biasHh[hiddenSize + h];
            float sumG = biasIh[2 * hiddenSize + h] + biasHh[2 * hiddenSize + h];
            float sumO = biasIh[3 * hiddenSize + h] + biasHh[3 * hiddenSize + h];

            // Input contribution at this timestep. The engine feeds the input in its native
            // [batch, seq, in] (batch-major) layout — element (b, t) is at (b*seqLen + t) — so the
            // offset must be batch-major too. It was seq-major (t*batch + b), which reads the wrong
            // element whenever batch != seq and grossly corrupted the forward output.
            int inputOffset = (b * seqLen + t) * inputSize;
            for (int j = 0; j < inputSize; j++) {
                float inVal = input[inputOffset + j];
                sumI += inVal * weightsIh[h * inputSize + j];
                sumF += inVal * weightsIh[(hiddenSize + h) * inputSize + j];
                sumG += inVal * weightsIh[(2 * hiddenSize + h) * inputSize + j];
                sumO += inVal * weightsIh[(3 * hiddenSize + h) * inputSize + j];
            }

            // Previous hidden state contribution (with barrier synchronization)
            for (int j = 0; j < hiddenSize; j++) {
                float hVal = (j == h) ? hPrev : allH[t * batch * hiddenSize + b * hiddenSize + j];
                sumI += hVal * weightsHh[h * hiddenSize + j];
                sumF += hVal * weightsHh[(hiddenSize + h) * hiddenSize + j];
                sumG += hVal * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
                sumO += hVal * weightsHh[(3 * hiddenSize + h) * hiddenSize + j];
            }

            // Apply activations
            float i = sigmoid_fn(sumI);
            float f = sigmoid_fn(sumF);
            float g = tanh(sumG);
            float o = sigmoid_fn(sumO);

            // Cell state update
            float newC = f * cPrev + i * g;

            // Hidden state update
            float newH = o * tanh(newC);

            // Store output in the engine's native [batch, seq, hidden] (batch-major) layout —
            // element (b, t, h) at (b*seqLen + t)*hidden + h — matching how the C# side reads bufOut
            // as [B, S, Hd] with no permute. The previous seq-major index (t*batch + b) transposed
            // the sequence whenever batch != seq.
            int outIdx = (b * seqLen + t) * hiddenSize + h;
            output[outIdx] = newH;

            // Store all states for backward pass
            int stateIdx = (t + 1) * batch * hiddenSize + gid;
            allH[stateIdx] = newH;
            allC[stateIdx] = newC;

            // Cache gate values for backward
            int gateIdx = (t * batch * hiddenSize + gid) * 4;
            cacheGates[gateIdx + 0] = i;
            cacheGates[gateIdx + 1] = f;
            cacheGates[gateIdx + 2] = g;
            cacheGates[gateIdx + 3] = o;

            // Update for next iteration
            hPrev = newH;
            cPrev = newC;
        }

        // Barrier to ensure all threads have written new states before next iteration
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Store final states (only valid threads)
    if (isValid) {
        hFinal[gid] = hPrev;
        cFinal[gid] = cPrev;
    }
}

// ===========================================================================
// LSTM CELL BACKWARD KERNEL (Single Timestep)
// ===========================================================================

__kernel void lstm_cell_backward(
    __global const float* gradH,       // [batch, hidden_size] - gradient from next layer
    __global const float* gradCNext,   // [batch, hidden_size] - gradient from next timestep cell
    __global const float* gateI,       // [batch, hidden_size]
    __global const float* gateF,       // [batch, hidden_size]
    __global const float* gateG,       // [batch, hidden_size]
    __global const float* gateO,       // [batch, hidden_size]
    __global const float* prevC,       // [batch, hidden_size]
    __global const float* newC,        // [batch, hidden_size]
    __global float* gradPrevC,         // [batch, hidden_size] - gradient to previous cell state
    __global float* gradGateI,         // [batch, hidden_size] - gradient for input gate
    __global float* gradGateF,         // [batch, hidden_size] - gradient for forget gate
    __global float* gradGateG,         // [batch, hidden_size] - gradient for cell gate
    __global float* gradGateO,         // [batch, hidden_size] - gradient for output gate
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    float i = gateI[gid];
    float f = gateF[gid];
    float g = gateG[gid];
    float o = gateO[gid];
    float cPrev = prevC[gid];
    float cNew = newC[gid];

    float dH = gradH[gid];
    float tanhC = tanh(cNew);

    // Gradient through output gate
    float dO = dH * tanhC * sigmoid_derivative(o);

    // Gradient to cell state
    float dC = gradCNext[gid] + dH * o * tanh_derivative(tanhC);

    // Gradients through gates
    float dF = dC * cPrev * sigmoid_derivative(f);
    float dI = dC * g * sigmoid_derivative(i);
    float dG = dC * i * tanh_derivative(g);

    // Gradient to previous cell state
    float dPrevC = dC * f;

    // Store gradients
    gradPrevC[gid] = dPrevC;
    gradGateI[gid] = dI;
    gradGateF[gid] = dF;
    gradGateG[gid] = dG;
    gradGateO[gid] = dO;
}

// ===========================================================================
// LSTM BACKWARD INPUT GRADIENT KERNEL
// ===========================================================================

__kernel void lstm_backward_input(
    __global const float* gradGateI,   // [batch, hidden_size]
    __global const float* gradGateF,   // [batch, hidden_size]
    __global const float* gradGateG,   // [batch, hidden_size]
    __global const float* gradGateO,   // [batch, hidden_size]
    __global const float* weightsIh,   // [4 * hidden_size, input_size]
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

        float dI = gradGateI[batchHiddenIdx];
        float dF = gradGateF[batchHiddenIdx];
        float dG = gradGateG[batchHiddenIdx];
        float dO = gradGateO[batchHiddenIdx];

        gradSum += dI * weightsIh[h * inputSize + j];
        gradSum += dF * weightsIh[(hiddenSize + h) * inputSize + j];
        gradSum += dG * weightsIh[(2 * hiddenSize + h) * inputSize + j];
        gradSum += dO * weightsIh[(3 * hiddenSize + h) * inputSize + j];
    }

    gradInput[gid] = gradSum;
}

// ===========================================================================
// LSTM BACKWARD PREVIOUS HIDDEN GRADIENT KERNEL
// ===========================================================================

__kernel void lstm_backward_prevh(
    __global const float* gradGateI,   // [batch, hidden_size]
    __global const float* gradGateF,   // [batch, hidden_size]
    __global const float* gradGateG,   // [batch, hidden_size]
    __global const float* gradGateO,   // [batch, hidden_size]
    __global const float* weightsHh,   // [4 * hidden_size, hidden_size]
    __global float* gradPrevH,         // [batch, hidden_size]
    const int batch,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int totalElements = batch * hiddenSize;

    if (gid >= totalElements) return;

    int b = gid / hiddenSize;
    int j = gid % hiddenSize;

    float gradSum = 0.0f;

    for (int h = 0; h < hiddenSize; h++) {
        int batchHiddenIdx = b * hiddenSize + h;

        float dI = gradGateI[batchHiddenIdx];
        float dF = gradGateF[batchHiddenIdx];
        float dG = gradGateG[batchHiddenIdx];
        float dO = gradGateO[batchHiddenIdx];

        gradSum += dI * weightsHh[h * hiddenSize + j];
        gradSum += dF * weightsHh[(hiddenSize + h) * hiddenSize + j];
        gradSum += dG * weightsHh[(2 * hiddenSize + h) * hiddenSize + j];
        gradSum += dO * weightsHh[(3 * hiddenSize + h) * hiddenSize + j];
    }

    gradPrevH[gid] = gradSum;
}

// ===========================================================================
// LSTM BACKWARD SEQUENCE (correct, race-free BPTT) -- two kernels
// ===========================================================================
// The forward caches (allH/allC [(S+1),B,H], cacheGates [S,B,H,4] gate order i,f,g,o) are
// seq-major; gradOutput/input/gradInput are the engine's native batch-major [B,S,*]. The
// recurrence over t is inherently sequential, so kernel A runs ONE work-item per batch element
// (each independent -> no barriers, no cross-work-item writes) and emits the gate-preactivation
// gradients dGates[S,B,4H], grad_input, and the initial-state grads (gradHInit/gradCInit double
// as the reverse-time carry). Kernel B then sums dGates into the weight/bias gradients with one
// work-item per output element -- each an independent reduction over (t,b), so no atomics.

// Kernel A: per-(batch) reverse-time BPTT. global size = batch.
__kernel void lstm_backward_dgates(
    __global const float* gradOutput,  // [batch, seqLen, hidden] (batch-major)
    __global const float* allC,        // [seqLen + 1, batch, hidden]  (allC[0]=c0, allC[t+1]=c_t)
    __global const float* cacheGates,  // [seqLen, batch, hidden, 4]   (slots: i,f,g,o)
    __global const float* weightsIh,   // [4*hidden, inputSize]
    __global const float* weightsHh,   // [4*hidden, hidden]
    __global const float* input,       // [batch, seqLen, inputSize] (batch-major) -- unused here
    __global float* dGates,            // [seqLen, batch, 4*hidden]  (out: gate-preact grads)
    __global float* gradInput,         // [batch, seqLen, inputSize] (batch-major, out)
    __global float* gradHInit,         // [batch, hidden] (out; also the reverse-time dH carry)
    __global float* gradCInit,         // [batch, hidden] (out; also the reverse-time dC carry)
    const int seqLen,
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int b = get_global_id(0);
    if (b >= batch) return;

    int H = hiddenSize;
    int G = 4 * H;

    // Carries start at zero: h_{S-1}/c_{S-1} receive no gradient from beyond the sequence end.
    for (int h = 0; h < H; h++) {
        gradHInit[b * H + h] = 0.0f;
        gradCInit[b * H + h] = 0.0f;
    }

    for (int t = seqLen - 1; t >= 0; t--) {
        int goBase   = (b * seqLen + t) * H;              // gradOutput[b, t, :]
        int cCurr    = ((t + 1) * batch + b) * H;         // allC[t+1] = c_t
        int cPrev    = (t * batch + b) * H;               // allC[t]   = c_{t-1}
        int gateBase = ((t * batch + b) * H) * 4;         // cacheGates[t, b, :, :]
        int dgBase   = (t * batch + b) * G;               // dGates[t, b, :]

        // Phase 1: per hidden unit -> gate-preactivation gradients; update the dC carry.
        for (int h = 0; h < H; h++) {
            float dh = gradOutput[goBase + h] + gradHInit[b * H + h];   // upstream + recurrent
            int gI = gateBase + h * 4;
            float i = cacheGates[gI + 0];
            float f = cacheGates[gI + 1];
            float g = cacheGates[gI + 2];
            float o = cacheGates[gI + 3];
            float cT    = allC[cCurr + h];
            float cPrv  = allC[cPrev + h];
            float tanhC = tanh(cT);

            float dO    = dh * tanhC;
            float dPreO = dO * o * (1.0f - o);
            float dc    = gradCInit[b * H + h] + dh * o * (1.0f - tanhC * tanhC);
            float dI    = dc * g;
            float dPreI = dI * i * (1.0f - i);
            float dG    = dc * i;
            float dPreG = dG * (1.0f - g * g);
            float dF    = dc * cPrv;
            float dPreF = dF * f * (1.0f - f);

            // dc flows to the previous (earlier) timestep; at t==0 this becomes grad w.r.t. c0.
            gradCInit[b * H + h] = dc * f;

            // Store gate-preact grads; row = gate*H + h, gate order i,f,g,o (matches the weights).
            dGates[dgBase + 0 * H + h] = dPreI;
            dGates[dgBase + 1 * H + h] = dPreF;
            dGates[dgBase + 2 * H + h] = dPreG;
            dGates[dgBase + 3 * H + h] = dPreO;
        }

        // Phase 2a: recurrent dh for the previous step: dh_prev[hh] = sum_row dPre[row] * Whh[row, hh].
        // Overwrites gradHInit (Phase 1 already consumed the old value for every h).
        for (int hh = 0; hh < H; hh++) {
            float s = 0.0f;
            for (int row = 0; row < G; row++) {
                s += dGates[dgBase + row] * weightsHh[row * H + hh];
            }
            gradHInit[b * H + hh] = s;   // at t==0 this becomes grad w.r.t. h0
        }

        // Phase 2b: grad_input[b, t, j] = sum_row dPre[row] * Wih[row, j].
        int giBase = (b * seqLen + t) * inputSize;
        for (int j = 0; j < inputSize; j++) {
            float s = 0.0f;
            for (int row = 0; row < G; row++) {
                s += dGates[dgBase + row] * weightsIh[row * inputSize + j];
            }
            gradInput[giBase + j] = s;
        }
    }
}

// Kernel B: weight + bias gradients from the precomputed dGates. One work-item per output
// element across the concatenated ranges [dWih | dWhh | dBias]; each is an independent sum.
__kernel void lstm_backward_dweights(
    __global const float* dGates,      // [seqLen, batch, 4*hidden]
    __global const float* input,       // [batch, seqLen, inputSize] (batch-major)
    __global const float* allH,        // [seqLen + 1, batch, hidden]  (allH[t] = h_{t-1})
    __global float* gradWeightsIh,     // [4*hidden, inputSize]
    __global float* gradWeightsHh,     // [4*hidden, hidden]
    __global float* gradBiasIh,        // [4*hidden]
    __global float* gradBiasHh,        // [4*hidden]
    const int seqLen,
    const int batch,
    const int inputSize,
    const int hiddenSize)
{
    int gid = get_global_id(0);
    int H = hiddenSize;
    int G = 4 * H;
    int nWih = G * inputSize;
    int nWhh = G * H;

    if (gid < nWih) {
        // dWih[row, j] = sum_{t,b} dGates[t,b,row] * input[b,t,j]
        int row = gid / inputSize;
        int j   = gid % inputSize;
        float s = 0.0f;
        for (int t = 0; t < seqLen; t++) {
            for (int b = 0; b < batch; b++) {
                s += dGates[(t * batch + b) * G + row] * input[(b * seqLen + t) * inputSize + j];
            }
        }
        gradWeightsIh[row * inputSize + j] = s;
    } else if (gid < nWih + nWhh) {
        // dWhh[row, hh] = sum_{t,b} dGates[t,b,row] * h_{t-1}[b,hh]   (h_{t-1} = allH[t])
        int k   = gid - nWih;
        int row = k / H;
        int hh  = k % H;
        float s = 0.0f;
        for (int t = 0; t < seqLen; t++) {
            for (int b = 0; b < batch; b++) {
                s += dGates[(t * batch + b) * G + row] * allH[(t * batch + b) * H + hh];
            }
        }
        gradWeightsHh[row * H + hh] = s;
    } else if (gid < nWih + nWhh + G) {
        // dBih[row] = dBhh[row] = sum_{t,b} dGates[t,b,row]   (both biases share the gate grad)
        int row = gid - nWih - nWhh;
        float s = 0.0f;
        for (int t = 0; t < seqLen; t++) {
            for (int b = 0; b < batch; b++) {
                s += dGates[(t * batch + b) * G + row];
            }
        }
        gradBiasIh[row] = s;
        gradBiasHh[row] = s;
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
            "lstm_cell_backward",
            "lstm_backward_input",
            "lstm_backward_prevh",
            "lstm_backward_dgates",
            "lstm_backward_dweights"
        };
    }
}
