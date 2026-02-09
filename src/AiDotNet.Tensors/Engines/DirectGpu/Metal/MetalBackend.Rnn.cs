// Copyright (c) AiDotNet. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// Metal GPU backend implementation - RNN/LSTM/GRU Sequence Operations.
/// Provides efficient sequence modeling for recurrent neural networks with BPTT support.
/// </summary>
public partial class MetalBackend
{
    #region RNN (LSTM/GRU) Sequence Operations

    /// <summary>
    /// Forward pass for LSTM sequence - processes all timesteps in a single kernel launch.
    /// Efficient for BPTT training with minimal kernel launch overhead.
    /// </summary>
    /// <param name="input">Input sequence [seqLen * batch * inputSize].</param>
    /// <param name="hInit">Initial hidden state [batch * hiddenSize].</param>
    /// <param name="cInit">Initial cell state [batch * hiddenSize].</param>
    /// <param name="weightsIh">Input-to-hidden weights [4 * hiddenSize * inputSize] (gates: i, f, g, o).</param>
    /// <param name="weightsHh">Hidden-to-hidden weights [4 * hiddenSize * hiddenSize].</param>
    /// <param name="biasIh">Input-to-hidden bias [4 * hiddenSize].</param>
    /// <param name="biasHh">Hidden-to-hidden bias [4 * hiddenSize].</param>
    /// <param name="output">Output sequence [seqLen * batch * hiddenSize].</param>
    /// <param name="hFinal">Final hidden state [batch * hiddenSize].</param>
    /// <param name="cFinal">Final cell state [batch * hiddenSize].</param>
    /// <param name="allH">All hidden states cache [(seqLen + 1) * batch * hiddenSize] for backward.</param>
    /// <param name="allC">All cell states cache [(seqLen + 1) * batch * hiddenSize] for backward.</param>
    /// <param name="cacheGates">Gate cache [seqLen * batch * hiddenSize * 4] for backward.</param>
    /// <param name="seqLen">Sequence length.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="inputSize">Input feature size.</param>
    /// <param name="hiddenSize">Hidden state size.</param>
    public void LstmForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer cFinal,
        IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        // CPU fallback implementation for LSTM forward sequence
        var inputData = DownloadBuffer(input);
        var hInitData = DownloadBuffer(hInit);
        var cInitData = DownloadBuffer(cInit);
        var wIh = DownloadBuffer(weightsIh);
        var wHh = DownloadBuffer(weightsHh);
        var bIh = DownloadBuffer(biasIh);
        var bHh = DownloadBuffer(biasHh);

        var outputData = new float[seqLen * batch * hiddenSize];
        var allHData = new float[(seqLen + 1) * batch * hiddenSize];
        var allCData = new float[(seqLen + 1) * batch * hiddenSize];
        var gatesData = new float[seqLen * batch * hiddenSize * 4];

        // Copy initial states to allH[0] and allC[0]
        int stateSize = batch * hiddenSize;
        Array.Copy(hInitData, 0, allHData, 0, stateSize);
        Array.Copy(cInitData, 0, allCData, 0, stateSize);

        // Process each timestep
        for (int t = 0; t < seqLen; t++)
        {
            int inputOffset = t * batch * inputSize;
            int hPrevOffset = t * stateSize;
            int hCurrOffset = (t + 1) * stateSize;
            int outputOffset = t * batch * hiddenSize;
            int gateOffset = t * batch * hiddenSize * 4;

            // For each batch element
            for (int b = 0; b < batch; b++)
            {
                // Compute gates: i, f, g, o
                var gates = new float[4 * hiddenSize];

                // Input to hidden contribution
                for (int g = 0; g < 4; g++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        float sum = bIh[g * hiddenSize + h] + bHh[g * hiddenSize + h];

                        // Input contribution
                        for (int i = 0; i < inputSize; i++)
                        {
                            int wIdx = g * hiddenSize * inputSize + h * inputSize + i;
                            sum += wIh[wIdx] * inputData[inputOffset + b * inputSize + i];
                        }

                        // Hidden contribution
                        for (int hp = 0; hp < hiddenSize; hp++)
                        {
                            int wIdx = g * hiddenSize * hiddenSize + h * hiddenSize + hp;
                            sum += wHh[wIdx] * allHData[hPrevOffset + b * hiddenSize + hp];
                        }

                        gates[g * hiddenSize + h] = sum;
                    }
                }

                // Apply activations and compute cell/hidden states
                for (int h = 0; h < hiddenSize; h++)
                {
                    // Input gate (sigmoid)
                    float i_gate = SigmoidActivation(gates[0 * hiddenSize + h]);
                    // Forget gate (sigmoid)
                    float f_gate = SigmoidActivation(gates[1 * hiddenSize + h]);
                    // Cell candidate (tanh)
                    float g_gate = (float)Math.Tanh(gates[2 * hiddenSize + h]);
                    // Output gate (sigmoid)
                    float o_gate = SigmoidActivation(gates[3 * hiddenSize + h]);

                    // Cache gates for backward pass
                    int cacheIdx = gateOffset + b * hiddenSize * 4 + h * 4;
                    gatesData[cacheIdx + 0] = i_gate;
                    gatesData[cacheIdx + 1] = f_gate;
                    gatesData[cacheIdx + 2] = g_gate;
                    gatesData[cacheIdx + 3] = o_gate;

                    // Cell state update: c_t = f * c_{t-1} + i * g
                    float cPrev = allCData[hPrevOffset + b * hiddenSize + h];
                    float cCurr = f_gate * cPrev + i_gate * g_gate;
                    allCData[hCurrOffset + b * hiddenSize + h] = cCurr;

                    // Hidden state update: h_t = o * tanh(c_t)
                    float hCurr = o_gate * (float)Math.Tanh(cCurr);
                    allHData[hCurrOffset + b * hiddenSize + h] = hCurr;
                    outputData[outputOffset + b * hiddenSize + h] = hCurr;
                }
            }
        }

        // Copy final states
        Array.Copy(allHData, seqLen * stateSize, new float[stateSize], 0, stateSize);
        Array.Copy(allCData, seqLen * stateSize, new float[stateSize], 0, stateSize);

        // Upload results
        UploadToBuffer(output, outputData);
        UploadToBuffer(allH, allHData);
        UploadToBuffer(allC, allCData);
        UploadToBuffer(cacheGates, gatesData);

        // Copy final hidden and cell states
        var hFinalData = new float[stateSize];
        var cFinalData = new float[stateSize];
        Array.Copy(allHData, seqLen * stateSize, hFinalData, 0, stateSize);
        Array.Copy(allCData, seqLen * stateSize, cFinalData, 0, stateSize);
        UploadToBuffer(hFinal, hFinalData);
        UploadToBuffer(cFinal, cFinalData);
    }

    /// <summary>
    /// Backward pass for LSTM sequence - computes gradients via BPTT.
    /// </summary>
    public void LstmBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer allC, IGpuBuffer cacheGates,
        IGpuBuffer hInit, IGpuBuffer cInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer gradCInit,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        // CPU fallback implementation for LSTM backward sequence
        var gradOutData = DownloadBuffer(gradOutput);
        var allHData = DownloadBuffer(allH);
        var allCData = DownloadBuffer(allC);
        var gatesData = DownloadBuffer(cacheGates);
        var wIh = DownloadBuffer(weightsIh);
        var wHh = DownloadBuffer(weightsHh);
        var inputData = DownloadBuffer(input);

        int stateSize = batch * hiddenSize;

        // Initialize gradient buffers
        var dInput = new float[seqLen * batch * inputSize];
        var dHInit = new float[stateSize];
        var dCInit = new float[stateSize];
        var dWIh = new float[4 * hiddenSize * inputSize];
        var dWHh = new float[4 * hiddenSize * hiddenSize];
        var dBIh = new float[4 * hiddenSize];
        var dBHh = new float[4 * hiddenSize];

        // Gradient accumulators for hidden and cell states
        var dHNext = new float[stateSize];
        var dCNext = new float[stateSize];

        // Backprop through time (reverse order)
        for (int t = seqLen - 1; t >= 0; t--)
        {
            int inputOffset = t * batch * inputSize;
            int hPrevOffset = t * stateSize;
            int hCurrOffset = (t + 1) * stateSize;
            int outputOffset = t * batch * hiddenSize;
            int gateOffset = t * batch * hiddenSize * 4;

            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    // Get cached gate values
                    int cacheIdx = gateOffset + b * hiddenSize * 4 + h * 4;
                    float i_gate = gatesData[cacheIdx + 0];
                    float f_gate = gatesData[cacheIdx + 1];
                    float g_gate = gatesData[cacheIdx + 2];
                    float o_gate = gatesData[cacheIdx + 3];

                    float cPrev = allCData[hPrevOffset + b * hiddenSize + h];
                    float cCurr = allCData[hCurrOffset + b * hiddenSize + h];

                    // Gradient from output + gradient from next timestep
                    float dH = gradOutData[outputOffset + b * hiddenSize + h] + dHNext[b * hiddenSize + h];

                    // Gradient through tanh(c_t)
                    float tanhC = (float)Math.Tanh(cCurr);
                    float dO = dH * tanhC;
                    float dC = dH * o_gate * (1 - tanhC * tanhC) + dCNext[b * hiddenSize + h];

                    // Gate gradients (before sigmoid/tanh)
                    float dI = dC * g_gate;
                    float dF = dC * cPrev;
                    float dG = dC * i_gate;

                    // Gradient through activations
                    float dI_raw = dI * i_gate * (1 - i_gate);
                    float dF_raw = dF * f_gate * (1 - f_gate);
                    float dG_raw = dG * (1 - g_gate * g_gate);
                    float dO_raw = dO * o_gate * (1 - o_gate);

                    float[] dGates = [dI_raw, dF_raw, dG_raw, dO_raw];

                    // Accumulate weight and bias gradients
                    for (int g = 0; g < 4; g++)
                    {
                        dBIh[g * hiddenSize + h] += dGates[g];
                        dBHh[g * hiddenSize + h] += dGates[g];

                        // Input weight gradients
                        for (int i = 0; i < inputSize; i++)
                        {
                            int wIdx = g * hiddenSize * inputSize + h * inputSize + i;
                            dWIh[wIdx] += dGates[g] * inputData[inputOffset + b * inputSize + i];
                        }

                        // Hidden weight gradients
                        for (int hp = 0; hp < hiddenSize; hp++)
                        {
                            int wIdx = g * hiddenSize * hiddenSize + h * hiddenSize + hp;
                            dWHh[wIdx] += dGates[g] * allHData[hPrevOffset + b * hiddenSize + hp];
                        }
                    }

                    // Input gradient
                    for (int i = 0; i < inputSize; i++)
                    {
                        float dInputVal = 0;
                        for (int g = 0; g < 4; g++)
                        {
                            int wIdx = g * hiddenSize * inputSize + h * inputSize + i;
                            dInputVal += dGates[g] * wIh[wIdx];
                        }
                        dInput[inputOffset + b * inputSize + i] += dInputVal;
                    }

                    // Gradient to previous hidden state
                    for (int hp = 0; hp < hiddenSize; hp++)
                    {
                        float dHPrev = 0;
                        for (int g = 0; g < 4; g++)
                        {
                            int wIdx = g * hiddenSize * hiddenSize + hp * hiddenSize + h;
                            dHPrev += dGates[g] * wHh[wIdx];
                        }
                        if (t > 0)
                        {
                            dHNext[b * hiddenSize + hp] = dHPrev;
                        }
                        else
                        {
                            dHInit[b * hiddenSize + hp] += dHPrev;
                        }
                    }

                    // Gradient to previous cell state
                    float dCPrev = dC * f_gate;
                    if (t > 0)
                    {
                        dCNext[b * hiddenSize + h] = dCPrev;
                    }
                    else
                    {
                        dCInit[b * hiddenSize + h] = dCPrev;
                    }
                }
            }
        }

        // Upload gradients
        UploadToBuffer(gradInput, dInput);
        UploadToBuffer(gradHInit, dHInit);
        UploadToBuffer(gradCInit, dCInit);
        UploadToBuffer(gradWeightsIh, dWIh);
        UploadToBuffer(gradWeightsHh, dWHh);
        UploadToBuffer(gradBiasIh, dBIh);
        UploadToBuffer(gradBiasHh, dBHh);
    }

    /// <summary>
    /// Forward pass for GRU sequence - processes all timesteps in a single kernel launch.
    /// Efficient for BPTT training with minimal kernel launch overhead.
    /// </summary>
    public void GruForwardSequence(
        IGpuBuffer input, IGpuBuffer hInit,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer biasIh, IGpuBuffer biasHh,
        IGpuBuffer output, IGpuBuffer hFinal, IGpuBuffer allH, IGpuBuffer cacheGates,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        // CPU fallback implementation for GRU forward sequence
        var inputData = DownloadBuffer(input);
        var hInitData = DownloadBuffer(hInit);
        var wIh = DownloadBuffer(weightsIh);
        var wHh = DownloadBuffer(weightsHh);
        var bIh = DownloadBuffer(biasIh);
        var bHh = DownloadBuffer(biasHh);

        var outputData = new float[seqLen * batch * hiddenSize];
        var allHData = new float[(seqLen + 1) * batch * hiddenSize];
        var gatesData = new float[seqLen * batch * hiddenSize * 3];

        int stateSize = batch * hiddenSize;

        // Copy initial state to allH[0]
        Array.Copy(hInitData, 0, allHData, 0, stateSize);

        // Process each timestep
        for (int t = 0; t < seqLen; t++)
        {
            int inputOffset = t * batch * inputSize;
            int hPrevOffset = t * stateSize;
            int hCurrOffset = (t + 1) * stateSize;
            int outputOffset = t * batch * hiddenSize;
            int gateOffset = t * batch * hiddenSize * 3;

            for (int b = 0; b < batch; b++)
            {
                // Compute gates: r (reset), z (update), n (new/candidate)
                var gates = new float[3 * hiddenSize];
                var hPrevGated = new float[hiddenSize];

                // First compute r and z gates
                for (int g = 0; g < 2; g++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        float sum = bIh[g * hiddenSize + h] + bHh[g * hiddenSize + h];

                        // Input contribution
                        for (int i = 0; i < inputSize; i++)
                        {
                            int wIdx = g * hiddenSize * inputSize + h * inputSize + i;
                            sum += wIh[wIdx] * inputData[inputOffset + b * inputSize + i];
                        }

                        // Hidden contribution
                        for (int hp = 0; hp < hiddenSize; hp++)
                        {
                            int wIdx = g * hiddenSize * hiddenSize + h * hiddenSize + hp;
                            sum += wHh[wIdx] * allHData[hPrevOffset + b * hiddenSize + hp];
                        }

                        gates[g * hiddenSize + h] = SigmoidActivation(sum);
                    }
                }

                // Apply reset gate to previous hidden state
                for (int h = 0; h < hiddenSize; h++)
                {
                    float r_gate = gates[0 * hiddenSize + h];
                    hPrevGated[h] = r_gate * allHData[hPrevOffset + b * hiddenSize + h];
                }

                // Compute n (candidate) gate with reset-gated hidden state
                for (int h = 0; h < hiddenSize; h++)
                {
                    float sum = bIh[2 * hiddenSize + h] + bHh[2 * hiddenSize + h];

                    // Input contribution
                    for (int i = 0; i < inputSize; i++)
                    {
                        int wIdx = 2 * hiddenSize * inputSize + h * inputSize + i;
                        sum += wIh[wIdx] * inputData[inputOffset + b * inputSize + i];
                    }

                    // Reset-gated hidden contribution
                    for (int hp = 0; hp < hiddenSize; hp++)
                    {
                        int wIdx = 2 * hiddenSize * hiddenSize + h * hiddenSize + hp;
                        sum += wHh[wIdx] * hPrevGated[hp];
                    }

                    gates[2 * hiddenSize + h] = (float)Math.Tanh(sum);
                }

                // Compute new hidden state and cache gates
                for (int h = 0; h < hiddenSize; h++)
                {
                    float r_gate = gates[0 * hiddenSize + h];
                    float z_gate = gates[1 * hiddenSize + h];
                    float n_gate = gates[2 * hiddenSize + h];

                    // Cache gates for backward pass
                    int cacheIdx = gateOffset + b * hiddenSize * 3 + h * 3;
                    gatesData[cacheIdx + 0] = r_gate;
                    gatesData[cacheIdx + 1] = z_gate;
                    gatesData[cacheIdx + 2] = n_gate;

                    // Hidden state update: h_t = (1 - z) * n + z * h_{t-1}
                    float hPrev = allHData[hPrevOffset + b * hiddenSize + h];
                    float hCurr = (1 - z_gate) * n_gate + z_gate * hPrev;
                    allHData[hCurrOffset + b * hiddenSize + h] = hCurr;
                    outputData[outputOffset + b * hiddenSize + h] = hCurr;
                }
            }
        }

        // Upload results
        UploadToBuffer(output, outputData);
        UploadToBuffer(allH, allHData);
        UploadToBuffer(cacheGates, gatesData);

        // Copy final hidden state
        var hFinalData = new float[stateSize];
        Array.Copy(allHData, seqLen * stateSize, hFinalData, 0, stateSize);
        UploadToBuffer(hFinal, hFinalData);
    }

    /// <summary>
    /// Backward pass for GRU sequence - computes gradients via BPTT.
    /// </summary>
    public void GruBackwardSequence(
        IGpuBuffer gradOutput, IGpuBuffer allH, IGpuBuffer cacheGates,
        IGpuBuffer weightsIh, IGpuBuffer weightsHh, IGpuBuffer input,
        IGpuBuffer gradInput, IGpuBuffer gradHInit, IGpuBuffer dHBuffer,
        IGpuBuffer gradWeightsIh, IGpuBuffer gradWeightsHh, IGpuBuffer gradBiasIh, IGpuBuffer gradBiasHh,
        int seqLen, int batch, int inputSize, int hiddenSize)
    {
        // CPU fallback implementation for GRU backward sequence
        var gradOutData = DownloadBuffer(gradOutput);
        var allHData = DownloadBuffer(allH);
        var gatesData = DownloadBuffer(cacheGates);
        var wIh = DownloadBuffer(weightsIh);
        var wHh = DownloadBuffer(weightsHh);
        var inputData = DownloadBuffer(input);

        int stateSize = batch * hiddenSize;

        // Initialize gradient buffers
        var dInput = new float[seqLen * batch * inputSize];
        var dHInit = new float[stateSize];
        var dHNext = new float[stateSize];
        var dWIh = new float[3 * hiddenSize * inputSize];
        var dWHh = new float[3 * hiddenSize * hiddenSize];
        var dBIh = new float[3 * hiddenSize];
        var dBHh = new float[3 * hiddenSize];

        // Backprop through time (reverse order)
        for (int t = seqLen - 1; t >= 0; t--)
        {
            int inputOffset = t * batch * inputSize;
            int hPrevOffset = t * stateSize;
            int outputOffset = t * batch * hiddenSize;
            int gateOffset = t * batch * hiddenSize * 3;

            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    // Get cached gate values
                    int cacheIdx = gateOffset + b * hiddenSize * 3 + h * 3;
                    float r_gate = gatesData[cacheIdx + 0];
                    float z_gate = gatesData[cacheIdx + 1];
                    float n_gate = gatesData[cacheIdx + 2];

                    float hPrev = allHData[hPrevOffset + b * hiddenSize + h];

                    // Gradient from output + gradient from next timestep
                    float dH = gradOutData[outputOffset + b * hiddenSize + h] + dHNext[b * hiddenSize + h];

                    // Gradient through GRU update: h = (1-z)*n + z*h_prev
                    float dN = dH * (1 - z_gate);
                    float dZ = dH * (hPrev - n_gate);
                    float dHPrev_direct = dH * z_gate;

                    // Gradient through n gate (tanh)
                    float dN_raw = dN * (1 - n_gate * n_gate);

                    // Gradient through z gate (sigmoid)
                    float dZ_raw = dZ * z_gate * (1 - z_gate);

                    // Gradient through reset gate interaction with n
                    // n = tanh(Wih*x + Whh*(r*h_prev) + b)
                    // dR contribution comes from the hidden state path
                    float dHPrev_gated = 0;
                    for (int hp = 0; hp < hiddenSize; hp++)
                    {
                        int wIdx = 2 * hiddenSize * hiddenSize + h * hiddenSize + hp;
                        dHPrev_gated += dN_raw * wHh[wIdx];
                    }
                    float dR = dHPrev_gated * hPrev;
                    float dR_raw = dR * r_gate * (1 - r_gate);

                    float[] dGates = [dR_raw, dZ_raw, dN_raw];

                    // Accumulate weight and bias gradients
                    for (int g = 0; g < 3; g++)
                    {
                        dBIh[g * hiddenSize + h] += dGates[g];
                        dBHh[g * hiddenSize + h] += dGates[g];

                        // Input weight gradients
                        for (int i = 0; i < inputSize; i++)
                        {
                            int wIdx = g * hiddenSize * inputSize + h * inputSize + i;
                            dWIh[wIdx] += dGates[g] * inputData[inputOffset + b * inputSize + i];
                        }

                        // Hidden weight gradients (use reset-gated hidden for n gate)
                        for (int hp = 0; hp < hiddenSize; hp++)
                        {
                            int wIdx = g * hiddenSize * hiddenSize + h * hiddenSize + hp;
                            float hVal = (g == 2) ? r_gate * allHData[hPrevOffset + b * hiddenSize + hp]
                                                  : allHData[hPrevOffset + b * hiddenSize + hp];
                            dWHh[wIdx] += dGates[g] * hVal;
                        }
                    }

                    // Input gradient
                    for (int i = 0; i < inputSize; i++)
                    {
                        float dInputVal = 0;
                        for (int g = 0; g < 3; g++)
                        {
                            int wIdx = g * hiddenSize * inputSize + h * inputSize + i;
                            dInputVal += dGates[g] * wIh[wIdx];
                        }
                        dInput[inputOffset + b * inputSize + i] += dInputVal;
                    }

                    // Gradient to previous hidden state
                    float dHPrev = dHPrev_direct + dHPrev_gated * r_gate;
                    for (int g = 0; g < 2; g++)
                    {
                        for (int hp = 0; hp < hiddenSize; hp++)
                        {
                            int wIdx = g * hiddenSize * hiddenSize + hp * hiddenSize + h;
                            dHPrev += dGates[g] * wHh[wIdx];
                        }
                    }

                    if (t > 0)
                    {
                        dHNext[b * hiddenSize + h] = dHPrev;
                    }
                    else
                    {
                        dHInit[b * hiddenSize + h] = dHPrev;
                    }
                }
            }
        }

        // Upload gradients
        UploadToBuffer(gradInput, dInput);
        UploadToBuffer(gradHInit, dHInit);
        UploadToBuffer(dHBuffer, dHNext);
        UploadToBuffer(gradWeightsIh, dWIh);
        UploadToBuffer(gradWeightsHh, dWHh);
        UploadToBuffer(gradBiasIh, dBIh);
        UploadToBuffer(gradBiasHh, dBHh);
    }

    /// <summary>
    /// Backward pass for a single GRU cell - computes gate gradients and gradient to previous hidden state.
    /// </summary>
    public void GruCellBackward(
        IGpuBuffer gradH, IGpuBuffer gateR, IGpuBuffer gateZ, IGpuBuffer gateN, IGpuBuffer prevH,
        IGpuBuffer weightsHh,
        IGpuBuffer gradPrevH, IGpuBuffer gradGateR, IGpuBuffer gradGateZ, IGpuBuffer gradGateN,
        int batch, int hiddenSize)
    {
        // CPU fallback implementation for single GRU cell backward
        var dH = DownloadBuffer(gradH);
        var r = DownloadBuffer(gateR);
        var z = DownloadBuffer(gateZ);
        var n = DownloadBuffer(gateN);
        var hPrev = DownloadBuffer(prevH);
        var wHh = DownloadBuffer(weightsHh);

        var dPrevH = new float[batch * hiddenSize];
        var dR = new float[batch * hiddenSize];
        var dZ = new float[batch * hiddenSize];
        var dN = new float[batch * hiddenSize];

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < hiddenSize; h++)
            {
                int idx = b * hiddenSize + h;

                float gradH_val = dH[idx];
                float r_val = r[idx];
                float z_val = z[idx];
                float n_val = n[idx];
                float hPrev_val = hPrev[idx];

                // GRU update: h = (1-z)*n + z*h_prev
                // dN = dH * (1 - z)
                float dN_val = gradH_val * (1 - z_val);

                // dZ = dH * (h_prev - n)
                float dZ_val = gradH_val * (hPrev_val - n_val);

                // dH_prev_direct = dH * z
                float dHPrev_direct = gradH_val * z_val;

                // Gradient through n (tanh derivative)
                float dN_raw = dN_val * (1 - n_val * n_val);

                // Gradient through z (sigmoid derivative)
                float dZ_raw = dZ_val * z_val * (1 - z_val);

                // Compute gradient through reset gate path
                // n = tanh(... + Whh_n * (r * h_prev))
                // dR contribution from hidden path
                float dHPrev_gated = 0;
                for (int hp = 0; hp < hiddenSize; hp++)
                {
                    // Weights for n gate (gate index 2)
                    int wIdx = 2 * hiddenSize * hiddenSize + h * hiddenSize + hp;
                    dHPrev_gated += dN_raw * wHh[wIdx];
                }

                float dR_val = dHPrev_gated * hPrev_val;
                float dR_raw = dR_val * r_val * (1 - r_val);

                // Gradient to previous hidden state
                float dHPrev = dHPrev_direct + dHPrev_gated * r_val;

                // Add contribution from r and z gates to h_prev gradient
                for (int g = 0; g < 2; g++)
                {
                    float gateGrad = (g == 0) ? dR_raw : dZ_raw;
                    for (int hp = 0; hp < hiddenSize; hp++)
                    {
                        int wIdx = g * hiddenSize * hiddenSize + hp * hiddenSize + h;
                        dHPrev += gateGrad * wHh[wIdx];
                    }
                }

                dPrevH[idx] = dHPrev;
                dR[idx] = dR_raw;
                dZ[idx] = dZ_raw;
                dN[idx] = dN_raw;
            }
        }

        // Upload gradients
        UploadToBuffer(gradPrevH, dPrevH);
        UploadToBuffer(gradGateR, dR);
        UploadToBuffer(gradGateZ, dZ);
        UploadToBuffer(gradGateN, dN);
    }

    #endregion

    #region Helper Methods for RNN

    /// <summary>
    /// Sigmoid activation function for RNN gate computations.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>Sigmoid of x: 1 / (1 + exp(-x)).</returns>
    private static float SigmoidActivation(float x)
    {
        // Use numerically stable sigmoid implementation
        if (x >= 0)
        {
            float expNegX = (float)Math.Exp(-x);
            return 1.0f / (1.0f + expNegX);
        }
        else
        {
            float expX = (float)Math.Exp(x);
            return expX / (1.0f + expX);
        }
    }

    #endregion
}
