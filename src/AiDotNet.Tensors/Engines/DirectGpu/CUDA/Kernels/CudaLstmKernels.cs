// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
        // An embedded resource: as a literal the text counted against the assembly's string heap (CS8103).
        return KernelSourceResources.Read("CudaLstm.c");
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
            "lstm_forward_recurrence_reg_h32_b1",
            "lstm_backward_recurrence_reg_h32_b1",
            "lstm_forward_recurrence_reg_h64_b1",
            "lstm_forward_recurrence_shfl_h32",
            "lstm_forward_recurrence_shfl_h64",
            "lstm_backward_last_recurrence_reg_h32_b1",
            "lstm_backward_last_recurrence_reg_h64_b1",
            "lstm_backward_recurrence_reg_h64_b1",
            "lstm_accumulate_weight_gradients",
            "lstm_accumulate_weight_gradients_deterministic",
            "lstm_compute_gate_gradients",
            "gru_forward_recurrence_reg_h32",
            "gru_backward_recurrence_reg_h32",
            "gru_forward_recurrence_reg_h64",
            "gru_backward_recurrence_reg_h64",
            "gru_backward_last_recurrence_reg_h32",
            "gru_backward_last_recurrence_reg_h64",
            "gru_backward_gemms"
        };
    }
}
