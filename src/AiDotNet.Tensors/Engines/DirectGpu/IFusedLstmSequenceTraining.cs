namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend whose <see cref="IDirectGpuBackend.LstmForwardSequence"/> and <see cref="IDirectGpuBackend.LstmBackwardSequence"/>
/// meet the contract the engine's fused LSTM training path relies on: batch-major input [B, T, in] and output
/// [B, T, H], PyTorch gate order (i, f, g, o), the two biases summed, and full BPTT gradients for the input, both
/// weights and the bias, for any hidden size up to <see cref="MaxFusedLstmHidden"/>.
/// </summary>
/// <remarks>
/// Declared by every backend after checking against a CPU forward + BPTT reference (LstmSequenceBackendParityTests).
/// CUDA, HIP, OpenCL and Vulkan run their own sequence kernels. Metal and WebGPU run the shared lstm_p_ kernels
/// (PortableLstmSequence, #1107 / #1108); the test runs that same source on OpenCL, Vulkan and HIP as well.
/// </remarks>
internal interface IFusedLstmSequenceTraining
{
    /// <summary>The largest hidden size the sequence kernels handle correctly on this device.</summary>
    int MaxFusedLstmHidden { get; }

    /// <summary>
    /// Whether <see cref="IDirectGpuBackend.LstmBackwardSequence"/> writes every gradient output rather than
    /// accumulating into it, so the caller need not zero them first (on CUDA that zeroing was five memsets per LSTM
    /// layer in every captured training step).
    /// </summary>
    bool LstmBackwardOverwritesGradients { get; }
}
