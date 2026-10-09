namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend whose <see cref="IDirectGpuBackend.LstmForwardSequence"/> and <see cref="IDirectGpuBackend.LstmBackwardSequence"/>
/// meet the contract the engine's fused LSTM training path relies on: batch-major input [B, T, in] and output
/// [B, T, H], PyTorch gate order (i, f, g, o), the two biases summed, and full BPTT gradients for the input, both
/// weights and the bias, for any hidden size up to <see cref="MaxFusedLstmHidden"/>.
/// </summary>
/// <remarks>
/// Declared only by backends verified against a CPU forward + BPTT reference (LstmSequenceBackendParityTests):
/// CUDA, HIP, OpenCL and Vulkan. Metal and WebGPU are left out until their kernels are checked on hardware; there the
/// engine keeps the per-timestep ops, which give the same gradients with more launches. Tracked per backend: Metal in
/// #1107, WebGPU in #1108.
/// </remarks>
internal interface IFusedLstmSequenceTraining
{
    /// <summary>The largest hidden size the sequence kernels handle correctly on this device.</summary>
    int MaxFusedLstmHidden { get; }
}