using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Phase 5.2: Activation memory optimization for compiled training plans.
///
/// For training, forward activations must be stored for backward. However,
/// some activations can be stored more efficiently:
///
/// - ReLU: only needs a bitmask (1 bit/element vs 32 bits for float)
///   backward: grad * (input > 0) — the bitmask IS the derivative
///
/// - Sigmoid/Tanh: backward can be computed from the OUTPUT alone
///   sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
///   tanh'(x) = 1 - tanh(x)^2
///   No need to store the input — just reuse the forward output
///
/// This can reduce activation memory by 4-32x for common models.
/// </summary>
internal static class ActivationCheckpoint
{
    /// <summary>
    /// Determines the optimal storage strategy for a forward activation.
    /// Returns the storage type and any required metadata.
    /// </summary>
    internal static ActivationStorageType GetStorageType(string opName)
    {
        return opName switch
        {
            "ReLU" => ActivationStorageType.Bitmask,
            "Sigmoid" => ActivationStorageType.ReuseOutput,
            "Tanh" => ActivationStorageType.ReuseOutput,
            _ => ActivationStorageType.Full
        };
    }

    /// <summary>
    /// Creates a bitmask from a ReLU forward pass.
    /// Each bit indicates whether the corresponding element was positive.
    /// </summary>
    internal static byte[] CreateReluBitmask(float[] activationData, int length)
    {
        int byteCount = (length + 7) / 8;
        var mask = new byte[byteCount];

        for (int i = 0; i < length; i++)
        {
            if (activationData[i] > 0f)
                mask[i / 8] |= (byte)(1 << (i % 8));
        }

        return mask;
    }

    /// <summary>
    /// Applies ReLU backward using a bitmask instead of the full activation tensor.
    /// grad_input = grad_output * bitmask (element-wise)
    /// </summary>
    internal static void ApplyReluBackwardFromBitmask(
        float[] gradOutput, byte[] bitmask, float[] gradInput, int length)
    {
        for (int i = 0; i < length; i++)
        {
            bool active = (bitmask[i / 8] & (1 << (i % 8))) != 0;
            gradInput[i] = active ? gradOutput[i] : 0f;
        }
    }

    /// <summary>
    /// Memory savings factor for this storage type vs full storage.
    /// </summary>
    internal static int MemorySavingsFactor(ActivationStorageType type)
    {
        return type switch
        {
            ActivationStorageType.Bitmask => 32,  // 1 bit vs 32 bits (float)
            ActivationStorageType.ReuseOutput => int.MaxValue, // No extra storage
            _ => 1
        };
    }
}

/// <summary>How to store forward activations for backward pass.</summary>
internal enum ActivationStorageType
{
    /// <summary>Store the full activation tensor (default).</summary>
    Full,

    /// <summary>Store only a bitmask of positive elements (ReLU).</summary>
    Bitmask,

    /// <summary>Reuse the forward output tensor (Sigmoid, Tanh).</summary>
    ReuseOutput,

    /// <summary>Recompute from a checkpoint during backward (gradient checkpointing).</summary>
    Recompute
}
