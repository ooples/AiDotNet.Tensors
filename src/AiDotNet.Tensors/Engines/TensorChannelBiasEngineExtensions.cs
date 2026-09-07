using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Semantic channel-first bias operations that preserve compatibility with third-party
/// <see cref="IEngine"/> implementations.
/// </summary>
public static class TensorChannelBiasEngineExtensions
{
    /// <summary>
    /// Adds a rank-1 channel bias to a channel-first tensor without creating a storage-sharing
    /// reshape of the bias. For an input shaped <c>[N, C, ...]</c>, computes
    /// <c>output[n, c, ...] = input[n, c, ...] + bias[c]</c>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="engine">The engine that performs the operation.</param>
    /// <param name="input">A contiguous or strided channel-first tensor with rank at least two.</param>
    /// <param name="bias">A rank-1 tensor whose length equals the input's channel dimension.</param>
    /// <returns>A tensor with the same logical shape as <paramref name="input"/>.</returns>
    public static Tensor<T> TensorChannelBiasAdd<T>(
        this IEngine engine,
        Tensor<T> input,
        Tensor<T> bias)
    {
        if (engine is null)
            throw new ArgumentNullException(nameof(engine));
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (bias is null)
            throw new ArgumentNullException(nameof(bias));
        if (input.Rank < 2)
            throw new ArgumentException("Channel bias input must have rank at least two.", nameof(input));
        if (bias.Rank != 1)
            throw new ArgumentException("Channel bias must be rank one.", nameof(bias));
        if (bias.Length != input.Shape[1])
        {
            throw new ArgumentException(
                $"Channel bias length {bias.Length} does not match input channel count {input.Shape[1]}.",
                nameof(bias));
        }

        // Every built-in engine derives from CpuEngine; virtual dispatch reaches the resident GPU
        // implementation where available. Keeping the primitive off IEngine avoids an ABI break for
        // external engines compiled against an earlier package.
        if (engine is CpuEngine builtIn)
            return builtIn.TensorChannelBiasAdd(input, bias);

        // A third-party engine can express the same differentiable operation using only the existing
        // IEngine contract: move channels to the broadcast-last axis, add the rank-1 bias directly,
        // then restore channel-first order. Crucially, this never creates a view of the parameter.
        int rank = input.Rank;
        if (rank == 2)
            return engine.TensorAdd(input, bias);

        var channelsLast = new int[rank];
        channelsLast[0] = 0;
        for (int axis = 1; axis < rank - 1; axis++)
            channelsLast[axis] = axis + 1;
        channelsLast[rank - 1] = 1;

        var channelFirst = new int[rank];
        channelFirst[0] = 0;
        channelFirst[1] = rank - 1;
        for (int axis = 2; axis < rank; axis++)
            channelFirst[axis] = axis - 1;

        var permuted = engine.TensorPermute(input, channelsLast);
        var biased = engine.TensorAdd(permuted, bias);
        return engine.TensorPermute(biased, channelFirst);
    }
}
