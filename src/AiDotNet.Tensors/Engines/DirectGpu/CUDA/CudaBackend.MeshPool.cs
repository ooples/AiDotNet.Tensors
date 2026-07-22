using System;
using AiDotNet.Tensors.Engines.DirectGpu;
#if NET5_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
#endif

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    /// <summary>Computes one importance score per contiguous edge row.</summary>
    public void MeshPoolComputeScores(IGpuBuffer input, IGpuBuffer importanceWeights, IGpuBuffer scores,
        int numEdges, int inputChannels)
    {
        RequireMeshBuffer(input, checked((long)numEdges * inputChannels), nameof(input));
        RequireMeshBuffer(importanceWeights, inputChannels, nameof(importanceWeights));
        RequireMeshBuffer(scores, numEdges, nameof(scores));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.ComputeScores, input, importanceWeights, scores, null,
            numEdges, PtxMeshPoolF32Kernel.NumKept, inputChannels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_compute_scores", input, importanceWeights, scores, null,
            numEdges, 0, inputChannels);
    }

    /// <summary>Gathers kept edge rows into contiguous output storage.</summary>
    public void MeshPoolGather(IGpuBuffer input, IGpuBuffer keptIndices, IGpuBuffer output,
        int numKept, int numEdges, int inputChannels)
    {
        RequireMeshBuffer(input, checked((long)numEdges * inputChannels), nameof(input));
        RequireMeshBuffer(keptIndices, numKept, nameof(keptIndices));
        RequireMeshBuffer(output, checked((long)numKept * inputChannels), nameof(output));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.Gather, input, keptIndices, output, null,
            numEdges, numKept, inputChannels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_gather", input, keptIndices, output, null,
            numEdges, numKept, inputChannels);
    }

    /// <summary>Scatters pooled gradients, selecting deterministic ownership when requested.</summary>
    public void MeshPoolBackward(IGpuBuffer gradOutput, IGpuBuffer keptIndices, IGpuBuffer gradInput,
        int numKept, int numEdges, int inputChannels, bool deterministic)
    {
        RequireMeshBuffer(gradOutput, checked((long)numKept * inputChannels), nameof(gradOutput));
        RequireMeshBuffer(keptIndices, numKept, nameof(keptIndices));
        RequireMeshBuffer(gradInput, checked((long)numEdges * inputChannels), nameof(gradInput));
        string name = deterministic ? "mesh_pool_backward_deterministic" : "mesh_pool_backward";
#if NET5_0_OR_GREATER
        DirectPtxMeshPoolOperation operation = deterministic
            ? DirectPtxMeshPoolOperation.BackwardDeterministic : DirectPtxMeshPoolOperation.BackwardAtomic;
        if (TryDirectPtxMeshPool(operation, gradOutput, keptIndices, gradInput, null,
            numEdges, numKept, inputChannels)) return;
#endif
        LaunchMeshPoolFallback(name, gradOutput, keptIndices, gradInput, null,
            numEdges, numKept, inputChannels);
    }

    /// <summary>Computes the importance-weight gradient in ascending kept-edge order.</summary>
    public void MeshPoolImportanceBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer keptIndices,
        IGpuBuffer gradImportanceWeights, int numKept, int numEdges, int inputChannels)
    {
        RequireMeshBuffer(gradOutput, checked((long)numKept * inputChannels), nameof(gradOutput));
        RequireMeshBuffer(input, checked((long)numEdges * inputChannels), nameof(input));
        RequireMeshBuffer(keptIndices, numKept, nameof(keptIndices));
        RequireMeshBuffer(gradImportanceWeights, inputChannels, nameof(gradImportanceWeights));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.ImportanceBackward, gradOutput, input, keptIndices,
            gradImportanceWeights, numEdges, numKept, inputChannels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_importance_backward", gradOutput, input, keptIndices,
            gradImportanceWeights, numEdges, numKept, inputChannels);
    }

    /// <summary>Zeroes a contiguous MeshPool edge-gradient tensor.</summary>
    public void MeshPoolZeroGrad(IGpuBuffer gradInput, int numEdges, int inputChannels)
    {
        RequireMeshBuffer(gradInput, checked((long)numEdges * inputChannels), nameof(gradInput));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.ZeroGrad, gradInput, null, null, null,
            numEdges, PtxMeshPoolF32Kernel.NumKept, inputChannels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_zero_grad", gradInput, null, null, null,
            numEdges, 0, inputChannels);
    }

    public void MeshPoolSoftmaxFindMax(IGpuBuffer scores, IGpuBuffer partialMax, int numEdges)
    {
        int partials = checked((numEdges + (int)DefaultBlockSize - 1) / (int)DefaultBlockSize);
        RequireMeshBuffer(scores, numEdges, nameof(scores)); RequireMeshBuffer(partialMax, partials, nameof(partialMax));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.SoftmaxFindMax, scores, partialMax, null, null,
            numEdges, PtxMeshPoolF32Kernel.NumKept, PtxMeshPoolF32Kernel.Channels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_softmax_find_max", scores, partialMax, null, null,
            numEdges, partials, 0);
    }

    public void MeshPoolSoftmaxFinalMax(IGpuBuffer partialMax, IGpuBuffer globalMax, int numPartials)
    {
        RequireMeshBuffer(partialMax, numPartials, nameof(partialMax)); RequireMeshBuffer(globalMax, 1, nameof(globalMax));
#if NET5_0_OR_GREATER
        if (numPartials == PtxMeshPoolF32Kernel.Partials &&
            TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.SoftmaxFinalMax, partialMax, globalMax, null, null,
            PtxMeshPoolF32Kernel.NumEdges, PtxMeshPoolF32Kernel.NumKept, PtxMeshPoolF32Kernel.Channels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_softmax_final_max", partialMax, globalMax, null, null,
            0, numPartials, 0);
    }

    public void MeshPoolSoftmaxExpSum(IGpuBuffer scores, IGpuBuffer globalMax, IGpuBuffer expValues,
        IGpuBuffer partialSum, float temperature, int numEdges)
    {
        int partials = checked((numEdges + (int)DefaultBlockSize - 1) / (int)DefaultBlockSize);
        RequireMeshBuffer(scores, numEdges, nameof(scores)); RequireMeshBuffer(globalMax, 1, nameof(globalMax));
        RequireMeshBuffer(expValues, numEdges, nameof(expValues)); RequireMeshBuffer(partialSum, partials, nameof(partialSum));
        if (!(temperature > 0f) || float.IsNaN(temperature)) throw new ArgumentOutOfRangeException(nameof(temperature));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.SoftmaxExpSum, scores, globalMax, expValues, partialSum,
            numEdges, PtxMeshPoolF32Kernel.NumKept, PtxMeshPoolF32Kernel.Channels, temperature)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_softmax_exp_sum", scores, globalMax, expValues, partialSum,
            numEdges, partials, 0, temperature);
    }

    public void MeshPoolSoftmaxFinalSum(IGpuBuffer partialSum, IGpuBuffer globalSum, int numPartials)
    {
        RequireMeshBuffer(partialSum, numPartials, nameof(partialSum)); RequireMeshBuffer(globalSum, 1, nameof(globalSum));
#if NET5_0_OR_GREATER
        if (numPartials == PtxMeshPoolF32Kernel.Partials &&
            TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.SoftmaxFinalSum, partialSum, globalSum, null, null,
            PtxMeshPoolF32Kernel.NumEdges, PtxMeshPoolF32Kernel.NumKept, PtxMeshPoolF32Kernel.Channels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_softmax_final_sum", partialSum, globalSum, null, null,
            0, numPartials, 0);
    }

    public void MeshPoolSoftmaxNormalize(IGpuBuffer expValues, IGpuBuffer globalSum, IGpuBuffer softmaxScores,
        int numEdges)
    {
        RequireMeshBuffer(expValues, numEdges, nameof(expValues)); RequireMeshBuffer(globalSum, 1, nameof(globalSum));
        RequireMeshBuffer(softmaxScores, numEdges, nameof(softmaxScores));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.SoftmaxNormalize, expValues, globalSum, softmaxScores, null,
            numEdges, PtxMeshPoolF32Kernel.NumKept, PtxMeshPoolF32Kernel.Channels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_softmax_normalize", expValues, globalSum, softmaxScores, null,
            numEdges, 0, 0);
    }

    public void MeshPoolSoftmaxScores(IGpuBuffer scores, IGpuBuffer softmaxScores, float temperature, int numEdges)
    {
        RequireMeshBuffer(scores, numEdges, nameof(scores)); RequireMeshBuffer(softmaxScores, numEdges, nameof(softmaxScores));
        if (!(temperature > 0f) || float.IsNaN(temperature)) throw new ArgumentOutOfRangeException(nameof(temperature));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.SoftmaxScores, scores, softmaxScores, null, null,
            numEdges, PtxMeshPoolF32Kernel.NumKept, PtxMeshPoolF32Kernel.Channels, temperature)) return;
#endif
        if (numEdges > DefaultBlockSize)
            throw new ArgumentOutOfRangeException(nameof(numEdges), "The legacy CUDA fallback supports at most 256 edges; use the five-stage softmax surface.");
        LaunchMeshPoolFallback("mesh_pool_softmax_scores", scores, softmaxScores, null, null,
            numEdges, 0, 0, temperature);
    }

    public void MeshPoolWeightedGather(IGpuBuffer input, IGpuBuffer scores, IGpuBuffer keptIndices,
        IGpuBuffer output, int numKept, int numEdges, int inputChannels)
    {
        RequireMeshBuffer(input, checked((long)numEdges * inputChannels), nameof(input));
        RequireMeshBuffer(scores, numEdges, nameof(scores)); RequireMeshBuffer(keptIndices, numKept, nameof(keptIndices));
        RequireMeshBuffer(output, checked((long)numKept * inputChannels), nameof(output));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.WeightedGather, input, scores, keptIndices, output,
            numEdges, numKept, inputChannels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_weighted_gather", input, scores, keptIndices, output,
            numEdges, numKept, inputChannels);
    }

    public void MeshPoolWeightedBackward(IGpuBuffer gradOutput, IGpuBuffer scores, IGpuBuffer keptIndices,
        IGpuBuffer gradInput, int numKept, int numEdges, int inputChannels, bool deterministic)
    {
        RequireMeshBuffer(gradOutput, checked((long)numKept * inputChannels), nameof(gradOutput));
        RequireMeshBuffer(scores, numEdges, nameof(scores)); RequireMeshBuffer(keptIndices, numKept, nameof(keptIndices));
        RequireMeshBuffer(gradInput, checked((long)numEdges * inputChannels), nameof(gradInput));
        string name = deterministic ? "mesh_pool_weighted_backward_deterministic" : "mesh_pool_weighted_backward";
#if NET5_0_OR_GREATER
        DirectPtxMeshPoolOperation operation = deterministic
            ? DirectPtxMeshPoolOperation.WeightedBackwardDeterministic : DirectPtxMeshPoolOperation.WeightedBackwardAtomic;
        if (TryDirectPtxMeshPool(operation, gradOutput, scores, keptIndices, gradInput,
            numEdges, numKept, inputChannels)) return;
#endif
        LaunchMeshPoolFallback(name, gradOutput, scores, keptIndices, gradInput,
            numEdges, numKept, inputChannels);
    }

    public void MeshPoolScoresBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer keptIndices,
        IGpuBuffer gradScores, int numKept, int numEdges, int inputChannels)
    {
        RequireMeshBuffer(gradOutput, checked((long)numKept * inputChannels), nameof(gradOutput));
        RequireMeshBuffer(input, checked((long)numEdges * inputChannels), nameof(input));
        RequireMeshBuffer(keptIndices, numKept, nameof(keptIndices)); RequireMeshBuffer(gradScores, numEdges, nameof(gradScores));
#if NET5_0_OR_GREATER
        if (TryDirectPtxMeshPool(DirectPtxMeshPoolOperation.ScoresBackward, gradOutput, input, keptIndices, gradScores,
            numEdges, numKept, inputChannels)) return;
#endif
        LaunchMeshPoolFallback("mesh_pool_scores_backward", gradOutput, input, keptIndices, gradScores,
            numEdges, numKept, inputChannels);
    }

    private static void RequireMeshBuffer(IGpuBuffer buffer, long requiredElements, string parameter)
    {
        if (buffer is null) throw new ArgumentNullException(parameter);
        if (requiredElements <= 0) throw new ArgumentOutOfRangeException(parameter, "MeshPool extents must be positive.");
        if (buffer.Size < requiredElements)
            throw new ArgumentException($"MeshPool requires {requiredElements} elements; the buffer has {buffer.Size}.", parameter);
    }

    private unsafe void LaunchMeshPoolFallback(string kernelName, IGpuBuffer p0, IGpuBuffer? p1,
        IGpuBuffer? p2, IGpuBuffer? p3, int numEdges, int numKept, int channels, float temperature = 1f)
    {
        if (!_kernelCache.TryGetValue(kernelName, out IntPtr kernel))
            throw new InvalidOperationException($"CUDA kernel not found: {kernelName}");
        using var _ = PushContext();
        IntPtr a = p0.Handle, b = p1?.Handle ?? IntPtr.Zero, c = p2?.Handle ?? IntPtr.Zero, d = p3?.Handle ?? IntPtr.Zero;
        int size = checked(numEdges * channels);
        void** args = stackalloc void*[8];
        uint grid;
        switch (kernelName)
        {
            case "mesh_pool_compute_scores":
                args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &numEdges; args[4] = &channels;
                grid = checked((uint)((numEdges + (int)DefaultBlockSize - 1) / (int)DefaultBlockSize)); break;
            case "mesh_pool_gather": case "mesh_pool_backward":
                args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &numKept; args[4] = &numEdges; args[5] = &channels;
                grid = checked((uint)(((long)numKept * channels + DefaultBlockSize - 1) / DefaultBlockSize)); break;
            case "mesh_pool_backward_deterministic":
                args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &numKept; args[4] = &numEdges; args[5] = &channels;
                LaunchKernel2D(kernel, checked((uint)((channels + 15) / 16)), checked((uint)((numEdges + 15) / 16)), 16, 16, args); return;
            case "mesh_pool_importance_backward": case "mesh_pool_weighted_gather":
            case "mesh_pool_weighted_backward": case "mesh_pool_scores_backward":
                args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &d; args[4] = &numKept; args[5] = &numEdges; args[6] = &channels;
                long work = kernelName == "mesh_pool_importance_backward" ? channels :
                    kernelName == "mesh_pool_scores_backward" ? numEdges : (long)numKept * channels;
                grid = checked((uint)((work + DefaultBlockSize - 1) / DefaultBlockSize)); break;
            case "mesh_pool_weighted_backward_deterministic":
                args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &d; args[4] = &numKept; args[5] = &numEdges; args[6] = &channels;
                LaunchKernel2D(kernel, checked((uint)((channels + 15) / 16)), checked((uint)((numEdges + 15) / 16)), 16, 16, args); return;
            case "mesh_pool_zero_grad":
                args[0] = &a; args[1] = &size; grid = checked((uint)((size + (int)DefaultBlockSize - 1) / (int)DefaultBlockSize)); break;
            case "mesh_pool_softmax_find_max":
                args[0] = &a; args[1] = &b; args[2] = &numEdges; grid = checked((uint)numKept);
                LaunchKernelWithSharedMem(kernel, grid, DefaultBlockSize, DefaultBlockSize * sizeof(float), args); return;
            case "mesh_pool_softmax_final_max": case "mesh_pool_softmax_final_sum":
                args[0] = &a; args[1] = &b; args[2] = &numKept;
                LaunchKernelWithSharedMem(kernel, 1, DefaultBlockSize, DefaultBlockSize * sizeof(float), args); return;
            case "mesh_pool_softmax_exp_sum":
                args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &d; args[4] = &temperature; args[5] = &numEdges;
                grid = checked((uint)numKept); LaunchKernelWithSharedMem(kernel, grid, DefaultBlockSize, DefaultBlockSize * sizeof(float), args); return;
            case "mesh_pool_softmax_normalize":
                args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &numEdges;
                grid = checked((uint)((numEdges + (int)DefaultBlockSize - 1) / (int)DefaultBlockSize)); break;
            case "mesh_pool_softmax_scores":
                args[0] = &a; args[1] = &b; args[2] = &temperature; args[3] = &numEdges;
                LaunchKernelWithSharedMem(kernel, 1, DefaultBlockSize, DefaultBlockSize * sizeof(float), args); return;
            default: throw new ArgumentOutOfRangeException(nameof(kernelName));
        }
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
    }
}
