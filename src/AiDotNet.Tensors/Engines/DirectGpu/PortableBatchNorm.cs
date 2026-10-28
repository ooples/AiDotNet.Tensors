// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The register-resident training BatchNorm (<see cref="IBatchNormVarianceBackend"/>) on the backends that run
/// <see cref="BatchNormRegisterKernelTemplate"/>'s kernels: the host side shared by HIP, Metal, OpenCL, Vulkan and WebGPU.
/// It packs the per-channel parameters into one buffer per direction (two copies of C floats each, so the backward binds
/// eight storage buffers), launches one 256-wide group per channel, and unpacks the statistics or parameter gradients.
/// </summary>
internal static class PortableBatchNorm
{
    /// <summary>One launch: the kernel's buffers in binding order and its ints, over <paramref name="groups"/> groups of
    /// <see cref="BatchNormRegisterKernelTemplate.GroupSize"/>.</summary>
    internal delegate void Launcher(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints);

    // Every API's guaranteed 1-D grid; past it (more channels) the caller's composed path runs.
    private const int MaxGroups = 65535;

    private static int[] Ints(int batch, int channels, int spatial, float epsilon, long count, int mode) => new[]
    {
        batch, channels, spatial, SingleToInt32BitsCompat(epsilon),
        SingleToInt32BitsCompat((float)(1.0 / count)), mode,
    };

#if !NETCOREAPP
    // BitConverter.SingleToInt32Bits is unavailable on net471; reinterpret the bits via a pointer there.
    private static unsafe int SingleToInt32BitsCompat(float value) => *(int*)&value;
#else
    private static int SingleToInt32BitsCompat(float value) => BitConverter.SingleToInt32Bits(value);
#endif

    /// <summary>As <see cref="IBatchNormVarianceBackend.TryBatchNormTrainingWithVariance"/>.</summary>
    internal static bool TryForward(IDirectGpuBackend backend, Launcher launch, IGpuBuffer input, IGpuBuffer output,
        IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer saveMean, IGpuBuffer saveVariance,
        int batch, int channels, int spatial, float epsilon, bool relu, IGpuBuffer? residual)
    {
        long count = (long)batch * spatial, n = count * channels;
        int k = BatchNormRegisterKernelTemplate.VariantFor(count);
        if (k == 0 || batch <= 0 || spatial <= 0 || channels <= 0 || channels > MaxGroups || n > int.MaxValue
            || input.Size < n || output.Size < n || gamma.Size < channels || beta.Size < channels
            || saveMean.Size < channels || saveVariance.Size < channels
            || (residual is not null && (!relu || residual.Size < n)))
            return false;
        using var prm = backend.AllocateBuffer(2 * channels);
        using var stats = backend.AllocateBuffer(2 * channels);
        backend.Copy(gamma, 0, prm, 0, channels);
        backend.Copy(beta, 0, prm, channels, channels);
        int mode = (relu ? 1 : 0) | (residual is not null ? 2 : 0);
        launch(BatchNormRegisterKernelTemplate.ForwardName(k), channels,
            new[] { input, residual ?? input, output, prm, stats }, Ints(batch, channels, spatial, epsilon, count, mode));
        backend.Copy(stats, 0, saveMean, 0, channels);
        backend.Copy(stats, channels, saveVariance, 0, channels);
        return true;
    }

    /// <summary>As <see cref="IBatchNormVarianceBackend.TryBatchNormBackwardFromVariance"/>.</summary>
    internal static bool TryBackward(IDirectGpuBackend backend, Launcher launch, IGpuBuffer gradOutput, IGpuBuffer input,
        IGpuBuffer gamma, IGpuBuffer saveMean, IGpuBuffer saveVariance, IGpuBuffer gradInput, IGpuBuffer gradGamma,
        IGpuBuffer gradBeta, int batch, int channels, int spatial, float epsilon, IGpuBuffer? reluOutput,
        IGpuBuffer? gradResidual, IGpuBuffer? beta, IGpuBuffer? gradOutput2, bool canRebuildMask = true)
    {
        long count = (long)batch * spatial, n = count * channels;
        int k = BatchNormRegisterKernelTemplate.VariantFor(count);
        if (k == 0 || batch <= 0 || spatial <= 0 || channels <= 0 || channels > MaxGroups || n > int.MaxValue
            || gradOutput.Size < n || input.Size < n || gradInput.Size < n || gamma.Size < channels
            || saveMean.Size < channels || saveVariance.Size < channels || gradGamma.Size < channels
            || gradBeta.Size < channels || (reluOutput is not null && reluOutput.Size < n)
            || (gradResidual is not null && (reluOutput is null || gradResidual.Size < n))
            || (gradOutput2 is not null && gradOutput2.Size < n) || (beta is not null && beta.Size < channels))
            return false;
        // The mask is rebuilt from x (one activation read fewer) when beta is supplied for a ReLU without a residual, on a
        // backend whose fma rounds once (canRebuildMask; see BatchNormRegisterKernelTemplate), else the ReLU output is read.
        bool rebuildMask = canRebuildMask && beta is not null && reluOutput is not null && gradResidual is null;
        using var prm = backend.AllocateBuffer(4 * channels);
        using var dprm = backend.AllocateBuffer(2 * channels);
        using var noResidual = gradResidual is null ? backend.AllocateBuffer(1) : null;
        backend.Copy(gamma, 0, prm, 0, channels);
        if (beta is not null) backend.Copy(beta, 0, prm, channels, channels);
        backend.Copy(saveMean, 0, prm, 2 * channels, channels);
        backend.Copy(saveVariance, 0, prm, 3 * channels, channels);
        int mode = (reluOutput is not null ? 1 : 0) | (gradResidual is not null ? 2 : 0) | (rebuildMask ? 4 : 0)
                   | (gradOutput2 is not null ? 8 : 0);
        var mask = rebuildMask ? input : reluOutput ?? input;
        launch(BatchNormRegisterKernelTemplate.BackwardName(k), channels,
            new[] { gradOutput, mask, gradResidual ?? noResidual!, input, prm, gradInput, dprm, gradOutput2 ?? gradOutput },
            Ints(batch, channels, spatial, epsilon, count, mode));
        backend.Copy(dprm, 0, gradGamma, 0, channels);
        backend.Copy(dprm, channels, gradBeta, 0, channels);
        return true;
    }
}
