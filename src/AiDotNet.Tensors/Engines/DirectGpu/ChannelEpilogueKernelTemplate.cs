// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The per-channel convolution epilogue of IChannelEpilogueBackend (CUDA's channel_epilogue_fwd / _bwd in
// CudaActivationKernels), written once in WinogradF23KernelTemplate's body conventions and emitted as OpenCL C, HIP C++,
// Vulkan GLSL and Metal Shading Language. WGSL keeps its own copy (WebGpu/WebGpuChannelEpilogueKernels.cs).
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <c>y = act(((z + bias[c]) * scale[c]) + shift[c])</c> and <c>dz = act'(y) * dy * scale[c]</c>, one thread per element
/// over a 1-D grid of 256-wide groups. The mode word: bit 0 bias, bit 1 scale, bit 2 shift (forward; the backward reads
/// bit 1 only), bits 3-4 the activation (<see cref="ChannelEpilogueActivation"/>); the LeakyReLU slope rides as float bits.
/// An absent operand's binding is a placeholder the mode never reads. The element offset lets the host run a tensor past
/// one launch's grid in chunks (GLSL has no pointer arithmetic).
/// </summary>
/// <remarks>
/// Each step is its own rounded operation, as the separate passes compute it: <c>PRECISE</c> keeps GLSL from contracting
/// the multiply and add into a fused multiply-add. The activations are the relu, leaky_relu and leaky_relu_backward
/// kernels' own expressions.
/// </remarks>
internal static class ChannelEpilogueKernelTemplate
{
    internal const string Forward = "channel_epilogue_fwd";
    internal const string Backward = "channel_epilogue_bwd";

    /// <summary>Every kernel's group size.</summary>
    internal const int GroupSize = 256;

    private const string ForwardBody = @"
    int i = offset + gid;
    if (gid < count)
    {
        int c = (i / spatial) % channels;
        PRECISE float v = zin[i];
        if ((mode & 1) != 0) v = v + bia[c];
        if ((mode & 2) != 0) v = v * scl[c];
        if ((mode & 4) != 0) v = v + shf[c];
        int act = (mode >> 3) & 3;
        if (act == 1) v = v > 0.0f ? v : (v != v ? v : 0.0f);
        else if (act == 2) v = v > 0.0f ? v : IBITS(slopeBits) * v;
        yout[i] = v;
    }
";

    private const string BackwardBody = @"
    int i = offset + gid;
    if (gid < count)
    {
        int c = (i / spatial) % channels;
        PRECISE float g = dyin[i];
        int act = (mode >> 3) & 3;
        if (act == 1) g = yin[i] > 0.0f ? g : 0.0f;
        else if (act == 2) g = yin[i] > 0.0f ? g : IBITS(slopeBits) * g;
        if ((mode & 2) != 0) g = g * scl[c];
        dzout[i] = g;
    }
";

    internal static readonly string[] Ints = { "count", "channels", "spatial", "mode", "slopeBits", "offset" };

    /// <summary>The kernels, in this template's conventions (binding 1 written, every other binding read).</summary>
    internal static readonly WinogradF23KernelTemplate.Spec[] Kernels =
    {
        new(Forward, new[] { "zin", "yout", "bia", "scl", "shf" }, Ints, ForwardBody),
        new(Backward, new[] { "dyin", "dzout", "yin", "scl" }, Ints, BackwardBody),
    };

    internal static string[] GetKernelNames() => new[] { Forward, Backward };

    /// <summary>The kernels in one OpenCL C, HIP C++ or Metal Shading Language module.</summary>
    internal static string GetModuleSource(WinogradKernelDialect dialect)
        => WinogradF23KernelTemplate.EmitModule(Kernels, dialect);

    /// <summary>One kernel as a Vulkan compute shader (bindings in buffer order, the ints as push constants).</summary>
    internal static string GetGlslSource(string name)
    {
        foreach (var spec in Kernels)
            if (spec.Name == name) return WinogradF23KernelTemplate.EmitGlsl(spec);
        throw new KeyNotFoundException(name);
    }
}

/// <summary>
/// The host side of <see cref="IChannelEpilogueBackend"/> on HIP, Metal, OpenCL, Vulkan and WebGPU: the shape checks,
/// placeholders for absent operands, and the launches in chunks of at most <see cref="MaxGroups"/> groups.
/// </summary>
internal static class PortableChannelEpilogue
{
    /// <summary>One launch: the kernel's buffers in binding order and its ints, over <paramref name="groups"/> groups of
    /// <see cref="ChannelEpilogueKernelTemplate.GroupSize"/>.</summary>
    internal delegate void Launcher(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints);

    // Every API's guaranteed 1-D grid (WebGPU's 65535 workgroups per dimension).
    private const int MaxGroups = 65535;

    private const int G = ChannelEpilogueKernelTemplate.GroupSize;

#if !NETCOREAPP
    private static unsafe int Bits(float value) => *(int*)&value;
#else
    private static int Bits(float value) => System.BitConverter.SingleToInt32Bits(value);
#endif

    private static bool Shape(int batch, int channels, int spatial, out long n)
    {
        n = (long)batch * channels * spatial;
        return batch > 0 && channels > 0 && spatial > 0 && n <= int.MaxValue;
    }

    private static void Run(Launcher launch, string name, IGpuBuffer[] buffers, int n, int channels, int spatial, int mode,
        float slope)
    {
        const int chunk = MaxGroups * G;
        for (int offset = 0; offset < n; offset += chunk)
        {
            int count = System.Math.Min(chunk, n - offset);
            launch(name, (count + G - 1) / G, buffers,
                new[] { count, channels, spatial, mode, Bits(slope), offset });
        }
    }

    /// <summary>As <see cref="IChannelEpilogueBackend.TryChannelEpilogue"/>.</summary>
    internal static bool TryForward(IDirectGpuBackend backend, Launcher launch, IGpuBuffer input, IGpuBuffer? bias,
        IGpuBuffer? scale, IGpuBuffer? shift, IGpuBuffer output, int batch, int channels, int spatial, int activation,
        float slope)
    {
        if (!ChannelEpilogueActivation.Accepts(activation, slope) || !Shape(batch, channels, spatial, out long n)
            || input.Size < n || output.Size < n || (bias is not null && bias.Size < channels)
            || (scale is not null && scale.Size < channels) || (shift is not null && shift.Size < channels))
            return false;
        int mode = (bias is not null ? 1 : 0) | (scale is not null ? 2 : 0) | (shift is not null ? 4 : 0) | (activation << 3);
        IGpuBuffer? placeholder = bias is null || scale is null || shift is null ? backend.AllocateBuffer(1) : null;
        using (placeholder)
        {
            Run(launch, ChannelEpilogueKernelTemplate.Forward,
                new[] { input, output, bias ?? placeholder!, scale ?? placeholder!, shift ?? placeholder! }, (int)n, channels,
                spatial, mode, activation == ChannelEpilogueActivation.LeakyRelu ? slope : 0f);
        }
        return true;
    }

    /// <summary>As <see cref="IChannelEpilogueBackend.TryChannelEpilogueBackward"/>.</summary>
    internal static bool TryBackward(IDirectGpuBackend backend, Launcher launch, IGpuBuffer gradOutput, IGpuBuffer output,
        IGpuBuffer? scale, IGpuBuffer gradInput, int batch, int channels, int spatial, int activation, float slope)
    {
        if (!ChannelEpilogueActivation.Accepts(activation, slope) || !Shape(batch, channels, spatial, out long n)
            || gradOutput.Size < n || output.Size < n || gradInput.Size < n || (scale is not null && scale.Size < channels))
            return false;
        int mode = (scale is not null ? 2 : 0) | (activation << 3);
        IGpuBuffer? placeholder = scale is null ? backend.AllocateBuffer(1) : null;
        using (placeholder)
        {
            Run(launch, ChannelEpilogueKernelTemplate.Backward, new[] { gradOutput, gradInput, output, scale ?? placeholder! },
                (int)n, channels, spatial, mode, activation == ChannelEpilogueActivation.LeakyRelu ? slope : 0f);
        }
        return true;
    }
}
