using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A fused conv2d + bias + ReLU recorded on the GPU engine leaves its output (and the output gradient) on the device.
/// When the backward then runs on a CPU engine (the tape's engine is the current one when gradients are computed),
/// the one-pass CPU epilogue backward read those tensors' host arrays, found none and threw "needs contiguous CPU
/// tensors". It must fall back to the per-op backward, which reads them through the ordinary accessors, and give the
/// gradients the CPU engine gives for the same values.
/// </summary>
[Collection("DirectGpuSerial")]
public class FusedConv2DDeviceOutputBackwardTests
{
    private static Tensor<float> Values(int[] shape, int seed, float scale)
    {
        var t = new Tensor<float>(shape);
        var rng = new Random(seed);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, (float)(rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static (float[] Input, float[] Kernel, float[] Bias) Grads(IEngine forward, IEngine backward)
    {
        var input = Values(new[] { 1, 2, 5, 5 }, 17, 1f);
        var kernel = Values(new[] { 3, 2, 3, 3 }, 18, 1f);
        var bias = Values(new[] { 3 }, 19, 0.2f);
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = forward;
            using var tape = new GradientTape<float>();
            var output = forward.FusedConv2D(input, kernel, bias, 1, 1, 0, 0, 1, 1, FusedActivationType.ReLU);
            var loss = forward.ReduceSum(forward.TensorMultiply(output, output), null);
            AiDotNetEngine.Current = backward;
            var grads = tape.ComputeGradients(loss, sources: new[] { input, kernel, bias });
            return (grads[input].ToArray(), grads[kernel].ToArray(), grads[bias].ToArray());
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [SkippableFact]
    public void A_device_output_backpropagates_on_a_cpu_engine_as_on_the_cpu()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No GPU available");
            var cpu = new CpuEngine();
            var expected = Grads(cpu, cpu);
            var got = Grads(gpu, cpu);
            void Near(float[] e, float[] g, string what)
            {
                Assert.Equal(e.Length, g.Length);
                for (int i = 0; i < e.Length; i++)
                    Assert.True(Math.Abs(e[i] - g[i]) <= 1e-4f * (1 + Math.Abs(e[i])), $"{what}[{i}]: {g[i]} vs {e[i]}");
            }
            Near(expected.Input, got.Input, "dInput");
            Near(expected.Kernel, got.Kernel, "dKernel");
            Near(expected.Bias, got.Bias, "dBias");
        }
    }

    /// <summary>
    /// The same backward called directly with a CPU engine on a device-resident output and output gradient (GPU op
    /// results whose host copy was never read): the one-pass CPU epilogue found no host array and threw "needs
    /// contiguous CPU tensors". A test that captured the process-wide engine while another test had a GPU engine
    /// current, and ran its backward after that engine closed, hit exactly this in the CUDA lane.
    /// </summary>
    [SkippableFact]
    public void The_cpu_epilogue_backward_reads_device_resident_tensors()
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { Skip.If(true, "No GPU backend"); return; }
        using (gpu)
        {
            Skip.IfNot(gpu.IsGpuAvailable, "No GPU available");
            var input = Values(new[] { 1, 2, 5, 5 }, 17, 1f);
            var kernel = Values(new[] { 3, 2, 3, 3 }, 18, 1f);
            var bias = Values(new[] { 3 }, 19, 0.2f);
            var cpu = new CpuEngine();
            var hostOutput = cpu.FusedConv2D(input, kernel, bias, 1, 1, 0, 0, 1, 1, FusedActivationType.ReLU);
            var hostGrad = Values(hostOutput.Shape.ToArray(), 20, 1f);
            var zeros = new Tensor<float>(hostOutput.Shape.ToArray());
            // Device-resident copies: results of a GPU op, never read on the host.
            var deviceOutput = gpu.TensorAdd(hostOutput, zeros);
            var deviceGrad = gpu.TensorAdd(hostGrad, zeros);
            Skip.If(deviceOutput.GetCpuBackingForStridedRead(out _) is not null && deviceGrad.GetCpuBackingForStridedRead(out _) is not null,
                "The GPU engine downloaded its results; nothing device-resident to test.");

            object[] saved = { new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }, true };
            var expected = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>();
            var got = new System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>>();
            var inputs = new[] { input, kernel, bias };
            BackwardFunctions<float>.FusedConv2DBiasActivationBackward(hostGrad, inputs, hostOutput, saved, cpu, expected);
            BackwardFunctions<float>.FusedConv2DBiasActivationBackward(deviceGrad, inputs, deviceOutput, saved, cpu, got);
            foreach (var t in inputs)
            {
                var e = expected[t].ToArray(); var g = got[t].ToArray();
                for (int i = 0; i < e.Length; i++)
                    Assert.True(Math.Abs(e[i] - g[i]) <= 1e-4f * (1 + Math.Abs(e[i])), $"[{i}]: {g[i]} vs {e[i]}");
            }
        }
    }
}
