using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The tape's backward engine defaulted to AiDotNetEngine.Current at tape creation and, once it saw GPU-resident
/// data, assumed that default was the engine that produced it. A forward on a specific GPU engine while an earlier
/// caller had left a CpuEngine global therefore ran its backward on the host - measured in the full suite as
/// "dA was computed on the host" (MatMulTransposedTapeGpuTests) and "dQ was computed on the host" (GlaScanTapeGpuTests),
/// both passing in isolation. The backward now runs on the engine that produced the data.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class GradientTapeFollowsDataEngineTests
{
    [SkippableFact]
    public void The_backward_runs_on_the_gpu_engine_that_produced_the_data_whatever_engine_is_global()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception) { }
        Skip.IfNot(gpu is not null && gpu.IsGpuAvailable, "No GPU device.");
        var prior = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();   // what an earlier caller left global
            var rng = new Random(9);
            Tensor<float> R(params int[] s)
            {
                var t = new Tensor<float>(s);
                for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
                return t;
            }
            // Above the CPU fast path's work threshold, so a host backward would take it.
            var a = R(512, 256);
            var b = R(1024, 256);
            var r = R(512, 1024);
            using var tape = new GradientTape<float>();
            var loss = gpu!.ReduceSum(gpu.TensorMultiply(gpu.TensorMatMulTransposed(a, b), r), [0, 1], keepDims: false);
            var grads = tape.ComputeGradients(loss, [a, b]);
            foreach (var (t, name) in new[] { (a, "dA"), (b, "dB") })
                Assert.True(grads[t].IsGpuResident || grads[t].HasPendingGpuData, $"{name} was computed on the host");
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            gpu?.Dispose();
        }
    }
}
