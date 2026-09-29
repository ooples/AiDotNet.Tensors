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
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
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

    /// <summary>
    /// A persistent tape records again after each backward. It used to resolve its data engine once, on the first GPU
    /// tensor, and keep it through Reset: after that engine was disposed, the next step's backward still ran on it.
    /// </summary>
    [SkippableFact]
    public void A_persistent_tape_resolves_its_engine_per_recording_and_never_uses_a_disposed_one()
    {
        AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine? direct = null;
        try { direct = new AiDotNet.Tensors.Engines.DirectGpu.DirectGpuEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.IfNot(direct is not null && direct.IsAvailable, "No GPU device.");
        var prior = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var older = new DirectGpuTensorEngine(direct!);
            var newer = new DirectGpuTensorEngine(direct!);
            var rng = new Random(3);
            Tensor<float> R(params int[] s)
            {
                var t = new Tensor<float>(s);
                for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
                return t;
            }
            var a = R(256, 128);
            var b = R(128, 64);
            using var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true });

            var loss = newer.ReduceSum(newer.TensorMatMul(a, b), [0, 1], keepDims: false);
            Assert.Equal(2, tape.ComputeGradients(loss, [a, b]).Count);
            Assert.Same(newer, tape.Engine);

            newer.Dispose();
            tape.Reset();
            loss = older.ReduceSum(older.TensorMatMul(a, b), [0, 1], keepDims: false);
            var grads = tape.ComputeGradients(loss, [a, b]);
            Assert.False(tape.Engine is DirectGpuTensorEngine { IsDisposed: true }, "the backward ran on a disposed engine");
            Assert.Same(older, tape.Engine);
            // The backward chain a persistent tape caches belonged to the recording Reset dropped: replaying it for the
            // new loss returned no gradients at all.
            Assert.True(grads.ContainsKey(a) && grads.ContainsKey(b), $"the second recording produced {grads.Count} gradients");
            Assert.True(grads[a].IsGpuResident || grads[a].HasPendingGpuData, "dA was computed on the host");
            older.Dispose();
        }
        finally
        {
            AiDotNetEngine.Current = prior;
            direct?.Dispose();
        }
    }
}
