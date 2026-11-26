#if NET6_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// LogSumExp (forward, and backward under a tape) and in-place addition of device tensors on the GPU engine: each must
/// match the CPU engine and make no host-device transfer. A recording tape used to send LogSumExp to the CPU
/// composition, so a [rows x vocabulary] input and all of its gradients were downloaded, and every gradient
/// accumulation into it went through a download, a host copy and a fresh device buffer.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class LogSumExpAndInPlaceAddResidencyTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public LogSumExpAndInPlaceAddResidencyTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Rand(int[] shape, int seed, double scale = 4.0)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, string name)
    {
        Assert.Equal(expected._shape, actual._shape);
        var e = expected.ToArray();
        var a = actual.ToArray();
        double scale = Math.Max(e.Max(v => Math.Abs(v)), 1e-6);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= 1e-4 * scale, $"{name}[{i}]: cpu {e[i]}, gpu {a[i]} (scale {scale})");
    }

    private static string Transfers(GpuResidencyScope scope) =>
        string.Join(", ", scope.Events.Select(e => $"{e.Kind} {e.Bytes} B {e.Operation}"));

    private static void AssertNoTransfers(GpuResidencyScope scope, string what) =>
        Assert.True(scope.Uploads + scope.Downloads == 0, $"{what} crossed the host-device boundary: {Transfers(scope)}");

    private DirectGpuTensorEngine Gpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        return _fixture.Engine ?? throw new InvalidOperationException("The GPU fixture reported a device but has no engine.");
    }

    [SkippableTheory]
    [InlineData(new[] { 4, 33 }, 1, false)]
    [InlineData(new[] { 4, 33 }, -1, true)]
    [InlineData(new[] { 3, 5, 7 }, 1, false)]      // interior axis
    [InlineData(new[] { 3, 5, 7 }, 0, true)]
    [InlineData(new[] { 64, 4099 }, 1, false)]     // a vocabulary-sized row
    public void LogSumExp_OfADeviceTensor_StaysOnTheDevice_AndMatchesTheCpu(int[] shape, int axis, bool keepDims)
    {
        var gpu = Gpu();
        var x = Rand(shape, 1);
        var expected = new CpuEngine().TensorLogSumExp(x, axis, keepDims);
        var dx = gpu.UploadToGpu(x, GpuTensorRole.General);
        _ = gpu.TensorLogSumExp(dx, axis, keepDims).ToArray();   // warm-up: kernel compilation and pooled buffers

        Tensor<float> actual;
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
            actual = gpu.TensorLogSumExp(dx, axis, keepDims);
        AssertNoTransfers(scope, "LogSumExp");
        Assert.True(actual.IsGpuResident || actual.HasPendingGpuData, "LogSumExp returned a host tensor");
        AssertClose(expected, actual, "lse");
    }

    // loss = sum(LSE(z, axis) * r) + sum(z * q): the LogSumExp backward runs on the device, and its gradient and the
    // product's are accumulated into dz on the device.
    private static (Tensor<float> Loss, Tensor<float> Dz) Taped(IEngine engine, Tensor<float> z, Tensor<float> r,
        Tensor<float> q, int axis)
    {
        using var tape = new GradientTape<float>();
        var lse = engine.TensorLogSumExp(z, axis, keepDims: false);
        var a = engine.ReduceSum(engine.TensorMultiply(lse, r), Enumerable.Range(0, lse.Rank).ToArray(), keepDims: false);
        var b = engine.ReduceSum(engine.TensorMultiply(z, q), Enumerable.Range(0, z.Rank).ToArray(), keepDims: false);
        var loss = engine.TensorAdd(a, b);
        var grads = tape.ComputeGradients(loss, new[] { z });
        return (loss, grads[z]);
    }

    [SkippableTheory]
    [InlineData(new[] { 4, 33 }, 1)]
    [InlineData(new[] { 3, 5, 7 }, 1)]
    [InlineData(new[] { 64, 4099 }, 1)]
    public void TapedLogSumExp_ForwardBackwardAndAccumulation_StayOnTheDevice_AndMatchTheCpu(int[] shape, int axis)
    {
        var gpu = Gpu();
        var z = Rand(shape, 2);
        var q = Rand(shape, 3, 1.0);
        var reducedShape = shape.Where((_, i) => i != axis).ToArray();
        var r = Rand(reducedShape, 4, 1.0);
        var expected = Taped(new CpuEngine(), z, r, q, axis);

        var dz = gpu.UploadToGpu(z, GpuTensorRole.General);
        var dq = gpu.UploadToGpu(q, GpuTensorRole.General);
        var dr = gpu.UploadToGpu(r, GpuTensorRole.General);
        _ = Taped(gpu, dz, dr, dq, axis).Dz.ToArray();   // warm-up

        (Tensor<float> Loss, Tensor<float> Dz) actual;
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
            actual = Taped(gpu, dz, dr, dq, axis);
        Assert.True(scope.Downloads == 0, $"the taped LogSumExp downloaded: {Transfers(scope)}");
        Assert.True(actual.Dz.IsGpuResident || actual.Dz.HasPendingGpuData, "dz was computed on the host");
        AssertClose(expected.Loss, actual.Loss, "loss");
        AssertClose(expected.Dz, actual.Dz, "dz");
    }

    // TakeAlongDim alone (ce: false), and the per-token cross-entropy LSE(z) - z[ids] (ce: true), whose two gradients
    // into z, the scatter-add and the LogSumExp backward, are accumulated on the device.
    private static (Tensor<float> Loss, Tensor<float> Dz) TapedGather(IEngine engine, Tensor<float> z, Tensor<int> ids,
        Tensor<float> r, bool ce)
    {
        using var tape = new GradientTape<float>();
        int rows = z._shape[0];
        var picked = engine.Reshape(engine.TensorTakeAlongDim(z, ids, 1), new[] { rows });
        var perToken = ce ? engine.TensorSubtract(engine.TensorLogSumExp(z, 1, keepDims: false), picked) : picked;
        var loss = engine.ReduceSum(engine.TensorMultiply(perToken, r), new[] { 0 }, keepDims: false);
        var grads = tape.ComputeGradients(loss, new[] { z });
        return (loss, grads[z]);
    }

    [SkippableTheory]
    [InlineData(8, 33, false)]
    [InlineData(8, 33, true)]
    [InlineData(64, 4099, true)]
    public void TapedTakeAlongDim_GradientStaysOnTheDevice_AndMatchesTheCpu(int rows, int vocab, bool ce)
    {
        var gpu = Gpu();
        var z = Rand(new[] { rows, vocab }, 7);
        var r = Rand(new[] { rows }, 8, 1.0);
        var ids = new Tensor<int>(new[] { rows, 1 });
        var rng = new Random(9);
        for (int i = 0; i < rows; i++) ids[i, 0] = i % 4 == 3 ? ids[i - 1, 0] : rng.Next(vocab);   // repeated targets too
        var expected = TapedGather(new CpuEngine(), z, ids, r, ce);

        var dz = gpu.UploadToGpu(z, GpuTensorRole.General);
        var dr = gpu.UploadToGpu(r, GpuTensorRole.General);
        _ = TapedGather(gpu, dz, ids, dr, ce).Dz.ToArray();   // warm-up

        (Tensor<float> Loss, Tensor<float> Dz) actual;
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
            actual = TapedGather(gpu, dz, ids, dr, ce);
        // The host index tensor may be uploaded; nothing may come back.
        Assert.True(scope.Downloads == 0, $"the taped gather downloaded: {Transfers(scope)}");
        Assert.True(actual.Dz.IsGpuResident || actual.Dz.HasPendingGpuData, "dz was computed on the host");
        AssertClose(expected.Loss, actual.Loss, "loss");
        AssertClose(expected.Dz, actual.Dz, "dz");
    }

    // The gated linear units under a tape: the kernel's result carries the CPU op's node, and the backward runs on the
    // device. A recording tape used to send the forward to the CPU op, which downloaded the input.
    [SkippableTheory]
    [InlineData("GLU")]
    [InlineData("GeGLU")]
    [InlineData("ReGLU")]
    [InlineData("SwiGLU")]
    public void TapedGatedLinearUnit_StaysOnTheDevice_AndMatchesTheCpu(string op)
    {
        var gpu = Gpu();
        int[] shape = { 6, 64 };
        var x = Rand(shape, 11, 2.0);
        var r = Rand(new[] { 6, 32 }, 12, 1.0);
        (Tensor<float> Y, Tensor<float> Dx) Run(IEngine e, Tensor<float> xi, Tensor<float> ri)
        {
            using var tape = new GradientTape<float>();
            // A computed input: on the device its host copy is stale, so a host fallback has to download it (a freshly
            // uploaded input still has a current host copy and hid the fallback).
            var h = e.TensorAdd(xi, xi);
            var y = op switch
            {
                "GLU" => e.GLU(h, -1),
                "GeGLU" => e.GeGLU(h, -1),
                "ReGLU" => e.ReGLU(h, -1),
                _ => e.SwiGLU(h, -1),
            };
            var loss = e.ReduceSum(e.TensorMultiply(y, ri), new[] { 0, 1 }, keepDims: false);
            return (y, tape.ComputeGradients(loss, new[] { xi })[xi]);
        }
        var expected = Run(new CpuEngine(), x, r);
        var dx = gpu.UploadToGpu(x, GpuTensorRole.General);
        var dr = gpu.UploadToGpu(r, GpuTensorRole.General);
        _ = Run(gpu, dx, dr).Dx.ToArray();   // warm-up
        (Tensor<float> Y, Tensor<float> Dx) actual;
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
            actual = Run(gpu, dx, dr);
        Assert.True(scope.Downloads == 0, $"the taped {op} downloaded: {Transfers(scope)}");
        Assert.True(actual.Dx.IsGpuResident || actual.Dx.HasPendingGpuData, "dx was computed on the host");
        AssertClose(expected.Y, actual.Y, "y");
        AssertClose(expected.Dx, actual.Dx, "dx");
    }

    public static IEnumerable<object[]> InPlaceOperands() => new[]
    {
        new object[] { "uploaded", "uploaded" },
        new object[] { "computed", "computed" },
        new object[] { "uploaded", "computed" },
        new object[] { "computed", "uploaded" },
    };

    [SkippableTheory]
    [MemberData(nameof(InPlaceOperands))]
    public void TensorAddInPlace_OfDeviceTensors_StaysOnTheDevice(string aKind, string bKind)
    {
        var gpu = Gpu();
        int[] shape = { 37, 129 };
        var x = Rand(shape, 5);
        var y = Rand(shape, 6);
        var ones = new Tensor<float>(shape);
        ones.Fill(1f);
        var dOnes = gpu.UploadToGpu(ones, GpuTensorRole.General);
        Tensor<float> Make(Tensor<float> host, string kind)
        {
            var up = gpu.UploadToGpu(host, GpuTensorRole.General);
            // A device op's result: a pending device buffer with no host copy.
            return kind == "computed" ? gpu.TensorMultiply(up, dOnes) : up;
        }

        var warmA = Make(x, aKind);
        gpu.TensorAddInPlace(warmA, Make(y, bKind));
        _ = warmA.ToArray();

        var a = Make(x, aKind);
        var b = Make(y, bKind);
        GpuResidencyScope scope;
        using (scope = GpuResidencyScope.Begin(captureOperations: true))
            gpu.TensorAddInPlace(a, b);
        AssertNoTransfers(scope, "TensorAddInPlace");
        AssertClose(new CpuEngine().TensorAdd(x, y), a, "a + b");
    }
}
#endif
