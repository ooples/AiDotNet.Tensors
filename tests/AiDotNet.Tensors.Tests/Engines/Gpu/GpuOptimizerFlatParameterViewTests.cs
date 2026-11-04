// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Model parameters are views (offset into one flat ParameterBuffer). The fused GPU AdamW refused every view, so
/// eager training ran op-by-op AdamW on the host copies (measured: 25 of 26 parameters of an LM, every step). The
/// flat array now lives on the device as one buffer and each parameter is updated through a view: training through
/// views must match training standalone copies, move nothing to the host, and leave the parameters device-resident.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class GpuOptimizerFlatParameterViewTests : IClassFixture<DirectGpuTensorEngineTestFixture>, IDisposable
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public GpuOptimizerFlatParameterViewTests(DirectGpuTensorEngineTestFixture fixture)
    {
        _fixture = fixture;
        if (fixture.IsAvailable) AiDotNetEngine.Current = fixture.Engine!;
    }

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    // loss = sum(tanh(x·w1)·w2)
    private static void Train(IEngine e, Tensor<float>[] ps, Tensor<float>[] m, Tensor<float>[] v, Tensor<float> x, int steps)
    {
        for (int s = 0; s < steps; s++)
        {
            using var tape = new GradientTape<float>();
            var h = e.TensorTanh(e.TensorMatMul(x, ps[0]));
            var y = e.TensorMatMul(h, ps[1]);
            var loss = e.ReduceSum(y, new[] { 0, 1 }, keepDims: false);
            var grads = tape.ComputeGradients(loss, ps);
            for (int i = 0; i < ps.Length; i++)
                Assert.True(GpuOptimizer.TryAdamWStep(ps[i], grads[ps[i]], m[i], v[i], 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, s + 1),
                    $"the fused GPU AdamW declined parameter {i}: " + string.Join("; ", GpuLaunchProbe.Fallbacks));
        }
    }

    [SkippableFact]
    public void AdamWThroughParameterBufferViews_MatchesStandaloneParameters_WithoutHostTraffic()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        int[][] shapes = { new[] { 8, 16 }, new[] { 16, 4 } };
        var w1 = Rand(shapes[0], 1);
        var w2 = Rand(shapes[1], 2);
        var x = Rand(new[] { 32, 8 }, 3);

        var buffer = new ParameterBuffer<float>(shapes);
        var views = buffer.CreateAllViews();
        buffer.CopyFrom(new[] { w1, w2 });
        var standalone = new[] { new Tensor<float>(shapes[0]), new Tensor<float>(shapes[1]) };
        for (int i = 0; i < w1.Length; i++) standalone[0][i] = w1[i];
        for (int i = 0; i < w2.Length; i++) standalone[1][i] = w2[i];

        Tensor<float>[] State() => new[] { GpuOptimizer.CreateStateTensor(shapes[0]), GpuOptimizer.CreateStateTensor(shapes[1]) };
        Train(gpu, standalone, State(), State(), x, 3);

        bool savedCapture = GpuLaunchProbe.CaptureReadbackSites;
        long readbackBytes;
        string sites;
        try
        {
            GpuLaunchProbe.CaptureReadbackSites = true;
            GpuLaunchProbe.Reset();
            Train(gpu, views, State(), State(), x, 3);
            readbackBytes = GpuLaunchProbe.ReadbackBytes;
            sites = string.Join("; ", GpuLaunchProbe.ReadbackSites);
        }
        finally
        {
            GpuLaunchProbe.CaptureReadbackSites = savedCapture;
        }
        Assert.True(readbackBytes == 0, $"training through views read back {readbackBytes} bytes: {sites}");
        Assert.True(gpu.IsDeviceAuthoritative(views[0]) && gpu.IsDeviceAuthoritative(views[1]),
            "after on-device AdamW the parameter views are not device-authoritative");

        for (int p = 0; p < 2; p++)
        {
            var got = views[p].ToArray();          // host read: downloads the flat buffer once
            var expected = standalone[p].ToArray();
            for (int i = 0; i < got.Length; i++)
                Assert.True(Math.Abs(got[i] - expected[i]) < 1e-5f, $"param {p}[{i}]: view {got[i]} standalone {expected[i]}");
        }
        // The flat vector reflects the same values (views and buffer share storage).
        Assert.Equal(views[1][0], buffer.AsVector()[shapes[0][0] * shapes[0][1]]);
    }

    /// <summary>
    /// Every fused step, not only AdamW, places an eager model's parameter on the device: a ParameterBuffer view (at an
    /// offset OpenCL cannot address as a sub-buffer, so through a staged copy) and a standalone host-authoritative
    /// tensor. They took only a parameter already bound to a device buffer, so both declined and the step ran on the
    /// host. Both placements must run the same kernel on the same values and leave the parameter device-authoritative.
    /// </summary>
    [SkippableTheory]
    [InlineData("Adam")]
    [InlineData("Sgd")]
    [InlineData("SgdMomentum")]
    [InlineData("Rmsprop")]
    [InlineData("Adagrad")]
    [InlineData("Lion")]
    [InlineData("Nadam")]
    [InlineData("Amsgrad")]
    public void FusedStep_PlacesViewAndHostParameters_OnTheDevice(string kind)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        int[][] shapes = { new[] { 3, 5 }, new[] { 7, 3 } };
        var w = Rand(shapes[1], 8);
        var buffer = new ParameterBuffer<float>(shapes);
        var views = buffer.CreateAllViews();
        buffer.CopyFrom(new[] { Rand(shapes[0], 9), w });
        var standalone = new Tensor<float>(shapes[1]);
        for (int i = 0; i < w.Length; i++) standalone[i] = w[i];

        bool Step(Tensor<float> p)
        {
            var g = gpu.TensorMultiplyScalar(Rand(shapes[1], 10), 1f);   // a device-resident gradient
            Tensor<float> S() => GpuOptimizer.CreateStateTensor(shapes[1]);
            return kind switch
            {
                "Adam" => GpuOptimizer.TryAdamStep(p, g, S(), S(), 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
                "Sgd" => GpuOptimizer.TrySgdStep(p, g, 0.1f),
                "SgdMomentum" => GpuOptimizer.TrySgdMomentumStep(p, g, S(), 0.1f, 0.9f, 0f),
                "Rmsprop" => GpuOptimizer.TryRmspropStep(p, g, S(), 0.01f, 0.99f, 1e-8f, 0f),
                "Adagrad" => GpuOptimizer.TryAdagradStep(p, g, S(), 0.1f, 1e-10f, 0f),
                "Lion" => GpuOptimizer.TryLionStep(p, g, S(), 0.01f, 0.9f, 0.99f, 0f),
                "Nadam" => GpuOptimizer.TryNadamStep(p, g, S(), S(), 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
                "Amsgrad" => GpuOptimizer.TryAmsgradStep(p, g, S(), S(), S(), 0.01f, 0.9f, 0.999f, 1e-8f, 0f, 1),
                _ => throw new ArgumentOutOfRangeException(nameof(kind)),
            };
        }

        GpuLaunchProbe.Reset();
        Assert.True(Step(views[1]), $"{kind} declined a ParameterBuffer view: " + string.Join("; ", GpuLaunchProbe.Fallbacks));
        Assert.True(Step(standalone), $"{kind} declined a host parameter: " + string.Join("; ", GpuLaunchProbe.Fallbacks));
        Assert.True(gpu.IsDeviceAuthoritative(views[1]) && gpu.IsDeviceAuthoritative(standalone),
            $"{kind}: the updated parameters are not device-authoritative");

        var got = views[1].ToArray();
        var expected = standalone.ToArray();
        Assert.NotEqual(w.ToArray(), got);
        for (int i = 0; i < got.Length; i++)
            Assert.Equal(expected[i], got[i]);
        // The neighbouring parameter in the flat array is untouched by the staged write-back.
        Assert.Equal(Rand(shapes[0], 9).ToArray(), views[0].ToArray());
    }

    /// <summary>
    /// A copy-on-write snapshot of a parameter (what a model DeepCopy takes) keeps its values through a device
    /// optimizer step. The step wrote the parameter's device buffer in place without detaching the share, and the
    /// snapshot read the same array, device copy and pending download, so it received the training update too.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeviceStep_LeavesACopyOnWriteSnapshotUnchanged(bool alreadyOnDevice)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        int[] shape = { 6, 4 };
        var param = Rand(shape, 11);
        Tensor<float> Grad() => gpu.TensorMultiplyScalar(Rand(shape, 12), 1f);
        // Optionally make the parameter device-bound and device-authoritative before the snapshot.
        if (alreadyOnDevice)
            Assert.True(GpuOptimizer.TrySgdStep(param, Grad(), 0.1f), string.Join("; ", GpuLaunchProbe.Fallbacks));

        var before = param.ToArray();
        var snapshot = (Tensor<float>)param.CloneShared();
        Assert.True(param.IsCowShared && snapshot.IsCowShared);

        Assert.True(GpuOptimizer.TryAdamWStep(param, Grad(), GpuOptimizer.CreateStateTensor(shape),
            GpuOptimizer.CreateStateTensor(shape), 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 1),
            string.Join("; ", GpuLaunchProbe.Fallbacks));

        Assert.Equal(before, snapshot.ToArray());
        Assert.NotEqual(before, param.ToArray());
    }

    /// <summary>
    /// A snapshot taken while the parameter's newest value exists only on the device (a pending download, never read
    /// on the host) must hold that value, and the next device step must start from it: the detach clones the host
    /// array and drops the device binding, so cloning it before the download resolves would lose the first step.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeviceStep_AfterASnapshotOfAPendingDeviceValue_StartsFromTheDeviceValue(bool parameterView)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        int[] shape = { 6, 4 };
        Tensor<float> Grad(int seed) => gpu.TensorMultiplyScalar(Rand(shape, seed), 1f);
        string Fallbacks() => string.Join("; ", GpuLaunchProbe.Fallbacks);
        // A ParameterBuffer view's pending download is keyed on the shared flat array, not on its own storage.
        Tensor<float> Make()
        {
            if (!parameterView) return Rand(shape, 21);
            var buffer = new ParameterBuffer<float>(new[] { new[] { 4, 8 }, shape });
            buffer.CopyFrom(new[] { Rand(new[] { 4, 8 }, 20), Rand(shape, 21) });
            return buffer.CreateAllViews()[1];
        }

        // Reference: the same two steps with the device value read on the host before the snapshot.
        var reference = Make();
        Assert.True(GpuOptimizer.TrySgdStep(reference, Grad(22), 0.1f), Fallbacks());
        var afterFirst = reference.ToArray();
        var referenceSnapshot = (Tensor<float>)reference.CloneShared();
        Assert.True(GpuOptimizer.TrySgdStep(reference, Grad(23), 0.1f), Fallbacks());
        Assert.Equal(afterFirst, referenceSnapshot.ToArray());

        var param = Make();
        Assert.True(GpuOptimizer.TrySgdStep(param, Grad(22), 0.1f), Fallbacks());
        Assert.True(gpu.IsDeviceAuthoritative(param));
        Assert.True(param.HasPendingGpuData, "the first step's value must still be device-only");
        var snapshot = (Tensor<float>)param.CloneShared();
        // A standalone tensor's snapshot shares the pending download, so the step's detach is what must resolve it. A
        // view's CloneShared downloads the flat array itself; for it only the values below are in question.
        if (!parameterView)
            Assert.True(param.HasPendingGpuData && snapshot.HasPendingGpuData, "the snapshot must not resolve the download");
        Assert.True(GpuOptimizer.TrySgdStep(param, Grad(23), 0.1f), Fallbacks());

        Assert.Equal(afterFirst, snapshot.ToArray());
        Assert.Equal(reference.ToArray(), param.ToArray());
    }

    /// <summary>
    /// The compressed-moment (int8) AdamW step writes the parameter's device buffer in place through its own
    /// placement path; a copy-on-write snapshot of a device-bound parameter must keep its values through it too, and the
    /// detached parameter must still be stepped on the device.
    /// </summary>
    [SkippableFact]
    public void CompressedMomentStep_LeavesACopyOnWriteSnapshotUnchanged()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        int[] shape = { 8, 8 };
        const int blockSize = 32;
        var param = Rand(shape, 13);
        Tensor<float> Grad() => gpu.TensorMultiplyScalar(Rand(shape, 14), 1f);
        Assert.True(GpuOptimizer.TrySgdStep(param, Grad(), 0.1f), string.Join("; ", GpuLaunchProbe.Fallbacks));
        Skip.IfNot(GpuOptimizer.TryAllocAdam8BitState(param.Length, blockSize, out var mQ, out var vQ, out var mS, out var vS),
            "The backend has no compressed optimizer moments.");
        try
        {
            var before = param.ToArray();
            var snapshot = (Tensor<float>)param.CloneShared();
            // The detach drops the shared device binding; the step must place the private copy, not decline.
            Assert.True(GpuOptimizer.TryAdam8BitStep(param, Grad(), mQ, vQ, mS, vS,
                0.01f, 0.9f, 0.999f, 1e-8f, 1f - 0.9f, 1f - 0.999f, blockSize),
                string.Join("; ", GpuLaunchProbe.Fallbacks));
            Assert.True(gpu.IsDeviceAuthoritative(param));

            Assert.Equal(before, snapshot.ToArray());
            Assert.NotEqual(before, param.ToArray());
        }
        finally
        {
            foreach (var buffer in new[] { mQ, vQ, mS, vS })
                if (buffer is not null) GpuOptimizer.FreeGpuBuffer(buffer);
        }
    }

    /// <summary>
    /// A write through the vector <see cref="ParameterBuffer{T}.AsVector"/> returned, made after a GPU read uploaded
    /// the flat array, must reach the next GPU read. The version was advanced only when AsVector returned, so the
    /// device copy uploaded in between was reused with the old element.
    /// </summary>
    [SkippableFact]
    public void WriteThroughAsVector_AfterAGpuRead_ReachesTheNextGpuRead()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        int[][] shapes = { new[] { 4, 8 }, new[] { 8, 2 } };
        var buffer = new ParameterBuffer<float>(shapes);
        var views = buffer.CreateAllViews();
        buffer.CopyFrom(new[] { Rand(shapes[0], 4), Rand(shapes[1], 5) });

        var flat = buffer.AsVector();
        Assert.Equal(views[1][3], gpu.TensorMultiplyScalar(views[1], 1f).ToArray()[3]);
        int at = shapes[0][0] * shapes[0][1] + 3;
        flat[at] = 42f;
        Assert.Equal(42f, gpu.TensorMultiplyScalar(views[1], 1f).ToArray()[3]);
        flat.MultiplyInPlace(2f);
        Assert.Equal(84f, gpu.TensorMultiplyScalar(views[1], 1f).ToArray()[3]);
    }

    /// <summary>
    /// The resident FP16 forward casts a ParameterBuffer parameter on the device, from its offset in the flat device
    /// buffer. Views are never bound to a device buffer, so the cast declined every one and the caller cast on the
    /// host each step.
    /// </summary>
    [SkippableFact]
    public void ResidentHalfCast_OfAParameterBufferView_RunsOnTheDevice()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        int[][] shapes = { new[] { 4, 8 }, new[] { 8, 2 } };
        var buffer = new ParameterBuffer<float>(shapes);
        var views = buffer.CreateAllViews();
        buffer.CopyFrom(new[] { Rand(shapes[0], 6), Rand(shapes[1], 7) });

        var half = new Tensor<Half>(shapes[1]);
        Assert.True(gpu.TryCastResidentForwardHalf(views[1], half),
            "the resident FP16 cast declined a ParameterBuffer view: " + string.Join("; ", GpuLaunchProbe.Fallbacks));
        var got = half.ToArray();
        var expected = views[1].ToArray();
        for (int i = 0; i < got.Length; i++)
            Assert.Equal((float)(Half)expected[i], (float)got[i]);
    }
}
