#if NET10_0_OR_GREATER
using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("DirectGpuSerial")]
public sealed class CompiledCudaOptimizerEnrollmentTests
{
    private static CudaBackend RequireCuda()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            // Absence is a failure in any lane that requires a GPU, never a skipped validation.
            bool required = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1"
                || Environment.GetEnvironmentVariable("AIDOTNET_EXPECT_GPU_BACKEND") == "CUDA";
            Assert.False(required, "The validation lane requires a working CUDA backend.");
            Skip.If(true, "No CUDA device available.");
        }
        return backend;
    }

    [SkippableTheory]
    [InlineData(false, 7u)]
    [InlineData(true, 7u)]
    [InlineData(false, 16777217u)]
    [InlineData(true, 16777217u)]
    public void Captured_Adam_uses_leaf_ages_and_one_finite_verdict(bool decoupled, uint globalAge)
    {
        using var backend = RequireCuda();
        var capture = (IGpuGraphCapture)backend;
        Assert.True(capture.SupportsGraphCapture);
        using var p0 = backend.AllocateBuffer(new[] { 2f });
        using var p1 = backend.AllocateBuffer(new[] { 3f });
        using var g0 = backend.AllocateBuffer(new[] { 2f });
        using var g1 = backend.AllocateBuffer(new[] { 4f });
        using var m0 = backend.AllocateBuffer(new[] { 0.3f });
        using var m1 = backend.AllocateBuffer(new[] { 0f });
        using var v0 = backend.AllocateBuffer(new[] { 0.4f });
        using var v1 = backend.AllocateBuffer(new[] { 0f });
        IGpuBuffer[] parameters = [p0, p1], gradients = [g0, g1], first = [m0, m1], second = [v0, v1];
        int[] sizes = [1, 1];
        using var io = capture.CreateStepIo([], []);
        Assert.NotNull(io);
        io.InitializeAppliedSteps(globalAge);
        long[] offsets = [0, -(long)globalAge];
        io.PrepareOptimizer(parameters, gradients, first, second, sizes, offsets);
        offsets[1] = 0; // The graph must own a snapshot, not this mutable host list.
        const float beta1 = 0.9f, beta2 = 0.99f, epsilon = 1e-6f, decay = 0.1f;
        var graph = capture.CaptureGraph(() =>
        {
            io.CapturePrologue(0);
            io.CaptureAdamUpdate(decoupled, parameters, gradients, first, second, sizes, beta1, beta2, epsilon, decay);
            io.CaptureEpilogue(null, 0);
        });
        Assert.True(graph != IntPtr.Zero, capture.LastCaptureFailure);
        try
        {
            Assert.True(capture.LastCaptureNodeCounts?.Kernel > 0);
            float[] expectedP = [2f, 3f], expectedM = [0.3f, 0f], expectedV = [0.4f, 0f];
            // Finite / rejected / finite: rejected work changes neither leaf nor moment nor global age.
            for (uint replay = 0; replay < 3; replay++)
            {
                bool finite = replay != 1;
                float[][] before = [backend.DownloadBuffer(p0), backend.DownloadBuffer(p1),
                    backend.DownloadBuffer(m0), backend.DownloadBuffer(m1), backend.DownloadBuffer(v0), backend.DownloadBuffer(v1)];
                float learningRate = replay == 2 ? 0.03f : 0.01f;
                uint appliedBefore = globalAge + (replay == 0 ? 0u : 1u);
                io.WriteLearningRates(replay, checked((int)appliedBefore + 1), new[] { learningRate });
                backend.UploadBufferInPlace(new[] { finite ? 4f : float.NaN }, g1);
                capture.LaunchCapturedGraph(graph);
                io.WaitForReplays(replay + 1);
                Assert.Equal(replay + 1, io.CompletedReplays); // Actual captured graph publication, no eager fallback.
                Assert.Equal(appliedBefore + (finite ? 1u : 0u), io.AppliedSteps);
                Assert.Equal(finite, double.IsFinite(io.SumOfSquares(replay)));
                if (finite)
                {
                    int accepted = replay == 0 ? 1 : 2;
                    Update(ref expectedP[0], ref expectedM[0], ref expectedV[0], 2f,
                        (long)globalAge + accepted, learningRate, decoupled, beta1, beta2, epsilon, decay);
                    Update(ref expectedP[1], ref expectedM[1], ref expectedV[1], 4f,
                        accepted, learningRate, decoupled, beta1, beta2, epsilon, decay);
                }
                for (int i = 0; i < 2; i++)
                {
                    Close(expectedP[i], backend.DownloadBuffer(parameters[i])[0]);
                    Close(expectedM[i], backend.DownloadBuffer(first[i])[0]);
                    Close(expectedV[i], backend.DownloadBuffer(second[i])[0]);
                }
                if (!finite)
                {
                    IGpuBuffer[] all = [p0, p1, m0, m1, v0, v1];
                    for (int i = 0; i < all.Length; i++) Assert.Equal(before[i], backend.DownloadBuffer(all[i]));
                }
            }
        }
        finally { capture.DestroyCapturedGraph(graph); }
    }

    private static void Update(ref float p, ref float m, ref float v, float g, long age, float lr,
        bool decoupled, float b1, float b2, float eps, float decay)
    {
        if (decoupled) p *= 1 - lr * decay;
        else g += decay * p;
        m = b1 * m + (1 - b1) * g;
        v = b2 * v + (1 - b2) * g * g;
        double mh = m / (1 - Math.Pow(b1, age));
        double vh = v / (1 - Math.Pow(b2, age));
        p -= (float)(lr * mh / (Math.Sqrt(vh) + eps));
    }

    private static void Close(float expected, float actual)
        => Assert.True(float.IsFinite(actual) && Math.Abs(expected - actual) <= 3e-5f * (1 + Math.Abs(expected)),
            $"expected {expected:R}, actual {actual:R}");

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Shared_optimizer_handover_recaptures_staggered_ages_and_freezes_inactive_leaf(bool adamW)
    {
        using var backendProof = RequireCuda();
        using var engine = new DirectGpuTensorEngine();
        Assert.IsType<CudaBackend>(engine.QueryBackend);
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var p = new Tensor<float>(new[] { 2f }, new[] { 1 });
            var q = new Tensor<float>(new[] { 3f }, new[] { 1 });
            p.Gpu(); q.Gpu();
            CompiledTrainingPlan<float> Compile(bool freezeP)
            {
                using var scope = GraphMode.EnableTraining(new[] { p, q });
                Tensor<float> twiceP;
                if (freezeP)
                {
                    using var noGrad = new NoGradScope<float>();
                    twiceP = engine.TensorMultiplyScalar(p, 2f);
                }
                else twiceP = engine.TensorMultiplyScalar(p, 2f);
                var loss = engine.ReduceSum(engine.TensorMultiply(twiceP, q), null, false);
                return (CompiledTrainingPlan<float>)scope.CompileTraining(new[] { p, q }, loss);
            }
            using var a = Compile(true);
            using var b = Compile(false);
            const float lr = 0.01f, beta1 = 0.9f, beta2 = 0.99f, epsilon = 1e-6f, decay = 0.02f;
            void Apply(ref float value, ref float firstMoment, ref float secondMoment, float gradient, long age)
            {
                if (adamW) Update(ref value, ref firstMoment, ref secondMoment, gradient, age,
                    lr, true, beta1, beta2, epsilon, decay);
                else value -= lr * (gradient + decay * value);
            }
            a.ConfigureOptimizer(adamW ? OptimizerType.AdamW : OptimizerType.SGD, lr, beta1, beta2, epsilon, decay);
            a.Step(); // Materialize the initial moments before seeding a large accepted global age.
            var checkpoint = a.CaptureFusedOptimizerCheckpoint();
            Assert.NotNull(checkpoint);
            const int global = (1 << 24) + 1;
            checkpoint.OptimizerStep = global;
            checkpoint.Parameters[0].AcceptedUpdates = 0;
            checkpoint.Parameters[1].AcceptedUpdates = global;
            a.RestoreFusedOptimizerCheckpoint(checkpoint);
            float ep = p.ToArray()[0], eq = q.ToArray()[0];
            float mp = checkpoint.Parameters[0].MFloat?[0] ?? 0, vp = checkpoint.Parameters[0].VFloat?[0] ?? 0;
            float mq = checkpoint.Parameters[1].MFloat?[0] ?? 0, vq = checkpoint.Parameters[1].VFloat?[0] ?? 0;
            long beforeInitialA = CompiledTrainingPlan<float>.GraphReplayCount;
            for (int i = 1; i <= 12; i++)
            {
                Apply(ref eq, ref mq, ref vq, 2 * ep, (long)global + i);
                a.Step();
            }
            Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > beforeInitialA, a.GraphCaptureFailureReason);
            Assert.NotNull(a.StepGraphNodeCounts);
            Assert.Equal(2f, p.ToArray()[0]);
            Close(eq, q.ToArray()[0]);
            Assert.Equal(0, a.ParameterAcceptedUpdates[0]);
            Assert.Equal((long)global + 12, a.ParameterAcceptedUpdates[1]);
            // No restore/reset between this live captured A and the two handovers below.
            b.ContinueOptimizerFrom(a);
            long before = CompiledTrainingPlan<float>.GraphReplayCount;
            for (int i = 1; i <= 12; i++)
            {
                float gp = 2 * eq, gq = 2 * ep;
                Apply(ref ep, ref mp, ref vp, gp, i);
                Apply(ref eq, ref mq, ref vq, gq, (long)global + 12 + i);
                b.Step();
            }
            Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > before, b.GraphCaptureFailureReason);
            Assert.NotNull(b.StepGraphNodeCounts);
            Close(ep, p.ToArray()[0]); Close(eq, q.ToArray()[0]);
            var afterB = Assert.IsType<FusedOptimizerCheckpoint>(b.CaptureFusedOptimizerCheckpoint());
            Assert.Equal(12, afterB.Parameters[0].AcceptedUpdates);
            Assert.Equal((long)global + 24, afterB.Parameters[1].AcceptedUpdates);
            if (adamW)
            {
                Close(mp, Assert.IsType<float[]>(afterB.Parameters[0].MFloat)[0]); Close(vp, Assert.IsType<float[]>(afterB.Parameters[0].VFloat)[0]);
                Close(mq, Assert.IsType<float[]>(afterB.Parameters[1].MFloat)[0]); Close(vq, Assert.IsType<float[]>(afterB.Parameters[1].VFloat)[0]);
            }
            a.ContinueOptimizerFrom(b);
            float frozen = p.ToArray()[0];
            before = CompiledTrainingPlan<float>.GraphReplayCount;
            for (int i = 25; i <= 36; i++)
            {
                Apply(ref eq, ref mq, ref vq, 2 * frozen, (long)global + i);
                a.Step();
            }
            Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > before, a.GraphCaptureFailureReason);
            Assert.NotNull(a.StepGraphNodeCounts);
            Assert.Equal(frozen, p.ToArray()[0]);
            Close(eq, q.ToArray()[0]);
            var afterA = Assert.IsType<FusedOptimizerCheckpoint>(a.CaptureFusedOptimizerCheckpoint());
            Assert.Equal(afterB.Parameters[0].AcceptedUpdates, afterA.Parameters[0].AcceptedUpdates);
            Assert.Equal(afterB.Parameters[0].MFloat, afterA.Parameters[0].MFloat);
            Assert.Equal(afterB.Parameters[0].VFloat, afterA.Parameters[0].VFloat);
            Assert.Equal((long)global + 36, afterA.Parameters[1].AcceptedUpdates);
            if (adamW)
            {
                Close(mq, Assert.IsType<float[]>(afterA.Parameters[1].MFloat)[0]);
                Close(vq, Assert.IsType<float[]>(afterA.Parameters[1].VFloat)[0]);
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [SkippableFact]
    public void Offset_validation_precedes_table_replacement_and_zero_offsets_allocate_no_table()
    {
        using var backend = RequireCuda();
        using var p = backend.AllocateBuffer(new[] { 1f });
        using var g = backend.AllocateBuffer(new[] { 2f });
        using var m = backend.AllocateBuffer(new[] { 0f });
        using var v = backend.AllocateBuffer(new[] { 0f });
        using var io = ((IGpuGraphCapture)backend).CreateStepIo([], []);
        Assert.NotNull(io);
        Assert.Throws<ArgumentOutOfRangeException>(() => io.PrepareOptimizer([p], [g], [m], [v], [1], [-1]));
        io.InitializeAppliedSteps(4);
        io.PrepareOptimizer([p], [g], [m], [v], [1], [0]);
        var field = io.GetType().GetField("_acceptedOffsets", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        Assert.Null(field.GetValue(io));
        io.PrepareOptimizer([p], [g], [m], [v], [1], [-4]);
        object? table = field.GetValue(io);
        Assert.NotNull(table);
        Assert.Throws<ArgumentException>(() => io.PrepareOptimizer([p], [g], [m], [v], [1], []));
        Assert.Throws<ArgumentOutOfRangeException>(() => io.PrepareOptimizer([p], [g], [m], [v], [1], [-5]));
        Assert.Throws<ArgumentOutOfRangeException>(() => io.PrepareOptimizer([p], [g], [m], [v], [1], [1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => io.PrepareOptimizer([p], [g], [m], [v], [1], [long.MinValue]));
        Assert.Throws<ArgumentOutOfRangeException>(() => io.PrepareOptimizer([p], [g], [m], [v], [1], [long.MaxValue]));
        Assert.Same(table, field.GetValue(io));
        Assert.Equal(new[] { 1f }, backend.DownloadBuffer(p));
        Assert.Equal(4u, io.AppliedSteps);
        io.PrepareOptimizer([p], [g], [m], [v], [1]);
        Assert.Null(field.GetValue(io));
    }

    [SkippableFact]
    public void Frozen_parameter_read_in_sgd_step_graph_preserves_ping_pong_parity_graphs()
    {
        using var probe = RequireCuda();
        var prior = AiDotNetEngine.Current;
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.GetBackend() is CudaBackend, "DirectGpuTensorEngine did not select CudaBackend.");
        AiDotNetEngine.Current = gpu;
        try
        {
            var frozen = new Tensor<float>(new[] { 1.5f, -0.5f, 2.0f, 0.25f }, new[] { 2, 2 });
            var active = new Tensor<float>(new[] { 0.75f, -1.25f, 0.5f, 1.0f }, new[] { 2, 2 });
            frozen.Gpu();
            active.Gpu();
            var parameters = new[] { frozen, active };

            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                Tensor<float> frozenTerm;
                using (new NoGradScope<float>())
                    frozenTerm = gpu.TensorMultiplyScalar(frozen, 2f);
                var prod = gpu.TensorMultiply(frozenTerm, active);
                var loss = gpu.ReduceSum(prod, null, false);
                plan = Assert.IsType<CompiledTrainingPlan<float>>(scope.CompileTraining(parameters, loss));
            }

            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05);
                for (int s = 0; s < 6; s++) plan.Step();

                if (Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") != "0"
                    && Environment.GetEnvironmentVariable("AIDOTNET_SGD_PINGPONG") != "0")
                {
                    Assert.True(plan.ReplaysParityGraphs, plan.PingPongRejectionReason ?? plan.GraphCaptureFailureReason);
                }
                Assert.Equal(new[] { 1.5f, -0.5f, 2.0f, 0.25f }, frozen.ToArray());
                // Six SGD updates with constant gradient 2 * frozen.
                var expected = new[] { -0.15f, -0.95f, -0.7f, 0.85f };
                var actual = active.ToArray();
                for (int i = 0; i < expected.Length; i++) Close(expected[i], actual[i]);
                Assert.Equal(new long[] { 0, 6 }, plan.ParameterAcceptedUpdates);
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
#endif
