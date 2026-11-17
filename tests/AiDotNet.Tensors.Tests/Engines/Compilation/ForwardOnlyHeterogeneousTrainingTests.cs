using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("DirectGpuSerial")]
public sealed class ForwardOnlyHeterogeneousTrainingTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Image_area_is_counted_in_double_without_half_overflow(bool half)
    {
        if (half) CheckArea<Half>((Half)2, (Half)1);
        else CheckArea<float>(2f, 1f);
    }

    private static void CheckArea<T>(T two, T one)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<T>(new[] { two }, new[] { 1 });
        var input = Tensor<T>.CreateDefault(new[] { 256, 256 }, one);
        Tensor<double> area;
        CompiledTrainingPlan<T> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            area = engine.ReduceSum(DataCast<T, double>(engine,
                engine.TensorGreaterThan(input, default(T)!)), null, false);
            var ratio = DataCast<double, T>(engine, engine.TensorMultiplyScalar(area, 1.0 / 65536));
            var loss = engine.ReduceSum(engine.TensorMultiply(parameter, ratio), null, false);
            Assert.Null(scope.HostReadOp);
            plan = scope.CompileTraining(new[] { parameter }, loss);
        }
        using (plan)
        {
            Assert.True(plan.HasForwardOnlyHeterogeneousReplay);
            Assert.Equal(two, plan.Step().ToArray()[0]);
            Assert.Equal(65536.0, area.ToArray()[0]);
            Assert.Equal(one, plan.Gradients[0].ToArray()[0]);
        }
    }

    [Fact]
    public void Foreign_data_bridges_stop_gradients_while_the_ordinary_parameter_path_remains_live()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>(new[] { 2f }, new[] { 1 });
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            var data = DataCast<float, double>(engine, parameter);
            var squaredData = engine.TensorMultiply(data, data);
            var coefficient = DataCast<double, float>(engine, squaredData);
            plan = scope.CompileTraining(new[] { parameter }, engine.ReduceSum(engine.TensorMultiply(parameter, coefficient), null, false));
        }
        using (plan)
        {
            Assert.Equal(8f, plan.Step().ToArray()[0]);
            Assert.Equal(4f, plan.Gradients[0].ToArray()[0]); // d(w * stop(w^2))/dw = w^2, not 3*w^2.
            parameter[0] = 3f;
            Assert.Equal(27f, plan.Step().ToArray()[0]);
            Assert.Equal(9f, plan.Gradients[0].ToArray()[0]);
        }
    }

    [Fact]
    public void A_singular_derivative_behind_a_data_stop_is_never_evaluated()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>(new[] { 0f }, new[] { 1 });
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            var singular = engine.TensorSqrt(parameter); // Finite value, infinite derivative at zero.
            var data = DataCast<float, double>(engine, singular);
            var coefficient = DataCast<double, float>(engine, engine.TensorAddScalar(data, 2.0));
            plan = scope.CompileTraining(new[] { parameter }, engine.ReduceSum(engine.TensorMultiply(parameter, coefficient), null, false));
        }
        using (plan)
        {
            Assert.Equal(0f, plan.Step().ToArray()[0]);
            Assert.Equal(2f, plan.Gradients[0].ToArray()[0]);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Closure_guard_rebinds_new_foreign_draw_leaves_across_A_B_A_without_losing_double_bits(bool gradientPooling)
    {
        var previous = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableBackwardGradientPooling = gradientPooling });
            CheckGuardedReplay(new CpuEngine());
        }
        finally { TensorCodecOptions.SetCurrent(previous); }
    }

    internal static void CheckGuardedReplay(IEngine engine)
    {
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>(new[] { 3f }, new[] { 1 });
        if (engine.SupportsGpu) parameter.Gpu();
        var parameters = new[] { parameter };
        double below = 0.5 - Math.Pow(2, -30), above = 0.5 + Math.Pow(2, -30);
        Assert.Equal((float)below, (float)above); // A float draw would select the wrong pixel.
        var initialA = new ObjectiveData(new[] { 2f, 7f }, below);
        var initialB = new ObjectiveData(new[] { 1f, 3f, 5f, 11f }, 0.75);
        var (planA, guardA) = Compile(initialA.Objective(engine, parameter), parameters);
        var (planB, guardB) = Compile(initialB.Objective(engine, parameter), parameters);
        using (planA)
        using (planB)
        {
            Assert.True(guardA.HasForeignLeaves);
            Assert.True(guardB.HasForeignLeaves);
            Assert.NotNull(planA.StaticScheduleUnsupportedReason);
            Assert.False(planA.IsGraphStepEligible);
            Assert.False(planB.IsGraphStepEligible);
            long capturesBefore = CompiledTrainingPlan<float>.GraphReplayCount;
            planA.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1);
            planB.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1);
            long originalDrawVersion = initialA.Draw.Version;
            float expectedWeight = 3;
            var steps = new[]
            {
                (Data: new ObjectiveData(new[] { 2f, 7f }, below), Plan: planA, Guard: guardA, Coefficient: 2f),
                (Data: new ObjectiveData(new[] { 1f, 3f, 5f, 11f }, 0.75), Plan: planB, Guard: guardB, Coefficient: 11f),
                (Data: new ObjectiveData(new[] { 2f, 7f }, above), Plan: planA, Guard: guardA, Coefficient: 7f),
            };
            foreach (var step in steps)
            {
                Assert.Equal(ClosureGuardResult.Matched,
                    step.Guard.TryRebind(step.Data.Objective(engine, parameter), parameters, out _));
                float expectedLoss = expectedWeight * step.Coefficient;
                Assert.Equal(expectedLoss, step.Plan.Step().ToArray()[0], 4);
                Assert.Equal(step.Coefficient, step.Plan.Gradients[0].ToArray()[0], 5);
                expectedWeight -= 0.1f * step.Coefficient;
                Assert.Equal(expectedWeight, parameter.ToArray()[0], 5);
            }
            Assert.Equal(above, initialA.Draw.ToArray()[0]);
            Assert.True(initialA.Draw.Version > originalDrawVersion);
            Assert.Equal(2, planA.OptimizerStep);
            Assert.Equal(1, planB.OptimizerStep);
            Assert.Equal(capturesBefore, CompiledTrainingPlan<float>.GraphReplayCount);
        }
    }

    private sealed class ObjectiveData
    {
        internal readonly Tensor<float> Input;
        internal readonly Tensor<double> Draw;
        internal ObjectiveData(float[] values, double draw)
        {
            Input = new Tensor<float>(values, new[] { values.Length });
            Draw = new Tensor<double>(new[] { draw }, new[] { 1 });
        }
        internal Func<Tensor<float>> Objective(IEngine engine, Tensor<float> parameter) => () =>
        {
            var values = DataCast<float, double>(engine, Input);
            var selected = SelectByDraw(engine, values, Draw);
            var coefficient = DataCast<double, float>(engine, selected);
            return engine.ReduceSum(engine.TensorMultiply(parameter, coefficient), null, false);
        };
    }

    [Fact]
    public void No_grad_in_place_writes_preserve_the_live_prior_writer_through_foreign_data_replay()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var bias = new Tensor<float>(new[] { 1f }, new[] { 1 });
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter, bias }))
        {
            var live = engine.TensorMultiplyScalar(parameter, 3f);
            using (new NoGradScope<float>()) engine.TensorAddInPlace(live, bias);
            var coefficient = DataCast<double, float>(engine, new Tensor<double>(new[] { 2.0 }, new[] { 1 }));
            plan = scope.CompileTraining(new[] { parameter, bias }, engine.ReduceSum(engine.TensorMultiply(live, coefficient), null, false));
        }
        using (plan)
        {
            Assert.Equal(14f, plan.Step().ToArray()[0]);
            Assert.Equal(6f, plan.Gradients[0].ToArray()[0]);
            Assert.Equal(0f, plan.Gradients[1].ToArray()[0]);
        }
    }

    [Fact]
    public void Single_schedule_executes_rng_foreign_nodes_and_in_place_writes_exactly_once_in_order()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var state = new Tensor<double>(new[] { 0.0 }, new[] { 1 });
        var one = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var random = new Random(17);
        var oracle = new Random(17);
        var events = new List<string>();
        Tensor<double> foreign;
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            var randomValue = scope.RecordUnary(LazyNodeType.Custom, "CountingRandom", parameter, new[] { 1 },
                (_, output) => { events.Add("random"); output[0] = (float)random.NextDouble(); });
            var doubleRandom = DataCast<float, double>(engine, randomValue);
            foreign = scope.RecordBinary(LazyNodeType.Custom, "ForeignBeforeWrite", state, doubleRandom, new[] { 1 },
                (_, output) => { events.Add("foreign"); output[0] = state[0] + doubleRandom[0]; });
            scope.RecordInPlace(LazyNodeType.Custom, "CountingWrite", state, new[] { one },
                (e, _) => { events.Add("write"); e.TensorAddInPlace(state, one); });
            var coefficient = DataCast<double, float>(engine, engine.TensorAdd(foreign, state));
            var loss = engine.ReduceSum(engine.TensorMultiply(parameter, coefficient), null, false);
            plan = scope.CompileTraining(new[] { parameter }, loss);
        }
        using (plan)
        {
            Assert.Empty(events);
            object storage = foreign._storage;
            long version = foreign.Version;
            for (int step = 0; step < 4; step++)
            {
                float coefficient = (float)((float)oracle.NextDouble() + 2.0 * step + 1);
                Assert.Equal(2 * coefficient, plan.Step().ToArray()[0], 5);
                Assert.Equal(coefficient, plan.Gradients[0].ToArray()[0], 5);
                Assert.Equal(step + 1.0, state.ToArray()[0]);
                Assert.Equal(Enumerable.Range(0, step + 1).SelectMany(_ => new[] { "random", "foreign", "write" }), events);
                Assert.Same(storage, foreign._storage);
                Assert.True(foreign.Version > version);
                version = foreign.Version;
            }
        }
    }

    [Fact]
    public void Foreign_output_storage_survives_arena_exit_and_pool_reuse_and_releases_its_plan_lease()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>(new[] { 2f }, new[] { 1 });
        CompiledTrainingPlan<float> plan;
        Tensor<double> foreign;
        using (var arena = TensorArena.Create())
        {
            using (var scope = GraphMode.EnableTraining(new[] { parameter }))
            {
                var input = Tensor<float>.CreateDefault(new[] { 256, 256 }, 1f);
                foreign = DataCast<float, double>(engine, input);
                var area = engine.ReduceSum(foreign, null, false);
                var ratio = DataCast<double, float>(engine, engine.TensorMultiplyScalar(area, 1.0 / 65536));
                plan = scope.CompileTraining(new[] { parameter }, engine.ReduceSum(engine.TensorMultiply(parameter, ratio), null, false));
            }
            arena.Reset();
        }
        var storage = foreign._storage;
        int heldReferences = storage.RefCount;
        Assert.True(heldReferences >= 2);
        using (var next = TensorArena.Create())
            for (int i = 0; i < 4; i++) next.TryRentTensor<double>(65536, new[] { 256, 256 })!.AsWritableSpan().Fill(-99);
        for (int step = 0; step < 3; step++)
        {
            Assert.Equal(2f, plan.Step().ToArray()[0]);
            Assert.Equal(1f, plan.Gradients[0].ToArray()[0]);
            Assert.All(foreign.ToArray(), value => Assert.Equal(1.0, value));
            Assert.Same(storage, foreign._storage);
            Assert.Equal(heldReferences, storage.RefCount);
        }
        plan.Dispose();
        Assert.Equal(heldReferences - 1, storage.RefCount);
        Assert.Throws<ObjectDisposedException>(() => plan.Step());
    }

    [Fact]
    public void Failed_foreign_forward_does_not_update_and_next_step_overwrites_stale_outputs()
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var input = new Tensor<double>(new[] { 3.0 }, new[] { 1 });
        bool fail = true;
        int executions = 0;
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(new[] { parameter }))
        {
            var data = scope.RecordUnary(LazyNodeType.Custom, "FailingData", input, new[] { 1 }, (_, output) =>
            {
                executions++;
                output[0] = input[0];
                if (fail) throw new InvalidOperationException("Injected forward failure.");
            });
            var coefficient = DataCast<double, float>(engine, data);
            plan = scope.CompileTraining(new[] { parameter }, engine.ReduceSum(engine.TensorMultiply(parameter, coefficient), null, false));
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.1);
            Assert.Throws<InvalidOperationException>(() => plan.Step());
            Assert.Equal(2f, parameter.ToArray()[0]);
            Assert.Equal(0, plan.OptimizerStep);
            fail = false;
            input[0] = 7.0;
            Assert.Equal(14f, plan.Step().ToArray()[0]);
            Assert.Equal(7f, plan.Gradients[0].ToArray()[0]);
            Assert.Equal(1.3f, parameter.ToArray()[0], 5);
            Assert.Equal(1, plan.OptimizerStep);
            Assert.Equal(2, executions);
        }
    }

    [Theory]
    [InlineData("checkpoint")]
    [InlineData("frozen")]
    [InlineData("serialize")]
    public void Typed_only_schedule_APIs_refuse_before_execution_or_state_mutation(string feature)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var data = new ObjectiveData(new[] { 3f, 7f }, 0.0);
        var (plan, _) = Compile(data.Objective(engine, parameter), new[] { parameter });
        using (plan)
        using (var stream = new MemoryStream())
        {
            Assert.Throws<NotSupportedException>(() =>
            {
                if (feature == "checkpoint") plan.EnableCheckpointing();
                else if (feature == "frozen") plan.EnableFrozenWeightOptimizations();
                else _ = plan.SaveAsync(stream); // Preflight rejects synchronously before creating/writing a task.
            });
            Assert.Equal(0, stream.Length);
            Assert.Equal(2f, parameter.ToArray()[0]);
            Assert.Equal(0, plan.OptimizerStep);
            Assert.Equal(6f, plan.Step().ToArray()[0]);
            Assert.Equal(3f, plan.Gradients[0].ToArray()[0]);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Unsupported_differentiable_cross_type_graphs_are_refused_before_any_execution(bool mixedFp16AndDouble)
    {
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        var parameter = new Tensor<float>(new[] { 2f }, new[] { 1 });
        int executions = 0;
        using var scope = GraphMode.EnableTraining(new[] { parameter });
        Tensor<float> result;
        if (mixedFp16AndDouble)
        {
            var half = scope.RecordCrossTypeWithBackward<float, Half>(LazyNodeType.Custom, "DifferentiableHalf", parameter,
                new[] { 1 }, (_, output) => { executions++; output[0] = (Half)2; },
                (gradient, _, _, _, _) => new Tensor<float>(new[] { (float)gradient[0] }, new[] { 1 }));
            result = DataCast<Half, float>(engine, half);
            result = engine.TensorAdd(result, DataCast<double, float>(engine,
                new Tensor<double>(new[] { 1.0 }, new[] { 1 })));
        }
        else
        {
            var value = scope.RecordCrossTypeWithBackward<float, double>(LazyNodeType.Custom, "DifferentiableDouble", parameter,
                new[] { 1 }, (_, output) => { executions++; output[0] = 2.0; },
                (gradient, _, _, _, _) => new Tensor<float>(new[] { (float)gradient[0] }, new[] { 1 }));
            result = DataCast<double, float>(engine, value);
        }
        var loss = engine.ReduceSum(engine.TensorMultiply(parameter, result), null, false);
        Assert.Throws<NotSupportedException>(() => scope.CompileTraining(new[] { parameter }, loss));
        Assert.Equal(0, executions);
        Assert.Equal(2f, parameter.ToArray()[0]);
    }

    private static (CompiledTrainingPlan<float>, ClosureObjectiveGuard<float>) Compile(
        Func<Tensor<float>> objective, Tensor<float>[] parameters)
    {
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var loss = objective();
            Assert.Null(scope.HostReadOp);
            plan = scope.CompileTraining(parameters, loss);
            Assert.Null(scope.HostReadOp);
        }
        var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason);
        Assert.True(guard is not null, reason);
        guard!.AttachTo(plan);
        return (plan, guard);
    }

    internal static Tensor<TOut> DataCast<TIn, TOut>(IEngine engine, Tensor<TIn> input)
    {
        if (GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(engine);
            return scope.RecordCrossType<TIn, TOut>(LazyNodeType.Custom, "ForwardDataCast", input, input.Shape.ToArray(),
                (e, output) => DirectGpuTensorEngine.CopyResultInto(e, input.Cast<TOut>(), output));
        }
        return input.Cast<TOut>();
    }

    private static Tensor<double> SelectByDraw(IEngine engine, Tensor<double> values, Tensor<double> draw)
    {
        if (GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(engine);
            return scope.RecordBinary(LazyNodeType.Custom, "SelectByDoubleDraw", values, draw, new[] { 1 }, (_, output) =>
                output[0] = values[(int)(draw[0] * values.Length)]);
        }
        return new Tensor<double>(new[] { values[(int)(draw[0] * values.Length)] }, new[] { 1 });
    }
}

// Keep optional backend coverage separate from the mandatory, zero-skip CPU gate.
[Collection("DirectGpuSerial")]
public sealed class ForwardOnlyHeterogeneousGpuTrainingTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    [SkippableFact]
    public void Ordinary_gpu_replay_refreshes_host_written_foreign_data_and_leaf_versions()
    {
        DirectGpuTensorEngine? device = null;
        try { device = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        using (device)
        {
            bool available = device?.IsGpuAvailable == true;
            if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
                Assert.True(available, "GPU tests were required but no direct GPU backend resolved.");
            Skip.If(!available, "No direct GPU backend available.");
            // Preserve default opportunistic graph/resident settings and the original GPU engine.
            ForwardOnlyHeterogeneousTrainingTests.CheckGuardedReplay(device!);
        }
    }

    [SkippableFact]
    public void Foreign_gpu_outputs_are_detached_but_persistent_state_and_its_views_remain_current()
    {
        DirectGpuTensorEngine? device = null;
        try { device = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        using (device)
        {
            bool available = device?.IsGpuAvailable == true;
            if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
                Assert.True(available, "GPU tests were required but no direct GPU backend resolved.");
            Skip.If(!available, "No direct GPU backend available.");
            var engine = device!;
            AiDotNetEngine.Current = engine;
            var backend = engine.GetBackend()!;
            using var outputBinding = backend.AllocateBuffer(new[] { 0f });
            using var stateBinding = backend.AllocateBuffer(new[] { 0f });
            var parameter = new Tensor<float>(new[] { 2f }, new[] { 1 });
            using var state = new Tensor<Half>(new[] { (Half)0 }, new[] { 1 });
            var one = new Tensor<Half>(new[] { (Half)1 }, new[] { 1 });
            parameter.Gpu();
            engine.BindResidentBuffer(state, stateBinding, backend);
            Tensor<Half> before;
            Tensor<Half> stateView;
            int observedDeviceOutputs = 0;
            CompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { parameter }))
            {
                // Tensor.Reshape records a metadata-only RecordView with shared storage; the
                // engine's traced Reshape instead owns a replay output slot and copies into it.
                var oldView = state.Reshape(1, 1);
                Assert.Same(state._storage, oldView._storage);
                before = engine.TensorMultiplyScalar(oldView, (Half)1);
                var observed = scope.RecordUnary(LazyNodeType.Custom, "ObserveForeignDeviceOutput", before, new[] { 1, 1 },
                    (e, output) =>
                    {
                        // Ordinary GPU results may use a pending download/cache rather than an explicit
                        // tensor binding. Deliberately bind this replay's real value to a controlled GPU
                        // buffer, so step-end detachment is exercised regardless of that representation.
                        backend.UploadBufferInPlace(new[] { (float)before.ToArray()[0] }, outputBinding);
                        engine.BindResidentBuffer(before, outputBinding, backend);
                        Assert.Same(outputBinding, before._gpuBuffer);
                        observedDeviceOutputs++;
                        e.TensorCopy(before, output);
                    });
                engine.TensorAddInPlace(state, one);
                stateView = state.Reshape(1, 1);
                Assert.Same(state._storage, stateView._storage);
                var sum = engine.TensorAdd(observed, stateView);
                var coefficient = ForwardOnlyHeterogeneousTrainingTests.DataCast<Half, float>(engine,
                    engine.Reshape(sum, new[] { 1 }));
                plan = scope.CompileTraining(new[] { parameter }, engine.ReduceSum(engine.TensorMultiply(parameter, coefficient), null, false));
            }
            using (plan)
            {
                Assert.True(plan.HasForwardOnlyHeterogeneousReplay);
                Assert.False(plan.IsGraphStepEligible);
                object storage = state._storage;
                for (int step = 0; step < 4; step++)
                {
                    float coefficient = 2 * step + 1;
                    Assert.Equal(2 * coefficient, plan.Step().ToArray()[0], 5);
                    Assert.Equal(coefficient, plan.Gradients[0].ToArray()[0], 5);
                    Assert.Equal((Half)(step + 1), state.ToArray()[0]);
                    Assert.Equal((Half)(step + 1), stateView.ToArray()[0]);
                    Assert.Same(storage, state._storage);
                    Assert.Same(storage, stateView._storage);
                    Assert.Null(before._gpuBuffer); // Dead outputs must not keep a released device binding.
                    Assert.Equal(step + 1, observedDeviceOutputs);
                }
            }
        }
    }
}
