using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("DirectGpuSerial")]
public sealed class CompiledEnrollmentCheckpointTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Roundtrip_preserves_frozen_and_connected_zero_membership_and_parameter_ages(bool useDouble)
    {
        if (useDouble) await CheckRoundtrip<double>();
        else await CheckRoundtrip<float>();
    }

    private static async Task CheckRoundtrip<T>()
    {
        var prior = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            async Task<double[][]> Run(bool roundtrip)
            {
                var parameters = new[] { Scalar<T>(2), Scalar<T>(3), Scalar<T>(4) };
                var expected = new[] { 2.0, 3.0, 4.0 };
                var m = new double[3];
                var v = new double[3];
                var ages = new long[3];
                var trajectory = new List<double[]>();
                var plan = Compile(engine, parameters, freezeFirst: false);
                plan.ConfigureOptimizer(OptimizerType.AdamW, LrSchedule.Constant(0.1),
                    beta1: 0.5, beta2: 0.75, eps: 0.01, weightDecay: 0.2);
                try
                {
                    for (int step = 0; step < 5; step++)
                    {
                        if (step is 1 or 4)
                        {
                            var next = Compile(engine, parameters, freezeFirst: step == 1);
                            next.ContinueOptimizerFrom(plan);
                            plan.Dispose();
                            plan = next;
                        }
                        if (step == 3 && roundtrip)
                        {
                            using var saved = new MemoryStream();
                            await plan.SaveAsync(saved);
                            parameters = parameters.Select(parameter => parameter.Clone()).ToArray();
                            saved.Position = 0;
                            var loaded = await CompiledPlanLoader.LoadTrainingAsync<T>(saved, engine, parameters);
                            Assert.NotNull(loaded);
                            plan.Dispose();
                            plan = Assert.IsType<CompiledTrainingPlan<T>>(loaded);
                        }

                        plan.Step();
                        for (int p = 0; p < 3; p++)
                        {
                            if (p != 0 || step is 0 or 4)
                            {
                                // Independent scalar AdamW: a connected zero gradient still enrolls and decays.
                                double gradient = p == 1 ? 0 : 2 * expected[p];
                                long age = ++ages[p];
                                m[p] = 0.5 * m[p] + 0.5 * gradient;
                                v[p] = 0.75 * v[p] + 0.25 * gradient * gradient;
                                double correctedM = m[p] / (1 - Math.Pow(0.5, age));
                                double correctedV = v[p] / (1 - Math.Pow(0.75, age));
                                expected[p] = 0.98 * expected[p] - 0.1 * correctedM / (Math.Sqrt(correctedV) + 0.01);
                            }
                            double actual = Convert.ToDouble(parameters[p].ToArray()[0]);
                            Assert.Equal(expected[p], actual, typeof(T) == typeof(double) ? 10 : 5);
                        }
                        var checkpoint = Assert.IsType<FusedOptimizerCheckpoint>(plan.CaptureFusedOptimizerCheckpoint());
                        Assert.Equal(step + 1, checkpoint.OptimizerStep);
                        Assert.Equal(3, checkpoint.Parameters.Length);
                        Assert.Equal(ages, checkpoint.Parameters.Select(parameter => parameter.AcceptedUpdates).ToArray());
                        trajectory.Add(parameters.Select(parameter => Convert.ToDouble(parameter.ToArray()[0])).ToArray());
                    }
                    return trajectory.ToArray();
                }
                finally { plan.Dispose(); }
            }

            var uninterrupted = await Run(false);
            var restored = await Run(true);
            for (int step = 0; step < uninterrupted.Length; step++)
                Assert.Equal(uninterrupted[step], restored[step]);
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    private static Tensor<T> Scalar<T>(double value)
        => new(new[] { typeof(T) == typeof(float) ? (T)(object)(float)value : (T)(object)value }, new[] { 1 });

    private static CompiledTrainingPlan<T> Compile<T>(IEngine engine, Tensor<T>[] parameters, bool freezeFirst,
        bool permanentStop = false)
    {
        using var scope = GraphMode.EnableTraining(parameters);
        Tensor<T> first;
        if (permanentStop)
            first = scope.RecordBinary(LazyNodeType.Custom, "TensorMultiply", parameters[0], parameters[0], new[] { 1 },
                (e, output) => DirectGpuTensorEngine.CopyResultInto(e, e.TensorMultiply(parameters[0], parameters[0]), output));
        else if (freezeFirst)
        {
            using var noGrad = new NoGradScope<T>();
            first = engine.TensorMultiply(parameters[0], parameters[0]);
        }
        else first = engine.TensorMultiply(parameters[0], parameters[0]);
        var zero = engine.TensorSubtract(parameters[1], parameters[1]);
        var live = engine.TensorMultiply(parameters[2], parameters[2]);
        var loss = engine.ReduceSum(engine.TensorAdd(first, engine.TensorAdd(zero, live)), null, false);
        return scope.CompileTraining(parameters, loss);
    }

    [Fact]
    public async Task Permanent_null_backward_stays_distinct_from_no_grad_after_roundtrip()
    {
        var engine = new CpuEngine();
        var parameters = new[] { Scalar<float>(2), Scalar<float>(3), Scalar<float>(4) };
        using var plan = Compile(engine, parameters, freezeFirst: false, permanentStop: true);
        plan.ConfigureOptimizer(OptimizerType.AdamW, LrSchedule.Constant(0.1), weightDecay: 0.2);
        using var saved = new MemoryStream();
        await plan.SaveAsync(saved);
        saved.Position = 0;
        using var loaded = await CompiledPlanLoader.LoadTrainingAsync<float>(saved, engine, parameters);
        var concrete = Assert.IsType<CompiledTrainingPlan<float>>(loaded);
        concrete.Step();
        Assert.Equal(2f, parameters[0].ToArray()[0]);
        Assert.Equal(2.94f, parameters[1].ToArray()[0], 5);
        var forwardSteps = Assert.IsAssignableFrom<IReadOnlyList<CompiledStep<float>>>(concrete.ForwardStepsForSerialization);
        var first = Assert.Single(forwardSteps, step =>
            ReferenceEquals(step.Inputs.FirstOrDefault(), parameters[0]));
        Assert.False(first.RecordedUnderNoGrad);
        Assert.Null(first.BackwardFn);
        Assert.Equal(new long[] { 0, 1, 1 }, Assert.IsType<FusedOptimizerCheckpoint>(concrete.CaptureFusedOptimizerCheckpoint()).Parameters
            .Select(parameter => parameter.AcceptedUpdates).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Partial_operand_edges_preserve_direct_branch_gradients_across_repeated_roundtrips(bool sameOperandTwice)
    {
        var engine = new CpuEngine();
        var parameters = new[] { Scalar<float>(2), Scalar<float>(3) };
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var excluded = sameOperandTwice ? parameters[0] : engine.TensorMultiply(parameters[0], parameters[0]);
            var included = sameOperandTwice ? parameters[0] : parameters[1];
            var product = scope.RecordVariadic(LazyNodeType.Custom, "TensorMultiply", new[] { excluded, included }, new[] { 1 },
                (e, output) => DirectGpuTensorEngine.CopyResultInto(e, e.TensorMultiply(excluded, included), output),
                backwardFn: (gradient, inputs, _, _, e, gradients) =>
                    DifferentiableOps.AccumulateGrad(gradients, inputs[1], e.TensorMultiply(gradient, inputs[0]), e),
                differentiableInputIndices: new[] { 1 });
            plan = scope.CompileTraining(parameters, engine.ReduceSum(engine.TensorAdd(product, parameters[0]), null, false));
        }
        try
        {
            for (int roundtrip = 0; roundtrip < 3; roundtrip++)
            {
                float p0 = 2 + roundtrip, p1 = 3 + 2 * roundtrip;
                parameters[0].CopyFromArray(new[] { p0 });
                parameters[1].CopyFromArray(new[] { p1 });
                float expectedLoss = sameOperandTwice ? p0 * p0 + p0 : p0 * p0 * p1 + p0;
                Assert.Equal(expectedLoss, plan.Step().ToArray()[0]);
                // The excluded producer remains live through the direct p0 branch. Enrollment alone cannot
                // hide a wrongly reconstructed derivative: shared-upstream Multiply would report 13 instead of 1.
                Assert.Equal(sameOperandTwice ? p0 + 1 : 1f, plan.Gradients[0].ToArray()[0]);
                Assert.Equal(sameOperandTwice ? 0f : p0 * p0, plan.Gradients[1].ToArray()[0]);
                using var saved = new MemoryStream();
                await plan.SaveAsync(saved);
                saved.Position = 0;
                var loaded = await CompiledPlanLoader.LoadTrainingAsync<float>(saved, engine, parameters);
                Assert.NotNull(loaded);
                plan.Dispose();
                plan = Assert.IsType<CompiledTrainingPlan<float>>(loaded);
            }
        }
        finally { plan.Dispose(); }
    }

    [Fact]
    public async Task Composite_partial_edges_remain_correct_when_replay_expands_the_registered_operator()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var weight = new Tensor<float>(new[] { 2f, 0f, 1f, 3f }, new[] { 2, 2 });
        var bias = new Tensor<float>(new[] { 0.5f, -0.5f }, new[] { 2 });
        var parameters = new[] { x, weight, bias };
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var product = scope.RecordVariadic(LazyNodeType.Custom, "FusedLinear", parameters, new[] { 2, 2 },
                (e, output) => DirectGpuTensorEngine.CopyResultInto(e, e.TensorAdd(e.TensorMatMul(x, weight), bias), output),
                backwardFn: (gradient, inputs, _, _, e, gradients) => DifferentiableOps.AccumulateGrad(gradients,
                    inputs[1], e.TensorMatMul(e.TensorTranspose(inputs[0]), gradient), e),
                differentiableInputIndices: new[] { 1 });
            plan = scope.CompileTraining(parameters, engine.ReduceSum(engine.TensorAdd(product, x), null, false));
        }
        try
        {
            for (int roundtrip = 0; roundtrip < 3; roundtrip++)
            {
                float offset = roundtrip;
                x.CopyFromArray(new[] { 1 + offset, 2 + offset, 3 + offset, 4 + offset });
                Assert.Equal(42 + 16 * offset, plan.Step().ToArray()[0]);
                Assert.Equal(new[] { 1f, 1f, 1f, 1f }, plan.Gradients[0].ToArray());
                Assert.Equal(new[] { 4 + 2 * offset, 4 + 2 * offset, 6 + 2 * offset, 6 + 2 * offset }, plan.Gradients[1].ToArray());
                Assert.Equal(new[] { 0f, 0f }, plan.Gradients[2].ToArray());
                using var saved = new MemoryStream();
                await plan.SaveAsync(saved);
                saved.Position = 0;
                var loaded = await CompiledPlanLoader.LoadTrainingAsync<float>(saved, engine, parameters);
                Assert.NotNull(loaded);
                plan.Dispose();
                plan = Assert.IsType<CompiledTrainingPlan<float>>(loaded);
            }
        }
        finally { plan.Dispose(); }
    }

    [Fact]
    public async Task No_grad_in_place_write_preserves_prior_writer_gradient_across_roundtrip()
    {
        var engine = new CpuEngine();
        var parameters = new[] { Scalar<float>(2), Scalar<float>(3), Scalar<float>(4) };
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var prior = engine.TensorMultiply(parameters[0], parameters[0]);
            using (new NoGradScope<float>()) engine.TensorAddInPlace(prior, parameters[1]);
            plan = scope.CompileTraining(parameters,
                engine.ReduceSum(engine.TensorMultiply(prior, parameters[2]), null, false));
        }
        try
        {
            for (int roundtrip = 0; roundtrip < 2; roundtrip++)
            {
                Assert.Equal(28f, plan.Step().ToArray()[0]);
                Assert.Equal(16f, plan.Gradients[0].ToArray()[0]);
                Assert.Equal(0f, plan.Gradients[1].ToArray()[0]);
                Assert.Equal(7f, plan.Gradients[2].ToArray()[0]);
                using var saved = new MemoryStream();
                await plan.SaveAsync(saved);
                var bytes = saved.ToArray();
                // A no-grad write can carry its prior writer's effective identity edge without an own derivative.
                // Both known bits are valid together and must not be treated as contradictory flags.
                bytes[FindGradientFlags(bytes, "TensorAddInPlace")] = 3;
                RewriteFooter(bytes);
                using var serialized = new MemoryStream(bytes);
                var loaded = await CompiledPlanLoader.LoadTrainingAsync<float>(serialized, engine, parameters);
                Assert.NotNull(loaded);
                plan.Dispose();
                plan = Assert.IsType<CompiledTrainingPlan<float>>(loaded);
            }
        }
        finally { plan.Dispose(); }
    }

    [Fact]
    public async Task In_place_metadata_cannot_detach_the_mutation_target()
    {
        var engine = new CpuEngine();
        var parameters = new[] { Scalar<float>(2), Scalar<float>(3), Scalar<float>(4) };
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var prior = engine.TensorMultiply(parameters[0], parameters[0]);
            using (new NoGradScope<float>()) engine.TensorAddInPlace(prior, parameters[1]);
            plan = scope.CompileTraining(parameters,
                engine.ReduceSum(engine.TensorMultiply(prior, parameters[2]), null, false));
        }
        using (plan)
        using (var saved = new MemoryStream())
        {
            await plan.SaveAsync(saved);
            var original = saved.ToArray();
            int flagsOffset = FindGradientFlags(original, "TensorAddInPlace");
            int countOffset = flagsOffset + 1;
            // Corrupt the native all-input contract to include only the non-target input. The otherwise-known
            // flags request the no-grad prior-writer identity, so rejecting arbitrary flag bits cannot hide this bug.
            var bytes = original.Take(countOffset + sizeof(int)).Concat(BitConverter.GetBytes(1))
                .Concat(original.Skip(countOffset + sizeof(int))).ToArray();
            bytes[flagsOffset] = 3;
            BitConverter.GetBytes(1).CopyTo(bytes, countOffset);
            RewriteFooter(bytes);
            using var corrupted = new MemoryStream(bytes);
            var error = Assert.Throws<InvalidDataException>(() => TrainingPlanReader.Read(corrupted, engine, parameters));
            Assert.Contains("cannot exclude its mutation target", error.Message);
            Assert.Equal(new[] { 2f, 3f, 4f }, parameters.Select(parameter => parameter.ToArray()[0]).ToArray());
            Assert.Equal(28f, plan.Step().ToArray()[0]);
            Assert.Equal(16f, plan.Gradients[0].ToArray()[0]);
        }
    }

    [Theory]
    [InlineData(-1L)]
    [InlineData(3L)]
    public void Serialized_checkpoint_rejects_invalid_parameter_ages(long invalidAge)
    {
        var checkpoint = new FusedOptimizerCheckpoint
        {
            OptimizerStep = 2,
            Parameters = new[] { new FusedOptimizerParameterCheckpoint { AcceptedUpdates = 1 } },
        };
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
            FusedOptimizerCheckpointSerializer.Write(writer, checkpoint);
        var bytes = stream.ToArray();
        using (var valid = new BinaryReader(new MemoryStream(bytes)))
            Assert.Equal(1L, Assert.IsType<FusedOptimizerCheckpoint>(FusedOptimizerCheckpointSerializer.Read(valid)).Parameters[0].AcceptedUpdates);
        // The null L-BFGS history presence byte follows the final parameter state.
        int ageOffset = bytes.Length - (sizeof(byte) + sizeof(long) + 12 * sizeof(int));
        Assert.Equal(1L, BitConverter.ToInt64(bytes, ageOffset));
        BitConverter.GetBytes(invalidAge).CopyTo(bytes, ageOffset);
        Assert.Equal(invalidAge, BitConverter.ToInt64(bytes, ageOffset));
        using var corrupted = new MemoryStream(bytes);
        using var reader = new BinaryReader(corrupted);
        Assert.Throws<InvalidDataException>(() => FusedOptimizerCheckpointSerializer.Read(reader));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(int.MaxValue)]
    public void Serialized_checkpoint_rejects_invalid_parameter_counts_before_allocation(int invalidCount)
    {
        var checkpoint = new FusedOptimizerCheckpoint
        {
            OptimizerStep = 2,
            Parameters = new[] { new FusedOptimizerParameterCheckpoint { AcceptedUpdates = 1 } },
        };
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
            FusedOptimizerCheckpointSerializer.Write(writer, checkpoint);
        var bytes = stream.ToArray();
        using (var valid = new BinaryReader(new MemoryStream(bytes)))
            Assert.Single(Assert.IsType<FusedOptimizerCheckpoint>(FusedOptimizerCheckpointSerializer.Read(valid)).Parameters);
        int countOffset = bytes.Length - (sizeof(byte) + sizeof(int) + sizeof(long) + 12 * sizeof(int));
        Assert.Equal(1, BitConverter.ToInt32(bytes, countOffset));
        BitConverter.GetBytes(invalidCount).CopyTo(bytes, countOffset);
        Assert.Equal(invalidCount, BitConverter.ToInt32(bytes, countOffset));
        using var corrupted = new MemoryStream(bytes);
        using var reader = new BinaryReader(corrupted);
        Assert.Throws<InvalidDataException>(() => FusedOptimizerCheckpointSerializer.Read(reader));
    }

    [Theory]
    [InlineData((byte)8)]
    [InlineData((byte)4)]
    public async Task Invalid_gradient_flags_are_rejected_before_retrace_or_parameter_mutation(byte flags)
    {
        var engine = new CpuEngine();
        var parameters = new[] { Scalar<float>(2), Scalar<float>(3), Scalar<float>(4) };
        using var plan = Compile(engine, parameters, freezeFirst: true);
        using var saved = new MemoryStream();
        await plan.SaveAsync(saved);
        var bytes = saved.ToArray();
        bytes[FindFirstGradientFlags(bytes)] = flags;
        RewriteFooter(bytes);
        var before = parameters.Select(parameter => parameter.ToArray()).ToArray();
        using var corrupted = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => TrainingPlanReader.Read(corrupted, engine, parameters));
        for (int p = 0; p < parameters.Length; p++) Assert.Equal(before[p], parameters[p].ToArray());
    }

    [Theory]
    [InlineData(-2)]
    [InlineData(int.MaxValue)]
    public async Task Invalid_derivative_counts_are_rejected_before_retrace(int count)
    {
        var engine = new CpuEngine();
        var parameters = new[] { Scalar<float>(2), Scalar<float>(3), Scalar<float>(4) };
        using var plan = Compile(engine, parameters, freezeFirst: true);
        using var saved = new MemoryStream();
        await plan.SaveAsync(saved);
        var bytes = saved.ToArray();
        BitConverter.GetBytes(count).CopyTo(bytes, FindFirstGradientFlags(bytes) + 1);
        RewriteFooter(bytes);
        using var corrupted = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => TrainingPlanReader.Read(corrupted, engine, parameters));
        Assert.Equal(2f, parameters[0].ToArray()[0]);
    }

    [Theory]
    [InlineData(new[] { -1 })]
    [InlineData(new[] { 2 })]
    [InlineData(new[] { 0, 0 })]
    [InlineData(new[] { 1, 0 })]
    public async Task Invalid_derivative_indices_are_rejected_before_retrace(int[] indices)
    {
        var engine = new CpuEngine();
        var parameters = new[] { Scalar<float>(2), Scalar<float>(3), Scalar<float>(4) };
        using var plan = Compile(engine, parameters, freezeFirst: true);
        using var saved = new MemoryStream();
        await plan.SaveAsync(saved);
        var original = saved.ToArray();
        int countOffset = FindFirstGradientFlags(original) + 1;
        var payload = indices.SelectMany(BitConverter.GetBytes).ToArray();
        var bytes = original.Take(countOffset + sizeof(int)).Concat(payload)
            .Concat(original.Skip(countOffset + sizeof(int))).ToArray();
        BitConverter.GetBytes(indices.Length).CopyTo(bytes, countOffset);
        RewriteFooter(bytes);
        using var corrupted = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => TrainingPlanReader.Read(corrupted, engine, parameters));
        Assert.Equal(2f, parameters[0].ToArray()[0]);
    }

    [Theory]
    [InlineData(OptimizerType.LBFGS)]
    [InlineData(OptimizerType.TrustRegion)]
    [InlineData(OptimizerType.ConjugateGradient)]
    [InlineData(OptimizerType.HypergradientSGD)]
    [InlineData(OptimizerType.DAdaptationSGD)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    public async Task Coupled_optimizer_checkpoint_resumes_the_same_canonical_subspace(OptimizerType optimizer)
    {
        var engine = new CpuEngine();
        var parameters = new[] { Scalar<float>(2), Scalar<float>(3), Scalar<float>(4) };
        using var original = Compile(engine, parameters, freezeFirst: true);
        original.ConfigureOptimizer(optimizer, LrSchedule.Constant(0.02));
        original.Step();
        original.Step();
        var checkpoint = Assert.IsType<FusedOptimizerCheckpoint>(original.CaptureFusedOptimizerCheckpoint());
        Assert.Equal(new[] { false, true, true }, checkpoint.FixedActiveDomain);
        if (optimizer == OptimizerType.LBFGS)
        {
            var history = Assert.IsType<FusedLbfgsHistoryCheckpoint>(checkpoint.LbfgsHistory);
            Assert.True(history.HasPrev);
            Assert.True(history.Count > 0);
            Assert.True(history.GammaDen > 0);
            int validCount = history.Count;
            history.Count = history.S.Length + 1;
            using (var malformed = new MemoryStream())
            using (var writer = new BinaryWriter(malformed))
            {
                Assert.Throws<InvalidDataException>(() => FusedOptimizerCheckpointSerializer.Write(writer, checkpoint));
                Assert.Equal(0, malformed.Length);
            }
            Assert.Equal(validCount, Assert.IsType<FusedLbfgsHistoryCheckpoint>(Assert.IsType<FusedOptimizerCheckpoint>(original.CaptureFusedOptimizerCheckpoint()).LbfgsHistory).Count);
            history.Count = validCount;
            float[] validVector = history.S[0];
            history.S[0] = Array.Empty<float>();
            using (var malformed = new MemoryStream())
            using (var writer = new BinaryWriter(malformed))
            {
                Assert.Throws<InvalidDataException>(() => FusedOptimizerCheckpointSerializer.Write(writer, checkpoint));
                Assert.Equal(0, malformed.Length);
            }
            // Captured arrays are independent: attempted corruption cannot change the running optimizer.
            Assert.Equal(validVector, Assert.IsType<FusedLbfgsHistoryCheckpoint>(Assert.IsType<FusedOptimizerCheckpoint>(original.CaptureFusedOptimizerCheckpoint()).LbfgsHistory).S[0]);
            history.S[0] = validVector;
        }
        using var saved = new MemoryStream();
        await original.SaveAsync(saved);
        var restoredParameters = parameters.Select(parameter => parameter.Clone()).ToArray();
        saved.Position = 0;
        using var loaded = await CompiledPlanLoader.LoadTrainingAsync<float>(saved, engine, restoredParameters);
        var restored = Assert.IsType<CompiledTrainingPlan<float>>(loaded);
        Assert.Equal(new[] { false, true, true }, Assert.IsType<FusedOptimizerCheckpoint>(restored.CaptureFusedOptimizerCheckpoint()).FixedActiveDomain);
        for (int step = 0; step < 2; step++)
        {
            original.Step();
            restored.Step();
            Assert.Equal(2f, parameters[0].ToArray()[0]);
            Assert.Equal(2f, restoredParameters[0].ToArray()[0]);
            for (int p = 0; p < parameters.Length; p++)
            {
                Assert.Equal(parameters[p].ToArray(), restoredParameters[p].ToArray());
            }
            Assert.Equal(original.ExportOptimizerState(), restored.ExportOptimizerState());
        }
        // A retraced successor hands over the same canonical tensor objects. The separately
        // restored plan remains the independent trajectory reference after the handover.
        using var continued = Compile(engine, parameters, freezeFirst: true);
        continued.ConfigureOptimizer(optimizer, LrSchedule.Constant(0.02));
        continued.ContinueOptimizerFrom(original);
        Assert.Equal(original.ExportOptimizerState(), continued.ExportOptimizerState());
        for (int step = 0; step < 2; step++)
        {
            continued.Step();
            restored.Step();
            Assert.Equal(2f, parameters[0].ToArray()[0]);
            for (int p = 0; p < parameters.Length; p++)
                Assert.Equal(parameters[p].ToArray(), restoredParameters[p].ToArray());
            Assert.Equal(continued.ExportOptimizerState(), restored.ExportOptimizerState());
        }
    }

    [Theory]
    [InlineData(OptimizerType.LBFGS)]
    [InlineData(OptimizerType.TrustRegion)]
    [InlineData(OptimizerType.ConjugateGradient)]
    [InlineData(OptimizerType.HypergradientSGD)]
    [InlineData(OptimizerType.DAdaptationSGD)]
    [InlineData(OptimizerType.ScheduleFreeSGD)]
    public void Coupled_optimizer_domain_mismatch_leaves_weights_and_existing_state_bitwise_unchanged(OptimizerType optimizer)
    {
        var engine = new CpuEngine();
        var parameters = new[] { Scalar<float>(2), Scalar<float>(3), Scalar<float>(4) };
        using var source = Compile(engine, parameters, freezeFirst: true);
        source.ConfigureOptimizer(optimizer, LrSchedule.Constant(0.02));
        source.Step();
        source.Step();
        using var target = Compile(engine, parameters, freezeFirst: false);
        target.ConfigureOptimizer(optimizer, LrSchedule.Constant(0.02));
        target.Step();
        var weights = parameters.Select(parameter => parameter.ToArray()).ToArray();
        var targetState = Assert.IsType<byte[]>(target.ExportOptimizerState());
        var sourceState = Assert.IsType<byte[]>(source.ExportOptimizerState());

        Assert.Throws<InvalidDataException>(() => target.ImportOptimizerState(sourceState));
        Assert.Equal(targetState, target.ExportOptimizerState());
        Assert.Equal(sourceState, source.ExportOptimizerState());
        for (int p = 0; p < parameters.Length; p++) Assert.Equal(weights[p], parameters[p].ToArray());

        Assert.Throws<InvalidOperationException>(() => target.ContinueOptimizerFrom(source));
        Assert.Equal(targetState, target.ExportOptimizerState());
        Assert.Equal(sourceState, source.ExportOptimizerState());
        for (int p = 0; p < parameters.Length; p++) Assert.Equal(weights[p], parameters[p].ToArray());
    }

    [Theory]
    [InlineData(-2)]
    [InlineData(int.MaxValue)]
    [InlineData(1)]
    public void Serialized_fixed_domain_rejects_invalid_widths(int invalidCount)
    {
        var checkpoint = DomainCheckpoint();
        var bytes = Serialize(checkpoint);
        using (var valid = new BinaryReader(new MemoryStream(bytes)))
            Assert.Equal(checkpoint.FixedActiveDomain, Assert.IsType<FusedOptimizerCheckpoint>(FusedOptimizerCheckpointSerializer.Read(valid)).FixedActiveDomain);
        int countOffset = FixedDomainOffset(bytes, checkpoint);
        BitConverter.GetBytes(invalidCount).CopyTo(bytes, countOffset);
        using var reader = new BinaryReader(new MemoryStream(bytes));
        Assert.Throws<InvalidDataException>(() => FusedOptimizerCheckpointSerializer.Read(reader));
    }

    [Fact]
    public void Serialized_fixed_domain_rejects_non_boolean_entries()
    {
        var checkpoint = DomainCheckpoint();
        var bytes = Serialize(checkpoint);
        bytes[FixedDomainOffset(bytes, checkpoint) + sizeof(int)] = 2;
        using var reader = new BinaryReader(new MemoryStream(bytes));
        Assert.Throws<InvalidDataException>(() => FusedOptimizerCheckpointSerializer.Read(reader));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Fixed_domain_presence_must_match_optimizer_coupling_before_writing(bool coupled)
    {
        var checkpoint = DomainCheckpoint();
        if (coupled) checkpoint.FixedActiveDomain = null;
        else checkpoint.OptimizerType = OptimizerType.AdamW;
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        Assert.Throws<InvalidDataException>(() => FusedOptimizerCheckpointSerializer.Write(writer, checkpoint));
        Assert.Equal(0, stream.Length);
    }

    private static FusedOptimizerCheckpoint DomainCheckpoint() => new()
    {
        OptimizerType = OptimizerType.ConjugateGradient,
        OptimizerStep = 2,
        FixedActiveDomain = new[] { false, true },
        Parameters = new[] { new FusedOptimizerParameterCheckpoint(), new FusedOptimizerParameterCheckpoint { AcceptedUpdates = 2 } },
    };

    private static byte[] Serialize(FusedOptimizerCheckpoint checkpoint)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
            FusedOptimizerCheckpointSerializer.Write(writer, checkpoint);
        return stream.ToArray();
    }

    private static int FixedDomainOffset(byte[] bytes, FusedOptimizerCheckpoint checkpoint)
        => bytes.Length - (checkpoint.Parameters.Length * (sizeof(long) + 12 * sizeof(int))
            + sizeof(int) + Assert.IsType<bool[]>(checkpoint.FixedActiveDomain).Length + sizeof(int));

    private static int FindFirstGradientFlags(byte[] bytes)
        => FindGradientFlags(bytes, null);

    private static int FindGradientFlags(byte[] bytes, string? operation)
    {
        using (var layout = new MemoryStream(bytes))
        using (var reader = new BinaryReader(layout))
        {
            layout.Position = sizeof(uint) + sizeof(ushort) + 2 * sizeof(byte) + sizeof(int);
            int rank = reader.ReadInt32();
            layout.Position += rank * sizeof(int);
            reader.ReadInt32(); // codec version
            int fingerprintLength = reader.ReadInt32();
            layout.Position += fingerprintLength;
            var tensors = TensorTableReader.Read<float>(reader);
            int count = reader.ReadInt32();
            for (int i = 0; i < count; i++)
            {
                reader.ReadByte();
                int nameLength = reader.ReadInt32();
                string name = System.Text.Encoding.UTF8.GetString(reader.ReadBytes(nameLength));
                int flags = checked((int)layout.Position);
                if (operation is null || operation == name) return flags;
                reader.ReadByte();
                int derivativeCount = reader.ReadInt32();
                if (derivativeCount >= 0) layout.Position += derivativeCount * sizeof(int);
                int inputs = reader.ReadInt32();
                layout.Position += (inputs + 1) * sizeof(int);
                SavedStateSerializer.Read<float>(reader, tensors);
            }
            throw new InvalidOperationException("The expected serialized operation was not found.");
        }
    }

    private static void RewriteFooter(byte[] bytes)
    {
        BitConverter.GetBytes((long)bytes.Length - 16).CopyTo(bytes, bytes.Length - 16);
        ulong checksum = XXHash64.Compute(bytes, 0, bytes.Length - 16);
        BitConverter.GetBytes(checksum).CopyTo(bytes, bytes.Length - sizeof(ulong));
    }
}
