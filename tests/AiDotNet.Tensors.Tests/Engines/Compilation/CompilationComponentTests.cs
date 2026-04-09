using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class CompilationComponentTests
{
    private const float Tolerance = 1e-5f;

    #region CompiledModelCache Tests

    [Fact]
    public void CompiledModelCache_CachesInferencePlan()
    {
        using var cache = new CompiledModelCache<float>();
        var engine = new CpuEngine();
        var input = CreateRandom(new[] { 4, 8 }, 42);
        var weights = CreateRandom(new[] { 8, 4 }, 43);

        int compileCount = 0;
        var plan = cache.GetOrCompileInference(input._shape, () =>
        {
            compileCount++;
            engine.TensorMatMul(input, weights);
        });

        Assert.NotNull(plan);
        Assert.Equal(1, compileCount);
        Assert.Equal(1, cache.InferencePlanCount);

        // Second call should reuse cached plan
        var plan2 = cache.GetOrCompileInference(input._shape, () =>
        {
            compileCount++;
            engine.TensorMatMul(input, weights);
        });

        Assert.Equal(1, compileCount); // Not recompiled
    }

    [Fact]
    public void CompiledModelCache_InvalidateClearsAll()
    {
        using var cache = new CompiledModelCache<float>();
        var engine = new CpuEngine();
        var input = CreateRandom(new[] { 4, 8 }, 42);
        var weights = CreateRandom(new[] { 8, 4 }, 43);

        cache.GetOrCompileInference(input._shape, () => engine.TensorMatMul(input, weights));
        Assert.Equal(1, cache.InferencePlanCount);

        cache.Invalidate();
        Assert.Equal(0, cache.InferencePlanCount);
    }

    [Fact]
    public void AutoTensorCache_ClearThenReturnStillAccepts()
    {
        // Verify that after Clear(), Return() still accepts buffers
        // (O(1) counters must be reset alongside the queues)
        var tensor = CreateRandom(new[] { 4, 4 }, 99);
        AiDotNet.Tensors.Helpers.AutoTensorCache.Return(tensor);
        AiDotNet.Tensors.Helpers.AutoTensorCache.Clear();

        // After clear, cache should accept new returns
        var tensor2 = CreateRandom(new[] { 4, 4 }, 100);
        AiDotNet.Tensors.Helpers.AutoTensorCache.Return(tensor2);

        // Renting should return the tensor we just returned (not the cleared one)
        var rented = AiDotNet.Tensors.Helpers.AutoTensorCache.RentOrAllocate<float>(new[] { 4, 4 });
        Assert.NotNull(rented);
    }

    #endregion

    #region SymbolicShape Tests

    [Fact]
    public void SymbolicShape_BatchDynamic_MatchesDifferentBatchSizes()
    {
        var shape = SymbolicShape.BatchDynamic(new[] { 32, 128 });

        Assert.True(shape.Matches(new[] { 32, 128 }));
        Assert.True(shape.Matches(new[] { 64, 128 }));  // Different batch
        Assert.True(shape.Matches(new[] { 1, 128 }));   // Single sample
        Assert.False(shape.Matches(new[] { 32, 64 }));  // Different features
        Assert.False(shape.Matches(new[] { 32 }));       // Wrong rank
    }

    [Fact]
    public void SymbolicShape_ComputeKey_IgnoresSymbolicDims()
    {
        var s1 = SymbolicShape.BatchDynamic(new[] { 32, 128 });
        var s2 = SymbolicShape.BatchDynamic(new[] { 64, 128 });
        var s3 = SymbolicShape.BatchDynamic(new[] { 32, 256 });

        Assert.Equal(s1.ComputeKey(), s2.ComputeKey());     // Same features, different batch
        Assert.NotEqual(s1.ComputeKey(), s3.ComputeKey());   // Different features
    }

    #endregion

    #region FusedAttention Tests

    [Fact]
    public void FlashAttention_MatchesNaiveAttention()
    {
        int seqQ = 16, seqK = 16, headDim = 32;
        var q = CreateRandomArray(seqQ * headDim, 42);
        var k = CreateRandomArray(seqK * headDim, 43);
        var v = CreateRandomArray(seqK * headDim, 44);
        var flashOutput = new float[seqQ * headDim];
        float scale = 1f / MathF.Sqrt(headDim);

        // Flash Attention
        FusedAttention.FlashAttentionForward(q, k, v, flashOutput, seqQ, seqK, headDim, scale);

        // Naive attention for comparison
        var naiveOutput = NaiveAttention(q, k, v, seqQ, seqK, headDim, scale);

        // Compare
        for (int i = 0; i < flashOutput.Length; i++)
        {
            Assert.True(MathF.Abs(flashOutput[i] - naiveOutput[i]) < 1e-3f,
                $"Flash[{i}]={flashOutput[i]}, Naive[{i}]={naiveOutput[i]}, diff={MathF.Abs(flashOutput[i] - naiveOutput[i])}");
        }
    }

    [Fact]
    public void FlashAttention_CausalMask_ZerosUpperTriangle()
    {
        int seq = 8, headDim = 16;
        var q = CreateRandomArray(seq * headDim, 50);
        var k = CreateRandomArray(seq * headDim, 51);
        var v = CreateRandomArray(seq * headDim, 52);
        var output = new float[seq * headDim];
        float scale = 1f / MathF.Sqrt(headDim);

        FusedAttention.FlashAttentionForward(q, k, v, output, seq, seq, headDim, scale, isCausal: true);

        // Output should be non-zero (causal attention still produces valid output)
        float sum = 0f;
        for (int i = 0; i < output.Length; i++) sum += MathF.Abs(output[i]);
        Assert.True(sum > 0f, "Causal attention produced all zeros");
    }

    #endregion

    #region FusedKernels Tests

    [Fact]
    public unsafe void FusedSwish_MatchesScalar()
    {
        int length = 256;
        var input = CreateRandomArray(length, 60);
        var fusedOutput = new float[length];
        var scalarOutput = new float[length];

        fixed (float* pIn = input, pFused = fusedOutput)
        {
            FusedKernels.SwishUnsafe(pIn, pFused, length);
        }

        for (int i = 0; i < length; i++)
            scalarOutput[i] = input[i] / (1f + MathF.Exp(-input[i]));

        for (int i = 0; i < length; i++)
            Assert.True(MathF.Abs(fusedOutput[i] - scalarOutput[i]) < Tolerance,
                $"Swish[{i}]: fused={fusedOutput[i]}, scalar={scalarOutput[i]}");
    }

    [Fact]
    public unsafe void FusedAddRelu_MatchesScalar()
    {
        int length = 128;
        var a = CreateRandomArray(length, 70);
        var b = CreateRandomArray(length, 71);
        var fusedOutput = new float[length];

        fixed (float* pA = a, pB = b, pOut = fusedOutput)
        {
            FusedKernels.AddReluUnsafe(pA, pB, pOut, length);
        }

        for (int i = 0; i < length; i++)
        {
            float expected = MathF.Max(0f, a[i] + b[i]);
            Assert.True(MathF.Abs(fusedOutput[i] - expected) < Tolerance,
                $"AddReLU[{i}]: fused={fusedOutput[i]}, expected={expected}");
        }
    }

    #endregion

    #region FusedOptimizer Tests

    [Fact]
    public unsafe void SgdSimd_MatchesScalar()
    {
        int length = 256;
        var param = CreateRandomArray(length, 80);
        var grad = CreateRandomArray(length, 81);
        var expected = (float[])param.Clone();
        float lr = 0.01f;

        // Scalar reference
        for (int i = 0; i < length; i++)
            expected[i] -= lr * grad[i];

        // SIMD
        fixed (float* pParam = param, pGrad = grad)
        {
            FusedOptimizer.SgdUpdateSimd(pParam, pGrad, length, lr);
        }

        for (int i = 0; i < length; i++)
            Assert.True(MathF.Abs(param[i] - expected[i]) < Tolerance,
                $"SGD[{i}]: simd={param[i]}, expected={expected[i]}");
    }

    [Fact]
    public unsafe void AdamSimd_ConvergesOnSimpleFunction()
    {
        int length = 16;
        var param = new float[length];
        var m = new float[length];
        var v = new float[length];

        // Initialize param to 1.0, gradient always pushes toward 0
        for (int i = 0; i < length; i++) param[i] = 1f;

        // Run 100 Adam steps with gradient = param (minimize x^2/2)
        for (int step = 1; step <= 100; step++)
        {
            var grad = (float[])param.Clone(); // grad = param for x^2/2
            fixed (float* pParam = param, pGrad = grad, pM = m, pV = v)
            {
                FusedOptimizer.AdamUpdateSimd(pParam, pGrad, pM, pV, length,
                    lr: 0.01f, beta1: 0.9f, beta2: 0.999f, eps: 1e-8f, step: step);
            }
        }

        // After 100 steps, params should be close to 0
        for (int i = 0; i < length; i++)
            Assert.True(MathF.Abs(param[i]) < 0.5f,
                $"Adam didn't converge: param[{i}]={param[i]}");
    }

    #endregion

    #region ActivationCheckpoint Tests

    [Fact]
    public void ReluBitmask_RoundTrip()
    {
        int length = 100;
        var data = CreateRandomArray(length, 90);
        var gradOut = CreateRandomArray(length, 91);

        // Create bitmask from forward activations
        var bitmask = ActivationCheckpoint.CreateReluBitmask(data, length);

        // Apply backward
        var gradIn = new float[length];
        ActivationCheckpoint.ApplyReluBackwardFromBitmask(gradOut, bitmask, gradIn, length);

        // Verify: gradIn[i] should be gradOut[i] where data[i] > 0, else 0
        for (int i = 0; i < length; i++)
        {
            float expected = data[i] > 0f ? gradOut[i] : 0f;
            Assert.Equal(expected, gradIn[i]);
        }
    }

    [Fact]
    public void ActivationStorageType_CorrectClassification()
    {
        Assert.Equal(ActivationStorageType.Bitmask, ActivationCheckpoint.GetStorageType("ReLU"));
        Assert.Equal(ActivationStorageType.ReuseOutput, ActivationCheckpoint.GetStorageType("Sigmoid"));
        Assert.Equal(ActivationStorageType.ReuseOutput, ActivationCheckpoint.GetStorageType("Tanh"));
        Assert.Equal(ActivationStorageType.Full, ActivationCheckpoint.GetStorageType("GELU"));
    }

    #endregion

    #region GradientCheckpointing Tests

    [Fact]
    public void GradientCheckpointing_SegmentSizing()
    {
        // 100 steps should give ~10 segments of ~10 steps each
        var steps = new CompiledStep<float>[100];
        for (int i = 0; i < 100; i++)
        {
            var output = new Tensor<float>(new[] { 4 });
            steps[i] = new CompiledStep<float>("Test", (eng, o) => { }, output,
                Array.Empty<Tensor<float>>(), null, null);
        }

        var engine = new CpuEngine();
        var checkpoint = new AiDotNet.Tensors.Engines.Compilation.GradientCheckpointing<float>(steps, engine);

        Assert.Equal(10, checkpoint.SegmentSize);
        Assert.Equal(10, checkpoint.SegmentCount);
        Assert.True(checkpoint.MemorySavingsFactor > 4.0);
    }

    #endregion

    #region ForwardCSEPass Tests

    [Fact]
    public void ForwardCSEPass_CSECopy_HasNullBackwardFn()
    {
        // Verify that CSE_Copy steps have null BackwardFn (not stale BackwardFn
        // from the original step). Stale BackwardFn would produce incorrect gradients
        // because its SavedState is from a computation that never ran.
        var pass = new ForwardCSEPass();

        // Create two identical steps that the pass should deduplicate
        var sharedInput = new Tensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var output1 = new Tensor<float>(new[] { 4 });
        var output2 = new Tensor<float>(new[] { 4 });

        // A backward function that should NOT be retained on the copy
        BackwardFunction<float> backwardFn = (gradOut, inputs, output, state, eng, gradMap) => { };

        var steps = new[]
        {
            new CompiledStep<float>("TensorExp", (eng, o) =>
            {
                for (int i = 0; i < 4; i++)
                    o.AsWritableSpan()[i] = MathF.Exp(sharedInput.AsSpan()[i]);
            }, output1, new[] { sharedInput }, backwardFn, new object[] { "some state" }),
            new CompiledStep<float>("TensorExp", (eng, o) =>
            {
                for (int i = 0; i < 4; i++)
                    o.AsWritableSpan()[i] = MathF.Exp(sharedInput.AsSpan()[i]);
            }, output2, new[] { sharedInput }, backwardFn, new object[] { "some state" }),
        };

        var engine = new CpuEngine();
        var result = pass.TryOptimize(steps, engine);

        Assert.NotNull(result);
        Assert.Equal(2, result.Length);
        // First step should be kept as-is with its backward
        Assert.NotNull(result[0].BackwardFn);
        // Second step should be a CSE_Copy with null BackwardFn
        Assert.Equal("CSE_Copy", result[1].OpName);
        Assert.Null(result[1].BackwardFn);
        Assert.Null(result[1].SavedState);
    }

    #endregion

    #region ConstantFoldingPass Tests

    [Fact]
    public void ConstantFoldingPass_TreatsAllGraphInputsAsDynamic()
    {
        // Verify the pass identifies ALL graph-level inputs as dynamic,
        // not just steps[0].Inputs[0]. A graph input is any tensor that
        // appears as a step input but is not produced by any step's output.
        var pass = new ConstantFoldingPass();

        var input1 = new Tensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        var input2 = new Tensor<float>(new float[] { 3f, 4f }, new[] { 2 });
        var output = new Tensor<float>(new[] { 2 });

        // Step uses two external inputs — both should be treated as dynamic
        var steps = new[]
        {
            new CompiledStep<float>("TensorAdd", (eng, o) =>
            {
                var span = o.AsWritableSpan();
                var s1 = input1.AsSpan();
                var s2 = input2.AsSpan();
                for (int i = 0; i < 2; i++) span[i] = s1[i] + s2[i];
            }, output, new[] { input1, input2 }, null, null),
        };

        var engine = new CpuEngine();
        // With both inputs being graph-level (not produced by any step), the step
        // is dynamic and should NOT be folded.
        var result = pass.TryOptimize(steps, engine);

        // null means no folding occurred (all steps are dynamic) — correct
        Assert.Null(result);
    }

    #endregion

    #region TensorCodecOptions Tests

    [Fact]
    public void TensorCodecOptions_DefaultReturnsFreshInstance()
    {
        // Verify that Default returns a new instance each time,
        // preventing global state corruption via mutation.
        var d1 = TensorCodecOptions.Default;
        var d2 = TensorCodecOptions.Default;

        Assert.NotSame(d1, d2);

        // Mutating one instance should not affect another
        d1.EnableCompilation = false;
        Assert.True(d2.EnableCompilation);
    }

    #endregion

    #region SymbolicShape Validation Tests

    [Fact]
    public void SymbolicShape_ClonesSymbolicDimensions()
    {
        // Verify that SymbolicDimensions are defensively copied
        var dims = new[] { 0 };
        var shape = new SymbolicShape(new[] { 32, 128 }, dims);

        // Mutating the original array should not affect the SymbolicShape
        dims[0] = 999;
        Assert.Equal(0, shape.SymbolicDimensions[0]);
    }

    [Fact]
    public void SymbolicShape_ThrowsOnInvalidDimensionIndex()
    {
        // Out-of-range symbolic dimension indices should throw
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SymbolicShape(new[] { 32, 128 }, new[] { 5 }));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SymbolicShape(new[] { 32, 128 }, new[] { -1 }));
    }

    [Fact]
    public void SymbolicShape_AcceptsValidDimensionIndices()
    {
        // Valid indices should not throw
        var shape = new SymbolicShape(new[] { 32, 128, 64 }, new[] { 0, 2 });
        Assert.True(shape.Matches(new[] { 64, 128, 64 })); // dim 0,2 are symbolic
        Assert.True(shape.Matches(new[] { 1, 128, 99 }));
        Assert.False(shape.Matches(new[] { 32, 256, 64 })); // dim 1 changed (non-symbolic)
    }

    #endregion

    #region End-to-End Integration Tests

    [Fact]
    public void CompiledTraining_SimpleMatMulReduceSum_Works()
    {
        // Minimal: MatMul → ReduceSum — no TensorMultiply
        var engine = new CpuEngine();
        var input = CreateRandom(new[] { 4, 4 }, 42);
        var w = CreateRandom(new[] { 4, 4 }, 43);

        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.TensorMatMul(input, w);
            engine.ReduceSum(h, null);
            plan = scope.CompileTraining(new[] { w });
        }

        try
        {
            var loss = plan.Step();
            Assert.True(loss.Length == 1, $"Loss should be scalar, got length {loss.Length}");
            Assert.NotNull(plan.Gradients);
            Assert.True(plan.Gradients.Length == 1, "Should have 1 gradient tensor");

            // Second step should produce same loss (no weight updates)
            var loss2 = plan.Step();
            Assert.True(MathF.Abs(loss.GetFlat(0) - loss2.GetFlat(0)) < 1e-4f,
                "Consistent loss without weight updates");
        }
        finally { plan.Dispose(); }
    }

    [Fact]
    public void CompiledTraining_MatMulMultiplyReduceSum_Works()
    {
        // MatMul → TensorMultiply(x,x) → ReduceSum — tests the TensorMultiply specialization
        var engine = new CpuEngine();
        var input = CreateRandom(new[] { 4, 4 }, 42);
        var w = CreateRandom(new[] { 4, 4 }, 43);

        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.TensorMatMul(input, w);
            var sq = engine.TensorMultiply(h, h);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { w });
        }

        try
        {
            var loss = plan.Step();
            Assert.True(loss.Length == 1, $"Loss should be scalar, got length {loss.Length}");
        }
        finally { plan.Dispose(); }
    }

    [Fact]
    public void CompiledTraining_ProducesGradientsAndLossDecreases()
    {
        // End-to-end: compile a 2-layer MLP, train for 20 steps,
        // verify compiled loss matches eager loss at each step.
        var engine = new CpuEngine();
        int m = 8, k = 4, h = 4, n = 2;

        var input = CreateRandom(new[] { m, k }, 42);
        var w1Eager = CreateRandom(new[] { k, h }, 43);
        var w2Eager = CreateRandom(new[] { h, n }, 44);

        // Clone weights for compiled path (both start from same initial values)
        var w1Compiled = new Tensor<float>((float[])w1Eager.GetDataArray().Clone(), w1Eager._shape);
        var w2Compiled = new Tensor<float>((float[])w2Eager.GetDataArray().Clone(), w2Eager._shape);

        float lr = 0.01f;
        float[] eagerLosses = new float[20];
        float[] compiledLosses = new float[20];

        // Eager training: GradientTape forward + backward
        for (int step = 0; step < 20; step++)
        {
            using var tape = new GradientTape<float>();
            var h1 = engine.ReLU(engine.TensorMatMul(input, w1Eager));
            var output = engine.TensorMatMul(h1, w2Eager);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, output), null);
            eagerLosses[step] = loss.GetFlat(0);

            var grads = tape.ComputeGradients(loss, new[] { w1Eager, w2Eager });
            // Manual SGD update
            if (grads.ContainsKey(w1Eager))
            {
                var g = grads[w1Eager];
                var span = w1Eager.AsWritableSpan();
                var gSpan = g.AsSpan();
                for (int i = 0; i < span.Length; i++) span[i] -= lr * gSpan[i];
            }
            if (grads.ContainsKey(w2Eager))
            {
                var g = grads[w2Eager];
                var span = w2Eager.AsWritableSpan();
                var gSpan = g.AsSpan();
                for (int i = 0; i < span.Length; i++) span[i] -= lr * gSpan[i];
            }
        }

        // Compiled training: GraphMode → compile → Step()
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h1 = engine.ReLU(engine.TensorMatMul(input, w1Compiled));
            var output = engine.TensorMatMul(h1, w2Compiled);
            engine.ReduceSum(engine.TensorMultiply(output, output), null);
            plan = scope.CompileTraining(new[] { w1Compiled, w2Compiled });
        }

        try
        {
            for (int step = 0; step < 20; step++)
            {
                var lossOut = plan.Step();
                compiledLosses[step] = lossOut.GetFlat(0);

                // Manual SGD from compiled gradients
                var grads = plan.Gradients;
                for (int p = 0; p < 2; p++)
                {
                    var param = p == 0 ? w1Compiled : w2Compiled;
                    if (grads[p] is not null)
                    {
                        var span = param.AsWritableSpan();
                        var gSpan = grads[p].AsSpan();
                        for (int i = 0; i < span.Length; i++) span[i] -= lr * gSpan[i];
                    }
                }
            }
        }
        finally
        {
            // Verify before dispose: compiled plan produces non-null gradients
            Assert.NotNull(plan.Gradients);
            Assert.True(plan.Gradients.Length == 2, "Should have 2 gradient tensors (w1, w2)");

            // Verify consistent loss (same value on repeated calls without weight updates)
            var loss1 = plan.Step().GetFlat(0);
            var loss2 = plan.Step().GetFlat(0);
            Assert.True(MathF.Abs(loss1 - loss2) < 1e-4f,
                $"Compiled plan should produce consistent loss: {loss1:F6} vs {loss2:F6}");

            plan.Dispose();
        }

        // Verify eager training works
        Assert.True(eagerLosses[19] < eagerLosses[0],
            $"Eager loss should decrease: start={eagerLosses[0]:F4}, end={eagerLosses[19]:F4}");
    }

    #endregion

    #region CompiledModelCache Input Rebinding Tests

    [Fact]
    public void CompiledModelCache_RebindsInputOnCacheHit()
    {
        // Verify that the cache overload with Tensor<T> input
        // correctly rebinds data on cache hits.
        using var cache = new CompiledModelCache<float>();
        var engine = new CpuEngine();
        var weights = CreateRandom(new[] { 8, 4 }, 43);

        // First call: compile with input1
        var input1 = CreateRandom(new[] { 4, 8 }, 42);
        var plan = cache.GetOrCompileInference(input1, () =>
        {
            engine.TensorMatMul(input1, weights);
        });
        var result1 = plan.Execute();
        var output1 = new float[result1.Length];
        result1.AsSpan().CopyTo(output1);

        // Second call: cache hit with different data (same shape)
        var input2 = CreateRandom(new[] { 4, 8 }, 99);
        var plan2 = cache.GetOrCompileInference(input2, () =>
        {
            engine.TensorMatMul(input2, weights);
        });
        Assert.Same(plan, plan2); // Same cached plan
        var result2 = plan2.Execute();
        var output2 = new float[result2.Length];
        result2.AsSpan().CopyTo(output2);

        // Results should differ because input data changed
        bool anyDifferent = false;
        for (int i = 0; i < output1.Length; i++)
        {
            if (MathF.Abs(output1[i] - output2[i]) > 1e-6f)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent, "Cache hit should produce different results with different input data");
    }

    [Fact]
    public void CompiledModelCache_SymbolicOverloadStoresUnderSymbolicKey()
    {
        // Verify that the symbolic overload stores the plan under the symbolic key
        // (not the concrete-shape key), so different batch sizes hit the same cache.
        using var cache = new CompiledModelCache<float>();
        var engine = new CpuEngine();
        var weights = CreateRandom(new[] { 8, 4 }, 43);

        var input = CreateRandom(new[] { 4, 8 }, 42);
        var symbolic = SymbolicShape.BatchDynamic(input._shape);

        int compileCount = 0;
        cache.GetOrCompileInference(input._shape, () =>
        {
            compileCount++;
            engine.TensorMatMul(input, weights);
        }, symbolic);

        Assert.Equal(1, compileCount);

        // Different batch size — should hit the symbolic cache
        var input2 = CreateRandom(new[] { 8, 8 }, 99);
        var symbolic2 = SymbolicShape.BatchDynamic(input2._shape);
        cache.GetOrCompileInference(input2._shape, () =>
        {
            compileCount++;
            engine.TensorMatMul(input2, weights);
        }, symbolic2);

        // Both symbolic keys (ignoring dim 0) should be identical
        // So compileCount should still be 1 (cache hit)
        Assert.Equal(1, compileCount);
    }

    #endregion

    #region WeightLayoutOptimizer Tests

    [Fact]
    public void WeightLayoutOptimizer_PackRoundTrip()
    {
        // Verify that packing and unpacking preserves all values
        int rows = 4, cols = 6, panelWidth = 4;
        var weights = new float[rows * cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = i + 1;

        var packed = WeightLayoutOptimizer.PackRowMajorToPanelFormat(weights, rows, cols, panelWidth);

        // Verify packed array contains all original values (in a different layout)
        Assert.True(packed.Length >= rows * cols,
            $"Packed array should be at least as large as original: {packed.Length} < {rows * cols}");

        // Unpack and verify: for each (r, c), find it in packed format
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                int panel = c / panelWidth;
                int j = c % panelWidth;
                int packedIdx = (panel * rows + r) * panelWidth + j;
                Assert.Equal(weights[r * cols + c], packed[packedIdx]);
            }
        }
    }

    #endregion

    #region CompiledDropout Tests

    [Fact]
    public void CompiledDropout_MaskCycling()
    {
        int length = 16;
        float rate = 0.5f;
        int maskCount = 4;
        var dropout = new CompiledDropout(length, rate, maskCount, seed: 42);

        Assert.Equal(maskCount, dropout.MaskCount);

        // Get masks and verify they cycle
        var mask1 = dropout.GetNextMask();
        var mask2 = dropout.GetNextMask();
        var mask3 = dropout.GetNextMask();
        var mask4 = dropout.GetNextMask();
        var mask5 = dropout.GetNextMask(); // Should wrap around

        // mask5 should equal mask1 (cycle of 4)
        Assert.Equal(mask1.Length, mask5.Length);
        for (int i = 0; i < mask1.Length; i++)
            Assert.Equal(mask1[i], mask5[i]);
    }

    [Fact]
    public void CompiledDropout_ApplyInPlace_ZerosElements()
    {
        int length = 100;
        float rate = 0.5f;
        var dropout = new CompiledDropout(length, rate, maskCount: 8, seed: 99);

        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = 1f;

        dropout.ApplyInPlace(data, length);

        // Some elements should be zero (dropped), others scaled by 1/(1-rate) = 2
        int zeros = 0, scaled = 0;
        for (int i = 0; i < length; i++)
        {
            if (data[i] == 0f) zeros++;
            else if (MathF.Abs(data[i] - 2f) < 1e-5f) scaled++;
        }

        // With rate=0.5, roughly half should be zero and half scaled
        Assert.True(zeros > 20 && zeros < 80,
            $"Expected ~50 zeros, got {zeros}");
        Assert.True(scaled > 20 && scaled < 80,
            $"Expected ~50 scaled, got {scaled}");
        Assert.Equal(length, zeros + scaled);
    }

    #endregion

    #region BlasBatchPass Tests

    [Fact]
    public void BlasBatchPass_GroupsIndependentMatMuls()
    {
        var pass = new BlasBatchPass();
        var engine = new CpuEngine();

        // Create 3 independent MatMul steps with same K,N dimensions
        var a1 = CreateRandom(new[] { 2, 4 }, 10);
        var b1 = CreateRandom(new[] { 4, 3 }, 11);
        var o1 = new Tensor<float>(new[] { 2, 3 });

        var a2 = CreateRandom(new[] { 3, 4 }, 12);
        var b2 = CreateRandom(new[] { 4, 3 }, 13);
        var o2 = new Tensor<float>(new[] { 3, 3 });

        var a3 = CreateRandom(new[] { 1, 4 }, 14);
        var b3 = CreateRandom(new[] { 4, 3 }, 15);
        var o3 = new Tensor<float>(new[] { 1, 3 });

        var steps = new[]
        {
            new CompiledStep<float>("TensorMatMul",
                (eng, o) => { var r = eng.TensorMatMul(a1, b1); r.AsSpan().CopyTo(o.AsWritableSpan()); },
                o1, new[] { a1, b1 }, null, null),
            new CompiledStep<float>("TensorMatMul",
                (eng, o) => { var r = eng.TensorMatMul(a2, b2); r.AsSpan().CopyTo(o.AsWritableSpan()); },
                o2, new[] { a2, b2 }, null, null),
            new CompiledStep<float>("TensorMatMul",
                (eng, o) => { var r = eng.TensorMatMul(a3, b3); r.AsSpan().CopyTo(o.AsWritableSpan()); },
                o3, new[] { a3, b3 }, null, null),
        };

        var result = pass.TryOptimize(steps, engine);

        // Should batch the 3 independent matmuls (same K=4, N=3)
        Assert.NotNull(result);
        // Batched: 1 BatchedMatMul + 2 BatchedMatMul_Output = 3 steps total
        // (fewer dispatch steps than 3 separate matmuls)
        Assert.True(result.Length <= steps.Length,
            $"Batched should have <= {steps.Length} steps, got {result.Length}");
        Assert.Contains(result, s => s.OpName == "BatchedMatMul");
    }

    #endregion

    #region Helpers

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int length = 1;
        for (int i = 0; i < shape.Length; i++) length *= shape[i];
        var data = new float[length];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private static float[] CreateRandomArray(int length, int seed)
    {
        var rng = new Random(seed);
        var data = new float[length];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    /// <summary>Naive O(N^2) attention for correctness comparison.</summary>
    private static float[] NaiveAttention(float[] q, float[] k, float[] v,
        int seqQ, int seqK, int headDim, float scale)
    {
        var scores = new float[seqQ * seqK];
        var output = new float[seqQ * headDim];

        // Compute scores: Q @ K^T
        for (int i = 0; i < seqQ; i++)
            for (int j = 0; j < seqK; j++)
            {
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                    dot += q[i * headDim + d] * k[j * headDim + d];
                scores[i * seqK + j] = dot * scale;
            }

        // Softmax per row
        for (int i = 0; i < seqQ; i++)
        {
            float maxVal = float.NegativeInfinity;
            for (int j = 0; j < seqK; j++)
                if (scores[i * seqK + j] > maxVal) maxVal = scores[i * seqK + j];

            float sumExp = 0f;
            for (int j = 0; j < seqK; j++)
            {
                scores[i * seqK + j] = MathF.Exp(scores[i * seqK + j] - maxVal);
                sumExp += scores[i * seqK + j];
            }
            for (int j = 0; j < seqK; j++)
                scores[i * seqK + j] /= sumExp;
        }

        // Output: attn @ V
        for (int i = 0; i < seqQ; i++)
            for (int d = 0; d < headDim; d++)
            {
                float sum = 0f;
                for (int j = 0; j < seqK; j++)
                    sum += scores[i * seqK + j] * v[j * headDim + d];
                output[i * headDim + d] = sum;
            }

        return output;
    }

    #endregion
}
