// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class GeneratedFusedKernelsAndPatternsTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(259)]
    public void RoslynGeneratedElementwiseKernels_MatchDoubleReferenceAcrossSimdBoundaries(int length)
    {
        var rng = new Random(42 + length);
        var a = RandomArray(rng, length, -2.5f, 2.5f);
        var b = RandomArray(rng, length, -2.5f, 2.5f);
        var c = RandomArray(rng, length, -1.5f, 1.5f);
        var dst = new float[length];

        // 1. SwiGluPair: swish(a) * b
        CompiledFusedKernels.SwiGluPair(a, b, dst);
        for (int i = 0; i < length; i++)
        {
            double expected = (a[i] / (1.0 + Math.Exp(-a[i]))) * b[i];
            Assert.True(Math.Abs(dst[i] - expected) < 2e-4, $"SwiGluPair mismatch at {i}: got {dst[i]}, expected {expected}");
        }

        // 2. GeGluPair: gelu(a) * b
        CompiledFusedKernels.GeGluPair(a, b, dst);
        for (int i = 0; i < length; i++)
        {
            double x = a[i];
            double gelu = 0.5 * x * (1.0 + Math.Tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
            double expected = gelu * b[i];
            Assert.True(Math.Abs(dst[i] - expected) < 5e-4, $"GeGluPair mismatch at {i}: got {dst[i]}, expected {expected}");
        }

        // 3. ResidualScaleAdd: a * scale + b
        const float scale = 0.375f;
        CompiledFusedKernels.ResidualScaleAdd(a, b, scale, dst);
        for (int i = 0; i < length; i++)
        {
            float expected = a[i] * scale + b[i];
            Assert.True(Math.Abs(dst[i] - expected) < 1e-5f);
        }

        // 4. AffineSwish: swish(a * b + c)
        CompiledFusedKernels.AffineSwish(a, b, c, dst);
        for (int i = 0; i < length; i++)
        {
            double z = (double)a[i] * b[i] + c[i];
            double expected = z / (1.0 + Math.Exp(-z));
            Assert.True(Math.Abs(dst[i] - expected) < 3e-4);
        }

        // 5. SwiGlu backward analytical vs central finite differences
        var dGate = new float[length];
        var dUp = new float[length];
        CompiledFusedKernels.SwiGluPairGradGate(c, a, b, dGate);
        CompiledFusedKernels.SwiGluPairGradUp(c, a, dUp);
        const double eps = 1e-5;
        for (int i = 0; i < length; i++)
        {
            double g = a[i];
            double u = b[i];
            double go = c[i];
            double fPlus = ((g + eps) / (1.0 + Math.Exp(-(g + eps)))) * u;
            double fMinus = ((g - eps) / (1.0 + Math.Exp(-(g - eps)))) * u;
            double numGradGate = go * (fPlus - fMinus) / (2.0 * eps);
            double numGradUp = go * (g / (1.0 + Math.Exp(-g)));
            Assert.True(Math.Abs(dGate[i] - numGradGate) < 1e-3, $"SwiGluGradGate mismatch at {i}: {dGate[i]} vs {numGradGate}");
            Assert.True(Math.Abs(dUp[i] - numGradUp) < 1e-3, $"SwiGluGradUp mismatch at {i}: {dUp[i]} vs {numGradUp}");
        }
    }

    [Fact]
    public void RoslynGeneratedCudaConstantsAndNvrtcGraphEmitter_ProduceValidVectorizedCudaC()
    {
        Assert.Contains("extern \"C\" __global__ void fused_swiglu_pair_f32", CompiledFusedKernels.SwiGluPairCudaSource);
        Assert.Contains("float4", CompiledFusedKernels.SwiGluPairCudaSource);
        Assert.Contains("fused_rmsnorm_residual_f32", CompiledFusedKernels.UniversalRowFusedCudaSource);
        Assert.Contains("fused_layernorm_residual_f32", CompiledFusedKernels.UniversalRowFusedCudaSource);
        Assert.Contains("fused_logsoftmax_ce_fwd_f32", CompiledFusedKernels.UniversalRowFusedCudaSource);
        Assert.Contains("__shfl_down_sync", CompiledFusedKernels.UniversalRowFusedCudaSource);

        // Verify NvrtcGraphEmitter lowers a same-shape Swish(gate) * up chain to vectorized (float4) CUDA C.
        var graph = new CodegenGraph();
        int inGate = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, new[] { 4, 64 }, 0));
        int inUp = graph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, new[] { 4, 64 }, 1));
        int swish = graph.AddNode(new CodegenNode(CodegenOpKind.Swish, new[] { inGate }, CodegenElementType.Float32, new[] { 4, 64 }));
        int mul = graph.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { swish, inUp }, CodegenElementType.Float32, new[] { 4, 64 }));
        graph.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { mul }, CodegenElementType.Float32, new[] { 4, 64 }, 0));

        var emitter = new NvrtcGraphEmitter();
        var emitted = emitter.Emit(graph, CodegenElementType.Float32);
        Assert.False(emitted.Declined);
        Assert.NotNull(emitted.Kernel);
        Assert.NotNull(emitted.Source);
        if (emitted.Source is not null)
        {
            Assert.Contains("extern \"C\" __global__ void aidn_nvrtc_fused_", emitted.Source);
            Assert.Contains("reinterpret_cast<const float4*>", emitted.Source);
            Assert.DoesNotContain("int cols", emitted.Source);
        }

        // SwiGLU + trailing [64] bias: the bias input is indexed by column, so the emitter takes the scalar
        // broadcast path with a cols parameter instead of the float4 path.
        var biasGraph = new CodegenGraph();
        int bGate = biasGraph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, new[] { 4, 64 }, 0));
        int bUp = biasGraph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, new[] { 4, 64 }, 1));
        int bBias = biasGraph.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, new[] { 64 }, 2));
        int bSwish = biasGraph.AddNode(new CodegenNode(CodegenOpKind.Swish, new[] { bGate }, CodegenElementType.Float32, new[] { 4, 64 }));
        int bMul = biasGraph.AddNode(new CodegenNode(CodegenOpKind.Mul, new[] { bSwish, bUp }, CodegenElementType.Float32, new[] { 4, 64 }));
        int bAdd = biasGraph.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { bMul, bBias }, CodegenElementType.Float32, new[] { 4, 64 }));
        biasGraph.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { bAdd }, CodegenElementType.Float32, new[] { 4, 64 }, 0));

        var biasEmitted = emitter.Emit(biasGraph, CodegenElementType.Float32);
        Assert.False(biasEmitted.Declined);
        Assert.NotNull(biasEmitted.Kernel);
        Assert.NotNull(biasEmitted.Source);
        if (biasEmitted.Source is not null)
        {
            Assert.Contains("int cols", biasEmitted.Source);
            Assert.Contains("int col = idx % cols;", biasEmitted.Source);
            Assert.Contains("in_2[col]", biasEmitted.Source);
            Assert.Contains("in_0[idx]", biasEmitted.Source);
            Assert.DoesNotContain("float4", biasEmitted.Source);
        }
    }

    [Fact]
    public void BiasActivationRowKernels_ShareOneActivationProtocol()
    {
        // The CUDA kernel's activation codes are emitted from the same table as the CPU enum.
        foreach (var (name, kind) in new[]
        {
            ("NONE", CompiledFusedKernels.RowActivationKind.None),
            ("RELU", CompiledFusedKernels.RowActivationKind.ReLU),
            ("GELU", CompiledFusedKernels.RowActivationKind.GELU),
            ("SWISH", CompiledFusedKernels.RowActivationKind.Swish),
            ("SIGMOID", CompiledFusedKernels.RowActivationKind.Sigmoid),
            ("TANH", CompiledFusedKernels.RowActivationKind.Tanh),
        })
        {
            Assert.Contains($"#define AIDN_ROW_ACT_{name} {(int)kind}\n",CompiledFusedKernels.UniversalRowFusedCudaSource);
        }
        Assert.Contains("act_kind == AIDN_ROW_ACT_RELU", CompiledFusedKernels.UniversalRowFusedCudaSource);
        Assert.Contains("act_kind == AIDN_ROW_ACT_TANH", CompiledFusedKernels.UniversalRowFusedCudaSource);
    }

    [Fact]
    public void RmsNormResidualFusionPattern_CollapsesNodes_AndMatchesEagerForwardAndBackward()
    {
        var engine = new CpuEngine();
        var rng = new Random(101);
        var x = new Tensor<float>(RandomArray(rng, 6 * 37, -1.5f, 1.5f), new[] { 6, 37 });
        var res = new Tensor<float>(RandomArray(rng, 6 * 37, -1.5f, 1.5f), new[] { 6, 37 });
        var gamma = new Tensor<float>(RandomArray(rng, 37, 0.5f, 1.5f), new[] { 37 });
        const double eps = 1e-5;

        // Reference eager forward + backward via GradientTape
        Tensor<float> expectedOut;
        Dictionary<Tensor<float>, Tensor<float>> expectedGrads;
        using (var tape = new GradientTape<float>())
        {
            var sumEager = engine.TensorAdd(x, res);
            expectedOut = engine.RMSNorm(sumEager, gamma, eps, out _);
            var lossEager = engine.ReduceSum(engine.TensorMultiply(expectedOut, expectedOut), new[] { 0, 1 }, keepDims: false);
            expectedGrads = tape.ComputeGradients(lossEager, new[] { x, res, gamma });
        }

        // GraphMode compilation (forTraining: true)
        List<ILazyNode> compiledNodes;
        Tensor<float> graphLoss;
        LazyTensorScope scope;
        using (scope = GraphMode.Enable())
        {
            var sumLazy = engine.TensorAdd(x, res);
            var normLazy = engine.RMSNorm(sumLazy, gamma, eps, out _);
            var sq = engine.TensorMultiply(normLazy, normLazy);
            graphLoss = engine.ReduceSum(sq, new[] { 0, 1 }, keepDims: false);
            var compiler = new LazyGraphCompiler(forTraining: true);
            compiledNodes = compiler.Compile(scope.Nodes);
        }

        Assert.Contains(compiledNodes, n => n.OpName == "FusedRmsNormResidual");
        Assert.DoesNotContain(compiledNodes, n => n.OpName == "RMSNorm");

        // Execute fused forward + backward
        var fusedNode = compiledNodes.OfType<LazyNode<float>>().First(n => n.OpName == "FusedRmsNormResidual");
        var fusedOut = new Tensor<float>(expectedOut._shape);
        fusedNode.Execute(engine, fusedOut);
        AssertClose(expectedOut.AsSpan(), fusedOut.AsSpan(), 1e-5f);

        var upstream = engine.TensorMultiplyScalar(fusedOut, 2.0f);
        var fusedGrads = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        Assert.NotNull(fusedNode.BackwardFn);
        fusedNode.BackwardFn?.Invoke(upstream, fusedNode.GetInputsArray(), fusedOut, fusedNode.SavedState ?? Array.Empty<object>(), engine, fusedGrads);

        AssertClose(expectedGrads[x].AsSpan(), fusedGrads[x].AsSpan(), 2e-4f);
        AssertClose(expectedGrads[res].AsSpan(), fusedGrads[res].AsSpan(), 2e-4f);
        AssertClose(expectedGrads[gamma].AsSpan(), fusedGrads[gamma].AsSpan(), 2e-4f);
    }

    [Fact]
    public void LayerNormResidualFusionPattern_CollapsesNodes_AndMatchesEagerForwardAndBackward()
    {
        var engine = new CpuEngine();
        var rng = new Random(202);
        var x = new Tensor<float>(RandomArray(rng, 5 * 40, -1.5f, 1.5f), new[] { 5, 40 });
        var res = new Tensor<float>(RandomArray(rng, 5 * 40, -1.5f, 1.5f), new[] { 5, 40 });
        var gamma = new Tensor<float>(RandomArray(rng, 40, 0.5f, 1.5f), new[] { 40 });
        var beta = new Tensor<float>(RandomArray(rng, 40, -0.5f, 0.5f), new[] { 40 });
        const double eps = 1e-5;

        Tensor<float> expectedOut;
        Dictionary<Tensor<float>, Tensor<float>> expectedGrads;
        using (var tape = new GradientTape<float>())
        {
            var sumEager = engine.TensorAdd(x, res);
            expectedOut = engine.LayerNorm(sumEager, gamma, beta, eps, out _, out _);
            var lossEager = engine.ReduceSum(engine.TensorMultiply(expectedOut, expectedOut), new[] { 0, 1 }, keepDims: false);
            expectedGrads = tape.ComputeGradients(lossEager, new[] { x, res, gamma, beta });
        }

        List<ILazyNode> compiledNodes;
        using (var scope = GraphMode.Enable())
        {
            var sumLazy = engine.TensorAdd(x, res);
            var lnLazy = engine.LayerNorm(sumLazy, gamma, beta, eps, out _, out _);
            _ = engine.ReduceSum(engine.TensorMultiply(lnLazy, lnLazy), new[] { 0, 1 }, keepDims: false);
            compiledNodes = new LazyGraphCompiler(forTraining: true).Compile(scope.Nodes);
        }

        Assert.Contains(compiledNodes, n => n.OpName == "FusedLayerNormResidual");
        Assert.DoesNotContain(compiledNodes, n => n.OpName == "LayerNorm");

        var fusedNode = compiledNodes.OfType<LazyNode<float>>().First(n => n.OpName == "FusedLayerNormResidual");
        var fusedOut = new Tensor<float>(expectedOut._shape);
        fusedNode.Execute(engine, fusedOut);
        AssertClose(expectedOut.AsSpan(), fusedOut.AsSpan(), 1e-5f);

        var upstream = engine.TensorMultiplyScalar(fusedOut, 2.0f);
        var fusedGrads = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        fusedNode.BackwardFn?.Invoke(upstream, fusedNode.GetInputsArray(), fusedOut, fusedNode.SavedState ?? Array.Empty<object>(), engine, fusedGrads);

        AssertClose(expectedGrads[x].AsSpan(), fusedGrads[x].AsSpan(), 2e-4f);
        AssertClose(expectedGrads[res].AsSpan(), fusedGrads[res].AsSpan(), 2e-4f);
        AssertClose(expectedGrads[gamma].AsSpan(), fusedGrads[gamma].AsSpan(), 2e-4f);
        AssertClose(expectedGrads[beta].AsSpan(), fusedGrads[beta].AsSpan(), 2e-4f);
    }

    [Fact]
    public void SwiGluFusionPattern_CollapsesBothDirectAndDecomposedVariants_AndMatchesGradients()
    {
        var engine = new CpuEngine();
        var rng = new Random(303);
        var gate = new Tensor<float>(RandomArray(rng, 4 * 48, -2.0f, 2.0f), new[] { 4, 48 });
        var up = new Tensor<float>(RandomArray(rng, 4 * 48, -2.0f, 2.0f), new[] { 4, 48 });

        Tensor<float> expectedOut;
        Dictionary<Tensor<float>, Tensor<float>> expectedGrads;
        using (var tape = new GradientTape<float>())
        {
            var sig = engine.Sigmoid(gate);
            var swish = engine.TensorMultiply(gate, sig);
            expectedOut = engine.TensorMultiply(swish, up);
            var loss = engine.ReduceSum(expectedOut, new[] { 0, 1 }, keepDims: false);
            expectedGrads = tape.ComputeGradients(loss, new[] { gate, up });
        }

        // Decomposed 3-node (Sigmoid -> Mul -> Mul) and direct 2-node (Swish -> Mul) both collapse to 1 FusedSwiGlu node.
        var variants = new (string Name, Action Record)[]
        {
            ("decomposed", () =>
            {
                var sig = engine.Sigmoid(gate);
                var swish = engine.TensorMultiply(gate, sig);
                _ = engine.TensorMultiply(swish, up);
            }),
            ("direct", () => _ = engine.TensorMultiply(engine.Swish(gate), up)),
        };

        foreach (var (name, record) in variants)
        {
            List<ILazyNode> compiled;
            using (var scope = GraphMode.Enable())
            {
                record();
                compiled = new LazyGraphCompiler(forTraining: true).Compile(scope.Nodes);
            }

            Assert.True(compiled.Count == 1, $"{name}: expected 1 fused node, got {compiled.Count}.");
            Assert.Equal("FusedSwiGlu", compiled[0].OpName);

            var fusedNode = (LazyNode<float>)compiled[0];
            var fusedOut = new Tensor<float>(expectedOut._shape);
            fusedNode.Execute(engine, fusedOut);
            AssertClose(expectedOut.AsSpan(), fusedOut.AsSpan(), 3e-4f);

            var ones = new Tensor<float>(Enumerable.Repeat(1.0f, expectedOut.Length).ToArray(), expectedOut._shape);
            var fusedGrads = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
            Assert.NotNull(fusedNode.BackwardFn);
            fusedNode.BackwardFn?.Invoke(ones, fusedNode.GetInputsArray(), fusedOut, fusedNode.SavedState ?? Array.Empty<object>(), engine, fusedGrads);
            AssertClose(expectedGrads[gate].AsSpan(), fusedGrads[gate].AsSpan(), 5e-4f);
            AssertClose(expectedGrads[up].AsSpan(), fusedGrads[up].AsSpan(), 5e-4f);
        }
    }

    [Fact]
    public void SwiGluFusionPattern_RebuildsOfOneShapeShareOneForwardGraph()
    {
        // The GPU engine caches compiled kernels by graph identity: rebuilding the same unit must not add entries.
        var first = SwiGluFusionPattern.GetForwardCodegenGraph(new[] { 4, 48 }, isGelu: false);
        Assert.Same(first, SwiGluFusionPattern.GetForwardCodegenGraph(new[] { 4, 48 }, isGelu: false));
        Assert.NotSame(first, SwiGluFusionPattern.GetForwardCodegenGraph(new[] { 4, 48 }, isGelu: true));
        Assert.NotSame(first, SwiGluFusionPattern.GetForwardCodegenGraph(new[] { 8, 48 }, isGelu: false));
    }

    [Fact]
    public void LayerNormResidualFusionPattern_DeclinesWhenATrainingTraceRetainsAStatistic()
    {
        var engine = new CpuEngine();
        var rng = new Random(212);
        var x = new Tensor<float>(RandomArray(rng, 5 * 40, -1.5f, 1.5f), new[] { 5, 40 });
        var res = new Tensor<float>(RandomArray(rng, 5 * 40, -1.5f, 1.5f), new[] { 5, 40 });
        var gamma = new Tensor<float>(RandomArray(rng, 40, 0.5f, 1.5f), new[] { 40 });
        var beta = new Tensor<float>(RandomArray(rng, 40, -0.5f, 0.5f), new[] { 40 });
        const double eps = 1e-5;

        engine.LayerNorm(engine.TensorAdd(x, res), gamma, beta, eps, out var expectedMean, out _);

        List<ILazyNode> compiledNodes;
        Tensor<float> mean;
        using (var scope = GraphMode.Enable())
        {
            var lnLazy = engine.LayerNorm(engine.TensorAdd(x, res), gamma, beta, eps, out mean, out _);
            _ = engine.ReduceSum(engine.TensorMultiply(lnLazy, lnLazy), new[] { 0, 1 }, keepDims: false);
            // The caller retains the mean: the training fusion pass blocks that node.
            var observable = new HashSet<ILazyNode> { mean.LazySource! };
            compiledNodes = new LazyGraphCompiler(forTraining: true, observable).Compile(scope.Nodes);
        }

        Assert.DoesNotContain(compiledNodes, n => n.OpName == "FusedLayerNormResidual");
        Assert.Contains(compiledNodes, n => n.OpName == "LayerNorm");
        Assert.Contains(compiledNodes, n => n.OpName == "LayerNormMean");

        // The retained statistic still gets its values from the LayerNorm writer.
        foreach (var node in compiledNodes.OfType<LazyNode<float>>())
            node.Execute(engine, node.Output);
        AssertClose(expectedMean.AsSpan(), mean.AsSpan(), 1e-5f);
    }

    [Fact]
    public void RmsNormGraphMode_RmsOutputDependsOnTheRmsNormNode()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 3.0f, 4.0f }, new[] { 1, 2 });
        var gamma = new Tensor<float>(new[] { 1.0f, 1.0f }, new[] { 2 });

        Tensor<float> rms;
        Tensor<float> consumer;
        using (var scope = GraphMode.Enable())
        {
            _ = engine.RMSNorm(x, gamma, 0.0, out rms);
            // A consumer of rms alone must schedule the RMSNorm node without anything reading its normalized output.
            consumer = engine.TensorMultiplyScalar(rms, 2.0f);
            Assert.NotNull(rms.LazySource);
            Assert.Contains(rms.LazySource!.GetInputNodes(), n => n.OpName == "RMSNorm");
        }

        float expectedRms = (float)Math.Sqrt(12.5);
        Assert.Equal(2.0f * expectedRms, consumer.AsSpan()[0], 4);
        Assert.Equal(expectedRms, rms.AsSpan()[0], 4);
    }

    [Fact]
    public void LogSoftmaxCrossEntropyFusionPattern_HandlesMaskedZeroRows_AndMatchesExactLossAndGradients()
    {
        var engine = new CpuEngine();
        var rng = new Random(404);
        int rows = 6;
        int vocab = 53;
        var logits = new Tensor<float>(RandomArray(rng, rows * vocab, -3.0f, 3.0f), new[] { rows, vocab });
        var targetData = new float[rows * vocab];
        // Make row 2 an all-zero masked padding token; other rows are valid one-hot / soft distributions.
        for (int r = 0; r < rows; r++)
        {
            if (r == 2) continue;
            int hot = (r * 11 + 3) % vocab;
            targetData[r * vocab + hot] = 0.8f;
            targetData[r * vocab + ((hot + 1) % vocab)] = 0.2f;
        }
        var target = new Tensor<float>(targetData, new[] { rows, vocab });

        Tensor<float> expectedLoss;
        Dictionary<Tensor<float>, Tensor<float>> expectedGrads;
        using (var tape = new GradientTape<float>())
        {
            var logP = engine.TensorLogSoftmax(logits, 1);
            var perRow = engine.ReduceSum(engine.TensorMultiply(target, logP), new[] { 1 }, keepDims: false);
            expectedLoss = engine.TensorNegate(engine.ReduceMean(perRow, new[] { 0 }, keepDims: false));
            expectedGrads = tape.ComputeGradients(expectedLoss, new[] { logits, target });
        }

        List<ILazyNode> compiled;
        using (var scope = GraphMode.Enable())
        {
            var logP = engine.TensorLogSoftmax(logits, 1);
            var perRow = engine.ReduceSum(engine.TensorMultiply(target, logP), new[] { 1 }, keepDims: false);
            _ = engine.TensorNegate(engine.ReduceMean(perRow, new[] { 0 }, keepDims: false));
            compiled = new LazyGraphCompiler(forTraining: true).Compile(scope.Nodes);
        }

        Assert.Single(compiled);
        Assert.Equal("FusedLogSoftmaxCrossEntropy", compiled[0].OpName);

        var fusedNode = (LazyNode<float>)compiled[0];
        var fusedLoss = new Tensor<float>(expectedLoss._shape);
        fusedNode.Execute(engine, fusedLoss);
        AssertClose(expectedLoss.AsSpan(), fusedLoss.AsSpan(), 1e-5f);

        var seed = new Tensor<float>(new[] { 1.0f }, expectedLoss._shape);
        var fusedGrads = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        fusedNode.BackwardFn?.Invoke(seed, fusedNode.GetInputsArray(), fusedLoss, fusedNode.SavedState ?? Array.Empty<object>(), engine, fusedGrads);
        AssertClose(expectedGrads[logits].AsSpan(), fusedGrads[logits].AsSpan(), 2e-4f);
        AssertClose(expectedGrads[target].AsSpan(), fusedGrads[target].AsSpan(), 2e-4f);
    }

    [Fact]
    public void RotaryPositionEmbeddingFusionPattern_CollapsesNineNodesToOne_AndMatchesAdjointBackward()
    {
        var engine = new CpuEngine();
        var rng = new Random(505);
        int batchHeads = 2;
        int seqLen = 5;
        int headDim = 16;
        int halfDim = headDim / 2;

        var x = new Tensor<float>(RandomArray(rng, batchHeads * seqLen * headDim, -1.0f, 1.0f), new[] { batchHeads, seqLen, headDim });
        var cos = new Tensor<float>(RandomArray(rng, seqLen * halfDim, -1.0f, 1.0f), new[] { 1, seqLen, halfDim });
        var sin = new Tensor<float>(RandomArray(rng, seqLen * halfDim, -1.0f, 1.0f), new[] { 1, seqLen, halfDim });

        Tensor<float> ApplyDecomposedRope(Tensor<float> inp)
        {
            var x0 = engine.TensorSlice(inp, new[] { 0, 0, 0 }, new[] { batchHeads, seqLen, halfDim });
            var x1 = engine.TensorSlice(inp, new[] { 0, 0, halfDim }, new[] { batchHeads, seqLen, halfDim });
            var y0 = engine.TensorSubtract(engine.TensorBroadcastMultiply(x0, cos), engine.TensorBroadcastMultiply(x1, sin));
            var y1 = engine.TensorAdd(engine.TensorBroadcastMultiply(x1, cos), engine.TensorBroadcastMultiply(x0, sin));
            return engine.Concat(new[] { y0, y1 }, 2);
        }

        Tensor<float> expectedOut;
        Dictionary<Tensor<float>, Tensor<float>> expectedGrads;
        using (var tape = new GradientTape<float>())
        {
            expectedOut = ApplyDecomposedRope(x);
            var loss = engine.ReduceSum(engine.TensorMultiply(expectedOut, expectedOut), new[] { 0, 1, 2 }, keepDims: false);
            expectedGrads = tape.ComputeGradients(loss, new[] { x });
        }

        List<ILazyNode> compiled;
        using (var scope = GraphMode.Enable())
        {
            _ = ApplyDecomposedRope(x);
            Assert.Equal(9, scope.Nodes.Count);
            compiled = new LazyGraphCompiler(forTraining: true).Compile(scope.Nodes);
        }

        Assert.Single(compiled);
        Assert.Equal("FusedRotaryPositionEmbedding", compiled[0].OpName);

        var fusedNode = (LazyNode<float>)compiled[0];
        var fusedOut = new Tensor<float>(expectedOut._shape);
        fusedNode.Execute(engine, fusedOut);
        AssertClose(expectedOut.AsSpan(), fusedOut.AsSpan(), 1e-5f);

        var upstream = engine.TensorMultiplyScalar(fusedOut, 2.0f);
        var fusedGrads = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        fusedNode.BackwardFn?.Invoke(upstream, fusedNode.GetInputsArray(), fusedOut, fusedNode.SavedState ?? Array.Empty<object>(), engine, fusedGrads);
        AssertClose(expectedGrads[x].AsSpan(), fusedGrads[x].AsSpan(), 1e-5f);
    }

    [Fact]
    public void BroadcastAddActivationAndElementwiseChainFusion_CollapseMultiOpInferenceChains()
    {
        var engine = new CpuEngine();
        var rng = new Random(606);
        var x = new Tensor<float>(RandomArray(rng, 4 * 32, -1.5f, 1.5f), new[] { 4, 32 });
        var bias = new Tensor<float>(RandomArray(rng, 32, -0.5f, 0.5f), new[] { 32 });
        var scale = new Tensor<float>(RandomArray(rng, 4 * 32, 0.5f, 1.5f), new[] { 4, 32 });

        var expectedBiasGelu = engine.GELU(engine.TensorBroadcastAdd(x, bias));
        var expectedChain = engine.TensorNegate(engine.Swish(engine.TensorAddScalar(engine.TensorMultiply(expectedBiasGelu, scale), 0.25f)));

        List<ILazyNode> compiled;
        using (var scope = GraphMode.EnableInference())
        {
            var bg = engine.GELU(engine.TensorBroadcastAdd(x, bias));
            _ = engine.TensorNegate(engine.Swish(engine.TensorAddScalar(engine.TensorMultiply(bg, scale), 0.25f)));
            Assert.Equal(6, scope.Nodes.Count);
            compiled = new LazyGraphCompiler(forTraining: false).Compile(scope.Nodes);
        }

        // 2 nodes (BroadcastAdd + GELU) -> 1 FusedBiasActivation, and 4 nodes (Mul + AddScalar + Swish + Negate) -> 1 FusedElementwiseChain!
        Assert.Equal(2, compiled.Count);
        Assert.Equal("FusedBiasActivation", compiled[0].OpName);
        Assert.Equal("FusedElementwiseChain", compiled[1].OpName);

        var step0 = (LazyNode<float>)compiled[0];
        var step1 = (LazyNode<float>)compiled[1];
        step0.Execute(engine, step0.Output);
        var actualFinal = new Tensor<float>(expectedChain._shape);
        step1.Execute(engine, actualFinal);

        AssertClose(expectedChain.AsSpan(), actualFinal.AsSpan(), 5e-4f);
    }

    private static float[] RandomArray(Random rng, int length, float min, float max)
    {
        var arr = new float[length];
        float span = max - min;
        for (int i = 0; i < length; i++)
        {
            arr[i] = min + (float)rng.NextDouble() * span;
        }
        return arr;
    }

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            Assert.True(diff <= tol, $"Index {i}: expected {expected[i]}, actual {actual[i]} (diff {diff} > {tol}).");
        }
    }
}
