// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
#if NETCOREAPP
using System.Reflection;
#endif
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Compilation.Autotune;
using AiDotNet.Tensors.Engines.Compilation.Codegen.CudaGraph;
using AiDotNet.Tensors.Engines.Compilation.FusionTileIr;
using AiDotNet.Tensors.Engines.Compilation.GeneratedKernels;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class FusionTileIrAndAutotuneTests
{
    [Fact]
    public void NormResidualRopeFusionPattern_CollapsesCrossLayerChain_AndMatchesEagerForwardAndBackward()
    {
        var engine = new CpuEngine();
        var rng = new Random(707);
        int batchHeads = 2;
        int seqLen = 4;
        int headDim = 16;
        int halfDim = headDim / 2;
        const double eps = 1e-5;

        var x = new Tensor<float>(RandomArray(rng, batchHeads * seqLen * headDim, -1.0f, 1.0f), new[] { batchHeads, seqLen, headDim });
        var res = new Tensor<float>(RandomArray(rng, batchHeads * seqLen * headDim, -1.0f, 1.0f), new[] { batchHeads, seqLen, headDim });
        var gamma = new Tensor<float>(RandomArray(rng, headDim, 0.5f, 1.5f), new[] { headDim });
        var cos = new Tensor<float>(RandomArray(rng, seqLen * halfDim, -1.0f, 1.0f), new[] { 1, seqLen, halfDim });
        var sin = new Tensor<float>(RandomArray(rng, seqLen * halfDim, -1.0f, 1.0f), new[] { 1, seqLen, halfDim });

        Tensor<float> ApplyCrossLayerChain(Tensor<float> inp, Tensor<float> residual, Tensor<float> weight)
        {
            var sum = engine.TensorAdd(inp, residual);
            var norm = engine.RMSNorm(sum, weight, eps, out _);
            var x0 = engine.TensorSlice(norm, new[] { 0, 0, 0 }, new[] { batchHeads, seqLen, halfDim });
            var x1 = engine.TensorSlice(norm, new[] { 0, 0, halfDim }, new[] { batchHeads, seqLen, halfDim });
            var y0 = engine.TensorSubtract(engine.TensorBroadcastMultiply(x0, cos), engine.TensorBroadcastMultiply(x1, sin));
            var y1 = engine.TensorAdd(engine.TensorBroadcastMultiply(x1, cos), engine.TensorBroadcastMultiply(x0, sin));
            return engine.Concat(new[] { y0, y1 }, 2);
        }

        Tensor<float> expectedOut;
        Dictionary<Tensor<float>, Tensor<float>> expectedGrads;
        using (var tape = new GradientTape<float>())
        {
            expectedOut = ApplyCrossLayerChain(x, res, gamma);
            var loss = engine.ReduceSum(engine.TensorMultiply(expectedOut, expectedOut), new[] { 0, 1, 2 }, keepDims: false);
            expectedGrads = tape.ComputeGradients(loss, new[] { x, res, gamma, cos, sin });
        }

        // A training plan keeps the residual add and the RMSNorm: the fused node never writes the residual sum,
        // which a training caller can still read after Step().
        List<ILazyNode> training;
        using (var scope = GraphMode.Enable())
        {
            _ = ApplyCrossLayerChain(x, res, gamma);
            training = new LazyGraphCompiler(forTraining: true).Compile(scope.Nodes);
        }
        Assert.DoesNotContain(training, n => n.OpName == "FusedRmsNormResidualRope");
        Assert.Contains(training, n => n.OpName == "RMSNorm");

        List<ILazyNode> compiled;
        using (var scope = GraphMode.Enable())
        {
            _ = ApplyCrossLayerChain(x, res, gamma);
            // 11 compute nodes plus the RMSNorm rms statistic node, which the fused node absorbs.
            Assert.Equal(12, scope.Nodes.Count);
            compiled = new LazyGraphCompiler(forTraining: false).Compile(scope.Nodes);
        }

        Assert.Single(compiled);
        Assert.Equal("FusedRmsNormResidualRope", compiled[0].OpName);

        var fusedNode = (LazyNode<float>)compiled[0];
        var fusedOut = new Tensor<float>(expectedOut._shape);
        fusedNode.Execute(engine, fusedOut);
        AssertClose(expectedOut.AsSpan(), fusedOut.AsSpan(), 1e-5f);

        var upstream = engine.TensorMultiplyScalar(fusedOut, 2.0f);
        var fusedGrads = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        Assert.NotNull(fusedNode.BackwardFn);
        fusedNode.BackwardFn?.Invoke(upstream, fusedNode.GetInputsArray(), fusedOut, fusedNode.SavedState ?? Array.Empty<object>(), engine, fusedGrads);

        AssertClose(expectedGrads[x].AsSpan(), fusedGrads[x].AsSpan(), 3e-4f);
        AssertClose(expectedGrads[res].AsSpan(), fusedGrads[res].AsSpan(), 3e-4f);
        AssertClose(expectedGrads[gamma].AsSpan(), fusedGrads[gamma].AsSpan(), 3e-4f);
        // The RoPE tables are trainable inputs too: their gradients sum over the broadcast batch-heads axis.
        AssertClose(expectedGrads[cos].AsSpan(), fusedGrads[cos].AsSpan(), 3e-4f);
        AssertClose(expectedGrads[sin].AsSpan(), fusedGrads[sin].AsSpan(), 3e-4f);
    }

    [Fact]
    public void NormResidualRopeFusionPattern_StridedInput_RunsTheFallbackAndBackwardWithoutAPriorForward()
    {
        var engine = new CpuEngine();
        var rng = new Random(709);
        int batchHeads = 2, seqLen = 4, headDim = 8, halfDim = headDim / 2;
        const double eps = 1e-5;

        // A transposed view: not contiguous, so the forward takes the engine fallback and the backward needs
        // logical-order reads.
        var x = new Tensor<float>(RandomArray(rng, batchHeads * seqLen * headDim, -1.0f, 1.0f), new[] { batchHeads, headDim, seqLen })
            .Transpose(new[] { 0, 2, 1 });
        Assert.False(x.IsContiguous);
        var res = new Tensor<float>(RandomArray(rng, batchHeads * seqLen * headDim, -1.0f, 1.0f), new[] { batchHeads, seqLen, headDim });
        var gamma = new Tensor<float>(RandomArray(rng, headDim, 0.5f, 1.5f), new[] { headDim });
        var cos = new Tensor<float>(RandomArray(rng, seqLen * halfDim, -1.0f, 1.0f), new[] { 1, seqLen, halfDim });
        var sin = new Tensor<float>(RandomArray(rng, seqLen * halfDim, -1.0f, 1.0f), new[] { 1, seqLen, halfDim });

        Tensor<float> Chain()
        {
            var norm = engine.RMSNorm(engine.TensorAdd(x, res), gamma, eps, out _);
            var x0 = engine.TensorSlice(norm, new[] { 0, 0, 0 }, new[] { batchHeads, seqLen, halfDim });
            var x1 = engine.TensorSlice(norm, new[] { 0, 0, halfDim }, new[] { batchHeads, seqLen, halfDim });
            var y0 = engine.TensorSubtract(engine.TensorBroadcastMultiply(x0, cos), engine.TensorBroadcastMultiply(x1, sin));
            var y1 = engine.TensorAdd(engine.TensorBroadcastMultiply(x1, cos), engine.TensorBroadcastMultiply(x0, sin));
            return engine.Concat(new[] { y0, y1 }, 2);
        }

        Tensor<float> expectedOut;
        Dictionary<Tensor<float>, Tensor<float>> expectedGrads;
        using (var tape = new GradientTape<float>())
        {
            expectedOut = Chain();
            var loss = engine.ReduceSum(engine.TensorMultiply(expectedOut, expectedOut), new[] { 0, 1, 2 }, keepDims: false);
            expectedGrads = tape.ComputeGradients(loss, new[] { x, res, gamma, cos, sin });
        }

        List<ILazyNode> compiled;
        using (var scope = GraphMode.Enable())
        {
            _ = Chain();
            compiled = new LazyGraphCompiler(forTraining: true).Compile(scope.Nodes);
        }
        var fusedNode = (LazyNode<float>)Assert.Single(compiled);
        Assert.Equal("FusedRmsNormResidualRope", fusedNode.OpName);

        // Backward first: a forward route that writes no statistics (the single-kernel CUDA one) leaves nothing
        // behind, so the backward must not depend on one.
        var upstream = engine.TensorMultiplyScalar(expectedOut, 2.0f);
        var fusedGrads = new Dictionary<Tensor<float>, Tensor<float>>(ReferenceEqualityComparer<Tensor<float>>.Instance);
        fusedNode.BackwardFn?.Invoke(upstream, fusedNode.GetInputsArray(), expectedOut, fusedNode.SavedState ?? Array.Empty<object>(), engine, fusedGrads);
        foreach (var t in new[] { x, res, gamma, cos, sin })
            AssertClose(expectedGrads[t].Contiguous().AsSpan(), fusedGrads[t].Contiguous().AsSpan(), 3e-4f);

        // The fallback forward applies RoPE to this step's normalized values, not to a stale RMSNorm output.
        var fusedOut = new Tensor<float>(expectedOut._shape);
        fusedNode.Execute(engine, fusedOut);
        AssertClose(expectedOut.AsSpan(), fusedOut.AsSpan(), 1e-5f);
    }

    [Fact]
    public void FusionTileAdjointBuilder_And_FusionTileLinker_ProduceExactGradientsAndLinkedTiles()
    {
        // Auto-differentiate GeGLU (exercises Gelu + Mul + Const immediates in FusionTileAdjointBuilder)
        var gegluFwd = CompiledFusedKernels.GeGluPairTileDescriptor;
        var gegluBwd = FusionTileAdjointBuilder.BuildBackwardTile(gegluFwd);
        Assert.Equal(3, gegluBwd.InputCount);
        Assert.Equal(2, gegluBwd.OutputCount);
        Assert.NotNull(gegluFwd.CpuExecutor);
        Assert.NotNull(gegluBwd.CpuExecutor);
        Assert.Contains("extern \"C\" __global__ void", gegluBwd.CudaDeviceFragment);

        var gate = new float[] { -1.2f, -0.3f, 0.25f, 1.1f };
        var up = new float[] { 0.7f, -1.4f, 0.9f, -0.5f };
        var dOut = new float[] { 0.4f, -0.6f, 1.2f, 0.8f };
        var dGate = new float[gate.Length];
        var dUp = new float[up.Length];

        gegluBwd.CpuExecutor?.Invoke(
            new[] { gate, up, dOut },
            Array.Empty<float>(),
            new[] { dGate, dUp });

        // Verify analytical adjoint matches central finite differences
        const float h = 1e-3f;
        var outPlus = new float[gate.Length];
        var outMinus = new float[gate.Length];
        for (int idx = 0; idx < gate.Length; idx++)
        {
            var gPlus = (float[])gate.Clone();
            var gMinus = (float[])gate.Clone();
            gPlus[idx] += h;
            gMinus[idx] -= h;
            gegluFwd.CpuExecutor?.Invoke(new[] { gPlus, up }, Array.Empty<float>(), new[] { outPlus });
            gegluFwd.CpuExecutor?.Invoke(new[] { gMinus, up }, Array.Empty<float>(), new[] { outMinus });
            float numGradGate = (outPlus[idx] - outMinus[idx]) / (2.0f * h) * dOut[idx];
            Assert.True(Math.Abs(numGradGate - dGate[idx]) <= 2e-3f,
                $"GeGLU dGate mismatch at {idx}: numeric={numGradGate}, auto={dGate[idx]}");

            var uPlus = (float[])up.Clone();
            var uMinus = (float[])up.Clone();
            uPlus[idx] += h;
            uMinus[idx] -= h;
            gegluFwd.CpuExecutor?.Invoke(new[] { gate, uPlus }, Array.Empty<float>(), new[] { outPlus });
            gegluFwd.CpuExecutor?.Invoke(new[] { gate, uMinus }, Array.Empty<float>(), new[] { outMinus });
            float numGradUp = (outPlus[idx] - outMinus[idx]) / (2.0f * h) * dOut[idx];
            Assert.True(Math.Abs(numGradUp - dUp[idx]) <= 2e-3f,
                $"GeGLU dUp mismatch at {idx}: numeric={numGradUp}, auto={dUp[idx]}");
        }

        // Verify vertical and horizontal cross-tile linking + epilogue/attention/scan tile synthesis
        var vertical = FusionTileLinker.LinkVertical(
            CompiledFusedKernels.AddGeluTileDescriptor,
            CompiledFusedKernels.AddSwishTileDescriptor,
            consumerInputPort: 0);
        Assert.Equal(3, vertical.InputCount);
        Assert.Equal(1, vertical.OutputCount);
        Assert.NotNull(vertical.CpuExecutor);

        var horizontal = FusionTileLinker.LinkHorizontalSharedInputs(
            CompiledFusedKernels.SwiGluPairTileDescriptor,
            CompiledFusedKernels.GeGluPairTileDescriptor);
        Assert.Equal(2, horizontal.InputCount);
        Assert.Equal(2, horizontal.OutputCount);

        var epilogue = FusionTileLinker.CreateGemmBiasActivationResidualEpilogueTile(FusionTileOpCode.Gelu);
        var epilogueBwd = FusionTileAdjointBuilder.BuildBackwardTile(epilogue);
        Assert.Equal(3, epilogueBwd.OutputCount);

        var flashCausal = FusionTileLinker.CreateTiledFlashAttentionTile(64, isCausal: true);
        Assert.Contains("kvRow > qIdx", flashCausal.CudaDeviceFragment);

        var linearScan = FusionTileLinker.CreateChunkedLinearScanTile(32);
        Assert.Contains("const int D = 32;", linearScan.CudaDeviceFragment);

        var crossLayer = FusionTileLinker.CreateCrossLayerNormResidualRopeTile(useLayerNorm: false);
        Assert.Contains("fused_rmsnorm_residual_rope_f32", crossLayer.CudaDeviceFragment);
    }

    [Fact]
    public void FusionTileArtifactCache_And_DirectPtxAutotuner_PersistAndHonorVendorIsolation()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "aidn_autotune_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        string? prevCacheOverride = FusionTileArtifactCache.OverrideCacheDirectory;
        try
        {
            FusionTileArtifactCache.OverrideCacheDirectory = tempDir;
            string key = FusionTileArtifactCache.ComputeArtifactKey("extern \"C\" __global__ void k() {}", 8, 9, fastMath: false);
            var payload = new byte[] { 1, 3, 5, 7, 9 };
            FusionTileArtifactCache.StoreArtifact(key, ".ptx", payload);
            Assert.True(File.Exists(Path.Combine(tempDir, key + ".ptx")));
            // Drop the in-memory tier so the lookup below must be served from the persisted file.
            FusionTileArtifactCache.ClearMemoryCache();
            Assert.True(FusionTileArtifactCache.TryLoadArtifact(key, ".ptx", out var loaded));
            Assert.Equal(new byte[] { 1, 3, 5, 7, 9 }, loaded);

            string jsonPath = Path.Combine(tempDir, "direct_ptx_tune.json");
            var shape = new DirectPtxShapeKey(89, TunedKernelOp.Gemm, 128, 256, 64, "gelu");

            int directCalls = 0;
            int vendorCalls = 0;
            var directCandidates = new[]
            {
                new AutotuneExecutableCandidate("direct_ptx_tile_128x128", TunedKernelOrigin.Generated, () => directCalls++)
            };
            var vendorProvider = new FakeVendorProvider(
                new AutotuneExecutableCandidate("vendor_cublaslt_algo0", TunedKernelOrigin.Vendor, () => vendorCalls++));

            // 1) Self-contained Direct-PTX mode excludes vendor libraries completely.
            var standaloneTuner = new DirectPtxAutotuner(
                new DirectPtxAutotunerOptions
                {
                    AllowExternalVendorLibraries = false,
                    WarmupRuns = 1,
                    MeasuredRuns = 3,
                    PersistenceFilePath = jsonPath
                },
                vendorProvider);

            var winnerStandalone = standaloneTuner.SelectOrTune(shape, directCandidates, out var standaloneRec);
            Assert.Equal("direct_ptx_tile_128x128", winnerStandalone.CandidateId);
            Assert.Equal(0, vendorCalls);
            Assert.True(directCalls >= 4);
            Assert.True(double.IsNaN(standaloneRec.DirectPtxVsVendorRatio));
            Assert.True(File.Exists(jsonPath));

            // A fresh tuner over the same JSON file must return the persisted winner without measuring again.
            int callsBeforeReload = directCalls;
            var reloadedTuner = new DirectPtxAutotuner(
                new DirectPtxAutotunerOptions
                {
                    AllowExternalVendorLibraries = false,
                    WarmupRuns = 1,
                    MeasuredRuns = 3,
                    PersistenceFilePath = jsonPath
                },
                vendorProvider);
            Assert.True(reloadedTuner.Records.ContainsKey(shape.ToString()));
            var winnerReloaded = reloadedTuner.SelectOrTune(shape, directCandidates, out var reloadedRec);
            Assert.Equal("direct_ptx_tile_128x128", winnerReloaded.CandidateId);
            Assert.Equal(callsBeforeReload, directCalls);
            Assert.Equal(0, vendorCalls);
            Assert.Equal(standaloneRec.CandidateSetSignature, reloadedRec.CandidateSetSignature);

            // 2) Vendor comparison mode records unclamped DirectPtxVsVendorRatio and persists to disk.
            string jsonVendorPath = Path.Combine(tempDir, "direct_ptx_vs_vendor.json");
            var compareTuner = new DirectPtxAutotuner(
                new DirectPtxAutotunerOptions
                {
                    AllowExternalVendorLibraries = true,
                    WarmupRuns = 1,
                    MeasuredRuns = 3,
                    PersistenceFilePath = jsonVendorPath
                },
                vendorProvider);

            _ = compareTuner.SelectOrTune(shape, directCandidates, out var compareRec);
            Assert.True(vendorCalls >= 4);
            Assert.False(double.IsNaN(compareRec.BestDirectPtxMedianMicroseconds));
            Assert.False(double.IsNaN(compareRec.BestVendorMedianMicroseconds));
        }
        finally
        {
            FusionTileArtifactCache.OverrideCacheDirectory = prevCacheOverride;
            try { Directory.Delete(tempDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void DevicePredicatedMaskKernels_CpuEquivalentsAndCudaSource_AreDeterministic()
    {
        Assert.Contains("accum_finite_flag_f32", DevicePredicatedMaskKernels.CudaSource);
        Assert.Contains("device_scalar_adamw_update_f32", DevicePredicatedMaskKernels.CudaSource);

        int finiteFlag = 1;
        DevicePredicatedMaskKernels.AccumulateFiniteFlagCpu(new float[] { 1f, -2f, 3.5f }, ref finiteFlag);
        Assert.Equal(1, finiteFlag);
        DevicePredicatedMaskKernels.AccumulateFiniteFlagCpu(new float[] { 1f, float.NaN, 2f }, ref finiteFlag);
        Assert.Equal(0, finiteFlag);

        var haltMask = new float[4];
        DevicePredicatedMaskKernels.ComputePredicatedHaltingMaskCpu(new float[] { 0.2f, 0.95f, 0.89f, 0.91f }, haltMask, 0.9f);
        Assert.Equal(new float[] { 1f, 0f, 1f, 0f }, haltMask);

        var topkMask = new float[6];
        DevicePredicatedMaskKernels.ComputePredicatedTopKMaskCpu(
            new float[] { 0.1f, 0.9f, 0.5f, 0.8f, 0.8f, 0.2f },
            topkMask,
            rows: 2,
            cols: 3,
            k: 2);
        Assert.Equal(new float[] { 0f, 1f, 1f, 1f, 1f, 0f }, topkMask);
    }

    private static FusionTileDescriptor IdentityTile(string name, int inputCount = 1, int scalarCount = 0) =>
        FusionTileLinker.CreatePrebuiltElementwiseTile(
            name,
            inputCount,
            scalarCount,
            new[]
            {
                new FusionTileInstruction(FusionTileOpCode.LoadInput, 0),
                new FusionTileInstruction(FusionTileOpCode.StoreOutput, 0, 0)
            },
            string.Empty);

    [Fact]
    public void AdjointBuilder_SumsGradientsAcrossEveryLoadOfAnInputPort()
    {
        // A horizontal link keeps one LoadInput(0) per sibling; dX must be the sum of both upstream gradients.
        var linked = FusionTileLinker.LinkHorizontalSharedInputs(IdentityTile("adj_id_left"), IdentityTile("adj_id_right"));
        var bwd = FusionTileAdjointBuilder.BuildBackwardTile(linked);
        Assert.Equal(3, bwd.InputCount);
        Assert.Equal(1, bwd.OutputCount);

        var dx = new float[2];
        bwd.CpuExecutor!(
            new[] { new float[] { 0.5f, -1f }, new float[] { 2f, 1f }, new float[] { 3f, 4f } },
            Array.Empty<float>(),
            new[] { dx });
        Assert.Equal(new float[] { 5f, 5f }, dx);
    }

    [Fact]
    public void AdjointBuilder_ZeroInputTile_BackwardWithNoOutputsIsANoOp()
    {
        var constTile = FusionTileLinker.CreatePrebuiltElementwiseTile(
            "adj_const7",
            inputCount: 0,
            scalarCount: 0,
            new[]
            {
                new FusionTileInstruction(FusionTileOpCode.Const, immediate: 7f),
                new FusionTileInstruction(FusionTileOpCode.StoreOutput, 0, 0)
            },
            string.Empty);
        var bwd = FusionTileAdjointBuilder.BuildBackwardTile(constTile);
        Assert.Equal(0, bwd.OutputCount);
        bwd.CpuExecutor!(new[] { new float[] { 1f, 2f } }, Array.Empty<float>(), Array.Empty<float[]>());
    }

    [Fact]
    public void FusionTileDescriptor_SnapshotsInstructions_SoCallerMutationCannotChangeTheTile()
    {
        var insts = new List<FusionTileInstruction>
        {
            new FusionTileInstruction(FusionTileOpCode.LoadInput, 0),
            new FusionTileInstruction(FusionTileOpCode.StoreOutput, 0, 0)
        };
        var tile = FusionTileLinker.CreatePrebuiltElementwiseTile("snapshot_identity", 1, 0, insts, string.Empty);
        string digest = tile.Sha256Digest;

        insts[0] = new FusionTileInstruction(FusionTileOpCode.Const, immediate: 7f);

        Assert.Equal(FusionTileOpCode.LoadInput, tile.Instructions[0].Op);
        Assert.False(tile.Instructions is List<FusionTileInstruction>);
        Assert.False(tile.Instructions is FusionTileInstruction[]);
        Assert.Equal(digest, new FusionTileDescriptor(tile.TileName, tile.Kind, 1, 0, 1, tile.Instructions, string.Empty).Sha256Digest);

        var y = new float[2];
        tile.CpuExecutor!(new[] { new float[] { 1.5f, -2f } }, Array.Empty<float>(), new[] { y });
        Assert.Equal(new float[] { 1.5f, -2f }, y);

        var linked = FusionTileLinker.LinkVertical(tile, IdentityTile("snapshot_consumer"));
        var y2 = new float[2];
        linked.CpuExecutor!(new[] { new float[] { 1.5f, -2f } }, Array.Empty<float>(), new[] { y2 });
        Assert.Equal(new float[] { 1.5f, -2f }, y2);
    }

    [Fact]
    public void FusionTileLinker_RejectsSpecializedTilesAndNonPointwiseOpcodes()
    {
        var flash = FusionTileLinker.CreateTiledFlashAttentionTile(16, isCausal: false);
        var scan = FusionTileLinker.CreateChunkedLinearScanTile(8);
        var id1 = IdentityTile("reject_id1");
        var id3 = IdentityTile("reject_id3", inputCount: 3, scalarCount: 1);

        Assert.Throws<NotSupportedException>(() => FusionTileLinker.LinkVertical(flash, id1));
        Assert.Throws<NotSupportedException>(() => FusionTileLinker.LinkVertical(id1, flash));
        Assert.Throws<NotSupportedException>(() => FusionTileLinker.LinkHorizontalSharedInputs(flash, id3));
        Assert.Throws<NotSupportedException>(() => FusionTileLinker.LinkVertical(scan, id1));

        var rowSum = new[]
        {
            new FusionTileInstruction(FusionTileOpCode.LoadInput, 0),
            new FusionTileInstruction(FusionTileOpCode.RowSum, 0),
            new FusionTileInstruction(FusionTileOpCode.StoreOutput, 1, 0)
        };
        Assert.Throws<NotSupportedException>(() => FusionTileLinker.CreateCpuExecutor(1, 0, 1, rowSum));
        Assert.Throws<NotSupportedException>(() => FusionTileLinker.EmitCudaKernelFromInstructions("k", 1, 0, 1, rowSum));
    }

    [Fact]
    public void GemmEpilogue_RejectsUnsupportedActivationAndUnexpandedBias_AndReducesDBiasPerColumn()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => FusionTileLinker.CreateGemmBiasActivationResidualEpilogueTile(FusionTileOpCode.Exp));
        Assert.Throws<ArgumentOutOfRangeException>(() => FusionTileLinker.CreateGemmBiasActivationResidualEpilogueTile(FusionTileOpCode.Add));

        var relu = FusionTileLinker.CreateGemmBiasActivationResidualEpilogueTile(FusionTileOpCode.Relu);
        var y = new float[4];

        // 2x2 output with a two-element (unexpanded) bias must be rejected rather than read out of range.
        Assert.Throws<ArgumentException>(() => relu.CpuExecutor!(
            new[] { new float[] { 1f, 2f, 3f, 4f }, new float[] { 0.5f, 0.5f }, new float[4] },
            Array.Empty<float>(),
            new[] { y }));

        // Expanded bias: rows = 2, cols = 2, bias = [0.5, 0.25] repeated per row.
        var gemmOut = new float[] { 1f, 2f, 3f, 4f };
        var biasExpanded = new float[] { 0.5f, 0.25f, 0.5f, 0.25f };
        var residual = new float[] { 1f, 2f, 3f, 4f };
        relu.CpuExecutor!(new[] { gemmOut, biasExpanded, residual }, Array.Empty<float>(), new[] { y });
        Assert.Equal(new float[] { 2.5f, 4.25f, 6.5f, 8.25f }, y);

        var bwd = FusionTileAdjointBuilder.BuildBackwardTile(relu);
        var dGemm = new float[4];
        var dBiasExpanded = new float[4];
        var dRes = new float[4];
        var upstream = new float[] { 1f, 2f, 3f, 4f };
        bwd.CpuExecutor!(new[] { gemmOut, biasExpanded, residual, upstream }, Array.Empty<float>(), new[] { dGemm, dBiasExpanded, dRes });
        Assert.Equal(upstream, dGemm);
        Assert.Equal(upstream, dBiasExpanded);
        Assert.Equal(upstream, dRes);

        var dBias = new float[2];
        FusionTileLinker.ReduceExpandedBiasGradient(dBiasExpanded, rows: 2, cols: 2, dBias);
        Assert.Equal(new float[] { 4f, 6f }, dBias);
        Assert.Throws<ArgumentException>(() => FusionTileLinker.ReduceExpandedBiasGradient(dBiasExpanded, 3, 2, dBias));

        // rows == 1 allows the destination to alias the source; the in-place reduction is the identity.
        var aliased = new float[] { 7f, -3f };
        FusionTileLinker.ReduceExpandedBiasGradient(aliased, rows: 1, cols: 2, aliased);
        Assert.Equal(new float[] { 7f, -3f }, aliased);

        // rows == 0 yields a zero gradient.
        var zeroRows = new float[] { 9f, 9f };
        FusionTileLinker.ReduceExpandedBiasGradient(Array.Empty<float>(), rows: 0, cols: 2, zeroRows);
        Assert.Equal(new float[] { 0f, 0f }, zeroRows);
    }

    [Fact]
    public void CudaEmitter_EmitsValidExpressionsForNonFiniteConstants()
    {
        var insts = new[]
        {
            new FusionTileInstruction(FusionTileOpCode.Const, immediate: float.NaN),
            new FusionTileInstruction(FusionTileOpCode.Const, immediate: float.PositiveInfinity),
            new FusionTileInstruction(FusionTileOpCode.Const, immediate: float.NegativeInfinity),
            new FusionTileInstruction(FusionTileOpCode.Const, immediate: 2f),
            new FusionTileInstruction(FusionTileOpCode.StoreOutput, 0, 0)
        };
        string src = FusionTileLinker.EmitCudaKernelFromInstructions("nonfinite_consts", 0, 0, 1, insts);
        Assert.Contains("float r0 = __int_as_float(0x7fc00000);", src);
        Assert.Contains("float r1 = __int_as_float(0x7f800000);", src);
        Assert.Contains("float r2 = (-__int_as_float(0x7f800000));", src);
        Assert.Contains("float r3 = 2.0f;", src);
        Assert.DoesNotContain("NaN", src);
        Assert.DoesNotContain("Infinity", src);

        // The CPU executor produces the same IEEE values.
        var y = new float[1];
        FusionTileLinker.CreateCpuExecutor(0, 0, 1, insts)(Array.Empty<float[]>(), Array.Empty<float>(), new[] { y });
        Assert.True(float.IsNaN(y[0]));
    }

    [Fact]
    public void CrossLayerLayerNormKernel_FinishesReadingReducedSumBeforeScratchReuse()
    {
        string src = FusionTileLinker.CreateCrossLayerNormResidualRopeTile(useLayerNorm: true).CudaDeviceFragment;
        int read = src.IndexOf("float total = smem[0];", StringComparison.Ordinal);
        Assert.True(read >= 0, "block sum must copy smem[0] into a register");
        int barrier = src.IndexOf("__syncthreads();", read, StringComparison.Ordinal);
        int ret = src.IndexOf("return total;", read, StringComparison.Ordinal);
        Assert.True(barrier > read && barrier < ret, "a block barrier must separate the smem[0] read from the return");
    }

    [Fact]
    public void FusionTileArtifactCache_KeepsPrivateCopies_AndRepairsEmptyArtifacts()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "aidn_artifact_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        string? prev = FusionTileArtifactCache.OverrideCacheDirectory;
        try
        {
            FusionTileArtifactCache.OverrideCacheDirectory = tempDir;
            string key = FusionTileArtifactCache.ComputeArtifactKey("artifact-copy-" + Guid.NewGuid().ToString("N"), 8, 6, fastMath: true);

            var producer = new byte[] { 10, 20, 30 };
            FusionTileArtifactCache.StoreArtifact(key, ".cubin", producer);
            producer[0] = 99; // producer reuses its buffer
            Assert.True(FusionTileArtifactCache.TryLoadArtifact(key, ".cubin", out var first));
            Assert.Equal(new byte[] { 10, 20, 30 }, first);
            first[1] = 77; // reader mutates the returned bytes
            Assert.True(FusionTileArtifactCache.TryLoadArtifact(key, ".cubin", out var second));
            Assert.Equal(new byte[] { 10, 20, 30 }, second);
            Assert.Empty(Directory.GetFiles(tempDir, "*.tmp"));

            // An empty on-disk artifact is a miss and must be replaced by the next successful store.
            string emptyKey = FusionTileArtifactCache.ComputeArtifactKey("artifact-empty-" + Guid.NewGuid().ToString("N"), 8, 6, fastMath: true);
            string emptyPath = Path.Combine(tempDir, emptyKey + ".cubin");
            File.WriteAllBytes(emptyPath, Array.Empty<byte>());
            Assert.False(FusionTileArtifactCache.TryLoadArtifact(emptyKey, ".cubin", out _));
            FusionTileArtifactCache.StoreArtifact(emptyKey, ".cubin", new byte[] { 4, 5 });
            Assert.Equal(new byte[] { 4, 5 }, File.ReadAllBytes(emptyPath));
            Assert.Empty(Directory.GetFiles(tempDir, "*.tmp"));
        }
        finally
        {
            FusionTileArtifactCache.OverrideCacheDirectory = prev;
            try { Directory.Delete(tempDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void DirectPtxAutotuner_RetunesWhenEligibleSetChanges_AndMatchesWinnerByOrigin()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "aidn_autotune_set_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        try
        {
            string jsonPath = Path.Combine(tempDir, "tune.json");
            var shape = new DirectPtxShapeKey(86, TunedKernelOp.Gemm, 64, 64, 64);
            int directCalls = 0, vendorCalls = 0;
            var direct = new[] { new AutotuneExecutableCandidate("shared_id", TunedKernelOrigin.Generated, () => directCalls++) };
            // The vendor candidate shares the id but is much slower, so the direct candidate wins.
            var vendor = new FakeVendorProvider(new AutotuneExecutableCandidate("shared_id", TunedKernelOrigin.Vendor, () =>
            {
                vendorCalls++;
                System.Threading.Thread.Sleep(20);
            }));
            DirectPtxAutotunerOptions Opts(bool allowVendor) => new()
            {
                AllowExternalVendorLibraries = allowVendor,
                WarmupRuns = 0,
                MeasuredRuns = 1,
                PersistenceFilePath = jsonPath
            };

            _ = new DirectPtxAutotuner(Opts(false), vendor).SelectOrTune(shape, direct, out var standaloneRec);
            Assert.Equal(0, vendorCalls);
            Assert.True(double.IsNaN(standaloneRec.DirectPtxVsVendorRatio));

            // Same file, vendors now eligible: the cached direct-only record must not be reused.
            var withVendor = new DirectPtxAutotuner(Opts(true), vendor);
            var winner = withVendor.SelectOrTune(shape, direct, out var vendorRec);
            Assert.Equal(1, vendorCalls);
            Assert.False(double.IsNaN(vendorRec.DirectPtxVsVendorRatio));
            Assert.Equal(TunedKernelOrigin.Generated, winner.Origin);
            Assert.Equal(TunedKernelOrigin.Generated, vendorRec.WinnerOrigin);

            // Unchanged eligible set: cache hit, no measurement, same-origin winner.
            int directBefore = directCalls;
            var again = withVendor.SelectOrTune(shape, direct, out _);
            Assert.Same(direct[0], again);
            Assert.Equal(directBefore, directCalls);
            Assert.Equal(1, vendorCalls);

            // Back to direct-only: the record carrying a finite vendor ratio must not be reused.
            var standaloneAgain = new DirectPtxAutotuner(Opts(false), vendor);
            _ = standaloneAgain.SelectOrTune(shape, direct, out var directOnlyRec);
            Assert.True(double.IsNaN(directOnlyRec.DirectPtxVsVendorRatio));
            Assert.Equal(1, vendorCalls);
            Assert.Empty(Directory.GetFiles(tempDir, "*.tmp"));

            // Duplicate (origin, id) identities are ambiguous and rejected.
            var dup = new[]
            {
                new AutotuneExecutableCandidate("dup", TunedKernelOrigin.Generated, () => { }),
                new AutotuneExecutableCandidate("dup", TunedKernelOrigin.Generated, () => { })
            };
            Assert.Throws<ArgumentException>(() => standaloneAgain.SelectOrTune(shape, dup, out _));

            // Reverse case: same id, but now the VENDOR candidate is faster and wins. A cache hit must return
            // the vendor candidate (matched by origin), not the same-id direct candidate, without re-measuring.
            var vendorShape = new DirectPtxShapeKey(86, TunedKernelOp.Gemm, 128, 128, 128);
            int slowDirectCalls = 0, fastVendorCalls = 0;
            var slowDirect = new[]
            {
                new AutotuneExecutableCandidate("shared_id", TunedKernelOrigin.Generated, () =>
                {
                    slowDirectCalls++;
                    System.Threading.Thread.Sleep(20);
                })
            };
            var fastVendorCandidate = new AutotuneExecutableCandidate("shared_id", TunedKernelOrigin.Vendor, () => fastVendorCalls++);
            var vendorWinsTuner = new DirectPtxAutotuner(Opts(true), new FakeVendorProvider(fastVendorCandidate));
            var vendorWinner = vendorWinsTuner.SelectOrTune(vendorShape, slowDirect, out var vendorWinRec);
            Assert.Same(fastVendorCandidate, vendorWinner);
            Assert.Equal(TunedKernelOrigin.Vendor, vendorWinRec.WinnerOrigin);
            Assert.Equal(1, slowDirectCalls);
            Assert.Equal(1, fastVendorCalls);

            var vendorHit = vendorWinsTuner.SelectOrTune(vendorShape, slowDirect, out var vendorHitRec);
            Assert.Same(fastVendorCandidate, vendorHit);
            Assert.Same(vendorWinRec, vendorHitRec);
            Assert.Equal(1, slowDirectCalls);
            Assert.Equal(1, fastVendorCalls);
        }
        finally
        {
            try { Directory.Delete(tempDir, recursive: true); } catch { }
        }
    }

#if NETCOREAPP
    [Fact]
    public void GpuResidentSchedule_CudaGraphTreePool_And_SegmentedRunner_ShareArenasAndPreservePointers()
    {
        var harness = FakeGraphBackendHarness.Create();

        // 1) GpuResidentScheduleScalars updates LR/step/bias-correction without mutating Buffer.Handle.
        float[] uploadedPayload = Array.Empty<float>();
        using var schedule = new GpuResidentScheduleScalars(
            harness.Backend,
            (_, host) => uploadedPayload = (float[])host.Clone());
        IntPtr initialPtr = schedule.Buffer.Handle;
        schedule.UpdateSchedule(learningRate: 3e-4f, weightDecay: 0.01f, beta1: 0.9f, beta2: 0.999f, step: 10, maxGradNorm: 1.0f);
        Assert.Equal(initialPtr, schedule.Buffer.Handle);
        Assert.Equal(3e-4f, schedule.GetSlotValue(GpuResidentScheduleScalars.LearningRateSlot), 6);
        Assert.Equal(10f, schedule.GetSlotValue(GpuResidentScheduleScalars.StepNumberSlot), 6);
        Assert.Equal(8, uploadedPayload.Length);

        // 2) CudaGraphTreePool allocates ONE shared arena of max(buckets)*unit and routes sizes cleanly.
        var opts = new StaticShapeCudaGraphOptions { WarmupIterations = 1, SynchronizeOnReplay = false };
        using var treePool = new CudaGraphTreePool(
            harness.Backend,
            new[] { 64, 128, 256 },
            workspaceElementsPerUnit: 16,
            CudaGraphExecutionScopeKind.InferenceForward,
            opts);

        Assert.Equal(3, treePool.BucketCount);
        Assert.Equal(256 * 16, treePool.SharedArenaCapacityElements);
        Assert.Equal(64, treePool.ResolveBucket(33));
        Assert.Equal(128, treePool.ResolveBucket(128));
        Assert.Equal(-1, treePool.ResolveBucket(257));

        for (int iter = 0; iter < 3; iter++)
        {
            int routed = treePool.ExecuteForSize(50, (bucket, arenaView) =>
            {
                Assert.Equal(64, bucket);
                Assert.Equal(64 * 16, arenaView.Size);
                Assert.Equal(treePool.SharedArenaHandle, arenaView.Handle);
            });
            Assert.Equal(64, routed);
        }
        Assert.True(treePool.GetBucketScope(64).HasCapturedGraph);
        Assert.False(treePool.GetBucketScope(256).HasCapturedGraph);

        // 3) SegmentedBlockCudaGraphRunner shares one block scratch buffer across all segments.
        using var segRunner = new SegmentedBlockCudaGraphRunner(harness.Backend, segmentCount: 2, sharedScratchElements: 512, opts);
        using var segBinding = harness.Backend.AllocateBuffer(64);
        var seg0Handles = new List<IntPtr>();
        var seg1Handles = new List<IntPtr>();
        for (int iter = 0; iter < 3; iter++)
        {
            segRunner.ExecuteSegment(0, scratch =>
            {
                Assert.Equal(512, scratch.Size);
                seg0Handles.Add(scratch.Handle);
            }, new[] { segBinding });
            segRunner.ExecuteSegment(1, scratch =>
            {
                Assert.Equal(512, scratch.Size);
                seg1Handles.Add(scratch.Handle);
            }, new[] { segBinding });
        }
        Assert.True(segRunner.GetSegmentScope(0).HasCapturedGraph);
        Assert.True(segRunner.GetSegmentScope(1).HasCapturedGraph);
        Assert.NotEmpty(seg0Handles);
        Assert.NotEmpty(seg1Handles);
        IntPtr sharedScratch = seg0Handles[0];
        Assert.NotEqual(IntPtr.Zero, sharedScratch);
        Assert.All(seg0Handles, h => Assert.Equal(sharedScratch, h));
        Assert.All(seg1Handles, h => Assert.Equal(sharedScratch, h));
    }

    private sealed class FakeBuffer : IGpuBuffer
    {
        public FakeBuffer(IntPtr handle, int size)
        {
            Handle = handle;
            Size = size;
        }

        public GpuElementType ElementType => GpuElementType.Float32;
        public int Size { get; }
        public long SizeInBytes => (long)Size * sizeof(float);
        public IntPtr Handle { get; private set; }
        public void Dispose() => Handle = IntPtr.Zero;
    }

    private class FakeGraphBackendProxy : DispatchProxy
    {
        internal FakeGraphBackendHarness? Harness { get; set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod is null || Harness is null)
                return null;

            switch (targetMethod.Name)
            {
                case "AllocateBuffer":
                {
                    Harness.NextHandle += 0x10000;
                    int len = args is { Length: > 0 } && args[0] is int sz
                        ? sz
                        : (args is { Length: > 0 } && args[0] is float[] arr ? arr.Length : 8);
                    return new FakeBuffer(new IntPtr(Harness.NextHandle), len);
                }
                case "UploadBufferInPlace":
                    return null;
                case "get_SupportsGraphCapture":
                case "get_ContextIsLive":
                case "get_IsAvailable":
                    return true;
                case "get_CaptureActiveOnThisThread":
                case "IsStreamCapturing":
                    return false;
                case "StreamCaptureStatusRaw":
                    return 0;
                case "get_LastCaptureNodeCounts":
                    return new GraphNodeCounts(2, 0, 0);
                case "CaptureGraph":
                {
                    Harness.Captures++;
                    if (args is { Length: > 0 } && args[0] is Action act)
                    {
                        act();
                    }
                    return new IntPtr(0x9000 + Harness.Captures);
                }
                case "LaunchCapturedGraph":
                case "EnqueueCapturedGraph":
                case "DestroyCapturedGraph":
                    return null;
                default:
                    return targetMethod.ReturnType.IsValueType
                        ? Activator.CreateInstance(targetMethod.ReturnType)
                        : null;
            }
        }
    }

    private sealed class FakeGraphBackendHarness
    {
        public IGpuGraphCapture Backend { get; }
        public long NextHandle { get; set; } = 0x100000;
        public int Captures { get; set; }

        private FakeGraphBackendHarness(IGpuGraphCapture backend) => Backend = backend;

        public static FakeGraphBackendHarness Create()
        {
            var proxy = DispatchProxy.Create<IGpuGraphCapture, FakeGraphBackendProxy>();
            var harness = new FakeGraphBackendHarness(proxy);
            if ((object)proxy is FakeGraphBackendProxy impl)
            {
                impl.Harness = harness;
            }
            return harness;
        }
    }
#endif

    private sealed class FakeVendorProvider : IExternalLibraryCandidateProvider
    {
        private readonly AutotuneExecutableCandidate _candidate;
        public FakeVendorProvider(AutotuneExecutableCandidate candidate) => _candidate = candidate;
        public string ProviderName => "SyntheticVendor";
        public bool IsAvailable => true;
        public IReadOnlyList<AutotuneExecutableCandidate> GetVendorCandidates(DirectPtxShapeKey shape) => new[] { _candidate };
    }

    private static float[] RandomArray(Random rng, int length, float min, float max)
    {
        var arr = new float[length];
        double span = max - min;
        for (int i = 0; i < length; i++)
        {
            arr[i] = (float)(min + rng.NextDouble() * span);
        }
        return arr;
    }

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = Math.Abs(expected[i] - actual[i]);
            Assert.True(diff <= tol, $"Mismatch at [{i}]: expected={expected[i]}, actual={actual[i]}, diff={diff} > {tol}");
        }
    }
}
