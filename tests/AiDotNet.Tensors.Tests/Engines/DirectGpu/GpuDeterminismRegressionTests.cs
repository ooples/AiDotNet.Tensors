// Copyright (c) AiDotNet. All rights reserved.
// Regression test for issue #382: GPU floating-point reductions must be bit-identical
// across runs at the same seed when AiDotNetEngine.SetDeterministicMode(true) is in effect.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Closed-loop regression test for issue #382. Verifies that when the global
/// <see cref="AiDotNetEngine.SetDeterministicMode(bool)"/> flag is set, the GPU
/// embedding gradient — the most directly visible site of the original bug —
/// produces bit-identical output across repeated invocations. The non-deterministic
/// atomic-add kernel would produce per-cell variation on the order of FP rounding
/// noise across runs; the deterministic variant pins accumulation order so the
/// output is byte-equal.
///
/// Skip semantics: only runs when a DirectGPU backend (CUDA, OpenCL, or HIP) is
/// available on the host. The test deliberately uses a shape with many index
/// collisions (multiple input positions targeting the same vocab row) so the atomic
/// kernel's nondeterminism is maximally exercised.
///
/// Mutates a process-wide static flag (DeterministicMode), so the test joins the
/// existing BlasGlobalState collection to serialize against other deterministic-mode
/// tests in the suite.
/// </summary>
[Collection("BlasGlobalState")]
public class GpuDeterminismRegressionTests
{
    private readonly bool _isDirectGpuAvailable;

    public GpuDeterminismRegressionTests()
    {
        try
        {
            using var probe = new DirectGpuTensorEngine();
            _isDirectGpuAvailable = probe.IsGpuAvailable;
        }
        catch
        {
            _isDirectGpuAvailable = false;
        }
    }

    [SkippableFact]
    public void EmbeddingBackward_DeterministicMode_BitIdenticalAcrossRuns()
    {
        Skip.IfNot(_isDirectGpuAvailable, "DirectGPU backend (CUDA/OpenCL/HIP) not available on this host");

        bool originalDet = AiDotNetEngine.DeterministicMode;
        IEngine originalEngine = AiDotNetEngine.Current;

        try
        {
            AiDotNetEngine.SetDeterministicMode(true);
            AiDotNetEngine.Current = new DirectGpuTensorEngine();

            // Shape chosen so multiple input positions target the same vocab row —
            // this is exactly the collision pattern that triggered the atomic
            // ordering nondeterminism in the original kernel. Small enough to run
            // quickly but large enough to maximize collision count.
            const int vocabSize = 16;
            const int embeddingDim = 32;
            const int numIndices = 256;

            var rng = new Random(42);
            int[] indicesData = new int[numIndices];
            for (int i = 0; i < numIndices; i++)
            {
                indicesData[i] = rng.Next(0, vocabSize);
            }

            var gradOutput = new Tensor<float>([numIndices, embeddingDim]);
            for (int i = 0; i < numIndices * embeddingDim; i++)
            {
                gradOutput[i] = (float)(rng.NextDouble() * 2 - 1);
            }

            var indices = new Tensor<int>([numIndices]);
            for (int i = 0; i < numIndices; i++) indices[i] = indicesData[i];

            // Run four times and collect outputs. Under the atomic kernel, runs 1-4
            // would differ in low bits; under the deterministic kernel they must be
            // byte-equal.
            var snapshots = new float[4][];
            for (int run = 0; run < 4; run++)
            {
                var result = AiDotNetEngine.Current.EmbeddingBackward(gradOutput, indices, vocabSize, embeddingDim);
                snapshots[run] = new float[vocabSize * embeddingDim];
                for (int i = 0; i < vocabSize * embeddingDim; i++) snapshots[run][i] = result[i];
            }

            // Run 0 is reference; runs 1-3 must be byte-equal.
            for (int run = 1; run < 4; run++)
            {
                for (int i = 0; i < vocabSize * embeddingDim; i++)
                {
                    Assert.Equal(snapshots[0][i], snapshots[run][i]);
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = originalEngine;
            AiDotNetEngine.SetDeterministicMode(originalDet);
        }
    }
}
