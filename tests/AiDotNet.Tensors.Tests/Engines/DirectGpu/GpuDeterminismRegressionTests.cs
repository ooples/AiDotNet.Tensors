// Copyright (c) AiDotNet. All rights reserved.
// Regression test for issue #382: GPU floating-point reductions must be bit-identical
// across runs at the same seed when AiDotNetEngine.SetDeterministicMode(true) is in effect.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
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
/// Mutates the process-wide <see cref="AiDotNetEngine.Current"/> and the
/// <see cref="AiDotNetEngine.SetDeterministicMode(bool)"/> static flag — which
/// propagates to <c>BlasProvider.IsDeterministicMode</c> and switches the GEMM
/// reduction/packing path — so the test joins the unified
/// <c>BlasManaged-Stats-Serial</c> collection to serialize against every other test
/// that touches those statics (the pre-pack / scalar-kernel GEMM tests, the CPU
/// DeterministicMode tests, …). This is the fix for the cross-test "drift" flakiness
/// those GEMM tests intermittently hit on CI.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
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

        // Capture the *raw* thread-local override (null = "no override"), not the
        // merged effective value from AiDotNetEngine.DeterministicMode. Restoring
        // the merged value would erase the distinction between "no override" and
        // "override = process-wide value", which behave differently when the
        // process-wide flag is later flipped by an unrelated test.
        bool? originalThreadLocalDet = BlasProvider.GetThreadLocalDeterministicMode();
        bool originalProcessDet = AiDotNetEngine.DeterministicMode;
        IEngine originalEngine = AiDotNetEngine.Current;
        // Holds the DirectGpuTensorEngine we install so it can be disposed in
        // `finally` after Current is restored — otherwise the native GPU context
        // leaks across tests.
        DirectGpuTensorEngine? installedEngine = null;

        try
        {
            AiDotNetEngine.SetDeterministicMode(true);
            installedEngine = new DirectGpuTensorEngine();
            AiDotNetEngine.Current = installedEngine;

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

            // Run 0 is reference; runs 1-3 must be bit-equal. Compare the raw IEEE-754
            // payload (single-precision word) so +0/-0 differ — Assert.Equal(float, float)
            // treats them as equal — and so NaN payload differences are caught.
            // Uses BitConverter.GetBytes/ToInt32 (works on both net471 and net10.0) in
            // place of SingleToInt32Bits (net6.0+ only).
            for (int run = 1; run < 4; run++)
            {
                for (int i = 0; i < vocabSize * embeddingDim; i++)
                {
                    int refBits = BitConverter.ToInt32(BitConverter.GetBytes(snapshots[0][i]), 0);
                    int runBits = BitConverter.ToInt32(BitConverter.GetBytes(snapshots[run][i]), 0);
                    Assert.Equal(refBits, runBits);
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = originalEngine;
            installedEngine?.Dispose();
            AiDotNetEngine.SetDeterministicMode(originalProcessDet);
            BlasProvider.SetThreadLocalDeterministicMode(originalThreadLocalDet);
        }
    }

    /// <summary>
    /// Issue #742 (cross-hardware determinism hardening): the scatter-add gradient reduction
    /// (CudaBackend.ScatterAdd, used by TensorScatterAdd / embedding scatter-add grad) accumulates
    /// via atomicAdd by default — bit-stable on some GPUs but NOT guaranteed reproducible across all
    /// GPUs/drivers because fp32 add is non-associative and atomic completion order is scheduler-
    /// dependent. Under SetDeterministicMode(true) it must route to a fixed-order accumulating variant
    /// (scatter_add_accumulate_deterministic) that is bit-identical across runs on ANY hardware.
    /// This test asserts (a) byte-identical output across repeated deterministic-mode runs, and
    /// (b) the deterministic result equals the fast atomicAdd result to fp32 tolerance (correctness
    /// preserved — same math, just a fixed reduction order).
    /// </summary>
    [SkippableFact]
    public void ScatterAdd_DeterministicMode_BitIdenticalAndMatchesAtomic()
    {
        Skip.IfNot(_isDirectGpuAvailable, "DirectGPU backend (CUDA/OpenCL/HIP) not available on this host");

        bool? originalThreadLocalDet = BlasProvider.GetThreadLocalDeterministicMode();
        bool originalProcessDet = AiDotNetEngine.DeterministicMode;
        IEngine originalEngine = AiDotNetEngine.Current;
        DirectGpuTensorEngine? installedEngine = null;
        try
        {
            installedEngine = new DirectGpuTensorEngine();
            AiDotNetEngine.Current = installedEngine;

            // Heavy-collision scatter: 512 updates onto 32 cells → ~16 duplicate indices per cell,
            // the pattern that maximizes atomicAdd ordering sensitivity. Destination is pre-seeded
            // (the += target) so this exercises the accumulating deterministic variant, not overwrite.
            const int dstSize = 32, n = 512;
            var rng = new Random(123);
            var dest = new Tensor<float>([dstSize]);
            for (int i = 0; i < dstSize; i++) dest[i] = (float)(rng.NextDouble() * 2 - 1);
            var updates = new Tensor<float>([n]);
            for (int i = 0; i < n; i++) updates[i] = (float)(rng.NextDouble() * 2 - 1);
            var indices = new Tensor<int>([n]);
            for (int i = 0; i < n; i++) indices[i] = rng.Next(0, dstSize);

            int Bits(float f) { var b = BitConverter.GetBytes(f); return BitConverter.ToInt32(b, 0); }

            // Fast (atomicAdd) reference.
            AiDotNetEngine.SetDeterministicMode(false);
            var atomicResult = AiDotNetEngine.Current.TensorScatterAdd(dest, indices, updates, 0);

            // Deterministic variant, 3 runs → must be byte-identical run-to-run.
            AiDotNetEngine.SetDeterministicMode(true);
            var snapshots = new float[3][];
            for (int run = 0; run < 3; run++)
            {
                var r = AiDotNetEngine.Current.TensorScatterAdd(dest, indices, updates, 0);
                snapshots[run] = new float[dstSize];
                for (int i = 0; i < dstSize; i++) snapshots[run][i] = r[i];
            }
            for (int run = 1; run < 3; run++)
                for (int i = 0; i < dstSize; i++)
                    Assert.True(Bits(snapshots[run][i]) == Bits(snapshots[0][i]),
                        $"deterministic ScatterAdd run {run} diverged from run 0 at [{i}]: {snapshots[run][i]} vs {snapshots[0][i]}");

            // Correctness: deterministic result matches the atomicAdd result to fp32 tolerance.
            double maxAbs = 0;
            for (int i = 0; i < dstSize; i++)
            {
                double d = Math.Abs((double)snapshots[0][i] - atomicResult[i]);
                if (d > maxAbs) maxAbs = d;
            }
            Assert.True(maxAbs < 1e-3,
                $"deterministic ScatterAdd diverged from atomicAdd result (max_abs={maxAbs:E3}) — reduction changed the math, not just the order");
        }
        finally
        {
            AiDotNetEngine.Current = originalEngine;
            installedEngine?.Dispose();
            AiDotNetEngine.SetDeterministicMode(originalProcessDet);
            BlasProvider.SetThreadLocalDeterministicMode(originalThreadLocalDet);
        }
    }
}
