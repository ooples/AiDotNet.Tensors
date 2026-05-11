// Copyright (c) AiDotNet. All rights reserved.
// Issue #319 — stress tests for the structural autograd refactor
// (BackwardScratch thread-local pool, GradNodePool with owning-tape
// stamping, TensorAllocator.Rent<T>(int[], T[]) adopt-not-copy).
//
// These tests are designed to TRIGGER cross-cutting issues that
// would not show up in the existing per-op autodiff suite:
//
//   1. Nested backward (createGraph + GradientCheckpointing) —
//      validates the owning-tape stamp on GradNode prevents inner
//      cleanups from returning outer-tape nodes to the pool.
//   2. Many op types in one tape — ensures the pool dispatch handles
//      heterogeneous unary/binary/variadic Record calls correctly.
//   3. Repeated iteration on the same tape topology — checks that
//      the pool's rent/return cycle doesn't accumulate state
//      across iterations.
//   4. Multi-thread independence — ThreadStatic pool isolation.

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public class Issue319BackwardPoolStressTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void ManyOpsInOneTape_NoCorruption()
    {
        // Exercises unary, binary, broadcast, reduction, and
        // activation ops in a single backward. Each op type takes
        // a different code path through DifferentiableOps (Record1
        // vs Record2 vs RecordIfActive variadic), so this catches
        // any pool-bookkeeping inconsistency.
        const int N = 64;
        var x = MakeRandom(N, 0.5, 42);
        var w = MakeRandom(N, 0.1, 43);
        var b = MakeRandom(N, 0.01, 44);

        using var tape = new GradientTape<float>();
        var z = _engine.TensorMultiply(x, w);          // binary
        z = _engine.TensorAdd(z, b);                    // binary
        z = _engine.ReLU(z);                            // unary
        z = _engine.TensorMultiplyScalar(z, 2f);        // unary + scalar savedState
        z = _engine.Sigmoid(z);                         // unary, fused-engine
        z = _engine.GELU(z);                            // unary, fast SIMD
        z = _engine.Tanh(z);                            // unary
        z = _engine.TensorNegate(z);                    // unary
        var loss = _engine.ReduceSum(z, axes: null, keepDims: false);

        var grads = tape.ComputeGradients(loss, sources: new[] { x, w, b });

        // Every source should have a gradient.
        Assert.True(grads.ContainsKey(x));
        Assert.True(grads.ContainsKey(w));
        Assert.True(grads.ContainsKey(b));

        // No NaN/Inf in any output gradient.
        foreach (var g in new[] { grads[x], grads[w], grads[b] })
        {
            var span = g.AsSpan();
            for (int i = 0; i < span.Length; i++)
            {
                Assert.False(float.IsNaN(span[i]),
                    $"Gradient contains NaN — pool reuse likely corrupted state");
                Assert.False(float.IsInfinity(span[i]),
                    $"Gradient contains Inf — pool reuse likely corrupted state");
            }
        }
    }

    [Fact]
    public void RepeatedBackwardOnSameTopology_NoPoolPoisoning()
    {
        // Runs the same training step 20 times — the pool should
        // rent and return nodes each iteration without accumulating
        // stale state. If the cleanup walk misses any node, the
        // pool eventually returns a node with stale fields and a
        // later RecordUnary overwrites them — but the topo sort
        // would already have seen the corrupted version.
        const int N = 32;
        const int Iters = 20;
        var x = MakeRandom(N, 0.5, 1);
        var w = MakeRandom(N, 0.1, 2);

        for (int iter = 0; iter < Iters; iter++)
        {
            using var tape = new GradientTape<float>();
            var z = _engine.TensorMultiply(x, w);
            z = _engine.ReLU(z);
            z = _engine.TensorMultiplyScalar(z, 1.1f);
            var loss = _engine.ReduceSum(z, axes: null, keepDims: false);

            var grads = tape.ComputeGradients(loss, sources: new[] { w });
            Assert.True(grads.ContainsKey(w),
                $"Iter {iter}: w missing from grads — pool corruption?");
            var gSpan = grads[w].AsSpan();
            for (int i = 0; i < gSpan.Length; i++)
                Assert.False(float.IsNaN(gSpan[i]) || float.IsInfinity(gSpan[i]),
                    $"Iter {iter} idx {i}: non-finite gradient");
        }
    }

    [Fact]
    public void NestedTape_CreateGraph_DoesNotPoolOuterNodes()
    {
        // createGraph=true makes the outer backward record its own
        // backward ops onto the tape — the resulting GradNodes
        // must NOT be returned to the pool during the outer
        // cleanup (the cleanup is gated on
        // !DifferentiableOps._isBackwardCreateGraph).
        const int N = 16;
        var x = MakeRandom(N, 0.5, 7);

        using var tape = new GradientTape<float>();
        var y = _engine.TensorMultiplyScalar(x, 3f);
        y = _engine.ReLU(y);
        var loss = _engine.ReduceSum(y, axes: null, keepDims: false);

        // First backward with createGraph=true — records backward
        // ops on the tape for a hypothetical second-order pass.
        var grads = tape.ComputeGradients(loss, sources: new[] { x }, createGraph: true);
        Assert.True(grads.ContainsKey(x));
        // First-order gradient should still be finite.
        var span = grads[x].AsSpan();
        for (int i = 0; i < span.Length; i++)
            Assert.False(float.IsNaN(span[i]) || float.IsInfinity(span[i]));
    }

    [Fact]
    public void ConcurrentTapes_DifferentThreads_NoCrossThreadPoolBleed()
    {
        // The BackwardScratch + GradNodePool are ThreadStatic.
        // Two threads doing independent tapes simultaneously must
        // not share pool state — otherwise one thread's Return
        // could corrupt the other thread's in-flight backward.
        const int N = 32;
        const int ThreadCount = 4;
        const int IterPerThread = 10;

        var tasks = new Task[ThreadCount];
        var errors = new List<string>();
        var errorsLock = new object();

        for (int t = 0; t < ThreadCount; t++)
        {
            int tid = t;
            tasks[t] = Task.Run(() =>
            {
                try
                {
                    var localEngine = new CpuEngine();
                    var x = MakeRandom(N, 0.5, 1000 + tid);
                    var w = MakeRandom(N, 0.1, 2000 + tid);
                    for (int iter = 0; iter < IterPerThread; iter++)
                    {
                        using var tape = new GradientTape<float>();
                        var z = localEngine.TensorMultiply(x, w);
                        z = localEngine.GELU(z);
                        var loss = localEngine.ReduceSum(z, axes: null, keepDims: false);
                        var grads = tape.ComputeGradients(loss, sources: new[] { w });
                        if (!grads.ContainsKey(w))
                            throw new Exception($"thread {tid} iter {iter}: w missing");
                        var span = grads[w].AsSpan();
                        for (int i = 0; i < span.Length; i++)
                            if (float.IsNaN(span[i]) || float.IsInfinity(span[i]))
                                throw new Exception($"thread {tid} iter {iter} idx {i}: non-finite");
                    }
                }
                catch (Exception ex)
                {
                    lock (errorsLock) errors.Add(ex.Message);
                }
            });
        }

        Task.WaitAll(tasks);
        Assert.Empty(errors);
    }

    [Fact]
    public void DeepChain_TopologicalSort_HandlesPooledNodesCorrectly()
    {
        // Long chain of unary ops — the topological sort recurses
        // through .Input0.GradFn pointers. With the pool active,
        // every node in the chain has been Rented; the cleanup walk
        // must visit each exactly once without infinite recursion
        // or missed nodes.
        const int N = 16;
        const int Depth = 50;
        var x = MakeRandom(N, 0.1, 99);

        using var tape = new GradientTape<float>();
        var z = x;
        for (int d = 0; d < Depth; d++)
        {
            z = _engine.TensorMultiplyScalar(z, 1.01f);
        }
        var loss = _engine.ReduceSum(z, axes: null, keepDims: false);
        var grads = tape.ComputeGradients(loss, sources: new[] { x });

        Assert.True(grads.ContainsKey(x));
        var span = grads[x].AsSpan();
        for (int i = 0; i < span.Length; i++)
        {
            Assert.False(float.IsNaN(span[i]) || float.IsInfinity(span[i]));
            // After 50 chained 1.01x scaling, every grad component
            // should be exactly 1.01^50 ≈ 1.6446 (each x is independent).
            float expected = (float)Math.Pow(1.01, Depth);
            Assert.InRange(span[i], expected * 0.99f, expected * 1.01f);
        }
    }

    [Fact]
    public void DiamondGraph_SharedInput_HandlesPooledNodesCorrectly()
    {
        // Diamond: z = f(x) + g(x). The topological sort visits x
        // (or its first GradFn) from TWO paths but must add it to
        // the result list exactly once. With pooled GradNodes,
        // this also confirms the visited HashSet's identity
        // comparison stays stable across rent/return.
        const int N = 16;
        var x = MakeRandom(N, 0.3, 11);

        using var tape = new GradientTape<float>();
        var a = _engine.TensorMultiplyScalar(x, 2f);
        var b = _engine.ReLU(x);
        var z = _engine.TensorAdd(a, b);
        var loss = _engine.ReduceSum(z, axes: null, keepDims: false);

        var grads = tape.ComputeGradients(loss, sources: new[] { x });
        Assert.True(grads.ContainsKey(x));
        var gx = grads[x].AsSpan();
        // d/dx (2x + ReLU(x)) = 2 + (x>0 ? 1 : 0) — for positive x, 3; for negative, 2.
        var xSpan = x.AsSpan();
        for (int i = 0; i < N; i++)
        {
            float expected = xSpan[i] > 0 ? 3f : 2f;
            Assert.InRange(gx[i], expected - 0.01f, expected + 0.01f);
        }
    }

    private static Tensor<float> MakeRandom(int length, double scale, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { length });
        var s = t.AsWritableSpan();
        for (int i = 0; i < length; i++) s[i] = (float)((rng.NextDouble() - 0.5) * 2 * scale);
        return t;
    }
}
