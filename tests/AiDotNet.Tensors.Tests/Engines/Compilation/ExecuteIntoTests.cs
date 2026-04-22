using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Tests for issue #199 — <see cref="ICompiledPlan{T}.ExecuteInto"/> and
/// <see cref="ICompiledPlan{T}.SetInputs"/>. These are the primitives that
/// make a compiled plan safely composable with CUDA Graph capture: both
/// rebind storage so every Replay reads and writes the caller's buffers
/// instead of the plan's compile-time allocations.
/// </summary>
public class ExecuteIntoTests
{
    [Fact]
    public void ExecuteInto_WritesIntoCallerBuffer()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([4, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);

        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.TensorMatMul(input, weight);
            plan = scope.CompileInference<float>();
        }

        using (plan)
        {
            var outputBuf = new Tensor<float>([4, 2]);
            plan.ExecuteInto(outputBuf);
            // Buffer must hold the matmul result — not all zeros.
            bool anyNonZero = false;
            foreach (var v in outputBuf.AsSpan()) if (v != 0f) { anyNonZero = true; break; }
            Assert.True(anyNonZero, "ExecuteInto should have written into the caller's buffer");
        }
    }

    [Fact]
    public void ExecuteInto_SameBuffer_ReplaysAreStableAcrossCalls()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);

        ICompiledPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.TensorMatMul(input, weight);
            plan = scope.CompileInference<float>();
        }

        using (plan)
        {
            var outputBuf = new Tensor<float>([2, 2]);
            plan.ExecuteInto(outputBuf);
            var first = outputBuf.AsSpan().ToArray();
            plan.ExecuteInto(outputBuf);
            var second = outputBuf.AsSpan().ToArray();
            Assert.Equal(first, second);
        }
    }

    [Fact]
    public void ExecuteInto_MatchesExecute_BitExact()
    {
        // ExecuteInto just rebinds the plan's final-output buffer — the
        // numerical result must be identical to Execute() (same kernels,
        // same accumulation order).
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([8, 4]);
        var weight = Tensor<float>.CreateRandom([4, 6]);

        using var planA = BuildMatmulPlan(engine, input, weight);
        using var planB = BuildMatmulPlan(engine, input, weight);

        var viaExecute = planA.Execute();
        var outBuf = new Tensor<float>([8, 6]);
        planB.ExecuteInto(outBuf);

        Assert.Equal(viaExecute.AsSpan().ToArray(), outBuf.AsSpan().ToArray());
    }

    [Fact]
    public void ExecuteInto_NullOutput_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        using var plan = BuildMatmulPlan(engine, input, weight);
        Assert.Throws<ArgumentNullException>(() => plan.ExecuteInto(null!));
    }

    [Fact]
    public void ExecuteInto_WrongShape_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 4]);
        using var plan = BuildMatmulPlan(engine, input, weight);
        var wrongShape = new Tensor<float>([2, 5]);
        Assert.Throws<ArgumentException>(() => plan.ExecuteInto(wrongShape));
    }

    [Fact]
    public void SetInputs_NewInputData_ChangesOutput()
    {
        // SetInputs rebinds the captured input tensor to the caller's buffer.
        // After the rebind, mutating the caller's buffer and running Execute
        // must produce output computed against the NEW data.
        var engine = new CpuEngine();
        var compiledInput = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);

        using var plan = BuildMatmulPlan(engine, compiledInput, weight);

        // Run once through the compile-time input — anchor output.
        var outAnchor = plan.Execute().AsSpan().ToArray();

        // Build a fresh input tensor, set via SetInputs, and run again.
        var newInput = new Tensor<float>([2, 3]);
        var ns = newInput.AsWritableSpan();
        for (int i = 0; i < ns.Length; i++) ns[i] = i * 0.5f;
        plan.SetInputs(new[] { newInput });
        var outNew = plan.Execute().AsSpan().ToArray();

        // Different input → different output (they must not trivially coincide).
        Assert.NotEqual(outAnchor, outNew);

        // Verify output matches the expected matmul on the new input directly.
        // The compiled plan and the eager TensorMatMul can use different GEMM
        // kernels (specialized SIMD vs reference), whose accumulation orders
        // differ by a ULP or two at single-precision. Compare with a float-
        // scale tolerance (1e-5 abs) rather than bit-exact equality, which
        // would make this test flake on any CPU/JIT state change.
        var expected = engine.TensorMatMul(newInput, weight).AsSpan().ToArray();
        Assert.Equal(expected.Length, outNew.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - outNew[i]);
            Assert.True(diff < 1e-5f,
                $"outNew[{i}] = {outNew[i]}, expected = {expected[i]}, diff = {diff}");
        }
    }

    [Fact]
    public void SetInputs_WrongCount_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        using var plan = BuildMatmulPlan(engine, input, weight);
        var two = new[] { new Tensor<float>([2, 2]), new Tensor<float>([2, 2]) };
        Assert.Throws<ArgumentException>(() => plan.SetInputs(two));
    }

    [Fact]
    public void SetInputs_NullArray_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        using var plan = BuildMatmulPlan(engine, input, weight);
        Assert.Throws<ArgumentNullException>(() => plan.SetInputs(null!));
    }

    [Fact]
    public void SetInputs_ShapeMismatch_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 3]);
        var weight = Tensor<float>.CreateRandom([3, 2]);
        using var plan = BuildMatmulPlan(engine, input, weight);
        var wrongShape = new Tensor<float>([3, 3]);
        // RebindStorageFrom throws ArgumentException on shape mismatch.
        Assert.Throws<ArgumentException>(() => plan.SetInputs(new[] { wrongShape }));
    }

    [Fact]
    public void ExecuteInto_AfterDispose_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        var plan = BuildMatmulPlan(engine, input, weight);
        plan.Dispose();
        var outBuf = new Tensor<float>([2, 2]);
        Assert.Throws<ObjectDisposedException>(() => plan.ExecuteInto(outBuf));
    }

    [Fact]
    public void SetInputs_AfterDispose_Throws()
    {
        var engine = new CpuEngine();
        var input = Tensor<float>.CreateRandom([2, 2]);
        var weight = Tensor<float>.CreateRandom([2, 2]);
        var plan = BuildMatmulPlan(engine, input, weight);
        plan.Dispose();
        Assert.Throws<ObjectDisposedException>(() => plan.SetInputs(new[] { new Tensor<float>([2, 2]) }));
    }

    private static ICompiledPlan<float> BuildMatmulPlan(CpuEngine engine, Tensor<float> input, Tensor<float> weight)
    {
        using var scope = GraphMode.Enable();
        engine.TensorMatMul(input, weight);
        return scope.CompileInference<float>();
    }
}
