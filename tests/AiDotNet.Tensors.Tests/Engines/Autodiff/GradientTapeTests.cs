using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Tests for the GradientTape infrastructure: lifecycle, nesting, recording, and gradient computation.
/// </summary>
public class GradientTapeTests
{
    private readonly CpuEngine _engine = new();

    // ──────────────────────────────────────────────────────────────
    // Tape lifecycle
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Tape_SetsCurrentOnConstruct()
    {
        Assert.Null(GradientTape<float>.Current);
        using var tape = new GradientTape<float>();
        Assert.NotNull(GradientTape<float>.Current);
        Assert.Same(tape, GradientTape<float>.Current);
    }

    [Fact]
    public void Tape_RestoresNullOnDispose()
    {
        using (var tape = new GradientTape<float>())
        {
            Assert.NotNull(GradientTape<float>.Current);
        }
        Assert.Null(GradientTape<float>.Current);
    }

    [Fact]
    public void Tape_NestedTapesWork()
    {
        using var outer = new GradientTape<float>();
        Assert.Same(outer, GradientTape<float>.Current);

        using (var inner = new GradientTape<float>())
        {
            Assert.Same(inner, GradientTape<float>.Current);
            Assert.NotSame(outer, GradientTape<float>.Current);
        }

        Assert.Same(outer, GradientTape<float>.Current);
    }

    [Fact]
    public void Tape_RecordCountIncreases()
    {
        using var tape = new GradientTape<float>();

        var a = new Tensor<float>(new[] { 2, 2 }, new Vector<float>(new float[] { 1, 2, 3, 4 }));
        var b = new Tensor<float>(new[] { 2, 2 }, new Vector<float>(new float[] { 5, 6, 7, 8 }));

        Assert.Equal(0, tape.EntryCount);
        _engine.TensorAdd(a, b);
        Assert.Equal(1, tape.EntryCount);
        _engine.TensorMultiply(a, b);
        Assert.Equal(2, tape.EntryCount);
    }

    [Fact]
    public void Tape_NoRecordingWhenNoTapeActive()
    {
        // No tape active — operations should work without any recording overhead
        var a = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 1, 2, 3 }));
        var b = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 4, 5, 6 }));
        var result = _engine.TensorAdd(a, b);

        Assert.Equal(5f, result[0]);
        Assert.Equal(7f, result[1]);
        Assert.Equal(9f, result[2]);
    }

    [Fact]
    public void Tape_ResetClearsEntries()
    {
        using var tape = new GradientTape<float>();
        var a = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 1, 2 }));
        var b = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 3, 4 }));

        _engine.TensorAdd(a, b);
        Assert.Equal(1, tape.EntryCount);

        tape.Reset();
        Assert.Equal(0, tape.EntryCount);
    }

    [Fact]
    public void Tape_MaxEntriesEnforced()
    {
        var options = new GradientTapeOptions { MaxEntries = 2 };
        using var tape = new GradientTape<float>(options);

        var a = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 1, 2 }));
        var b = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 3, 4 }));

        _engine.TensorAdd(a, b);
        _engine.TensorSubtract(a, b);
        _engine.TensorMultiply(a, b);

        Assert.Equal(2, tape.EntryCount);
    }

    // ──────────────────────────────────────────────────────────────
    // Gradient computation: simple operations
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_Add_CorrectGradients()
    {
        // z = a + b
        // dz/da = 1, dz/db = 1
        var a = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 1, 2, 3 }));
        var b = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 4, 5, 6 }));

        using var tape = new GradientTape<float>();
        var z = _engine.TensorAdd(a, b);
        var grads = tape.ComputeGradients(z, new[] { a, b });

        // Gradient should be all ones
        Assert.True(grads.ContainsKey(a));
        Assert.True(grads.ContainsKey(b));
        Assert.Equal(1f, grads[a][0], 5);
        Assert.Equal(1f, grads[a][1], 5);
        Assert.Equal(1f, grads[a][2], 5);
        Assert.Equal(1f, grads[b][0], 5);
        Assert.Equal(1f, grads[b][1], 5);
        Assert.Equal(1f, grads[b][2], 5);
    }

    [Fact]
    public void Gradient_Subtract_CorrectGradients()
    {
        // z = a - b
        // dz/da = 1, dz/db = -1
        var a = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 5, 3 }));
        var b = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 2, 1 }));

        using var tape = new GradientTape<float>();
        var z = _engine.TensorSubtract(a, b);
        var grads = tape.ComputeGradients(z, new[] { a, b });

        Assert.Equal(1f, grads[a][0], 5);
        Assert.Equal(1f, grads[a][1], 5);
        Assert.Equal(-1f, grads[b][0], 5);
        Assert.Equal(-1f, grads[b][1], 5);
    }

    [Fact]
    public void Gradient_Multiply_CorrectGradients()
    {
        // z = a * b (element-wise)
        // dz/da = b, dz/db = a
        var a = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 2, 3, 4 }));
        var b = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 5, 6, 7 }));

        using var tape = new GradientTape<float>();
        var z = _engine.TensorMultiply(a, b);
        var grads = tape.ComputeGradients(z, new[] { a, b });

        // dz/da = b
        Assert.Equal(5f, grads[a][0], 5);
        Assert.Equal(6f, grads[a][1], 5);
        Assert.Equal(7f, grads[a][2], 5);
        // dz/db = a
        Assert.Equal(2f, grads[b][0], 5);
        Assert.Equal(3f, grads[b][1], 5);
        Assert.Equal(4f, grads[b][2], 5);
    }

    [Fact]
    public void Gradient_Negate_CorrectGradient()
    {
        // z = -x
        // dz/dx = -1
        var x = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 1, 2, 3 }));

        using var tape = new GradientTape<float>();
        var z = _engine.TensorNegate(x);
        var grads = tape.ComputeGradients(z, new[] { x });

        Assert.Equal(-1f, grads[x][0], 5);
        Assert.Equal(-1f, grads[x][1], 5);
        Assert.Equal(-1f, grads[x][2], 5);
    }

    [Fact]
    public void Gradient_Exp_CorrectGradient()
    {
        // z = exp(x)
        // dz/dx = exp(x) = z
        var x = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 0f, 1f }));

        using var tape = new GradientTape<float>();
        var z = _engine.TensorExp(x);
        var grads = tape.ComputeGradients(z, new[] { x });

        // exp(0) = 1, exp(1) = e
        Assert.Equal(1f, grads[x][0], 4);
        Assert.Equal(MathF.E, grads[x][1], 3);
    }

    [Fact]
    public void Gradient_Log_CorrectGradient()
    {
        // z = log(x)
        // dz/dx = 1/x
        var x = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 1f, 2f }));

        using var tape = new GradientTape<float>();
        var z = _engine.TensorLog(x);
        var grads = tape.ComputeGradients(z, new[] { x });

        Assert.Equal(1f, grads[x][0], 4);      // 1/1 = 1
        Assert.Equal(0.5f, grads[x][1], 4);     // 1/2 = 0.5
    }

    [Fact]
    public void Gradient_MatMul_CorrectGradients()
    {
        // Z = A @ B
        // dZ/dA = grad @ B^T, dZ/dB = A^T @ grad
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var b = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });

        using var tape = new GradientTape<float>();
        var z = _engine.TensorMatMul(a, b);
        var grads = tape.ComputeGradients(z, new[] { a, b });

        // grad is all ones (2x2)
        // dZ/dA = ones @ B^T = [[5+7, 6+8], [5+7, 6+8]] = [[12,14],[12,14]]
        // Wait: ones @ B^T where B^T = [[5,7],[6,8]]
        // [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        Assert.True(grads.ContainsKey(a));
        Assert.True(grads.ContainsKey(b));

        // dZ/dA = ones(2,2) @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        Assert.Equal(11f, grads[a][0, 0], 3);
        Assert.Equal(15f, grads[a][0, 1], 3);
        Assert.Equal(11f, grads[a][1, 0], 3);
        Assert.Equal(15f, grads[a][1, 1], 3);

        // dZ/dB = A^T @ ones(2,2) = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        Assert.Equal(4f, grads[b][0, 0], 3);
        Assert.Equal(4f, grads[b][0, 1], 3);
        Assert.Equal(6f, grads[b][1, 0], 3);
        Assert.Equal(6f, grads[b][1, 1], 3);
    }

    // ──────────────────────────────────────────────────────────────
    // Chain rule: composite operations
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_ChainRule_AddThenMultiply()
    {
        // z = (a + b) * c
        // dz/da = c, dz/db = c, dz/dc = a + b
        var a = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 1, 2 }));
        var b = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 3, 4 }));
        var c = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 5, 6 }));

        using var tape = new GradientTape<float>();
        var sum = _engine.TensorAdd(a, b);          // [4, 6]
        var z = _engine.TensorMultiply(sum, c);      // [20, 36]
        var grads = tape.ComputeGradients(z, new[] { a, b, c });

        // dz/da = c = [5, 6]
        Assert.Equal(5f, grads[a][0], 4);
        Assert.Equal(6f, grads[a][1], 4);

        // dz/db = c = [5, 6]
        Assert.Equal(5f, grads[b][0], 4);
        Assert.Equal(6f, grads[b][1], 4);

        // dz/dc = a + b = [4, 6]
        Assert.Equal(4f, grads[c][0], 4);
        Assert.Equal(6f, grads[c][1], 4);
    }

    [Fact]
    public void Gradient_ChainRule_MultipleUses()
    {
        // z = a * a = a^2
        // dz/da = 2 * a
        var a = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 2, 3, 4 }));

        using var tape = new GradientTape<float>();
        var z = _engine.TensorMultiply(a, a);
        var grads = tape.ComputeGradients(z, new[] { a });

        // dz/da = 2a (because gradient accumulates: a appears as both inputs)
        Assert.Equal(4f, grads[a][0], 4);    // 2*2
        Assert.Equal(6f, grads[a][1], 4);    // 2*3
        Assert.Equal(8f, grads[a][2], 4);    // 2*4
    }

    // ──────────────────────────────────────────────────────────────
    // Activation gradient tests
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_ReLU_CorrectGradient()
    {
        // z = ReLU(x)
        // dz/dx = 1 if x > 0, else 0
        var x = new Tensor<float>(new[] { 4 }, new Vector<float>(new float[] { -1, 0, 1, 2 }));

        using var tape = new GradientTape<float>();
        var z = _engine.ReLU(x);
        var grads = tape.ComputeGradients(z, new[] { x });

        Assert.Equal(0f, grads[x][0], 5);   // x=-1: ReLU derivative is 0
        Assert.Equal(0f, grads[x][1], 5);   // x=0: ReLU derivative is 0
        Assert.Equal(1f, grads[x][2], 5);   // x=1: ReLU derivative is 1
        Assert.Equal(1f, grads[x][3], 5);   // x=2: ReLU derivative is 1
    }

    // ──────────────────────────────────────────────────────────────
    // Finite difference gradient verification
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_Sqrt_MatchesFiniteDifference()
    {
        // Verify sqrt gradient using finite differences
        var x = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 4f, 9f }));

        using var tape = new GradientTape<float>();
        var z = _engine.TensorSqrt(x);
        var grads = tape.ComputeGradients(z, new[] { x });

        // Numerical gradient: (sqrt(x+h) - sqrt(x-h)) / (2h)
        float h = 1e-3f;
        for (int i = 0; i < x.Length; i++)
        {
            float xi = x[i];
            float numerical = (MathF.Sqrt(xi + h) - MathF.Sqrt(xi - h)) / (2 * h);
            Assert.Equal(numerical, grads[x][i], 2);
        }
    }

    [Fact]
    public void Gradient_Power_MatchesFiniteDifference()
    {
        // z = x^3
        // dz/dx = 3x^2
        var x = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 2f, 3f }));

        using var tape = new GradientTape<float>();
        var z = _engine.TensorPower(x, 3f);
        var grads = tape.ComputeGradients(z, new[] { x });

        // 3 * 2^2 = 12, 3 * 3^2 = 27
        Assert.Equal(12f, grads[x][0], 2);
        Assert.Equal(27f, grads[x][1], 2);
    }

    // ──────────────────────────────────────────────────────────────
    // Integration: linear model y = Wx + b
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_LinearModel_EndToEnd()
    {
        // Simple linear model: z = W @ x + b
        var w = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var x = new Tensor<float>(new float[] { 1, 0 }, new[] { 2, 1 });
        var b = new Tensor<float>(new float[] { 0.5f, 0.5f }, new[] { 2, 1 });

        using var tape = new GradientTape<float>();
        var wx = _engine.TensorMatMul(w, x);  // [1, 3]
        var z = _engine.TensorAdd(wx, b);       // [1.5, 3.5]

        var grads = tape.ComputeGradients(z, new[] { w, b });

        // dz/db = 1 (identity)
        Assert.True(grads.ContainsKey(b));
        Assert.Equal(1f, grads[b][0], 4);
        Assert.Equal(1f, grads[b][1], 4);

        // dz/dW: grad @ x^T = ones(2,1) @ [[1,0]] = [[1,0],[1,0]]
        Assert.True(grads.ContainsKey(w));
        Assert.Equal(1f, grads[w][0, 0], 4);
        Assert.Equal(0f, grads[w][0, 1], 4);
        Assert.Equal(1f, grads[w][1, 0], 4);
        Assert.Equal(0f, grads[w][1, 1], 4);
    }

    // ──────────────────────────────────────────────────────────────
    // Persistent tape
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_PersistentTape_CanComputeMultipleTimes()
    {
        var options = new GradientTapeOptions { Persistent = true };
        using var tape = new GradientTape<float>(options);

        var a = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 1, 2 }));
        var b = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 3, 4 }));
        var z = _engine.TensorAdd(a, b);

        // First computation
        var grads1 = tape.ComputeGradients(z, new[] { a });
        Assert.Equal(1f, grads1[a][0], 5);

        // Second computation (should work because tape is persistent)
        var grads2 = tape.ComputeGradients(z, new[] { b });
        Assert.Equal(1f, grads2[b][0], 5);
    }

    [Fact]
    public void Gradient_NonPersistentTape_ClearsAfterCompute()
    {
        using var tape = new GradientTape<float>();

        var a = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 1, 2 }));
        var b = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 3, 4 }));
        var z = _engine.TensorAdd(a, b);

        tape.ComputeGradients(z, new[] { a });

        // Tape should be cleared (non-persistent)
        Assert.Equal(0, tape.EntryCount);

        // Second compute should fail (no entries)
        Assert.Throws<InvalidOperationException>(() => tape.ComputeGradients(z, new[] { b }));
    }
}
