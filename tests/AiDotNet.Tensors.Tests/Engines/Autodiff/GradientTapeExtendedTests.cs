using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Extended tests: Conv1D, activation gradients, SIMD backward kernels, in-place ops.
/// </summary>
public class GradientTapeExtendedTests
{
    private readonly CpuEngine _engine = new();

    // ──────────────────────────────────────────────────────────────
    // Conv1D forward correctness
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Conv1D_Forward_CorrectShape()
    {
        // Input: [batch=1, channels=1, length=5], Kernel: [out_ch=1, in_ch=1, kernel_len=3]
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5 }, new[] { 1, 1, 5 });
        var kernel = new Tensor<float>(new float[] { 1, 0, -1 }, new[] { 1, 1, 3 });

        var result = _engine.Conv1D(input, kernel);

        Assert.Equal(3, result.Rank);
        Assert.Equal(1, result._shape[0]); // batch
        Assert.Equal(1, result._shape[1]); // out_channels
        Assert.Equal(3, result._shape[2]); // output_length = (5 - 3) / 1 + 1 = 3
    }

    [Fact]
    public void Conv1D_Forward_CorrectValues()
    {
        // Input: [1, 1, 5] = [1, 2, 3, 4, 5]
        // Kernel: [1, 1, 3] = [1, 0, -1]
        // Expected: [1*1+2*0+3*(-1), 2*1+3*0+4*(-1), 3*1+4*0+5*(-1)] = [-2, -2, -2]
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5 }, new[] { 1, 1, 5 });
        var kernel = new Tensor<float>(new float[] { 1, 0, -1 }, new[] { 1, 1, 3 });

        var result = _engine.Conv1D(input, kernel);

        Assert.Equal(-2f, result[0, 0, 0], 3);
        Assert.Equal(-2f, result[0, 0, 1], 3);
        Assert.Equal(-2f, result[0, 0, 2], 3);
    }

    [Fact]
    public void Conv1D_WithPadding_CorrectLength()
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5 }, new[] { 1, 1, 5 });
        var kernel = new Tensor<float>(new float[] { 1, 1, 1 }, new[] { 1, 1, 3 });

        var result = _engine.Conv1D(input, kernel, stride: 1, padding: 1);

        // With padding=1: output_length = (5 + 2*1 - 3) / 1 + 1 = 5
        Assert.Equal(5, result._shape[2]);
    }

    [Fact]
    public void Conv1D_RecordedOnTape()
    {
        var input = new Tensor<float>(new float[] { 1, 2, 3, 4, 5 }, new[] { 1, 1, 5 });
        var kernel = new Tensor<float>(new float[] { 1, 0, -1 }, new[] { 1, 1, 3 });

        using var tape = new GradientTape<float>();
        var result = _engine.Conv1D(input, kernel);

        Assert.True(tape.EntryCount > 0);
    }

    // ──────────────────────────────────────────────────────────────
    // Sigmoid/Tanh gradient tests
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_Sigmoid_CorrectGradient()
    {
        // z = sigmoid(x)
        // dz/dx = sigmoid(x) * (1 - sigmoid(x))
        var x = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 0f, 1f }));

        using var tape = new GradientTape<float>();
        var z = _engine.Sigmoid(x);
        var grads = tape.ComputeGradients(z, new[] { x });

        // sigmoid(0) = 0.5, derivative = 0.5 * 0.5 = 0.25
        Assert.Equal(0.25f, grads[x][0], 3);
        // sigmoid(1) ≈ 0.7311, derivative ≈ 0.7311 * 0.2689 ≈ 0.1966
        Assert.Equal(0.1966f, grads[x][1], 2);
    }

    [Fact]
    public void Gradient_Tanh_CorrectGradient()
    {
        // z = tanh(x)
        // dz/dx = 1 - tanh(x)^2
        var x = new Tensor<float>(new[] { 2 }, new Vector<float>(new float[] { 0f, 1f }));

        using var tape = new GradientTape<float>();
        var z = _engine.Tanh(x);
        var grads = tape.ComputeGradients(z, new[] { x });

        // tanh(0) = 0, derivative = 1 - 0 = 1
        Assert.Equal(1f, grads[x][0], 3);
        // tanh(1) ≈ 0.7616, derivative ≈ 1 - 0.7616^2 ≈ 0.4200
        Assert.Equal(0.4200f, grads[x][1], 2);
    }

    [Fact]
    public void Gradient_GELU_MatchesFiniteDifference()
    {
        var x = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { -1f, 0f, 1f }));

        using var tape = new GradientTape<float>();
        var z = _engine.GELU(x);
        var grads = tape.ComputeGradients(z, new[] { x });

        // Verify against finite differences
        float h = 1e-3f;
        for (int i = 0; i < x.Length; i++)
        {
            var xPlus = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { x[0], x[1], x[2] }));
            xPlus[i] += h;
            var xMinus = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { x[0], x[1], x[2] }));
            xMinus[i] -= h;
            var fPlus = _engine.GELU(xPlus);
            var fMinus = _engine.GELU(xMinus);
            float numerical = (fPlus[i] - fMinus[i]) / (2 * h);
            Assert.Equal(numerical, grads[x][i], 1);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // SIMD backward kernel correctness
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void ReluBackward_SIMD_MatchesScalar()
    {
        // ReLU backward: mask * grad
        var input = new Tensor<float>(new float[] { -2, -1, 0, 1, 2, 3, -3, 4 }, new[] { 8 });
        var gradOutput = new Tensor<float>(new float[] { 1, 1, 1, 1, 1, 1, 1, 1 }, new[] { 8 });

        var result = _engine.ReluBackward(gradOutput, input);

        Assert.Equal(0f, result[0]); // input=-2: zero
        Assert.Equal(0f, result[1]); // input=-1: zero
        Assert.Equal(0f, result[2]); // input=0: zero
        Assert.Equal(1f, result[3]); // input=1: pass through
        Assert.Equal(1f, result[4]); // input=2: pass through
        Assert.Equal(1f, result[5]); // input=3: pass through
        Assert.Equal(0f, result[6]); // input=-3: zero
        Assert.Equal(1f, result[7]); // input=4: pass through
    }

    [Fact]
    public void GeluBackward_SIMD_ReasonableValues()
    {
        var input = new Tensor<float>(new float[] { -1, 0, 1, 2 }, new[] { 4 });
        var gradOutput = new Tensor<float>(new float[] { 1, 1, 1, 1 }, new[] { 4 });

        var result = _engine.GeluBackward(gradOutput, input);

        // GELU'(0) ≈ 0.5, GELU'(1) ≈ 1.083, GELU'(-1) ≈ -0.083
        Assert.Equal(0.5f, result[1], 1);     // x=0
        Assert.True(result[2] > 0.9f);         // x=1: positive and > 0.9
        Assert.True(result[0] < 0.1f);         // x=-1: small
    }

    // ──────────────────────────────────────────────────────────────
    // In-place operation gradient recording
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_TensorAddInPlace_RecordedWithSave()
    {
        var a = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 1, 2, 3 }));
        var b = new Tensor<float>(new[] { 3 }, new Vector<float>(new float[] { 4, 5, 6 }));

        using var tape = new GradientTape<float>();
        _engine.TensorAddInPlace(a, b);

        // In-place add should be recorded
        Assert.Equal(1, tape.EntryCount);

        // a is now [5, 7, 9]
        Assert.Equal(5f, a[0], 5);
        Assert.Equal(7f, a[1], 5);
        Assert.Equal(9f, a[2], 5);
    }

    // ──────────────────────────────────────────────────────────────
    // Broadcast gradient
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_BroadcastAdd_ReducesGradient()
    {
        // a: [2, 3], b: [1, 3] (broadcast along dim 0)
        var a = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var b = new Tensor<float>(new float[] { 10, 20, 30 }, new[] { 1, 3 });

        using var tape = new GradientTape<float>();
        var z = _engine.TensorBroadcastAdd(a, b);
        var grads = tape.ComputeGradients(z, new[] { a, b });

        // dz/da = ones (same shape as a: [2, 3])
        Assert.Equal(new[] { 2, 3 }, grads[a].Shape.ToArray());
        for (int i = 0; i < grads[a].Length; i++)
            Assert.Equal(1f, grads[a].GetFlat(i), 4);

        // dz/db = sum over broadcast dim 0 -> [1, 3] with values [2, 2, 2]
        Assert.True(grads.ContainsKey(b));
        Assert.Equal(new[] { 1, 3 }, grads[b].Shape.ToArray());
        Assert.Equal(2f, grads[b][0, 0], 4);
        Assert.Equal(2f, grads[b][0, 1], 4);
        Assert.Equal(2f, grads[b][0, 2], 4);
    }

    // ──────────────────────────────────────────────────────────────
    // Softmax gradient
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_Softmax_RecordedOnTape()
    {
        var x = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 1, 4 });

        using var tape = new GradientTape<float>();
        var z = _engine.Softmax(x, axis: 1);

        Assert.True(tape.EntryCount > 0);
        // Verify softmax output sums to 1
        float sum = 0;
        for (int i = 0; i < 4; i++) sum += z[0, i];
        Assert.Equal(1f, sum, 3);
    }

    // ──────────────────────────────────────────────────────────────
    // Transpose gradient
    // ──────────────────────────────────────────────────────────────

    [Fact]
    public void Gradient_Transpose_CorrectGradient()
    {
        // z = transpose(x), dz/dx = transpose(grad)
        var x = new Tensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });

        using var tape = new GradientTape<float>();
        var z = _engine.TensorTranspose(x);
        var grads = tape.ComputeGradients(z, new[] { x });

        // z is [3, 2], grad is ones [3, 2]
        // dx = transpose(grad) = ones [2, 3]
        Assert.Equal(new[] { 2, 3 }, grads[x].Shape.ToArray());
        Assert.Equal(1f, grads[x][0, 0], 5);
    }
}
