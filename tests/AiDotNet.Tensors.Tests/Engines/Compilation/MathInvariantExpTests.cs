using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase 5: Mathematical invariant tests for exp/sigmoid/tanh kernels.
/// These prove our implementations are mathematically correct, not just fast.
/// </summary>
public class MathInvariantExpTests
{
    private readonly ITestOutputHelper _output;
    public MathInvariantExpTests(ITestOutputHelper output) => _output = output;

    // ──────────────────────────────────────────────────────────
    // exp(x) invariants
    // ──────────────────────────────────────────────────────────

    [Fact]
    public unsafe void Exp_Identity_ExpZeroEqualsOne()
    {
        var input = new float[256];
        var output = new float[256];
        // All zeros
        fixed (float* pIn = input, pOut = output)
        {
            SimdKernels.ExpUnsafe(pIn, pOut, 256);
        }
        for (int i = 0; i < 256; i++)
            Assert.True(MathF.Abs(output[i] - 1.0f) < 1e-6f, $"exp(0) should be 1, got {output[i]}");
    }

    [Fact]
    public unsafe void Exp_Identity_ExpOneEqualsE()
    {
        var input = new float[256];
        var output = new float[256];
        for (int i = 0; i < 256; i++) input[i] = 1.0f;
        fixed (float* pIn = input, pOut = output)
        {
            SimdKernels.ExpUnsafe(pIn, pOut, 256);
        }
        float e = MathF.E;
        for (int i = 0; i < 256; i++)
            Assert.True(MathF.Abs(output[i] - e) / e < 1e-6f,
                $"exp(1) should be {e}, got {output[i]}, err={MathF.Abs(output[i] - e) / e:E2}");
    }

    [Fact]
    public unsafe void Exp_Property_ExpXTimesExpY_EqualsExpXPlusY()
    {
        // exp(x) * exp(y) = exp(x + y)
        var rng = new Random(42);
        int n = 1024;
        var x = new float[n];
        var y = new float[n];
        var expX = new float[n];
        var expY = new float[n];
        var xy = new float[n];
        var expXY = new float[n];

        for (int i = 0; i < n; i++)
        {
            x[i] = (float)(rng.NextDouble() * 10 - 5);
            y[i] = (float)(rng.NextDouble() * 10 - 5);
            xy[i] = x[i] + y[i];
        }

        fixed (float* px = x, py = y, pxy = xy, pex = expX, pey = expY, pexy = expXY)
        {
            SimdKernels.ExpUnsafe(px, pex, n);
            SimdKernels.ExpUnsafe(py, pey, n);
            SimdKernels.ExpUnsafe(pxy, pexy, n);
        }

        float maxErr = 0f;
        for (int i = 0; i < n; i++)
        {
            float product = expX[i] * expY[i];
            float expected = expXY[i];
            if (float.IsInfinity(product) || float.IsInfinity(expected)) continue;
            float relErr = MathF.Abs(product - expected) / MathF.Max(MathF.Abs(expected), 1e-30f);
            maxErr = MathF.Max(maxErr, relErr);
        }
        _output.WriteLine($"exp(x)*exp(y) vs exp(x+y): max relative error = {maxErr:E3}");
        Assert.True(maxErr < 1e-4f, $"exp(x)*exp(y) != exp(x+y), max error {maxErr:E3}");
    }

    [Fact]
    public unsafe void Exp_Property_ExpNegX_EqualsOneOverExpX()
    {
        // exp(-x) = 1 / exp(x)
        var rng = new Random(43);
        int n = 1024;
        var x = new float[n];
        var negX = new float[n];
        var expX = new float[n];
        var expNegX = new float[n];

        for (int i = 0; i < n; i++)
        {
            x[i] = (float)(rng.NextDouble() * 20 - 10);
            negX[i] = -x[i];
        }

        fixed (float* px = x, pnx = negX, pex = expX, penx = expNegX)
        {
            SimdKernels.ExpUnsafe(px, pex, n);
            SimdKernels.ExpUnsafe(pnx, penx, n);
        }

        float maxErr = 0f;
        for (int i = 0; i < n; i++)
        {
            if (expX[i] < 1e-30f || float.IsInfinity(expX[i])) continue;
            float reciprocal = 1.0f / expX[i];
            float relErr = MathF.Abs(expNegX[i] - reciprocal) / MathF.Max(MathF.Abs(reciprocal), 1e-30f);
            maxErr = MathF.Max(maxErr, relErr);
        }
        _output.WriteLine($"exp(-x) vs 1/exp(x): max relative error = {maxErr:E3}");
        Assert.True(maxErr < 1e-4f, $"exp(-x) != 1/exp(x), max error {maxErr:E3}");
    }

    // ──────────────────────────────────────────────────────────
    // sigmoid(x) invariants
    // ──────────────────────────────────────────────────────────

    [Fact]
    public unsafe void Sigmoid_BoundedZeroOne()
    {
        // sigmoid(x) ∈ (0, 1) for all finite x
        var rng = new Random(44);
        int n = 10000;
        var input = new float[n];
        var output = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = (float)(rng.NextDouble() * 200 - 100); // wide range

        fixed (float* pIn = input, pOut = output)
        {
            SimdKernels.SigmoidUnsafe(pIn, pOut, n);
        }

        for (int i = 0; i < n; i++)
        {
            Assert.True(output[i] >= 0f && output[i] <= 1f,
                $"sigmoid({input[i]}) = {output[i]} — must be in [0,1]");
            // For moderate inputs, sigmoid should be strictly between 0 and 1
            if (MathF.Abs(input[i]) < 15f) // float32 sigmoid saturates at ~16
                Assert.True(output[i] > 0f && output[i] < 1f,
                    $"sigmoid({input[i]}) = {output[i]} — must be in (0,1) for |x| < 30");
        }
    }

    [Fact]
    public unsafe void Sigmoid_SymmetryProperty()
    {
        // sigmoid(x) + sigmoid(-x) = 1
        var rng = new Random(45);
        int n = 1024;
        var x = new float[n];
        var negX = new float[n];
        var sigX = new float[n];
        var sigNegX = new float[n];

        for (int i = 0; i < n; i++)
        {
            x[i] = (float)(rng.NextDouble() * 20 - 10);
            negX[i] = -x[i];
        }

        fixed (float* px = x, pnx = negX, psx = sigX, psnx = sigNegX)
        {
            SimdKernels.SigmoidUnsafe(px, psx, n);
            SimdKernels.SigmoidUnsafe(pnx, psnx, n);
        }

        float maxErr = 0f;
        for (int i = 0; i < n; i++)
        {
            float sum = sigX[i] + sigNegX[i];
            maxErr = MathF.Max(maxErr, MathF.Abs(sum - 1.0f));
        }
        _output.WriteLine($"sigmoid(x) + sigmoid(-x) = 1: max error = {maxErr:E3}");
        Assert.True(maxErr < 1e-4f, $"sigmoid symmetry violated, max error {maxErr:E3}");
    }

    [Fact]
    public unsafe void Sigmoid_AtZero_EqualsHalf()
    {
        var input = new float[256];
        var output = new float[256];
        fixed (float* pIn = input, pOut = output)
        {
            SimdKernels.SigmoidUnsafe(pIn, pOut, 256);
        }
        for (int i = 0; i < 256; i++)
            Assert.True(MathF.Abs(output[i] - 0.5f) < 1e-5f,
                $"sigmoid(0) should be 0.5, got {output[i]}");
    }

    // ──────────────────────────────────────────────────────────
    // tanh(x) invariants
    // ──────────────────────────────────────────────────────────

    [Fact]
    public unsafe void Tanh_BoundedNegOneOne()
    {
        // tanh(x) ∈ (-1, 1) for all finite x
        var rng = new Random(46);
        int n = 10000;
        var input = new float[n];
        var output = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = (float)(rng.NextDouble() * 200 - 100);

        fixed (float* pIn = input, pOut = output)
        {
            SimdKernels.TanhUnsafe(pIn, pOut, n);
        }

        for (int i = 0; i < n; i++)
        {
            Assert.True(output[i] >= -1f && output[i] <= 1f,
                $"tanh({input[i]}) = {output[i]} — must be in [-1,1]");
        }
    }

    [Fact]
    public unsafe void Tanh_OddFunction()
    {
        // tanh(-x) = -tanh(x)
        var rng = new Random(47);
        int n = 1024;
        var x = new float[n];
        var negX = new float[n];
        var tanhX = new float[n];
        var tanhNegX = new float[n];

        for (int i = 0; i < n; i++)
        {
            x[i] = (float)(rng.NextDouble() * 20 - 10);
            negX[i] = -x[i];
        }

        fixed (float* px = x, pnx = negX, ptx = tanhX, ptnx = tanhNegX)
        {
            SimdKernels.TanhUnsafe(px, ptx, n);
            SimdKernels.TanhUnsafe(pnx, ptnx, n);
        }

        float maxErr = 0f;
        for (int i = 0; i < n; i++)
        {
            float err = MathF.Abs(tanhNegX[i] + tanhX[i]); // should be 0
            maxErr = MathF.Max(maxErr, err);
        }
        _output.WriteLine($"tanh(-x) + tanh(x) = 0: max error = {maxErr:E3}");
        Assert.True(maxErr < 1e-4f, $"tanh odd function violated, max error {maxErr:E3}");
    }

    [Fact]
    public unsafe void Tanh_AtZero_EqualsZero()
    {
        var input = new float[256];
        var output = new float[256];
        fixed (float* pIn = input, pOut = output)
        {
            SimdKernels.TanhUnsafe(pIn, pOut, 256);
        }
        for (int i = 0; i < 256; i++)
            Assert.True(MathF.Abs(output[i]) < 1e-5f,
                $"tanh(0) should be 0, got {output[i]}");
    }

    // ──────────────────────────────────────────────────────────
    // Cross-function consistency
    // ──────────────────────────────────────────────────────────

    [Fact]
    public unsafe void Tanh_EqualsToSigmoid_Relationship()
    {
        // tanh(x) = 2*sigmoid(2x) - 1
        var rng = new Random(48);
        int n = 1024;
        var x = new float[n];
        var twoX = new float[n];
        var tanhX = new float[n];
        var sig2X = new float[n];

        for (int i = 0; i < n; i++)
        {
            x[i] = (float)(rng.NextDouble() * 10 - 5);
            twoX[i] = 2f * x[i];
        }

        fixed (float* px = x, p2x = twoX, ptx = tanhX, ps2x = sig2X)
        {
            SimdKernels.TanhUnsafe(px, ptx, n);
            SimdKernels.SigmoidUnsafe(p2x, ps2x, n);
        }

        float maxErr = 0f;
        for (int i = 0; i < n; i++)
        {
            float fromSigmoid = 2f * sig2X[i] - 1f;
            float err = MathF.Abs(tanhX[i] - fromSigmoid);
            maxErr = MathF.Max(maxErr, err);
        }
        _output.WriteLine($"tanh(x) vs 2*sigmoid(2x)-1: max error = {maxErr:E3}");
        // Tolerance accounts for tanh and sigmoid potentially using different approximation
        // paths (e.g., Padé [3,3] for tanh vs table-driven for sigmoid on Intel CPUs)
        Assert.True(maxErr < 1e-3f, $"tanh/sigmoid relationship violated, max error {maxErr:E3}");
    }

    [Fact]
    public unsafe void Exp_Monotonically_Increasing()
    {
        // exp(x) is strictly monotonically increasing
        int n = 10000;
        var input = new float[n];
        var output = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = -50f + (100f * i / n); // sorted ascending

        fixed (float* pIn = input, pOut = output)
        {
            SimdKernels.ExpUnsafe(pIn, pOut, n);
        }

        for (int i = 1; i < n; i++)
        {
            if (float.IsInfinity(output[i]) || output[i] == 0f) continue;
            Assert.True(output[i] >= output[i - 1],
                $"exp not monotonic at i={i}: exp({input[i - 1]})={output[i - 1]} > exp({input[i]})={output[i]}");
        }
    }
}
