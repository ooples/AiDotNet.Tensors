using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Tier 1.1 correctness: the new Conv2DBackwardInputInto / Conv2DBackwardKernelInto
/// must produce bit-identical results to the allocating Conv2DBackwardInput /
/// Conv2DBackwardKernel they replace in compile-mode. Tests both overwrite
/// (accumulate=false) and accumulate (accumulate=true) paths for float and
/// double, on a typical 4D Conv shape.
/// </summary>
public class Conv2DBackwardIntoTests
{
    [Fact]
    public void Conv2DBackwardInputInto_FLOAT_Overwrite_MatchesAllocating()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<float>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(rng.NextDouble() - 0.5);
        var kernel = new Tensor<float>(new[] { outC, inC, kH, kW });
        for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(rng.NextDouble() - 0.5);

        var allocating = engine.Conv2DBackwardInput(gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var dest = new Tensor<float>(new[] { batch, inC, H, W });
        engine.Conv2DBackwardInputInto(dest, gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: false);

        for (int i = 0; i < allocating.Length; i++)
            Assert.True(System.Math.Abs(allocating[i] - dest[i]) < 1e-5f,
                $"[{i}] alloc={allocating[i]:F6} into={dest[i]:F6}");
    }

    [Fact]
    public void Conv2DBackwardInputInto_DOUBLE_Overwrite_MatchesAllocating()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var kernel = new Tensor<double>(new[] { outC, inC, kH, kW });
        for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() - 0.5;

        var allocating = engine.Conv2DBackwardInput(gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var dest = new Tensor<double>(new[] { batch, inC, H, W });
        engine.Conv2DBackwardInputInto(dest, gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: false);

        for (int i = 0; i < allocating.Length; i++)
            Assert.True(System.Math.Abs(allocating[i] - dest[i]) < 1e-12,
                $"[{i}] alloc={allocating[i]:F12} into={dest[i]:F12}");
    }

    [Fact]
    public void Conv2DBackwardInputInto_DOUBLE_Accumulate_AddsToExisting()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var kernel = new Tensor<double>(new[] { outC, inC, kH, kW });
        for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() - 0.5;

        var allocating = engine.Conv2DBackwardInput(gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });

        // Pre-fill dest with deterministic values; accumulate should add allocating into them.
        var dest = new Tensor<double>(new[] { batch, inC, H, W });
        var preExisting = new double[dest.Length];
        var rng2 = new System.Random(7);
        for (int i = 0; i < dest.Length; i++) { preExisting[i] = rng2.NextDouble() - 0.5; dest[i] = preExisting[i]; }

        engine.Conv2DBackwardInputInto(dest, gradOut, kernel,
            new[] { batch, inC, H, W }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: true);

        for (int i = 0; i < dest.Length; i++)
        {
            double expected = preExisting[i] + allocating[i];
            Assert.True(System.Math.Abs(expected - dest[i]) < 1e-12,
                $"[{i}] expected pre+alloc={expected:F12} but dest={dest[i]:F12}");
        }
    }

    [Fact]
    public void Conv2DBackwardKernelInto_FLOAT_Overwrite_MatchesAllocating()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<float>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(rng.NextDouble() - 0.5);
        var input = new Tensor<float>(new[] { batch, inC, H, W });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);

        var allocating = engine.Conv2DBackwardKernel(gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var dest = new Tensor<float>(new[] { outC, inC, kH, kW });
        engine.Conv2DBackwardKernelInto(dest, gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: false);

        for (int i = 0; i < allocating.Length; i++)
            Assert.True(System.Math.Abs(allocating[i] - dest[i]) < 1e-5f,
                $"[{i}] alloc={allocating[i]:F6} into={dest[i]:F6}");
    }

    [Fact]
    public void Conv2DBackwardKernelInto_DOUBLE_Overwrite_MatchesAllocating()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var input = new Tensor<double>(new[] { batch, inC, H, W });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;

        var allocating = engine.Conv2DBackwardKernel(gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });
        var dest = new Tensor<double>(new[] { outC, inC, kH, kW });
        engine.Conv2DBackwardKernelInto(dest, gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: false);

        for (int i = 0; i < allocating.Length; i++)
            Assert.True(System.Math.Abs(allocating[i] - dest[i]) < 1e-12,
                $"[{i}] alloc={allocating[i]:F12} into={dest[i]:F12}");
    }

    [Fact]
    public void Conv2DBackwardKernelInto_DOUBLE_Accumulate_AddsToExisting()
    {
        var engine = new CpuEngine();
        int batch = 1, inC = 3, H = 8, W = 8, outC = 4, kH = 3, kW = 3;
        int oH = H - kH + 1, oW = W - kW + 1;
        var rng = new System.Random(42);

        var gradOut = new Tensor<double>(new[] { batch, outC, oH, oW });
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = rng.NextDouble() - 0.5;
        var input = new Tensor<double>(new[] { batch, inC, H, W });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() - 0.5;

        var allocating = engine.Conv2DBackwardKernel(gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 });

        var dest = new Tensor<double>(new[] { outC, inC, kH, kW });
        var preExisting = new double[dest.Length];
        var rng2 = new System.Random(7);
        for (int i = 0; i < dest.Length; i++) { preExisting[i] = rng2.NextDouble() - 0.5; dest[i] = preExisting[i]; }

        engine.Conv2DBackwardKernelInto(dest, gradOut, input,
            new[] { outC, inC, kH, kW }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 },
            accumulate: true);

        for (int i = 0; i < dest.Length; i++)
        {
            double expected = preExisting[i] + allocating[i];
            Assert.True(System.Math.Abs(expected - dest[i]) < 1e-12,
                $"[{i}] expected pre+alloc={expected:F12} but dest={dest[i]:F12}");
        }
    }
}
