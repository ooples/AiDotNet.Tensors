using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression coverage for an untracked interleaved-RoPE boundary. Eager autodiff recorded the
/// rotation, but GraphMode did not, so compiled attention treated rotated Q/K as trace-time leaf
/// tensors. Once their pooled storage was reused, the first downstream permute read NaNs.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CompiledRoPEInterleavedTests
{
    [Fact]
    public void CpuTrainingPlan_RecordsRoPEAndReplaysFiniteForwardAndBackward()
    {
        var engine = new CpuEngine();
        var input = CreateDoubleInput();
        var (cos, sin) = CreateDoubleCache(maxSequenceLength: 32, headDimension: 8);
        const int startPosition = 3;

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(new[] { input }))
        {
            var rotated = engine.ApplyRoPEInterleaved(input, cos, sin, startPosition);
            Assert.NotNull(rotated.LazySource);

            // Match the transformer failure shape: RoPE feeds a key transpose before attention.
            var transposed = engine.TensorPermute(rotated, new[] { 0, 1, 3, 2 });
            var loss = engine.ReduceSum(transposed, null);
            plan = scope.CompileTraining(new[] { input }, loss);
        }

        using (plan)
        {
            for (int replay = 0; replay < 8; replay++)
            {
                for (int i = 0; i < input.Length; i++)
                    input[i] = 0.01 * (i + 1) + replay * 0.125;

                Tensor<double> loss;
                using (TensorArena.Create())
                    loss = plan.Step();

                Assert.True(!double.IsNaN(loss[0]) && !double.IsInfinity(loss[0]),
                    $"replay {replay} produced loss {loss[0]:G17}");
                Assert.Equal(ExpectedDoubleLoss(input, cos, sin, startPosition), loss[0], precision: 10);

                var gradient = Assert.IsType<Tensor<double>>(plan.Gradients[0]);
                AssertExpectedDoubleGradient(gradient, cos, sin, startPosition);
            }
        }
    }

    [Fact]
    public void GraphCapture_RejectsInvalidRoPERankBeforeRecording()
    {
        var engine = new CpuEngine();
        var input = new Tensor<double>(new[] { 8 });
        var (cos, sin) = CreateDoubleCache(maxSequenceLength: 4, headDimension: 8);

        using var scope = GraphMode.EnableTraining(new[] { input });
        var exception = Assert.Throws<ArgumentException>(
            () => engine.ApplyRoPEInterleaved(input, cos, sin));

        Assert.Equal("input", exception.ParamName);
        Assert.Contains("rank >= 2", exception.Message, StringComparison.Ordinal);
        Assert.Equal(0, scope.NodeCount);
    }

#if NET6_0_OR_GREATER
    [SkippableFact]
    public void DirectGpuTrainingPlan_RecordsRoPEAndMatchesCpuGradient()
    {
        var gpu = new DirectGpuTensorEngine();
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }

        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var input = CreateFloatInput();
            var (cos, sin) = CreateFloatCache(maxSequenceLength: 32, headDimension: 8);
            const int startPosition = 3;
            var engine = (IEngine)gpu;

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { input }))
            {
                var rotated = engine.ApplyRoPEInterleaved(input, cos, sin, startPosition);
                Assert.NotNull(rotated.LazySource);
                var transposed = engine.TensorPermute(rotated, new[] { 0, 1, 3, 2 });
                var loss = engine.ReduceSum(transposed, null);
                plan = scope.CompileTraining(new[] { input }, loss);
            }

            using (plan)
            {
                var loss = plan.Step();
                Assert.True(!float.IsNaN(loss[0]) && !float.IsInfinity(loss[0]),
                    $"GPU compiled RoPE loss was {loss[0]:G9}");
                float expectedLoss = ExpectedFloatLoss(input, cos, sin, startPosition);
                Assert.InRange(Math.Abs(loss[0] - expectedLoss), 0f,
                    Math.Max(1e-4f, Math.Abs(expectedLoss) * 1e-5f));

                var gradient = Assert.IsType<Tensor<float>>(plan.Gradients[0]);
                AssertExpectedFloatGradient(gradient, cos, sin, startPosition);
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }
#endif

    private static Tensor<double> CreateDoubleInput()
    {
        var input = new Tensor<double>(new[] { 1, 4, 22, 8 });
        for (int i = 0; i < input.Length; i++) input[i] = 0.01 * (i + 1);
        return input;
    }

    private static Tensor<float> CreateFloatInput()
    {
        var input = new Tensor<float>(new[] { 1, 4, 22, 8 });
        for (int i = 0; i < input.Length; i++) input[i] = 0.01f * (i + 1);
        return input;
    }

    private static (Tensor<double> Cos, Tensor<double> Sin) CreateDoubleCache(
        int maxSequenceLength, int headDimension)
    {
        int half = headDimension / 2;
        var cos = new Tensor<double>(new[] { maxSequenceLength, half });
        var sin = new Tensor<double>(new[] { maxSequenceLength, half });
        for (int position = 0; position < maxSequenceLength; position++)
        {
            for (int pair = 0; pair < half; pair++)
            {
                double angle = position / Math.Pow(10000.0, 2.0 * pair / headDimension);
                cos[position, pair] = Math.Cos(angle);
                sin[position, pair] = Math.Sin(angle);
            }
        }
        return (cos, sin);
    }

    private static (Tensor<float> Cos, Tensor<float> Sin) CreateFloatCache(
        int maxSequenceLength, int headDimension)
    {
        var (cosDouble, sinDouble) = CreateDoubleCache(maxSequenceLength, headDimension);
        var cos = new Tensor<float>(cosDouble._shape);
        var sin = new Tensor<float>(sinDouble._shape);
        for (int i = 0; i < cos.Length; i++)
        {
            cos[i] = (float)cosDouble[i];
            sin[i] = (float)sinDouble[i];
        }
        return (cos, sin);
    }

    private static double ExpectedDoubleLoss(
        Tensor<double> input, Tensor<double> cos, Tensor<double> sin, int startPosition)
    {
        double sum = 0.0;
        const int sequenceLength = 22;
        const int headDimension = 8;
        int rows = input.Length / headDimension;
        for (int row = 0; row < rows; row++)
        {
            int position = startPosition + row % sequenceLength;
            for (int pair = 0; pair < headDimension / 2; pair++)
            {
                double c = cos[position, pair];
                double s = sin[position, pair];
                double even = input[row * headDimension + 2 * pair];
                double odd = input[row * headDimension + 2 * pair + 1];
                sum += even * c - odd * s + even * s + odd * c;
            }
        }
        return sum;
    }

    private static float ExpectedFloatLoss(
        Tensor<float> input, Tensor<float> cos, Tensor<float> sin, int startPosition)
        => (float)ExpectedDoubleLoss(
            ToDouble(input), ToDouble(cos), ToDouble(sin), startPosition);

    private static void AssertExpectedDoubleGradient(
        Tensor<double> gradient, Tensor<double> cos, Tensor<double> sin, int startPosition)
    {
        Assert.Equal(1 * 4 * 22 * 8, gradient.Length);
        const int sequenceLength = 22;
        const int headDimension = 8;
        int rows = gradient.Length / headDimension;
        for (int row = 0; row < rows; row++)
        {
            int position = startPosition + row % sequenceLength;
            for (int pair = 0; pair < headDimension / 2; pair++)
            {
                Assert.Equal(cos[position, pair] + sin[position, pair],
                    gradient[row * headDimension + 2 * pair], precision: 10);
                Assert.Equal(cos[position, pair] - sin[position, pair],
                    gradient[row * headDimension + 2 * pair + 1], precision: 10);
            }
        }
    }

    private static void AssertExpectedFloatGradient(
        Tensor<float> gradient, Tensor<float> cos, Tensor<float> sin, int startPosition)
    {
        Assert.Equal(1 * 4 * 22 * 8, gradient.Length);
        const int sequenceLength = 22;
        const int headDimension = 8;
        int rows = gradient.Length / headDimension;
        for (int row = 0; row < rows; row++)
        {
            int position = startPosition + row % sequenceLength;
            for (int pair = 0; pair < headDimension / 2; pair++)
            {
                Assert.Equal(cos[position, pair] + sin[position, pair],
                    gradient[row * headDimension + 2 * pair], precision: 4);
                Assert.Equal(cos[position, pair] - sin[position, pair],
                    gradient[row * headDimension + 2 * pair + 1], precision: 4);
            }
        }
    }

    private static Tensor<double> ToDouble(Tensor<float> source)
    {
        var result = new Tensor<double>(source._shape);
        for (int i = 0; i < source.Length; i++) result[i] = source[i];
        return result;
    }
}
