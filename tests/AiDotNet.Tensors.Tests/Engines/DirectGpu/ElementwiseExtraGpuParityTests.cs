using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The GPU overrides of the ElementwiseExtra unary ops (acosh ... trunc) against the CPU engine: the values, and the
/// gradient each records on the tape. Skipped where no GPU device is present.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ElementwiseExtraGpuParityTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public ElementwiseExtraGpuParityTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    public static TheoryData<string> Ops => new() { "acosh", "asinh", "exp2", "expm1", "log10", "log1p", "log2", "rsqrt", "tan", "trunc" };

    private static Tensor<float> Apply(IEngine e, string op, Tensor<float> x) => op switch
    {
        "acosh" => e.TensorAcosh(x), "asinh" => e.TensorAsinh(x), "exp2" => e.TensorExp2(x),
        "expm1" => e.TensorExpm1(x), "log10" => e.TensorLog10(x), "log1p" => e.TensorLog1p(x),
        "log2" => e.TensorLog2(x), "rsqrt" => e.TensorRsqrt(x), "tan" => e.TensorTan(x),
        "trunc" => e.TensorTrunc(x), _ => throw new ArgumentException(op),
    };

    // Inside each op's domain, away from its poles.
    private static float[] Inputs(string op) => op switch
    {
        "acosh" => new[] { 1.5f, 2f, 3.25f, 10f },
        "log10" or "log1p" or "log2" or "rsqrt" => new[] { 0.25f, 1f, 2.5f, 7f },
        _ => new[] { -1.2f, -0.3f, 0.4f, 1.1f },
    };

    private IEngine Gpu()
    {
        Skip.IfNot(_fixture.IsAvailable && _fixture.Engine is not null, "No GPU device.");
        return _fixture.Engine ?? throw new InvalidOperationException("The fixture reported a device but no engine.");
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])),
                $"{what}[{i}]: CPU {expected[i]:R}, GPU {actual[i]:R}");
    }

    [SkippableTheory]
    [MemberData(nameof(Ops))]
    public void Values_MatchTheCpuEngine(string op)
    {
        IEngine gpu = Gpu();
        var xs = Inputs(op);
        var cpu = Apply(new CpuEngine(), op, new Tensor<float>((float[])xs.Clone(), new[] { xs.Length })).ToArray();
        var dev = Apply(gpu, op, new Tensor<float>((float[])xs.Clone(), new[] { xs.Length })).ToArray();
        AssertClose(cpu, dev, op);
    }

    [SkippableTheory]
    [MemberData(nameof(Ops))]
    public void RecordedGradient_MatchesTheCpuEngine(string op)
    {
        IEngine gpu = Gpu();
        var xs = Inputs(op);
        AssertClose(Gradient(new CpuEngine(), op, xs), Gradient(gpu, op, xs), op + " gradient");
    }

    private static float[] Gradient(IEngine engine, string op, float[] xs)
    {
        var x = new Tensor<float>((float[])xs.Clone(), new[] { xs.Length });
        using var tape = new GradientTape<float>();
        var loss = engine.ReduceSum(Apply(engine, op, x), null);
        return tape.ComputeGradients(loss, new[] { x })[x].ToArray();
    }
}
