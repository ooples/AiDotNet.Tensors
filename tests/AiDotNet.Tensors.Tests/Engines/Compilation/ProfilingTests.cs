using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Profiling tests using ProfilingCompiler to identify bottlenecks in compiled plans.
/// </summary>
public class ProfilingTests
{
    private readonly ITestOutputHelper _output;

    public ProfilingTests(ITestOutputHelper output) => _output = output;

    [Fact(Skip = "Profile benchmark — run manually")]
    public void Profile_MLP_Inference()
    {
        var engine = new CpuEngine();
        var input = CreateRandom(new[] { 32, 128 }, 42);
        var w1 = CreateRandom(new[] { 128, 64 }, 43);
        var w2 = CreateRandom(new[] { 64, 32 }, 44);

        // Compile MLP inference
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.ReLU(engine.TensorMatMul(input, w1));
            engine.TensorMatMul(h, w2);
            plan = scope.CompileInference<float>();
        }

        // Profile it
        var profiler = new ProfilingCompiler<float>();
        profiler.ProfileInference(plan, warmupSteps: 20, measureSteps: 100);
        var report = profiler.GetReport();

        _output.WriteLine("=== MLP Inference Profile ===");
        foreach (var profile in report.Profiles)
        {
            _output.WriteLine($"  {profile.Name}: {profile.MillisecondsPerExecution:F3}ms total, " +
                $"{profile.MillisecondsPerExecution:F4}ms/step, {profile.StepsPerSecond:F0} steps/sec");
        }

        plan.Dispose();
    }

    [Fact(Skip = "Profile benchmark — run manually")]
    public void Profile_MLP_Training()
    {
        var engine = new CpuEngine();
        var input = CreateRandom(new[] { 32, 128 }, 42);
        var w1 = CreateRandom(new[] { 128, 64 }, 43);
        var w2 = CreateRandom(new[] { 64, 32 }, 44);

        // Compile MLP training
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = engine.ReLU(engine.TensorMatMul(input, w1));
            var output = engine.TensorMatMul(h, w2);
            engine.ReduceSum(engine.TensorMultiply(output, output), null);
            plan = scope.CompileTraining(new[] { w1, w2 });
        }

        // Profile it
        var profiler = new ProfilingCompiler<float>();
        profiler.ProfileTraining(plan, warmupSteps: 10, measureSteps: 50);
        var report = profiler.GetReport();

        _output.WriteLine("=== MLP Training Profile ===");
        foreach (var profile in report.Profiles)
        {
            _output.WriteLine($"  {profile.Name}: {profile.MillisecondsPerExecution:F3}ms total, " +
                $"{profile.MillisecondsPerExecution:F4}ms/step, {profile.StepsPerSecond:F0} steps/sec");
        }

        plan.Dispose();
    }

    [Fact]
    public void Profile_CompilationBenchmark_ProducesResults()
    {
        // Non-skip'd test: verify ProfilingCompiler produces non-zero results
        var engine = new CpuEngine();
        var input = CreateRandom(new[] { 4, 8 }, 42);
        var w = CreateRandom(new[] { 8, 4 }, 43);

        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.TensorMatMul(input, w);
            plan = scope.CompileInference<float>();
        }

        var profiler = new ProfilingCompiler<float>();
        profiler.ProfileInference(plan, warmupSteps: 2, measureSteps: 5);
        var report = profiler.GetReport();

        Assert.NotNull(report);
        Assert.True(report.Profiles.Length > 0, "Profile should produce at least one entry");
        Assert.True(report.Profiles[0].MillisecondsPerExecution > 0, "Execution time should be positive");

        plan.Dispose();
    }

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int length = 1;
        for (int i = 0; i < shape.Length; i++) length *= shape[i];
        var data = new float[length];
        for (int i = 0; i < data.Length; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }
}
