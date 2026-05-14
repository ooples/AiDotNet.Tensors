using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Issue #340 regression: <c>CompiledTrainingPlan&lt;double&gt;.Step()</c> threw
/// <see cref="System.InvalidCastException"/> ("Unable to cast object of type
/// 'System.Double[]' to type 'System.Single[]'") on graphs that included a
/// ReLU op or a BLAS-Sgemm fast-path matmul backward. The ReLU forward and
/// backward branches, the MatMul backward BLAS path, the ReduceSum non-float
/// fallback, and the FusedMultiLayer triple-fuse pattern all assumed
/// <c>typeof(T) == typeof(float)</c> without checking — T=double crashed.
/// VGGNetwork&lt;double&gt; hit this exact path through Conv → BN → ReLU → ...
/// → Dense → Softmax and silently fell back to eager.
/// </summary>
public class CompiledTrainingPlanDoublePathTests
{
    /// <summary>
    /// Issue #340 primary repro — ReLU on the forward path with T=double.
    /// Pre-fix this threw <c>InvalidCastException</c> at plan.Step() because
    /// <c>CompiledTrainingPlan.BuildForwardAction</c>'s ReLU branch ran for any T
    /// then cast inputs to <c>Tensor&lt;float&gt;</c> in the replay closure.
    /// </summary>
    [Fact]
    public void Step_DoubleReLU_DoesNotThrowInvalidCast()
    {
        var engine = new CpuEngine();

        var input = Tensor<double>.CreateRandom([4, 3]);
        var weight = Tensor<double>.CreateRandom([3, 2]);

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var matmul = engine.TensorMatMul(input, weight);
            var activated = engine.ReLU(matmul);
            engine.ReduceSum(activated, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        using (plan)
        {
            // Pre-fix this would throw InvalidCastException
            // ("Unable to cast object of type 'System.Double[]' to type 'System.Single[]'").
            var loss = plan.Step();
            Assert.False(double.IsNaN(loss[0]), "Loss is NaN on T=double ReLU plan");
        }
    }

    /// <summary>
    /// Issue #340: validates the ReLU forward output on T=double matches a
    /// reference computation through the eager engine (the kernel itself
    /// works; the cast was the bug).
    /// </summary>
    [Fact]
    public void Step_DoubleReLU_ProducesCorrectForward()
    {
        var engine = new CpuEngine();

        var inputData = new[] { -1.0, 2.0, -3.0, 4.0 };
        var input = new Tensor<double>(inputData, new[] { 4 });
        var weight = new Tensor<double>(new[] { 1.0 }, new[] { 1, 1 });

        Tensor<double> reluOutput;
        using (var scope = GraphMode.Enable())
        {
            var reshaped = engine.Reshape(input, new[] { 4, 1 });
            var matmul = engine.TensorMatMul(reshaped, weight);
            reluOutput = engine.ReLU(matmul);
            engine.ReduceSum(reluOutput, null);
            using var plan = scope.CompileTraining(new[] { weight });
            plan.Step();
        }

        // ReLU(x) = max(0, x). For input [-1, 2, -3, 4] × 1.0 → [0, 2, 0, 4].
        var expected = new[] { 0.0, 2.0, 0.0, 4.0 };
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], reluOutput.GetFlat(i), precision: 9);
    }

    /// <summary>
    /// Issue #340: repeated Step() calls on a ReLU-bearing graph at T=double.
    /// Pre-fix this crashed on first Step inside the ReLU forward closure;
    /// post-fix the plan runs cleanly across multiple iterations.
    /// </summary>
    [Fact]
    public void Step_DoubleReLU_RunsMultipleStepsWithoutError()
    {
        var engine = new CpuEngine();

        var input = Tensor<double>.CreateRandom([8, 4]);
        var weight = Tensor<double>.CreateRandom([4, 2]);

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var hidden = engine.TensorMatMul(input, weight);
            var activated = engine.ReLU(hidden);
            engine.ReduceSum(activated, null);
            plan = scope.CompileTraining(new[] { weight });
        }

        using (plan)
        {
            // Pre-fix this throws InvalidCastException on the first Step's
            // ReLU forward closure replay. Multiple steps confirm the cached
            // closure stays valid across iterations.
            for (int i = 0; i < 5; i++)
            {
                var loss = plan.Step();
                Assert.False(double.IsNaN(loss[0]));
                Assert.False(double.IsInfinity(loss[0]));
            }
        }
    }

    /// <summary>
    /// Issue #340: TensorMatMul backward through the BLAS-Sgemm fast path on
    /// T=double. Pre-fix the path was reached for any T (no
    /// <c>typeof(T) == typeof(float)</c> gate) and crashed casting the
    /// double-storage arrays to float[].
    /// </summary>
    [Fact]
    public void Step_DoubleMatMulBackward_DoesNotThrowInvalidCast()
    {
        var engine = new CpuEngine();

        var inputA = Tensor<double>.CreateRandom([4, 3]);
        var inputB = Tensor<double>.CreateRandom([3, 2]);

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var output = engine.TensorMatMul(inputA, inputB);
            engine.ReduceSum(output, null);
            plan = scope.CompileTraining(new[] { inputB });
        }

        using (plan)
        {
            var loss = plan.Step();
            Assert.False(double.IsNaN(loss[0]));
        }
    }
}
