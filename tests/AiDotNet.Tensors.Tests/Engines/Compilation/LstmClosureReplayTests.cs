using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection("CompilationGlobalState")]
public sealed class LstmClosureReplayTests
{
    private const int Steps = 3, Hidden = 2;

    // Independent scalar recurrence; parameter groups are WIh, WHh, BIh, BHh, H0, C0.
    private static double Reference(double[] input, double[][] parameters, bool sequence, int optional)
    {
        var hidden = optional == 2 ? (double[])parameters[4].Clone() : new double[Hidden];
        var cell = optional == 2 ? (double[])parameters[5].Clone() : new double[Hidden];
        double result = 0;
        for (int step = 0; step < Steps; step++)
        {
            var gates = new double[4 * Hidden];
            for (int gate = 0; gate < gates.Length; gate++)
            {
                double value = parameters[0][gate] * input[step];
                for (int feature = 0; feature < Hidden; feature++)
                    value += parameters[1][gate * Hidden + feature] * hidden[feature];
                if (optional > 0) value += parameters[2][gate];
                if (optional == 2) value += parameters[3][gate];
                gates[gate] = value;
            }
            for (int feature = 0; feature < Hidden; feature++)
            {
                double Sigmoid(double value) => 1 / (1 + Math.Exp(-value));
                cell[feature] = Sigmoid(gates[Hidden + feature]) * cell[feature]
                    + Sigmoid(gates[feature]) * Math.Tanh(gates[2 * Hidden + feature]);
                hidden[feature] = Sigmoid(gates[3 * Hidden + feature]) * Math.Tanh(cell[feature]);
                if (sequence || step == Steps - 1)
                    result += hidden[feature] * (0.2 + (sequence ? step * Hidden + feature : feature) * 0.13);
            }
        }
        if (optional == 2)
            result += 0.17 * hidden.Sum() + 0.23 * cell.Sum();
        return result;
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 0)]
    [InlineData(false, 1)]
    [InlineData(true, 1)]
    [InlineData(false, 2)]
    [InlineData(true, 2)]
    public void Double_Lstm_closure_rebinds_real_inputs_and_preserves_all_gradients(bool sequence, int optional)
    {
        var engine = new CpuEngine();
        var values = new[]
        {
            new Tensor<double>(new[] { 4 * Hidden, 1 }),
            new Tensor<double>(new[] { 4 * Hidden, Hidden }),
            new Tensor<double>(new[] { 4 * Hidden }),
            new Tensor<double>(new[] { 4 * Hidden }),
            new Tensor<double>(new[] { 1, Hidden }),
            new Tensor<double>(new[] { 1, Hidden }),
        };
        for (int group = 0; group < values.Length; group++)
            for (int index = 0; index < values[group].Length; index++)
                values[group].SetFlat(index, 0.3 * Math.Sin(index + 0.7 * group));
        var parameters = values.Take(optional == 0 ? 2 : optional == 1 ? 3 : 6).ToArray();
        var input = new Tensor<double>(new[] { 0.3, -0.7, 1.1 }, new[] { 1, Steps, 1 });
        Tensor<double> Objective()
        {
            var h0 = optional == 2 ? values[4] : null;
            var c0 = optional == 2 ? values[5] : null;
            var bIh = optional > 0 ? values[2] : null;
            var bHh = optional == 2 ? values[3] : null;
            Tensor<double> output;
            Tensor<double>? finalHidden = null, finalCell = null;
            if (optional == 2)
                output = engine.LstmSequenceForward(input, h0, c0, values[0], values[1], bIh, bHh,
                    out finalHidden, out finalCell, sequence);
            else
                output = engine.LstmSequenceForward(input, h0, c0, values[0], values[1], bIh, bHh, sequence);
            var coefficients = new Tensor<double>(Enumerable.Range(0, output.Length)
                .Select(index => 0.2 + index * 0.13).ToArray(), output._shape);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, coefficients), null);
            if (optional == 2)
                loss = engine.TensorAdd(loss, engine.TensorAdd(
                    engine.TensorMultiplyScalar(engine.ReduceSum(finalHidden!, null), 0.17),
                    engine.TensorMultiplyScalar(engine.ReduceSum(finalCell!, null), 0.23)));
            return loss;
        }

        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            guard!.AttachTo(plan);
            for (int step = 0; step < 3; step++)
            {
                var inputValues = new[] { 0.3 + step * 0.2, -0.7 + step * 0.3, 1.1 - step * 0.4 };
                input = new Tensor<double>(inputValues, new[] { 1, Steps, 1 });
                for (int group = 0; group < parameters.Length; group++)
                {
                    for (int index = 0; index < parameters[group].Length; index++)
                        parameters[group].SetFlat(index, 0.3 * Math.Sin(index + 0.7 * group + 0.4 * step));
                    parameters[group].IncrementVersion();
                }
                Assert.True(guard.TryRebind(Objective, parameters, out _) == ClosureGuardResult.Matched, guard.LastReason);
                var point = values.Select(value => value.ToArray()).ToArray();
                Assert.Equal(Reference(inputValues, point, sequence, optional), plan.Step().ToArray()[0], 10);
                for (int group = 0; group < parameters.Length; group++)
                {
                    var actual = plan.Gradients[group].ToArray();
                    for (int index = 0; index < actual.Length; index++)
                    {
                        const double delta = 1e-5;
                        double saved = point[group][index];
                        point[group][index] = saved + delta;
                        double plus = Reference(inputValues, point, sequence, optional);
                        point[group][index] = saved - delta;
                        double minus = Reference(inputValues, point, sequence, optional);
                        point[group][index] = saved;
                        double expected = (plus - minus) / (2 * delta);
                        Assert.True(Math.Abs(actual[index] - expected) < 2e-8,
                            $"sequence={sequence}, optional={optional}, step={step}, group={group}[{index}]: expected {expected:R}, got {actual[index]:R}");
                    }
                    Assert.Equal(point[group], parameters[group].ToArray());
                }
            }
            sequence = !sequence;
            Assert.Equal(ClosureGuardResult.Mismatched, guard.TryRebind(Objective, parameters, out _));
        }
    }
}
