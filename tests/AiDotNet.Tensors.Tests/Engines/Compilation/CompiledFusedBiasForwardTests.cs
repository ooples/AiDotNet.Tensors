using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Engines.Compilation;

/// <summary>
/// The compiled plan's fused rank-2 bias forward (<c>x[R,C] op bias</c>, the bias a row <c>[C]</c>/<c>[1,C]</c> or a
/// column <c>[R,1]</c>, either operand order for add/multiply) must replay exactly what the eager engine computes, in
/// float and double, serial and row-parallel. The loss weights every output element by a distinct position-dependent
/// factor, so a bias applied to the wrong row or column changes it (a plain sum would not notice a transposed bias).
/// </summary>
public class CompiledFusedBiasForwardTests
{
    public enum Op { XPlusB, BPlusX, XMinusB, XTimesB, BTimesX }
    public enum BiasShape { Row, Row2D, Column }

    private static int[] BiasDims(BiasShape s, int r, int c) => s switch
    {
        BiasShape.Row => new[] { c },
        BiasShape.Row2D => new[] { 1, c },
        _ => new[] { r, 1 },
    };

    private static Tensor<TT> Fill<TT>(int[] shape, int salt) where TT : unmanaged
    {
        var t = new Tensor<TT>(shape);
        var ops = MathHelper.GetNumericOperations<TT>();
        for (int i = 0; i < t.Length; i++) t[i] = ops.FromDouble(Math.Sin((i + salt) * 0.37) + 0.1 * (i % 7));
        return t;
    }

    private static (double compiled, double eager) Run<TT>(Op op, BiasShape bs, int r, int c) where TT : unmanaged
    {
        var engine = new CpuEngine();
        var weights = Fill<TT>(new[] { r, c }, 991);   // position-dependent loss weights (a constant, not trained)

        (Tensor<TT> loss, Tensor<TT>[] ps) Build()
        {
            var x = Fill<TT>(new[] { r, c }, 3);
            var b = Fill<TT>(BiasDims(bs, r, c), 17);
            var y = op switch
            {
                Op.XPlusB => engine.TensorBroadcastAdd(x, b),
                Op.BPlusX => engine.TensorBroadcastAdd(b, x),
                Op.XMinusB => engine.TensorBroadcastSubtract(x, b),
                Op.XTimesB => engine.TensorBroadcastMultiply(x, b),
                _ => engine.TensorBroadcastMultiply(b, x),
            };
            return (engine.ReduceSum(engine.TensorMultiply(y, weights), null), new[] { x, b });
        }

        double eager = Convert.ToDouble(Build().loss[0]);
        ICompiledTrainingPlan<TT> plan;
        using (var scope = GraphMode.Enable())
        {
            var (_, ps) = Build();
            plan = scope.CompileTraining(ps);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);   // pins the forward replay
            var loss = new Tensor<TT>(new[] { 1 });
            plan.StepInto(loss);
            plan.StepInto(loss);                                              // second replay: recycled buffers
            return (Convert.ToDouble(loss[0]), eager);
        }
    }

    public static TheoryData<Op, BiasShape, int, int> Cases()
    {
        var d = new TheoryData<Op, BiasShape, int, int>();
        foreach (Op op in Enum.GetValues(typeof(Op)))
            foreach (BiasShape bs in Enum.GetValues(typeof(BiasShape)))
            {
                d.Add(op, bs, 5, 7);
                d.Add(op, bs, 256, 300);   // large enough for the row-parallel path
            }
        return d;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Double_CompiledMatchesEager(Op op, BiasShape bs, int r, int c)
    {
        var (compiled, eager) = Run<double>(op, bs, r, c);
        Assert.True(Math.Abs(compiled - eager) <= 1e-9 * Math.Max(1, Math.Abs(eager)), $"compiled {compiled:R} vs eager {eager:R}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Float_CompiledMatchesEager(Op op, BiasShape bs, int r, int c)
    {
        var (compiled, eager) = Run<float>(op, bs, r, c);
        Assert.True(Math.Abs(compiled - eager) <= 1e-3 * Math.Max(1, Math.Abs(eager)), $"compiled {compiled:R} vs eager {eager:R}");
    }
}
