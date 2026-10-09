using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// <see cref="OptimizerBase.Step(IReadOnlyDictionary{Tensor{float}, Tensor{float}})"/>: every optimizer stepped from
/// tensor gradients (no copy into the group's buffers) must match the array path bit for bit, must never write the
/// gradients it reads, and must skip a parameter that has no gradient. Plus the per-optimizer defects found while
/// moving them onto it.
/// </summary>
public class OptimizerTensorGradientTests
{
    public enum Kind
    {
        Sgd, SgdMomentum, SgdNesterovDampened, Adam, AmsGrad, AdamW, RAdam, NAdam, Adamax, Adagrad, RmsProp,
        RmsPropCenteredMomentum, AdaDelta, Lion, Asgd, Rprop, Lamb, Lars, Ftrl, SparseAdam, DAdaptAdam, Prodigy,
        BF16Adam, FP8Lion, Shampoo, SparseAdam24,
    }

    // Three full 64K chunks and a tail, so the chunked kernels split it; Shampoo's full-matrix path stays small.
    private static int[] ShapeFor(Kind kind) => kind == Kind.Shampoo ? new[] { 12, 16 } : new[] { 196, 1004 };

    private static OptimizerBase Create(Kind kind) => kind switch
    {
        Kind.Sgd or Kind.SgdMomentum or Kind.SgdNesterovDampened => new SgdOptimizer(),
        Kind.Adam or Kind.AmsGrad => new AdamOptimizer(),
        Kind.AdamW => new AdamWOptimizer(),
        Kind.RAdam => new RAdamOptimizer(),
        Kind.NAdam => new NAdamOptimizer(),
        Kind.Adamax => new AdamaxOptimizer(),
        Kind.Adagrad => new AdagradOptimizer(),
        Kind.RmsProp or Kind.RmsPropCenteredMomentum => new RmsPropOptimizer(),
        Kind.AdaDelta => new AdaDeltaOptimizer(),
        Kind.Lion => new LionOptimizer(),
        Kind.Asgd => new AsgdOptimizer(),
        Kind.Rprop => new RpropOptimizer(),
        Kind.Lamb => new LambOptimizer(),
        Kind.Lars => new LarsOptimizer(),
        Kind.Ftrl => new FtrlOptimizer(),
        Kind.SparseAdam => new SparseAdamOptimizer(),
        Kind.DAdaptAdam => new DAdaptAdamOptimizer(),
        Kind.Prodigy => new ProdigyOptimizer(),
        Kind.BF16Adam => new BF16AdamOptimizer(),
        Kind.FP8Lion => new FP8LionOptimizer(),
        Kind.Shampoo => new ShampooOptimizer(),
        Kind.SparseAdam24 => new SparseAdam24Optimizer(),
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    // Coupled weight decay and maximize on wherever the optimizer has them, so the scratch path is exercised.
    private static Dictionary<string, double> OptionsFor(Kind kind)
    {
        var options = new Dictionary<string, double> { ["lr"] = 1e-2, ["weight_decay"] = 0.01, ["maximize"] = 1.0 };
        switch (kind)
        {
            case Kind.SgdMomentum: options["momentum"] = 0.9; break;
            case Kind.SgdNesterovDampened: options["momentum"] = 0.9; options["dampening"] = 0.3; options["nesterov"] = 1.0; break;
            case Kind.AmsGrad: options["amsgrad"] = 1.0; break;
            case Kind.RmsPropCenteredMomentum: options["centered"] = 1.0; options["momentum"] = 0.5; break;
            case Kind.Adagrad: options["lr_decay"] = 0.1; break;
        }
        return options;
    }

    private static byte[] Pattern(int length)
    {
        // Positions 0 and 2 of every block: nibble 0b1000 = 0x8, two per byte.
        var pattern = new byte[(length / 4 + 1) / 2];
        for (int i = 0; i < pattern.Length; i++) pattern[i] = 0x88;
        return pattern;
    }

    private static float[] RandomArray(int length, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var a = new float[length];
        for (int i = 0; i < length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static void AddArray(OptimizerBase optimizer, Kind kind, int[] shape, float[] parameter, float[] gradient)
    {
        var options = OptionsFor(kind);
        if (optimizer is ShampooOptimizer shampoo) shampoo.Add2DParameter(shape[0], shape[1], parameter, gradient, options);
        else if (optimizer is SparseAdam24Optimizer sparse24) sparse24.AddSparse24Parameter(parameter, gradient, Pattern(parameter.Length), options);
        else optimizer.AddParamGroup(options).AddParameter(parameter, gradient);
    }

    private static void AddTensor(OptimizerBase optimizer, Kind kind, Tensor<float> parameter)
    {
        var options = OptionsFor(kind);
        if (optimizer is ShampooOptimizer shampoo) shampoo.Add2DParameter(parameter, options);
        else if (optimizer is SparseAdam24Optimizer sparse24) sparse24.AddSparse24Parameter(parameter, Pattern(parameter.Length), options);
        else optimizer.AddParamGroup(options).AddParameter(parameter);
    }

    [Theory]
    [InlineData(Kind.Sgd)] [InlineData(Kind.SgdMomentum)] [InlineData(Kind.SgdNesterovDampened)]
    [InlineData(Kind.Adam)] [InlineData(Kind.AmsGrad)] [InlineData(Kind.AdamW)] [InlineData(Kind.RAdam)]
    [InlineData(Kind.NAdam)] [InlineData(Kind.Adamax)] [InlineData(Kind.Adagrad)] [InlineData(Kind.RmsProp)]
    [InlineData(Kind.RmsPropCenteredMomentum)] [InlineData(Kind.AdaDelta)] [InlineData(Kind.Lion)]
    [InlineData(Kind.Asgd)] [InlineData(Kind.Rprop)] [InlineData(Kind.Lamb)] [InlineData(Kind.Lars)]
    [InlineData(Kind.Ftrl)] [InlineData(Kind.SparseAdam)] [InlineData(Kind.DAdaptAdam)] [InlineData(Kind.Prodigy)]
    [InlineData(Kind.BF16Adam)] [InlineData(Kind.FP8Lion)] [InlineData(Kind.Shampoo)] [InlineData(Kind.SparseAdam24)]
    public void TensorGradients_MatchTheArrayPath_AndAreNeverWritten(Kind kind)
    {
        var shape = ShapeFor(kind);
        int n = shape[0] * shape[1];
        var initial = RandomArray(n, 1);

        var arrayParameter = (float[])initial.Clone();
        var arrayGradient = new float[n];
        var arrayOptimizer = Create(kind);
        AddArray(arrayOptimizer, kind, shape, arrayParameter, arrayGradient);

        var tensorParameter = new Tensor<float>((float[])initial.Clone(), shape);
        var tensorOptimizer = Create(kind);
        AddTensor(tensorOptimizer, kind, tensorParameter);

        for (int step = 0; step < 3; step++)
        {
            var g = RandomArray(n, 100 + step);
            Array.Copy(g, arrayGradient, n);
            arrayOptimizer.Step();
            Assert.Equal(g, arrayGradient);   // the array path no longer writes the caller's gradient either

            // The gradient is the second half of a larger tensor: a contiguous view at a non-zero storage offset.
            var storage = new Tensor<float>(new[] { 2, shape[0], shape[1] });
            for (int i = 0; i < n; i++) storage.SetFlat(n + i, g[i]);
            var gradient = storage.Slice(0, 1, 2);
            int versionBefore = tensorParameter.Version;
            tensorOptimizer.Step(new Dictionary<Tensor<float>, Tensor<float>> { [tensorParameter] = gradient });

            Assert.True(tensorParameter.Version != versionBefore, "the stepped parameter was not marked modified");
            for (int i = 0; i < n; i++)
                Assert.True(gradient.GetFlat(i) == g[i], $"{kind}: gradient[{i}] was written");
            for (int i = 0; i < n; i++)
                Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(arrayParameter[i]) == TestHelpers.MathCompat.SingleToInt32Bits(tensorParameter.GetFlat(i)),
                    $"{kind} step {step}: parameter[{i}] array {arrayParameter[i]} vs tensor {tensorParameter.GetFlat(i)}");
        }
        Assert.NotEqual(initial, arrayParameter);
    }

    [Fact]
    public void AParameterWithNoGradient_IsSkipped()
    {
        var first = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var second = new Tensor<float>(new[] { 4f, 5f, 6f }, new[] { 3 });
        var optimizer = new AdamOptimizer();
        var group = optimizer.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.1 });
        group.AddParameter(first);
        group.AddParameter(second);
        int secondVersion = second.Version;

        optimizer.Step(new Dictionary<Tensor<float>, Tensor<float>>
        {
            [first] = new Tensor<float>(new[] { 1f, 1f, 1f }, new[] { 3 }),
        });

        Assert.NotEqual(1f, first.GetFlat(0));
        Assert.Equal(new[] { 4f, 5f, 6f }, new[] { second.GetFlat(0), second.GetFlat(1), second.GetFlat(2) });
        Assert.Equal(secondVersion, second.Version);
        Assert.False(optimizer.StateDict().State.ContainsKey(1), "a skipped parameter must not get optimizer state");
    }

    [Fact]
    public void AGradientOfTheWrongSize_IsRejected()
    {
        var parameter = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var optimizer = new SgdOptimizer();
        optimizer.AddParamGroup().AddParameter(parameter);
        Assert.Throws<ArgumentException>(() => optimizer.Step(new Dictionary<Tensor<float>, Tensor<float>>
        {
            [parameter] = new Tensor<float>(new[] { 1f, 1f }, new[] { 2 }),
        }));
    }

    [Fact]
    public void AViewIsRejectedAsAParameter()
    {
        var storage = new Tensor<float>(new[] { 2, 4 });
        var view = storage.Slice(0, 1, 2);
        Assert.Throws<ArgumentException>(() => new SgdOptimizer().AddParamGroup().AddParameter(view));
    }

    [Fact]
    public void Sgd_DampenedMomentum_StartsTheBufferAtTheFirstGradient()
    {
        // PyTorch: buf = grad on the first step (no dampening), then buf = momentum*buf + (1 - dampening)*grad.
        const float lr = 0.1f, momentum = 0.9f, dampening = 0.5f;
        var p = new[] { 1f };
        var grad = new float[1];
        var optimizer = new SgdOptimizer();
        optimizer.AddParamGroup(new Dictionary<string, double>
        {
            ["lr"] = lr, ["momentum"] = momentum, ["dampening"] = dampening,
        }).AddParameter(p, grad);

        float expected = 1f, buf = 0f;
        float[] gradients = { 0.4f, -0.2f, 0.7f };
        for (int t = 0; t < gradients.Length; t++)
        {
            grad[0] = gradients[t];
            optimizer.Step();
            buf = t == 0 ? gradients[t] : momentum * buf + (1f - dampening) * gradients[t];
            expected -= lr * buf;
            Assert.Equal(expected, p[0], 6);
        }
    }

    [Fact]
    public void Adagrad_LrDecay_ShrinksTheStepWithTheStepCount()
    {
        // PyTorch: clr = lr / (1 + (step - 1) * lr_decay); sum += g^2; p -= clr * g / (sqrt(sum) + eps).
        const float lr = 0.5f, decay = 0.25f, eps = 1e-10f;
        var p = new[] { 2f };
        var grad = new float[1];
        var optimizer = new AdagradOptimizer();
        optimizer.AddParamGroup(new Dictionary<string, double> { ["lr"] = lr, ["lr_decay"] = decay, ["eps"] = eps })
            .AddParameter(p, grad);

        float expected = 2f, sum = 0f;
        float[] gradients = { 0.3f, 0.6f, -0.9f, 0.2f };
        for (int t = 1; t <= gradients.Length; t++)
        {
            float g = gradients[t - 1];
            grad[0] = g;
            optimizer.Step();
            sum += g * g;
            expected -= lr / (1f + (t - 1) * decay) * g / (MathF.Sqrt(sum) + eps);
            Assert.Equal(expected, p[0], 6);
        }
    }

    [Fact]
    public void FP8Lion_AMomentThatOutgrowsTheScale_IsNotStuckAtTheOldClamp()
    {
        // Step 1's tiny gradient shrinks the moment scale to ~1e-5/224. Step 2's gradient of 1000 takes the moment to
        // ~10, far past what the old scale can encode. Step 3's gradient of -50 gives c = 0.9*m - 5: positive (a
        // descent step) only if the moment really is ~10. Quantizing against the old scale first clamped it to ~2e-5,
        // which made c negative and stepped the wrong way.
        const float lr = 0.01f;
        var p = new[] { 0f, 0f, 0f, 0f };
        var grad = new float[4];
        var optimizer = new FP8LionOptimizer();
        optimizer.AddParamGroup(new Dictionary<string, double> { ["lr"] = lr }).AddParameter(p, grad);

        foreach (float g in new[] { 1e-3f, 1000f, -50f })
        {
            for (int i = 0; i < 4; i++) grad[i] = g;
            optimizer.Step();
        }

        for (int i = 0; i < 4; i++) Assert.Equal(-3f * lr, p[i], 6);
    }
}
