using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class MesaScanTests
{
    private static Tensor<double> Create(int[] shape, int seed, double scale = 0.2)
    {
        int length = 1;
        foreach (int size in shape) length *= size;
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = Math.Sin(seed * 0.43 + i * 0.71) * scale;
        return new Tensor<double>(values, shape);
    }

    [Fact]
    public void Forward_MatchesIndependentWoodburyReference()
    {
        const int batch = 2, time = 3, heads = 2, dim = 2, model = heads * dim;
        var q = Create(new[] { batch, time, model }, 1);
        var k = Create(new[] { batch, time, model }, 2);
        var v = Create(new[] { batch, time, model }, 3);
        var w0 = Create(new[] { heads, dim, dim }, 4);

        var actual = new CpuEngine().MesaScanForward(q, k, v, w0, 0.7, heads);
        double[] expected = Reference(q, k, v, w0, 0.7, batch, time, heads, dim);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual.GetFlat(i), 11);
    }

    [Fact]
    public void Backward_AllDifferentiableInputsMatchCentralDifferences()
    {
        const int batch = 1, time = 3, heads = 2, dim = 2, model = heads * dim;
        Tensor<double>[] inputs =
        {
            Create(new[] { batch, time, model }, 1),
            Create(new[] { batch, time, model }, 2),
            Create(new[] { batch, time, model }, 3),
            Create(new[] { heads, dim, dim }, 4)
        };
        var projection = Create(new[] { batch, time, model }, 9, 0.8);
        var engine = new CpuEngine();
        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var weighted = engine.TensorMultiply(Scan(engine, inputs, heads), projection);
            var loss = engine.ReduceSum(weighted, new[] { 0, 1, 2 }, keepDims: false);
            gradients = tape.ComputeGradients(loss, inputs);
        }

        const double epsilon = 1e-6;
        foreach (Tensor<double> input in inputs)
        {
            double[] values = input.GetDataArray()!;
            for (int i = 0; i < values.Length; i++)
            {
                double original = values[i];
                values[i] = original + epsilon;
                double plus = Dot(Scan(engine, inputs, heads), projection);
                values[i] = original - epsilon;
                double minus = Dot(Scan(engine, inputs, heads), projection);
                values[i] = original;
                double numeric = (plus - minus) / (2 * epsilon);
                double analytic = gradients[input].GetFlat(i);
                Assert.True(Math.Abs(numeric - analytic) <= 3e-6 + 3e-5 * Math.Abs(numeric),
                    $"input {Array.IndexOf(inputs, input)}[{i}]: analytic={analytic:R}, numeric={numeric:R}");
            }
        }
    }

    [Fact]
    public void Backward_ProductionShapeFloatInputSamplesMatchCentralDifferences()
    {
        const int batch = 1, time = 4, heads = 8, dim = 32, model = heads * dim;
        var q = CreateFloat(new[] { batch, time, model }, 1, 1.2f);
        var k = CreateFloat(new[] { batch, time, model }, 2, 1.2f);
        var v = CreateFloat(new[] { batch, time, model }, 3, 1.2f);
        var w0 = CreateFloat(new[] { heads, dim, dim }, 4, 0.02f);
        var projection = CreateFloat(new[] { batch, time, model }, 9, 0.8f);
        var engine = new CpuEngine();

        Dictionary<Tensor<float>, Tensor<float>> gradients;
        using (var tape = new GradientTape<float>())
        {
            var output = engine.MesaScanForward(q, k, v, w0, 0.01f, heads);
            var weighted = engine.TensorMultiply(output, projection);
            var loss = engine.ReduceSum(weighted, new[] { 0, 1, 2 }, keepDims: false);
            gradients = tape.ComputeGradients(loss, new[] { q, k, v, w0 });
        }

        const float epsilon = 1e-3f;
        var inputs = new[] { q, k, v, w0 };
        foreach (var input in inputs)
        {
            for (int sample = 0; sample < 12; sample++)
            {
                int index = sample * (input.Length / 12);
                float original = input[index];
                input[index] = original + epsilon;
                double plus = DotFloat(engine.MesaScanForward(q, k, v, w0, 0.01f, heads), projection);
                input[index] = original - epsilon;
                double minus = DotFloat(engine.MesaScanForward(q, k, v, w0, 0.01f, heads), projection);
                input[index] = original;
                double numeric = (plus - minus) / (2 * epsilon);
                double analytic = gradients[input][index];
                double scale = Math.Max(Math.Max(Math.Abs(numeric), Math.Abs(analytic)), 1.0);
                Assert.True(Math.Abs(numeric - analytic) / scale < 5e-2,
                    $"input {Array.IndexOf(inputs, input)}[{index}]: analytic={analytic:R}, numeric={numeric:R}");
            }
        }
    }

    [Fact]
    public void Backward_ProductionShapeDoubleKeySamplesMatchCentralDifferences()
    {
        const int batch = 1, time = 4, heads = 8, dim = 32, model = heads * dim;
        var q = Create(new[] { batch, time, model }, 1, 1.2);
        var k = Create(new[] { batch, time, model }, 2, 1.2);
        var v = Create(new[] { batch, time, model }, 3, 1.2);
        var w0 = Create(new[] { heads, dim, dim }, 4, 0.02);
        var projection = Create(new[] { batch, time, model }, 9, 0.8);
        var engine = new CpuEngine();
        Tensor<double> gradient;
        using (var tape = new GradientTape<double>())
        {
            var output = engine.MesaScanForward(q, k, v, w0, 0.01, heads);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, projection), new[] { 0, 1, 2 }, false);
            gradient = tape.ComputeGradients(loss, new[] { k })[k];
        }
        const double epsilon = 1e-6;
        for (int sample = 0; sample < 12; sample++)
        {
            int index = sample * (k.Length / 12);
            double original = k[index];
            k[index] = original + epsilon;
            double plus = Dot(engine.MesaScanForward(q, k, v, w0, 0.01, heads), projection);
            k[index] = original - epsilon;
            double minus = Dot(engine.MesaScanForward(q, k, v, w0, 0.01, heads), projection);
            k[index] = original;
            double numeric = (plus - minus) / (2 * epsilon);
            double analytic = gradient[index];
            double scale = Math.Max(Math.Max(Math.Abs(numeric), Math.Abs(analytic)), 1.0);
            Assert.True(Math.Abs(numeric - analytic) / scale < 3e-5,
                $"k[{index}]: analytic={analytic:R}, numeric={numeric:R}");
        }
    }

    [Fact]
    public void Backward_LayerNormMesaCompositionInputSamplesMatchCentralDifferences()
    {
        const int batch = 1, time = 4, heads = 8, dim = 32, model = heads * dim;
        var input = CreateFloat(new[] { batch, time, model }, 11, 0.3f);
        var gamma = new Tensor<float>(new[] { model }); gamma.Fill(1f);
        var beta = new Tensor<float>(new[] { model });
        var qWeight = CreateFloat(new[] { model, model }, 12, 0.08f);
        var kWeight = CreateFloat(new[] { model, model }, 13, 0.08f);
        var vWeight = CreateFloat(new[] { model, model }, 14, 0.08f);
        var w0 = CreateFloat(new[] { heads, dim, dim }, 4, 0.02f);
        var projection = CreateFloat(new[] { batch, time, model }, 9, 0.8f);
        var engine = new CpuEngine();

        Tensor<float> gradient;
        using (var tape = new GradientTape<float>())
        {
            var output = Composed(engine, input, gamma, beta, qWeight, kWeight, vWeight, w0, heads);
            var loss = engine.ReduceSum(engine.TensorMultiply(output, projection), new[] { 0, 1, 2 }, false);
            gradient = tape.ComputeGradients(loss, new[] { input })[input];
        }
        const float epsilon = 1e-3f;
        for (int sample=0;sample<12;sample++)
        {
            int index=sample*(input.Length/12);float original=input[index];
            input[index]=original+epsilon;double plus=DotFloat(Composed(engine,input,gamma,beta,qWeight,kWeight,vWeight,w0,heads),projection);
            input[index]=original-epsilon;double minus=DotFloat(Composed(engine,input,gamma,beta,qWeight,kWeight,vWeight,w0,heads),projection);
            input[index]=original;double numeric=(plus-minus)/(2*epsilon),analytic=gradient[index];
            double scale=Math.Max(Math.Max(Math.Abs(numeric),Math.Abs(analytic)),1.0);
            Assert.True(Math.Abs(numeric-analytic)/scale<5e-2,$"input[{index}]: analytic={analytic:R}, numeric={numeric:R}");
        }
    }

    [Fact]
    public void Backward_FullMesaLayerCompositionInputSamplesMatchCentralDifferences()
    {
        const int batch=1,time=4,heads=8,dim=32,model=heads*dim;
        var input=CreateFloat(new[]{batch,time,model},11,0.3f);
        var gamma=new Tensor<float>(new[]{model});gamma.Fill(1f);var beta=new Tensor<float>(new[]{model});
        var qw=CreateFloat(new[]{model,model},12,0.08f);var kw=CreateFloat(new[]{model,model},13,0.08f);
        var vw=CreateFloat(new[]{model,model},14,0.08f);var gw=CreateFloat(new[]{model,model},15,0.08f);
        var ow=CreateFloat(new[]{model,model},16,0.08f);var w0=CreateFloat(new[]{heads,dim,dim},4,0.02f);
        var qb=CreateFloat(new[]{model},20,0.01f);var kb=CreateFloat(new[]{model},21,0.01f);
        var vb=CreateFloat(new[]{model},22,0.01f);var gb=CreateFloat(new[]{model},23,0.01f);
        var ob=CreateFloat(new[]{model},24,0.01f);var projection=CreateFloat(new[]{batch,time,model},9,0.8f);
        var engine=new CpuEngine();Tensor<float> gradient;
        using(var tape=new GradientTape<float>()){
            var output=FullComposed(engine,input,gamma,beta,qw,qb,kw,kb,vw,vb,gw,gb,ow,ob,w0,heads);
            var loss=engine.ReduceSum(engine.TensorMultiply(output,projection),new[]{0,1,2},false);
            gradient=tape.ComputeGradients(loss,new[]{input})[input];
        }
        const float epsilon=1e-3f;
        for(int sample=0;sample<12;sample++){
            int index=sample*(input.Length/12);float original=input[index];
            input[index]=original+epsilon;double plus=DotFloat(FullComposed(engine,input,gamma,beta,qw,qb,kw,kb,vw,vb,gw,gb,ow,ob,w0,heads),projection);
            input[index]=original-epsilon;double minus=DotFloat(FullComposed(engine,input,gamma,beta,qw,qb,kw,kb,vw,vb,gw,gb,ow,ob,w0,heads),projection);
            input[index]=original;double numeric=(plus-minus)/(2*epsilon),analytic=gradient[index];double scale=Math.Max(Math.Max(Math.Abs(numeric),Math.Abs(analytic)),1.0);
            Assert.True(Math.Abs(numeric-analytic)/scale<5e-2,$"input[{index}]: analytic={analytic:R}, numeric={numeric:R}");
        }
    }

    [Fact]
    public void Backward_MesaGateCompositionInputSamplesMatchCentralDifferences()
    {
        const int batch=1,time=4,heads=8,dim=32,model=heads*dim;
        var input=CreateFloat(new[]{batch,time,model},11,0.3f);var gamma=new Tensor<float>(new[]{model});gamma.Fill(1f);var beta=new Tensor<float>(new[]{model});
        var qw=CreateFloat(new[]{model,model},12,0.08f);var kw=CreateFloat(new[]{model,model},13,0.08f);var vw=CreateFloat(new[]{model,model},14,0.08f);
        var gw=CreateFloat(new[]{model,model},15,0.08f);var gb=CreateFloat(new[]{model},23,0.01f);var ow=CreateFloat(new[]{model,model},16,0.08f);var w0=CreateFloat(new[]{heads,dim,dim},4,0.02f);
        var projection=CreateFloat(new[]{batch,time,model},9,0.8f);var engine=new CpuEngine();Tensor<float> gradient;
        Tensor<float> Run(){var norm=engine.LayerNorm(input,gamma,beta,1e-5,out _,out _);var flat=engine.Reshape(norm,new[]{batch*time,model});
            Tensor<float> P(Tensor<float> w)=>engine.Reshape(engine.TensorMatMul(flat,w),new[]{batch,time,model});
            var mesa=engine.MesaScanForward(P(qw),P(kw),P(vw),w0,0.01f,heads);
            var gate=engine.Swish(engine.Reshape(engine.TensorBroadcastAdd(engine.TensorMatMul(flat,gw),engine.Reshape(gb,new[]{1,model})),new[]{batch,time,model}));
            var gated=engine.TensorMultiply(mesa,gate);var projected=engine.TensorMatMul(engine.Reshape(gated,new[]{batch*time,model}),ow);
            return engine.TensorAdd(engine.Reshape(projected,new[]{batch,time,model}),input);}
        using(var tape=new GradientTape<float>()){var loss=engine.ReduceSum(engine.TensorMultiply(Run(),projection),new[]{0,1,2},false);gradient=tape.ComputeGradients(loss,new[]{input})[input];}
        var gradientSnapshot = gradient.ToArray();
        const float epsilon=1e-3f;for(int sample=0;sample<12;sample++){int index=sample*(input.Length/12);float original=input[index];input[index]=original+epsilon;double plus=DotFloat(Run(),projection);input[index]=original-epsilon;double minus=DotFloat(Run(),projection);input[index]=original;
            Assert.Equal(gradientSnapshot[index], gradient[index]);
            double numeric=(plus-minus)/(2*epsilon),analytic=gradient[index],scale=Math.Max(Math.Max(Math.Abs(numeric),Math.Abs(analytic)),1.0);Assert.True(Math.Abs(numeric-analytic)/scale<5e-2,$"input[{index}]: analytic={analytic:R}, numeric={numeric:R}");}
    }

    private static Tensor<float> FullComposed(CpuEngine e,Tensor<float> input,Tensor<float> gamma,Tensor<float> beta,
        Tensor<float> qw,Tensor<float> qb,Tensor<float> kw,Tensor<float> kb,Tensor<float> vw,Tensor<float> vb,
        Tensor<float> gw,Tensor<float> gb,Tensor<float> ow,Tensor<float> ob,Tensor<float> w0,int heads)
    {
        int batch=input.Shape[0],time=input.Shape[1],model=input.Shape[2];var norm=e.LayerNorm(input,gamma,beta,1e-5,out _,out _);
        var flat=e.Reshape(norm,new[]{batch*time,model});
        Tensor<float> Project(Tensor<float> w,Tensor<float> b)=>e.Reshape(e.TensorBroadcastAdd(e.TensorMatMul(flat,w),e.Reshape(b,new[]{1,model})),new[]{batch,time,model});
        var q=Project(qw,qb);var k=Project(kw,kb);var v=Project(vw,vb);var gate=e.Swish(Project(gw,gb));
        var mesa=e.MesaScanForward(q,k,v,w0,0.01f,heads);var gated=e.TensorMultiply(mesa,gate);
        var projected=e.TensorBroadcastAdd(e.TensorMatMul(e.Reshape(gated,new[]{batch*time,model}),ow),e.Reshape(ob,new[]{1,model}));
        return e.TensorAdd(e.Reshape(projected,new[]{batch,time,model}),input);
    }

    private static Tensor<float> Composed(CpuEngine engine, Tensor<float> input, Tensor<float> gamma,
        Tensor<float> beta, Tensor<float> qWeight, Tensor<float> kWeight, Tensor<float> vWeight,
        Tensor<float> w0, int heads)
    {
        int batch=input.Shape[0],time=input.Shape[1],model=input.Shape[2];
        var normalized=engine.LayerNorm(input,gamma,beta,1e-5,out _,out _);
        var flat=engine.Reshape(normalized,new[]{batch*time,model});
        var q=engine.Reshape(engine.TensorMatMul(flat,qWeight),new[]{batch,time,model});
        var k=engine.Reshape(engine.TensorMatMul(flat,kWeight),new[]{batch,time,model});
        var v=engine.Reshape(engine.TensorMatMul(flat,vWeight),new[]{batch,time,model});
        return engine.MesaScanForward(q,k,v,w0,0.01f,heads);
    }

    private static Tensor<double> Scan(CpuEngine engine, Tensor<double>[] x, int heads)
        => engine.MesaScanForward(x[0], x[1], x[2], x[3], 0.7, heads);

    private static double Dot(Tensor<double> tensor, Tensor<double> projection)
    {
        double sum = 0;
        for (int i = 0; i < tensor.Length; i++) sum += tensor.GetFlat(i) * projection.GetFlat(i);
        return sum;
    }

    private static Tensor<float> CreateFloat(int[] shape, int seed, float scale)
    {
        int length = 1;
        foreach (int size in shape) length *= size;
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)Math.Sin(seed * 0.43 + i * 0.71) * scale;
        return new Tensor<float>(values, shape);
    }

    private static double DotFloat(Tensor<float> tensor, Tensor<float> projection)
    {
        double sum = 0;
        for (int i = 0; i < tensor.Length; i++) sum += tensor.GetFlat(i) * projection.GetFlat(i);
        return sum;
    }

    private static double[] Reference(
        Tensor<double> q, Tensor<double> k, Tensor<double> v, Tensor<double> w0,
        double lambda, int batch, int time, int heads, int dim)
    {
        int model = heads * dim;
        var result = new double[batch * time * model];
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
            {
                var w = new double[dim, dim];
                var p = new double[dim, dim];
                for (int i = 0; i < dim; i++)
                    for (int j = 0; j < dim; j++)
                    {
                        w[i, j] = w0[h, i, j];
                        p[i, j] = i == j ? 1 / lambda : 0;
                    }
                for (int t = 0; t < time; t++)
                {
                    int offset = (b * time + t) * model + h * dim;
                    var pk = new double[dim];
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++) pk[i] += p[i, j] * k.GetFlat(offset + j);
                    double denominator = 1;
                    for (int i = 0; i < dim; i++) denominator += k.GetFlat(offset + i) * pk[i];
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++) p[i, j] -= pk[i] * pk[j] / denominator;
                    var error = new double[dim];
                    for (int i = 0; i < dim; i++)
                    {
                        for (int j = 0; j < dim; j++) error[i] += w[i, j] * k.GetFlat(offset + j);
                        error[i] -= v.GetFlat(offset + i);
                    }
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++)
                        {
                            double row = 0;
                            for (int n = 0; n < dim; n++) row += k.GetFlat(offset + n) * p[n, j];
                            w[i, j] -= error[i] * row;
                        }
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++) result[offset + i] += w[i, j] * q.GetFlat(offset + j);
                }
            }
        return result;
    }
}
