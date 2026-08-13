using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public sealed class RoutedDiagonalSsmScanTests
{
    private static Tensor<float> CreateFloat(int[] shape,int seed,float scale)
    {
        int length=1;foreach(int size in shape)length*=size;var values=new float[length];
        for(int i=0;i<length;i++)values[i]=(float)Math.Sin(seed*0.37+i*0.61)*scale;
        return new Tensor<float>(values,shape);
    }

    private static Tensor<double> Create(int[] shape, int seed, double scale = 0.2)
    {
        int length = 1; foreach (int size in shape) length *= size;
        var values = new double[length];
        for (int i = 0; i < length; i++) values[i] = Math.Sin(seed * 0.37 + i * 0.61) * scale;
        return new Tensor<double>(values, shape);
    }

    [Fact]
    public void Backward_AllDifferentiableInputsMatchCentralDifferences()
    {
        const int batch=1,time=3,model=3,experts=2,state=2;
        Tensor<double>[] inputs =
        {
            Create(new[]{batch,time,model},1),
            new Tensor<double>(new[]{1.0,0.0, 1.0,1.0, 0.0,1.0},new[]{batch,time,experts}),
            Create(new[]{experts,state},2,0.6),
            Create(new[]{experts,state,model},3),
            Create(new[]{experts,model,state},4),
            Create(new[]{experts,model},5)
        };
        var projection=Create(new[]{batch,time,experts,model},9,0.8);var engine=new CpuEngine();
        System.Collections.Generic.Dictionary<Tensor<double>,Tensor<double>> gradients;
        using(var tape=new GradientTape<double>())
        {
            var output=Scan(engine,inputs);var loss=engine.ReduceSum(engine.TensorMultiply(output,projection),new[]{0,1,2,3},false);
            gradients=tape.ComputeGradients(loss,inputs);
        }
        const double eps=1e-6;
        foreach(var input in inputs)
            for(int i=0;i<input.Length;i++)
            {
                double original=input[i];input[i]=original+eps;double plus=Dot(Scan(engine,inputs),projection);
                input[i]=original-eps;double minus=Dot(Scan(engine,inputs),projection);input[i]=original;
                double numeric=(plus-minus)/(2*eps),analytic=gradients[input][i];
                Assert.True(Math.Abs(numeric-analytic)<=3e-6+3e-5*Math.Abs(numeric),
                    $"input {Array.IndexOf(inputs,input)}[{i}]: analytic={analytic:R}, numeric={numeric:R}");
            }
    }

    [Fact]
    public void RoutedMixtureComposition_InputVjpMatchesCentralDifferences()
    {
        const int batch=1,time=4,model=6,experts=4,state=3,topK=2;
        var input=Create(new[]{batch,time,model},11,0.5);var router=Create(new[]{model,experts},12,0.3);
        var transition=Create(new[]{experts,state},13,0.6);var inputMap=Create(new[]{experts,state,model},14,0.2);
        var outputMap=Create(new[]{experts,model,state},15,0.2);var skip=Create(new[]{experts,model},16,0.3);
        var gateWeight=Create(new[]{model,model},17,0.2);var outputWeight=Create(new[]{model,model},18,0.2);
        var projection=Create(new[]{batch,time,model},19,0.8);var engine=new CpuEngine();Tensor<double> gradient;
        Tensor<double> Run()
        {
            var flat=engine.Reshape(input,new[]{batch*time,model});var logits=engine.TensorMatMul(flat,router);
            var probabilities=engine.Softmax(logits,1);_=engine.TensorTopK(probabilities,topK,1,out Tensor<int> indices);
            var mask=new Tensor<double>(new[]{batch*time,experts});
            for(int token=0;token<batch*time;token++)for(int selected=0;selected<topK;selected++)mask[token,indices[token,selected]]=1;
            var masked=engine.TensorBroadcastMultiply(probabilities,mask);
            var weights=engine.TensorBroadcastDivide(masked,engine.TensorAddScalar(engine.ReduceSum(masked,new[]{1},true),1e-10));
            var scan=engine.RoutedDiagonalSsmScanForward(input,engine.Reshape(mask,new[]{batch,time,experts}),transition,inputMap,outputMap,skip);
            var mixed=engine.ReduceSum(engine.TensorBroadcastMultiply(scan,engine.Reshape(weights,new[]{batch,time,experts,1})),new[]{2},false);
            var gate=engine.Swish(engine.Reshape(engine.TensorMatMul(flat,gateWeight),new[]{batch,time,model}));
            var projected=engine.TensorMatMul(engine.Reshape(engine.TensorMultiply(mixed,gate),new[]{batch*time,model}),outputWeight);
            return engine.Reshape(projected,new[]{batch,time,model});
        }
        using(var tape=new GradientTape<double>()){var loss=engine.ReduceSum(engine.TensorMultiply(Run(),projection),new[]{0,1,2},false);gradient=tape.ComputeGradients(loss,new[]{input})[input];}
        const double eps=1e-6;for(int i=0;i<input.Length;i++)
        {
            double original=input[i];input[i]=original+eps;double plus=Dot(Run(),projection);input[i]=original-eps;double minus=Dot(Run(),projection);input[i]=original;
            double numeric=(plus-minus)/(2*eps),analytic=gradient[i];Assert.True(Math.Abs(numeric-analytic)<=3e-6+3e-5*Math.Abs(numeric),$"input[{i}]: analytic={analytic:R}, numeric={numeric:R}");
        }
    }

    [Fact]
    public void RoutedMixtureProductionFloat_InputVjpMatchesCentralDifferences()
    {
        const int batch=1,time=4,model=256,experts=8,state=16,topK=2;
        var input=CreateFloat(new[]{batch,time,model},11,0.8f);var router=CreateFloat(new[]{model,experts},12,0.1f);
        var transition=CreateFloat(new[]{experts,state},13,0.3f);var inputMap=CreateFloat(new[]{experts,state,model},14,0.08f);
        var outputMap=CreateFloat(new[]{experts,model,state},15,0.08f);var skip=CreateFloat(new[]{experts,model},16,0.3f);
        var gateWeight=CreateFloat(new[]{model,model},17,0.1f);var outputWeight=CreateFloat(new[]{model,model},18,0.1f);
        var projection=CreateFloat(new[]{batch,time,model},19,0.8f);var engine=new CpuEngine();Tensor<float> gradient;
        Tensor<float> Run()
        {
            var flat=engine.Reshape(input,new[]{batch*time,model});var logits=engine.TensorMatMul(flat,router);
            var probabilities=engine.Softmax(logits,1);_=engine.TensorTopK(probabilities,topK,1,out Tensor<int> indices);
            var mask=new Tensor<float>(new[]{batch*time,experts});for(int token=0;token<batch*time;token++)for(int selected=0;selected<topK;selected++)mask[token,indices[token,selected]]=1;
            var masked=engine.TensorBroadcastMultiply(probabilities,mask);var weights=engine.TensorBroadcastDivide(masked,engine.TensorAddScalar(engine.ReduceSum(masked,new[]{1},true),1e-10f));
            var scan=engine.RoutedDiagonalSsmScanForward(input,engine.Reshape(mask,new[]{batch,time,experts}),transition,inputMap,outputMap,skip);
            var mixed=engine.ReduceSum(engine.TensorBroadcastMultiply(scan,engine.Reshape(weights,new[]{batch,time,experts,1})),new[]{2},false);
            var gate=engine.Swish(engine.Reshape(engine.TensorMatMul(flat,gateWeight),new[]{batch,time,model}));
            return engine.Reshape(engine.TensorMatMul(engine.Reshape(engine.TensorMultiply(mixed,gate),new[]{batch*time,model}),outputWeight),new[]{batch,time,model});
        }
        using(var tape=new GradientTape<float>()){var loss=engine.ReduceSum(engine.TensorMultiply(Run(),projection),new[]{0,1,2},false);gradient=tape.ComputeGradients(loss,new[]{input})[input];}
        const float eps=1e-3f;for(int sample=0;sample<12;sample++)
        {
            int i=sample*(input.Length/12);float original=input[i];input[i]=original+eps;double plus=DotFloat(Run(),projection);input[i]=original-eps;double minus=DotFloat(Run(),projection);input[i]=original;
            double numeric=(plus-minus)/(2*eps),analytic=gradient[i],scale=Math.Max(Math.Max(Math.Abs(numeric),Math.Abs(analytic)),1.0);
            Assert.True(Math.Abs(numeric-analytic)/scale<5e-2,$"input[{i}]: analytic={analytic:R}, numeric={numeric:R}");
        }
    }

    private static Tensor<double> Scan(CpuEngine engine,Tensor<double>[] p)
        => engine.RoutedDiagonalSsmScanForward(p[0],p[1],p[2],p[3],p[4],p[5]);

    private static double Dot(Tensor<double> a,Tensor<double> b)
    {
        double result=0;for(int i=0;i<a.Length;i++)result+=a[i]*b[i];return result;
    }

    private static double DotFloat(Tensor<float> a,Tensor<float> b)
    {
        double result=0;for(int i=0;i<a.Length;i++)result+=a[i]*b[i];return result;
    }
}
