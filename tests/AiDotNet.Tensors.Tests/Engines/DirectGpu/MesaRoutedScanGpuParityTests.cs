#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class MesaRoutedScanGpuParityTests
{
    public enum BackendKind { OpenCL, Vulkan }
    public static TheoryData<BackendKind> Backends => new() { BackendKind.OpenCL, BackendKind.Vulkan };

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void MesaNativeForward_MatchesCpuEngine(BackendKind kind)
    {
        (IDirectGpuBackend? backend, Action dispose) = TryCreate(kind);
        Skip.If(backend is null, $"{kind} backend is unavailable on this host.");
        try
        {
            const int batch=2,time=4,heads=2,headDim=3,model=heads*headDim;
            float[] q=Values(batch*time*model,1),k=Values(batch*time*model,2),v=Values(batch*time*model,3);
            float[] w=Values(heads*headDim*headDim,4,0.1f),lambda={0.7f};
            var engine=new CpuEngine();
            float[] expected=engine.MesaScanForward(
                new Tensor<float>(q,[batch,time,model]),new Tensor<float>(k,[batch,time,model]),
                new Tensor<float>(v,[batch,time,model]),new Tensor<float>(w,[heads,headDim,headDim]),
                lambda[0],heads).GetDataArray()!;
            using var qb=backend!.AllocateBuffer(q);using var kb=backend.AllocateBuffer(k);using var vb=backend.AllocateBuffer(v);
            using var wb=backend.AllocateBuffer(w);using var lb=backend.AllocateBuffer(lambda);using var yb=backend.AllocateBuffer(expected.Length);
            using var work=backend.AllocateBuffer(batch*heads*headDim*headDim);using var covariance=backend.AllocateBuffer(batch*heads*headDim*headDim);
            backend.MesaScanForward(qb,kb,vb,wb,lb,yb,work,covariance,batch,time,model,heads,headDim);
            AssertClose(kind,expected,backend.DownloadBuffer(yb),3e-5f,3e-5f);
        }
        finally { dispose(); }
    }

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void RoutedNativeForward_MatchesCpuEngine(BackendKind kind)
    {
        (IDirectGpuBackend? backend, Action dispose) = TryCreate(kind);
        Skip.If(backend is null, $"{kind} backend is unavailable on this host.");
        try
        {
            const int batch=2,time=4,model=3,experts=3,state=2;
            float[] input=Values(batch*time*model,11),mask=Values(batch*time*experts,12,0.4f);
            for(int i=0;i<mask.Length;i++)mask[i]=i%3==0?1f:(i%3==1?0.35f:0f);
            float[] transition=Values(experts*state,13,0.6f),inputMap=Values(experts*state*model,14);
            float[] outputMap=Values(experts*model*state,15),skip=Values(experts*model,16);
            var engine=new CpuEngine();
            float[] expected=engine.RoutedDiagonalSsmScanForward(
                new Tensor<float>(input,[batch,time,model]),new Tensor<float>(mask,[batch,time,experts]),
                new Tensor<float>(transition,[experts,state]),new Tensor<float>(inputMap,[experts,state,model]),
                new Tensor<float>(outputMap,[experts,model,state]),new Tensor<float>(skip,[experts,model])).GetDataArray()!;
            using var xb=backend!.AllocateBuffer(input);using var mb=backend.AllocateBuffer(mask);using var ab=backend.AllocateBuffer(transition);
            using var bb=backend.AllocateBuffer(inputMap);using var cb=backend.AllocateBuffer(outputMap);using var db=backend.AllocateBuffer(skip);
            using var yb=backend.AllocateBuffer(expected.Length);using var scratch=backend.AllocateBuffer(batch*experts*state);
            backend.RoutedDiagonalSsmScanForward(xb,mb,ab,bb,cb,db,yb,scratch,batch,time,model,experts,state);
            AssertClose(kind,expected,backend.DownloadBuffer(yb),2e-5f,2e-5f);
        }
        finally { dispose(); }
    }

    private static void AssertClose(BackendKind kind,float[] expected,float[] actual,float atol,float rtol)
    {
        Assert.Equal(expected.Length,actual.Length);
        for(int i=0;i<expected.Length;i++)
            Assert.True(MathF.Abs(expected[i]-actual[i])<=atol+rtol*MathF.Abs(expected[i]),
                $"{kind} output[{i}]={actual[i]:R}, CPU={expected[i]:R}");
    }

    private static (IDirectGpuBackend?,Action) TryCreate(BackendKind kind)
    {
        try
        {
            if(kind==BackendKind.OpenCL){var backend=new OpenClBackend();if(backend.IsAvailable)return(backend,backend.Dispose);backend.Dispose();return(null,()=>{});}
            var vulkan=VulkanBackend.Instance;return vulkan.Initialize()&&vulkan.IsGlslCompilerAvailable?(vulkan,()=>{}):(null,()=>{});
        }
        catch{return(null,()=>{});}
    }

    private static float[] Values(int length,int seed,float scale=0.25f)
    {
        var values=new float[length];for(int i=0;i<length;i++)values[i]=(float)Math.Sin((i+1)*0.63+seed*0.41)*scale;return values;
    }
}
#endif
