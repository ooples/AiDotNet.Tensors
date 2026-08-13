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
public sealed class ComplexDiagonalSsmGpuParityTests
{
    public enum BackendKind { OpenCL, Vulkan }
    public static TheoryData<BackendKind> Backends => new() { BackendKind.OpenCL, BackendKind.Vulkan };

    [SkippableTheory]
    [MemberData(nameof(Backends))]
    public void NativeForward_MatchesCpuEngine(BackendKind kind)
    {
        (IDirectGpuBackend? backend, Action dispose) = TryCreate(kind);
        Skip.If(backend is null, $"{kind} backend is unavailable on this host.");
        try
        {
            const int batch=2, time=4, groups=2, width=3, state=4;
            float[] x=Values(batch*time*groups*width,1), ar=Values(groups*state,2,0.2f), ai=Values(groups*state,3,0.15f);
            float[] br=Values(groups*state*width,4), bi=Values(groups*state*width,5);
            float[] cr=Values(groups*width*state,6), ci=Values(groups*width*state,7), d=Values(groups*width,8);
            var engine=new CpuEngine();
            float[] expected=engine.ComplexDiagonalSsmScanForward(
                new Tensor<float>(x,[batch,time,groups,width]), new Tensor<float>(ar,[groups,state]),
                new Tensor<float>(ai,[groups,state]), new Tensor<float>(br,[groups,state,width]),
                new Tensor<float>(bi,[groups,state,width]), new Tensor<float>(cr,[groups,width,state]),
                new Tensor<float>(ci,[groups,width,state]), new Tensor<float>(d,[groups,width])).GetDataArray()!;

            using var xb=backend!.AllocateBuffer(x); using var arb=backend.AllocateBuffer(ar); using var aib=backend.AllocateBuffer(ai);
            using var brb=backend.AllocateBuffer(br); using var bib=backend.AllocateBuffer(bi); using var crb=backend.AllocateBuffer(cr);
            using var cib=backend.AllocateBuffer(ci); using var db=backend.AllocateBuffer(d); using var yb=backend.AllocateBuffer(expected.Length);
            backend.ComplexDiagonalSsmScanForward(xb,arb,aib,brb,bib,crb,cib,db,yb,batch,time,groups,width,state);
            float[] actual=backend.DownloadBuffer(yb);
            for(int i=0;i<expected.Length;i++)
                Assert.True(MathF.Abs(expected[i]-actual[i]) <= 2e-5f + 2e-5f*MathF.Abs(expected[i]),
                    $"{kind} output[{i}]={actual[i]:R}, CPU={expected[i]:R}");
        }
        finally { dispose(); }
    }

    private static (IDirectGpuBackend?,Action) TryCreate(BackendKind kind)
    {
        try
        {
            if(kind==BackendKind.OpenCL)
            {
                var backend=new OpenClBackend();
                if(backend.IsAvailable) return (backend,backend.Dispose);
                backend.Dispose(); return (null,()=>{});
            }
            var vulkan=VulkanBackend.Instance;
            return vulkan.Initialize() && vulkan.IsGlslCompilerAvailable ? (vulkan,()=>{}) : (null,()=>{});
        }
        catch { return (null,()=>{}); }
    }

    private static float[] Values(int length,int seed,float scale=0.3f)
    {
        var values=new float[length];
        for(int i=0;i<length;i++) values[i]=(float)Math.Sin((i+1)*0.63+seed*0.41)*scale;
        return values;
    }
}
#endif
