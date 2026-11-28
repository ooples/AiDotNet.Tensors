using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The engine's fused codegen kernel cache keeps an entry only while its graph (owned by a fused node) is alive, and a
/// live graph keeps hitting the kernel it compiled.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CodegenKernelCacheLifetimeTests
{
    [SkippableFact]
    public void RetiredGraph_IsCollectedWhileALiveGraphKeepsItsKernel()
    {
        DirectGpuTensorEngine? engine = null;
        try { engine = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(engine is null || !engine.IsGpuAvailable, "No direct GPU backend available.");
        using (engine)
        {
            Skip.If(engine.GetBackend() is not INativeGpuCodegenExecutor, "The active backend has no native codegen.");

            var live = AddReluGraph();
            Assert.True(Run(engine, live), "The backend declined a plain add + ReLU graph.");
            Assert.True(Run(engine, live));   // a replay of the same graph

            var retired = CompileAndDrop(engine);
            for (int i = 0; i < 3 && retired.IsAlive; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            Assert.False(retired.IsAlive, "The kernel cache kept a retired graph alive.");
            GC.KeepAlive(live);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference CompileAndDrop(DirectGpuTensorEngine engine)
    {
        var graph = AddReluGraph();
        Assert.True(Run(engine, graph));
        return new WeakReference(graph);
    }

    private static bool Run(DirectGpuTensorEngine engine, CodegenGraph graph)
    {
        var a = new Tensor<float>(new float[] { -1f, 2f, -3f, 4f }, new[] { 4 });
        var b = new Tensor<float>(new float[] { 0.5f, 0.5f, 0.5f, 0.5f }, new[] { 4 });
        var output = new Tensor<float>(new[] { 4 });
        if (!engine.TryExecuteCodegenGraphInto(graph, new[] { a, b }, output))
            return false;
        Assert.Equal(new[] { 0f, 2.5f, 0f, 4.5f }, output.ToArray());
        return true;
    }

    private static CodegenGraph AddReluGraph()
    {
        int[] shape = { 4 };
        var g = new CodegenGraph();
        int n0 = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, shape, 0));
        int n1 = g.AddNode(new CodegenNode(CodegenOpKind.LoadInput, Array.Empty<int>(), CodegenElementType.Float32, shape, 1));
        int add = g.AddNode(new CodegenNode(CodegenOpKind.Add, new[] { n0, n1 }, CodegenElementType.Float32, shape));
        int relu = g.AddNode(new CodegenNode(CodegenOpKind.ReLU, new[] { add }, CodegenElementType.Float32, shape));
        g.AddNode(new CodegenNode(CodegenOpKind.StoreOutput, new[] { relu }, CodegenElementType.Float32, shape, 0));
        return g;
    }
}
