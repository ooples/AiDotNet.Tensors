using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;
namespace AiDotNet.Tensors.Tests.Engines;
public class ZzRoute
{
    private readonly ITestOutputHelper _o; public ZzRoute(ITestOutputHelper o) => _o = o;
    [Fact] public void Route()
    {
        var e = new CpuEngine(); var r = new Random(1);
        var x = new Tensor<float>(new[] { 2, 256, 16, 16 }); var k = new Tensor<float>(new[] { 128, 256, 3, 3 });
        for (int i = 0; i < x.Length; i++) x[i] = (float)r.NextDouble(); for (int i = 0; i < k.Length; i++) k[i] = (float)r.NextDouble();
        void Run(string tag, Func<IDisposable?> scope)
        {
            int f0 = CpuEngine.FullIm2ColConvRunsOnThisThread, i0 = CpuEngine.ImplicitGemmConvRunsOnThisThread, b0 = CpuEngine.RowBlockConvRunsOnThisThread;
            using (scope()) e.Conv2D(x, k, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
            _o.WriteLine($"ROUTE {tag}: full+{CpuEngine.FullIm2ColConvRunsOnThisThread - f0} implicit+{CpuEngine.ImplicitGemmConvRunsOnThisThread - i0} rowblock+{CpuEngine.RowBlockConvRunsOnThisThread - b0}");
        }
        Run("default", () => null);
        Run("forceImplicit", () => CpuEngine.ForceImplicitGemmScope());
        Run("forceFull", () => CpuEngine.ForceFullIm2ColScope());
    }
}