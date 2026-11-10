using System;
using System.Reflection;
using System.Runtime.Intrinsics;
// Drives DirectConvAvx2's weight gradient and strided input gradient directly (internal API, via reflection) on one
// shape, so an SDE instruction mix (-mix) counts what those kernels execute. Arg: "dw" or "dx".
var t = typeof(AiDotNet.Tensors.Engines.CpuEngine).Assembly.GetType("AiDotNet.Tensors.Engines.Simd.DirectConvAvx2", true)!;
const BindingFlags F = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
Console.WriteLine($"Vector512={Vector512.IsHardwareAccelerated} uses512={t.GetProperty("UsesAvx512Kernel", F)!.GetValue(null)}");
int batch = 2, inC = 64, outC = 64, size = 16, k = 3, pad = 1;
string mode = args.Length > 0 ? args[0] : "dw";
int stride = mode == "dx" ? 2 : 1;
int o = (size + 2 * pad - k) / stride + 1;
var rng = new Random(1445);
float[] R(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1); return a; }
var x = R(batch * inC * size * size); var w = R(outC * inC * k * k); var dy = R(batch * outC * o * o);
for (int rep = 0; rep < 3; rep++)
{
    if (mode == "dw")
    {
        var dw = new float[w.Length];
        t.GetMethod("BackwardKernel", F)!.Invoke(null, new object[] { x, 0, dy, 0, dw, 0, false, batch, inC, size, size, outC, k, k,
            stride, stride, pad, pad, 1, 1, o, o, 0 });
        Console.WriteLine($"dw checksum {Sum(dw):R}");
    }
    else
    {
        var dx = new float[x.Length];
        t.GetMethod("BackwardInput", F)!.Invoke(null, new object[] { dy, 0, w, 0, dx, 0, false, batch, inC, size, size, outC, k, k,
            stride, stride, pad, pad, o, o, 0 });
        Console.WriteLine($"dx checksum {Sum(dx):R}");
    }
}
static double Sum(float[] a) { double s = 0; foreach (var v in a) s += v; return s; }
