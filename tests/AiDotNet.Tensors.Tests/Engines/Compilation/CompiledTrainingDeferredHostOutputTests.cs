using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A GPU training trace records its node outputs with deferred host storage: the compiled plan binds every output to a
/// device buffer, so a host array per output only cost host memory for the plan's lifetime (an Informer CUDA run kept
/// 2.9 GB of them, most of its 5.1 GB peak RSS, against torch.compile's 1.7 GB). After the whole-step graph is captured
/// and replayed, the device-bound intermediates must hold no host array, and a host read of one must still download
/// the current step's values.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledTrainingDeferredHostOutputTests : IDisposable
{
    private const int Batch = 32, Inputs = 24, Hidden = 48, Outputs = 6, Steps = 8;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static bool TryGpu(out DirectGpuTensorEngine? engine)
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (!candidate.IsGpuAvailable) { candidate.Dispose(); engine = null; return false; }
            engine = candidate;
            return true;
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { engine = null; return false; }
    }

    private static float[] Rand(int length, int seed, float scale)
    {
        var rng = new Random(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return values;
    }

    private static void Feed(Tensor<float> t, int seed)
    {
        Rand(t.Length, seed, 1f).AsSpan().CopyTo(t.AsWritableSpan());
        t.IncrementVersion();
    }

    // hidden = relu(x w1 + b1) and pred = hidden w2 + b2 on the host.
    private static (float[] Hidden, float[] Pred) CpuForward(float[] x, float[] w1, float[] b1, float[] w2, float[] b2)
    {
        var pred = new float[Batch * Outputs];
        var hidden = new float[Batch * Hidden];
        for (int n = 0; n < Batch; n++)
        {
            for (int j = 0; j < Hidden; j++)
            {
                double acc = b1[j];
                for (int i = 0; i < Inputs; i++) acc += x[n * Inputs + i] * w1[i * Hidden + j];
                hidden[n * Hidden + j] = (float)Math.Max(acc, 0);
            }
            for (int k = 0; k < Outputs; k++)
            {
                double acc = b2[k];
                for (int j = 0; j < Hidden; j++) acc += hidden[n * Hidden + j] * w2[j * Outputs + k];
                pred[n * Outputs + k] = (float)acc;
            }
        }
        return (hidden, pred);
    }

    [SkippableFact]
    public void Captured_gpu_plan_keeps_no_host_copies_of_device_intermediates_and_host_reads_download_them()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null, "GPU backend did not resolve.");
        using (gpu)
        {
            AiDotNetEngine.Current = gpu;
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend,
                "Whole-step graph capture is CUDA-only.");
            Skip.If(Environment.GetEnvironmentVariable("AIDOTNET_CUDA_GRAPH_STEP") == "0",
                "Graph capture is disabled for this process.");

            var x = new Tensor<float>(Rand(Batch * Inputs, 1, 1f), [Batch, Inputs]);
            var y = new Tensor<float>(Rand(Batch * Outputs, 2, 1f), [Batch, Outputs]);
            var w1 = new Tensor<float>(Rand(Inputs * Hidden, 3, 0.3f), [Inputs, Hidden]);
            var b1 = new Tensor<float>([Hidden]);
            var w2 = new Tensor<float>(Rand(Hidden * Outputs, 4, 0.3f), [Hidden, Outputs]);
            var b2 = new Tensor<float>([Outputs]);
            var parameters = new[] { w1, b1, w2, b2 };
            foreach (var p in parameters) p.Gpu();

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var h = gpu.FusedLinear(x, w1, b1, FusedActivationType.ReLU);
                var pred = gpu.FusedLinear(h, w2, b2, FusedActivationType.None);
                var diff = gpu.TensorSubtract(pred, y);
                gpu.ReduceMean(gpu.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
                plan = scope.CompileTraining(parameters);
            }

            using (plan)
            {
                var concrete = (CompiledTrainingPlan<float>)plan;
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-2f);
                float[][] weightsBeforeLastStep = Array.Empty<float[]>();
                double loss = 0;
                for (int s = 0; s < Steps; s++)
                {
                    Feed(x, 100 + s);
                    Feed(y, 200 + s);
                    if (s == Steps - 1) weightsBeforeLastStep = parameters.Select(p => p.ToArray()).ToArray();
                    loss = plan.Step().ToArray()[0];
                }

                var exec = (IntPtr)typeof(CompiledTrainingPlan<float>)
                    .GetField("_stepGraphExec", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)!;
                Assert.True(exec != IntPtr.Zero, "the plan never replayed a captured step graph: " + concrete.GraphCaptureFailureReason);

                // No device-bound forward intermediate holds a host array after capture and replay.
                var lossOutput = typeof(CompiledTrainingPlan<float>)
                    .GetField("_lossOutput", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete);
                var steps = (CompiledStep<float>[])typeof(CompiledTrainingPlan<float>)
                    .GetField("_forwardSteps", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(concrete)!;
                var intermediates = steps.Select(st => st.OutputBuffer)
                    .Where(o => !ReferenceEquals(o, lossOutput) && o._gpuBuffer is not null)
                    .Distinct().ToArray();
                Assert.NotEmpty(intermediates);
                var withHostCopy = intermediates.Where(o => o.GetBackingArrayForCacheLookupUnsafe() is not null)
                    .Select(o => $"[{string.Join(",", o.Shape.ToArray())}]").ToArray();
                Assert.True(withHostCopy.Length == 0,
                    $"{withHostCopy.Length} of {intermediates.Length} device-bound intermediates keep a host array: "
                    + string.Join(" ", withHostCopy));

                // A host read of each of them still returns the last step's values, downloaded from the device on demand.
                var xHost = x.ToArray();
                var yHost = y.ToArray();
                var w = weightsBeforeLastStep;
                var (cpuHidden, cpuPred) = CpuForward(xHost, w[0], w[1], w[2], w[3]);
                int checkedHidden = 0;
                foreach (var intermediate in intermediates)
                {
                    Assert.True(intermediate.HasDeferredHostStorage);
                    var host = intermediate.ToArray();
                    Assert.False(intermediate.HasDeferredHostStorage, "a host read must allocate the host array");
                    if (host.Length != cpuHidden.Length) continue;
                    for (int i = 0; i < host.Length; i++)
                        Assert.True(Math.Abs(host[i] - cpuHidden[i]) < 1e-3f, $"hidden[{i}] = {host[i]}, host recompute {cpuHidden[i]}");
                    checkedHidden++;
                }
                Assert.True(checkedHidden > 0, "no device-bound intermediate had the hidden activation's shape");
                double cpuLoss = 0;
                for (int i = 0; i < cpuPred.Length; i++)
                {
                    double d = cpuPred[i] - yHost[i];
                    cpuLoss += d * d;
                }
                cpuLoss /= cpuPred.Length;
                Assert.True(Math.Abs(loss - cpuLoss) < 1e-3 * Math.Max(1, cpuLoss), $"loss {loss}, host recompute {cpuLoss}");
            }
        }
    }
}
