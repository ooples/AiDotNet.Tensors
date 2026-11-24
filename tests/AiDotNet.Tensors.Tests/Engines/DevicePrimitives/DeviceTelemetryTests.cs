using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DevicePrimitives;

/// <summary>
/// What a training monitor reads about the device without reducing or copying anything itself: the device's free and
/// total memory as the driver reports them, its utilization (NVML on CUDA), and the global gradient norm a compiled
/// plan's clip already computed. None of them was public: CuBlasMatMul.GetMemoryInfo returned (0, 0), the backend's
/// free-memory query and the plan's norm record were internal.
/// </summary>
public class DeviceTelemetryTests
{
    private static bool TryCuda(out DirectGpuTensorEngine? engine)
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (!candidate.IsGpuAvailable || candidate.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)
            {
                candidate.Dispose();
                engine = null;
                return false;
            }
            engine = candidate;
            return true;
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { engine = null; return false; }
    }

    [SkippableFact]
    public void Device_memory_is_the_driver_figure_and_covers_this_process_allocations()
    {
        Skip.IfNot(TryCuda(out var gpu) && gpu is not null, "CUDA backend did not resolve.");
        using (gpu)
        {
            var backend = gpu.GetBackend()!;
            Assert.True(GpuMemoryStats.TryGetDeviceMemory(backend, out long free, out long total));
            Assert.True(total > 0 && free > 0 && free <= total, $"free {free} total {total}");
            Assert.Equal(backend.GlobalMemoryBytes, total);
            // Every process's use, so at least what this process's allocators hold.
            Assert.True(total - free >= GpuMemoryStats.Snapshot(backend).AllocatedBytes);
        }
    }

    [SkippableFact]
    public void CuBlasMatMul_GetMemoryInfo_reports_the_device_memory()
    {
        Skip.IfNot(CuBlasMatMul.IsAvailable, "cuBLAS is not available.");
        using var matMul = new CuBlasMatMul();
        var (used, total) = matMul.GetMemoryInfo();
        Assert.True(total > 0, "total was 0");
        Assert.True(used > 0 && used <= total, $"used {used} total {total}");
    }

    [SkippableFact]
    public void Utilization_reads_NVML_percentages_on_CUDA()
    {
        Skip.IfNot(TryCuda(out var gpu) && gpu is not null, "CUDA backend did not resolve.");
        using (gpu)
        {
            var backend = gpu.GetBackend()!;
            bool read = GpuUtilization.TryRead(backend, out int gpuPercent, out int memoryPercent);
            Skip.IfNot(read, "NVML is not installed with this driver.");
            Assert.InRange(gpuPercent, 0, 100);
            Assert.InRange(memoryPercent, 0, 100);
        }
    }

    // nvidia-smi's memory.used is what a monitor's device memory figure is checked against. The driver's free-memory
    // query is not it under Windows' display driver model (about 1.2 GB off on an RTX 3080 during training): NVML's is.
    [SkippableFact]
    public void Device_memory_in_use_is_the_figure_nvidia_smi_prints()
    {
        Skip.IfNot(TryCuda(out var gpu) && gpu is not null, "CUDA backend did not resolve.");
        using (gpu)
        {
            var backend = (AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend)gpu.GetBackend()!;
            using var held = backend.AllocateBuffer(64 << 20);   // 256 MiB, so this process is on the device
            backend.Synchronize();
            bool read = GpuUtilization.TryReadDeviceMemory(backend, out long used, out long total);
            Skip.IfNot(read, "NVML is not installed with this driver.");
            Skip.IfNot(TryNvidiaSmi(backend, out double smiUsedMb, out double smiTotalMb), "nvidia-smi is not on PATH.");
            double usedMb = used / (1024.0 * 1024.0), totalMb = total / (1024.0 * 1024.0);
            Assert.Equal(smiTotalMb, totalMb, 0.5);
            // Other processes can allocate between the two reads; 64 MiB of slack, far below the 1.2 GB it was off.
            Assert.True(Math.Abs(usedMb - smiUsedMb) <= 64, $"NVML used {usedMb:F1} MiB, nvidia-smi {smiUsedMb} MiB");
            Assert.True(used >= 256L << 20, $"used {used} does not cover this process's 256 MiB");
        }
    }

    private static bool TryNvidiaSmi(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend backend, out double usedMb, out double totalMb)
    {
        usedMb = totalMb = 0;
        string? uuid = backend.DeviceUuidForTests;
        if (uuid is null) return false;
        try
        {
            var info = new System.Diagnostics.ProcessStartInfo("nvidia-smi",
                $"-i {uuid} --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
            { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
            using var smi = System.Diagnostics.Process.Start(info)!;
            string line = smi.StandardOutput.ReadToEnd().Trim();
            smi.WaitForExit();
            var parts = line.Split(',');
            if (smi.ExitCode != 0 || parts.Length != 2) return false;
            usedMb = double.Parse(parts[0].Trim(), System.Globalization.CultureInfo.InvariantCulture);
            totalMb = double.Parse(parts[1].Trim(), System.Globalization.CultureInfo.InvariantCulture);
            return true;
        }
        catch (System.ComponentModel.Win32Exception) { return false; }
    }

    // loss = sum(p^2): the gradient is 2p, so the pre-clip norm is 2 ||p||. A learning rate of 0 keeps p, so every step
    // has that norm: through the warmup steps and the captured replays alike, and on the host plan.
    private static void AssertPlanReportsItsClipNorm(IEngine engine)
    {
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var rng = new Random(11);
            var p = new Tensor<float>(new[] { 48, 32 });
            double sq = 0;
            for (int i = 0; i < p.Length; i++)
            {
                p[i] = (float)(rng.NextDouble() * 2 - 1);
                sq += (double)p[i] * p[i];
            }
            double expected = 2 * Math.Sqrt(sq);
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(new[] { p }))
            {
                var loss = engine.ReduceSum(engine.TensorMultiply(p, p), null, false);
                plan = scope.CompileTraining(new[] { p }, loss);
            }
            using (plan as IDisposable)
            {
                var norms = Assert.IsAssignableFrom<ICompiledTrainingPlanGradientNorm>(plan);
                Assert.False(norms.TryGetLastGradientNorm(out _));
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0);
                plan.SetMaxGradNorm(1e-3);
                for (int step = 1; step <= 8; step++)
                {
                    plan.Step();
                    Assert.Equal(step, norms.GradientNormSteps);
                    Assert.True(norms.TryGetLastGradientNorm(out double norm));
                    Assert.True(Math.Abs(norm - expected) <= 1e-5 * expected, $"step {step}: norm {norm} vs {expected}");
                }
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [SkippableFact]
    public void A_compiled_device_plan_reports_the_norm_its_clip_computed_on_every_step()
    {
        Skip.IfNot(TryCuda(out var gpu) && gpu is not null, "CUDA backend did not resolve.");
        using (gpu) AssertPlanReportsItsClipNorm(gpu);
    }

    [Fact]
    public void A_compiled_host_plan_reports_the_norm_its_clip_computed_on_every_step()
        => AssertPlanReportsItsClipNorm(new CpuEngine());
}
