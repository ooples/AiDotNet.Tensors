using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private static readonly string? CpuWaveCaptureDirectory = Environment.GetEnvironmentVariable("AIDOTNET_CPU_WAVE_BOUNDARY_DIR");
    private static int s_waveCapture;
    private static Action<IEngine> ObserveCpuWaveBoundary(CompiledStep<T> step,
        Dictionary<Tensor<T>, Tensor<T>> gradients, IEngine engine, Action<IEngine> action)
    {
        if (string.IsNullOrEmpty(CpuWaveCaptureDirectory) || typeof(T) != typeof(float)
            || step.OpType != OpType.Tanh || step.OutputBuffer.Length != 8192
            || engine.Unwrap().GetType() != typeof(CpuEngine) || !ReferenceEquals(engine, engine.Unwrap())
            || !gradients.TryGetValue(step.OutputBuffer, out var upstream)) return action;
        return replayEngine =>
        {
            // This is the last consumer of the completed waveform gradient before its pool
            // slot may be reused. Copy only; no observers or alternative math paths are enabled.
            MaterializeIfGradPlaceholder(upstream);
            using var upstreamKeep = new KeepAliveScope(upstream);
            using var waveKeep = new KeepAliveScope(step.OutputBuffer);
            int index = Interlocked.Increment(ref s_waveCapture);
            Directory.CreateDirectory(CpuWaveCaptureDirectory!);
            using var process = System.Diagnostics.Process.GetCurrentProcess();
            string prefix = Path.Combine(CpuWaveCaptureDirectory!, "pid" + process.Id + "-wave" + index);
            var wave = ((Tensor<float>)(object)step.OutputBuffer).AsSpanUnmarked().ToArray();
            var seed = ((Tensor<float>)(object)upstream).AsSpanUnmarked().ToArray();
            var preTanh = ((Tensor<float>)(object)step.Inputs[0]).AsSpanUnmarked().ToArray();
            File.WriteAllBytes(prefix + "-wave.f32bin", MemoryMarshal.AsBytes(wave.AsSpan()).ToArray());
            File.WriteAllBytes(prefix + "-gradient.f32bin", MemoryMarshal.AsBytes(seed.AsSpan()).ToArray());
            File.WriteAllBytes(prefix + "-pre-tanh.f32bin", MemoryMarshal.AsBytes(preTanh.AsSpan()).ToArray());
            File.WriteAllText(prefix + ".txt", "diagnostic_only=true\nformal_performance_win=false\nshape="
                + string.Join(",", step.OutputBuffer._shape) + "\nindex=" + index.ToString(CultureInfo.InvariantCulture) + "\n");
            action(replayEngine);
        };
    }
}
