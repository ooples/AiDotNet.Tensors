using System.Diagnostics;
using System.Globalization;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Hardware-profiled evidence harness for PR #971. It separates cold end-to-end conversion from
/// warmed input-resident execution, forces every GPU result to materialize, and prints the actual
/// precision plan selected by the backend. It intentionally does not impose timing thresholds in CI.
/// </summary>
internal static class GpuPrecisionEvidence
{
    internal static int Run(string[] args)
    {
        int size = ParsePositive(args, 1, 512);
        int iterations = ParsePositive(args, 2, 15);
        string? outputPath = args.Length > 3 ? args[3] : null;
        var report = new StringBuilder();
        void Line(string value = "")
        {
            Console.WriteLine(value);
            report.AppendLine(value);
        }

        using var direct = new DirectGpuEngine();
        if (!direct.IsAvailable || direct.Backend is null)
        {
            Line("# PR #971 GPU precision evidence");
            Line();
            Line("No DirectGPU backend is available on this host. No hardware timing was recorded.");
            WriteReport(outputPath, report);
            return 2;
        }

        using var gpu = new DirectGpuTensorEngine(direct);
        var cpu = new CpuEngine();
        var backend = direct.Backend;
        Line("# PR #971 GPU precision evidence");
        Line();
        Line($"- UTC: {DateTime.UtcNow:O}");
        Line($"- OS: {Environment.OSVersion}");
        Line($"- Runtime: {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}");
        Line($"- CPU logical processors: {Environment.ProcessorCount}");
        Line($"- Backend: {direct.BackendName}");
        Line($"- Device: {direct.DeviceName}");
        Line($"- Vendor: {direct.DeviceVendor}");
        Line($"- Compute units: {direct.ComputeUnits}");
        Line($"- Global memory: {direct.GlobalMemoryBytes / (1024d * 1024 * 1024):F2} GiB");
        Line($"- Matrix: {size} x {size}; warmups: 3; measured iterations: {iterations}");
        Line();

        PrintCapabilityMatrix(backend, Line);
        PrintTypeMatrix(backend, Line);

        var random = new Random(971);
        int length = checked(size * size);
        var doubleA = new double[length];
        var doubleB = new double[length];
        var floatA = new float[length];
        var floatB = new float[length];
        for (int i = 0; i < length; i++)
        {
            doubleA[i] = random.NextDouble() - 0.5;
            doubleB[i] = random.NextDouble() - 0.5;
            floatA[i] = (float)doubleA[i];
            floatB[i] = (float)doubleB[i];
        }

        using var residentDoubleA = new Tensor<double>((double[])doubleA.Clone(), new[] { size, size });
        using var residentDoubleB = new Tensor<double>((double[])doubleB.Clone(), new[] { size, size });
        using var residentFloatA = new Tensor<float>((float[])floatA.Clone(), new[] { size, size });
        using var residentFloatB = new Tensor<float>((float[])floatB.Clone(), new[] { size, size });

        var cpuDouble = Measure(() =>
        {
            using var a = new Tensor<double>((double[])doubleA.Clone(), new[] { size, size });
            using var b = new Tensor<double>((double[])doubleB.Clone(), new[] { size, size });
            using var result = cpu.TensorMatMul(a, b);
            return result.GetDataArray()[length - 1];
        }, warmups: 1, iterations: Math.Max(3, Math.Min(iterations, 7)));

        var gpuFloatCold = Measure(() =>
        {
            using var a = new Tensor<float>((float[])floatA.Clone(), new[] { size, size });
            using var b = new Tensor<float>((float[])floatB.Clone(), new[] { size, size });
            using var result = gpu.TensorMatMul(a, b);
            return result.GetDataArray()[length - 1];
        }, 3, iterations);
        var gpuFloatColdPlan = GpuPrecisionDiagnostics.LastPlan;

        var gpuDoubleCold = Measure(() =>
        {
            using var a = new Tensor<double>((double[])doubleA.Clone(), new[] { size, size });
            using var b = new Tensor<double>((double[])doubleB.Clone(), new[] { size, size });
            using var result = gpu.TensorMatMul(a, b);
            return result.GetDataArray()[length - 1];
        }, 3, iterations);
        var gpuDoubleColdPlan = GpuPrecisionDiagnostics.LastPlan;

        var gpuFloatResident = Measure(() =>
        {
            using var result = gpu.TensorMatMul(residentFloatA, residentFloatB);
            return result.GetDataArray()[length - 1];
        }, 3, iterations);
        var gpuFloatResidentPlan = GpuPrecisionDiagnostics.LastPlan;

        var gpuDoubleResident = Measure(() =>
        {
            using var result = gpu.TensorMatMul(residentDoubleA, residentDoubleB);
            return result.GetDataArray()[length - 1];
        }, 3, iterations);
        var gpuDoubleResidentPlan = GpuPrecisionDiagnostics.LastPlan;

        TimingResult? gpuFp16Cold = null;
        TimingResult? gpuFp16Resident = null;
        GpuComputePlan? gpuFp16ColdPlan = null;
        GpuComputePlan? gpuFp16ResidentPlan = null;
        using (var autocast = new AutocastScope(PrecisionMode.Float16))
        {
            var probe = GpuPrecisionPlanner.CreatePlan<float>(backend, GpuPrecisionOperation.MatMul, "evidence-probe");
            if (probe.InputStorage == GpuScalarType.Float16)
            {
                gpuFp16Cold = Measure(() =>
                {
                    using var a = new Tensor<float>((float[])floatA.Clone(), new[] { size, size });
                    using var b = new Tensor<float>((float[])floatB.Clone(), new[] { size, size });
                    using var result = gpu.TensorMatMul(a, b);
                    return result.GetDataArray()[length - 1];
                }, 3, iterations);
                gpuFp16ColdPlan = GpuPrecisionDiagnostics.LastPlan;
                gpuFp16Resident = Measure(() =>
                {
                    using var result = gpu.TensorMatMul(residentFloatA, residentFloatB);
                    return result.GetDataArray()[length - 1];
                }, 3, iterations);
                gpuFp16ResidentPlan = GpuPrecisionDiagnostics.LastPlan;
            }
        }

        using var referenceA = new Tensor<double>((double[])doubleA.Clone(), new[] { size, size });
        using var referenceB = new Tensor<double>((double[])doubleB.Clone(), new[] { size, size });
        using var referenceTensor = cpu.TensorMatMul(referenceA, referenceB);
        var reference = referenceTensor.GetDataArray();
        using var autoTensor = gpu.TensorMatMul(residentDoubleA, residentDoubleB);
        var autoValues = autoTensor.GetDataArray();
        var autoError = RelativeMaxError(reference, autoValues);

        double? fp16Error = null;
        if (gpuFp16Resident is not null)
        {
            using var autocast = new AutocastScope(PrecisionMode.Float16);
            using var fp16Tensor = gpu.TensorMatMul(residentFloatA, residentFloatB);
            fp16Error = RelativeMaxError(reference, fp16Tensor.GetDataArray());
        }

        Line("## Timings");
        Line();
        Line("Every GPU sample includes output materialization/synchronization. Cold samples also include tensor creation, upload, and T↔compute-format conversion; resident samples reuse input tensors.");
        Line();
        Line("| Scenario | Median (ms) | P95 (ms) | Relative to native FP32 | Selected route |");
        Line("|---|---:|---:|---:|---|");
        WriteTiming("CPU declared double control", cpuDouble, gpuFloatCold.MedianMs, "CPU Float64", Line);
        WriteTiming("GPU native FP32 cold", gpuFloatCold, gpuFloatCold.MedianMs, Describe(gpuFloatColdPlan), Line);
        WriteTiming("GPU double→auto→double cold", gpuDoubleCold, gpuFloatCold.MedianMs, Describe(gpuDoubleColdPlan), Line);
        WriteTiming("GPU native FP32 resident inputs", gpuFloatResident, gpuFloatResident.MedianMs, Describe(gpuFloatResidentPlan), Line);
        WriteTiming("GPU double→auto→double resident inputs", gpuDoubleResident, gpuFloatResident.MedianMs, Describe(gpuDoubleResidentPlan), Line);
        if (gpuFp16Cold is { } fp16Cold)
            WriteTiming("GPU FP16 autocast cold", fp16Cold, gpuFloatCold.MedianMs, Describe(gpuFp16ColdPlan), Line);
        if (gpuFp16Resident is { } fp16Resident)
            WriteTiming("GPU FP16 autocast resident inputs", fp16Resident, gpuFloatResident.MedianMs, Describe(gpuFp16ResidentPlan), Line);
        Line();
        Line($"- End-to-end speedup, GPU converted double vs CPU declared double: {cpuDouble.MedianMs / gpuDoubleCold.MedianMs:F2}x");
        Line($"- Cold conversion overhead vs native FP32 GPU control: {(gpuDoubleCold.MedianMs / gpuFloatCold.MedianMs - 1) * 100:F1}%");
        Line($"- Resident-input conversion overhead vs native FP32 GPU control: {(gpuDoubleResident.MedianMs / gpuFloatResident.MedianMs - 1) * 100:F1}%");
        Line();
        Line("## Accuracy");
        Line();
        Line($"- GPU double→FP32→double relative max error vs CPU Float64: {autoError:E6}");
        if (fp16Error is not null)
            Line($"- GPU FP16 autocast relative max error vs CPU Float64: {fp16Error.Value:E6}");
        Line();
        Line("These measurements describe this named device and driver only. Packed/emulated formats and higher-precision fallbacks are labeled as such; they are not claimed as native hardware support.");

        WriteReport(outputPath, report);
        return 0;
    }

    private static void PrintCapabilityMatrix(IDirectGpuBackend backend, Action<string> line)
    {
        line("## Backend capability matrix");
        line(string.Empty);
        line("| Operation | Autocast format | Input | Multiply | Accumulator | Output | Implementation | Storage bytes reduced |");
        line("|---|---|---|---|---|---|---|---|");
        foreach (var operation in new[]
                 {
                     GpuPrecisionOperation.MatMul,
                     GpuPrecisionOperation.BatchMatMul,
                     GpuPrecisionOperation.MatMulTransposed,
                     GpuPrecisionOperation.Add,
                     GpuPrecisionOperation.Relu,
                     GpuPrecisionOperation.Gelu,
                     GpuPrecisionOperation.Convolution,
                     GpuPrecisionOperation.Reduction,
                 })
        {
            foreach (var capability in GpuPrecisionPlanner.GetCapabilities(backend, operation))
            {
                line($"| {operation} | {capability.ComputeFormat} | {capability.InputStorage} | {capability.MultiplyType} | {capability.AccumulatorType} | {capability.OutputStorage} | {capability.Implementation} | {capability.ReducesStorageBytes} |");
            }
        }
        line(string.Empty);
    }

    private static void PrintTypeMatrix(IDirectGpuBackend backend, Action<string> line)
    {
        line("## Public type parity matrix (speed-first Auto, MatMul)");
        line(string.Empty);
        line("| Public T | Route | Format | Input | Multiply | Accumulator | Output | Fallback |");
        line("|---|---|---|---|---|---|---|---|");
        PrintType<float>(backend, line);
        PrintType<double>(backend, line);
        PrintType<Half>(backend, line);
        PrintType<BFloat16>(backend, line);
        PrintType<Float8E4M3>(backend, line);
        PrintType<Float8E5M2>(backend, line);
        PrintType<int>(backend, line);
        PrintType<long>(backend, line);
        PrintType<decimal>(backend, line);
        line(string.Empty);
    }

    private static void PrintType<T>(IDirectGpuBackend backend, Action<string> line)
    {
        var plan = GpuPrecisionPlanner.CreatePlan<T>(backend, GpuPrecisionOperation.MatMul, "type-matrix");
        line($"| {typeof(T).Name} | {plan.Route} | {plan.ComputeFormat} | {plan.InputStorage} | {plan.MultiplyType} | {plan.AccumulatorType} | {plan.OutputStorage} | {plan.FallbackReason ?? "—"} |");
    }

    private static TimingResult Measure(Func<double> action, int warmups, int iterations)
    {
        double checksum = 0;
        for (int i = 0; i < warmups; i++) checksum += action();
        var samples = new double[iterations];
        for (int i = 0; i < iterations; i++)
        {
            var stopwatch = Stopwatch.StartNew();
            checksum += action();
            stopwatch.Stop();
            samples[i] = stopwatch.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        int p95Index = Math.Min(samples.Length - 1, (int)Math.Ceiling(samples.Length * 0.95) - 1);
        return new TimingResult(samples[samples.Length / 2], samples[p95Index], checksum);
    }

    private static void WriteTiming(
        string scenario,
        TimingResult timing,
        double controlMedian,
        string route,
        Action<string> line)
        => line($"| {scenario} | {timing.MedianMs:F3} | {timing.P95Ms:F3} | {timing.MedianMs / controlMedian:F3}x | {route} |");

    private static string Describe(GpuComputePlan? plan)
        => plan is null
            ? "No diagnostic"
            : $"{plan.Route} format={plan.ComputeFormat} physical={plan.InputStorage}/{plan.MultiplyType}/{plan.AccumulatorType}/{plan.OutputStorage}"
              + (plan.FallbackReason is null ? string.Empty : $" ({plan.FallbackReason})");

    private static double RelativeMaxError(double[] expected, double[] actual)
    {
        double maxError = 0;
        double maxReference = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            maxError = Math.Max(maxError, Math.Abs(expected[i] - actual[i]));
            maxReference = Math.Max(maxReference, Math.Abs(expected[i]));
        }
        return maxError / Math.Max(maxReference, double.Epsilon);
    }

    private static double RelativeMaxError(double[] expected, float[] actual)
    {
        var widened = new double[actual.Length];
        for (int i = 0; i < actual.Length; i++) widened[i] = actual[i];
        return RelativeMaxError(expected, widened);
    }

    private static int ParsePositive(string[] args, int index, int fallback)
        => args.Length > index
           && int.TryParse(args[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out int value)
           && value > 0
            ? value
            : fallback;

    private static void WriteReport(string? path, StringBuilder report)
    {
        if (string.IsNullOrWhiteSpace(path)) return;
        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        File.WriteAllText(fullPath, report.ToString());
        Console.WriteLine($"Evidence written to {fullPath}");
    }

    private readonly record struct TimingResult(double MedianMs, double P95Ms, double Checksum);
}
