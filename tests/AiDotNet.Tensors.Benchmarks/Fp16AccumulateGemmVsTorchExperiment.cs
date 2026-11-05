using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Helpers;
using TorchSharp;
using TorchTensor = TorchSharp.torch.Tensor;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #1549: FP16 GEMM against PyTorch on the same CUDA device and precision (FP16 operands, FP16 output). Tensors runs
/// <see cref="CudaBackend.GemmFp16HalfOut"/>, which takes the FP16-accumulate route on shapes where it beat cuBLAS on
/// this device; PyTorch runs <c>torch.matmul</c> on FP16 tensors (cuBLAS, FP32 accumulation). Both errors are measured
/// against an FP64 product of the same operands. Usage: <c>--fp16-accumulate-vs-torch [rounds]</c>.
/// </summary>
internal static class Fp16AccumulateGemmVsTorchExperiment
{
    // ResNet im2col convs, transformer projections and MLPs, DiT-XL blocks, and a square reference.
    private static readonly (string Model, int M, int N, int K)[] Shapes =
    {
        ("ResNet conv3x3 56x56 b4", 12544, 128, 1152),
        ("ResNet conv3x3 28x28 b4", 3136, 256, 2304),
        ("ResNet conv3x3 14x14 b4", 784, 512, 4608),
        ("BERT-base qkv", 4096, 2304, 768),
        ("BERT-base mlp up", 4096, 3072, 768),
        ("BERT-base mlp down", 4096, 768, 3072),
        ("DiT-XL attn proj", 4096, 1152, 1152),
        ("DiT-XL mlp up", 4096, 4608, 1152),
        ("DiT-XL mlp down", 4096, 1152, 4608),
        ("square", 4096, 4096, 4096),
    };

    public static void Run(string[] args)
    {
        int rounds = args.Length > 0 ? int.Parse(args[0], CultureInfo.InvariantCulture) : 7;
        if (!torch.cuda.is_available())
        {
            Console.WriteLine("TorchSharp has no CUDA device.");
            Environment.ExitCode = 1;
            return;
        }
        using var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            Console.WriteLine("No CUDA device for the Tensors backend.");
            Environment.ExitCode = 1;
            return;
        }
        var rate = backend.MeasureTensorCoreAccumulateRates();
        Console.WriteLine($"device {backend.DeviceName}; torch {torch.__version__}; tensor-core rate FP32-acc " +
                          $"{rate.Fp32AccumulateTflops:F1} / FP16-acc {rate.Fp16AccumulateTflops:F1} TFLOP/s; rounds {rounds}");
        Console.WriteLine();
        Console.WriteLine("| model | M×N×K | route | Tensors ms | PyTorch ms | Tensors TFLOP/s | PyTorch TFLOP/s | ratio (T/P) | Tensors rel err | PyTorch rel err |");
        Console.WriteLine("|---|---|---|---|---|---|---|---|---|---|");

        var random = RandomHelper.CreateSeededRandom(1549);
        var ratios = new List<double>();
        foreach (var (model, m, n, k) in Shapes)
        {
            ushort[] aHost = HalfValues(random, m * k);
            ushort[] bHost = HalfValues(random, k * n);
            using var a = backend.AllocateByteBuffer(aHost.Length * sizeof(ushort));
            using var b = backend.AllocateByteBuffer(bHost.Length * sizeof(ushort));
            using var c = backend.AllocateByteBuffer(m * n * sizeof(ushort));
            backend.UploadBytes(a, Bytes(aHost));
            backend.UploadBytes(b, Bytes(bHost));

            using TorchTensor ta = HalfTensor(aHost, m, k);
            using TorchTensor tb = HalfTensor(bHost, k, n);

            // The first eager call decides the route (paired rounds against cuBLAS); warm both sides after it.
            backend.GemmFp16HalfOut(a, b, c, m, n, k);
            backend.Synchronize();
            int reps = Math.Max(3, (int)Math.Min(200, 4e11 / (2.0 * m * n * k)));
            double TensorsRun() => Time(() => backend.GemmFp16HalfOut(a, b, c, m, n, k), backend.Synchronize, reps);
            double TorchRun() => Time(() => { using var _ = torch.matmul(ta, tb); }, () => torch.cuda.synchronize(), reps);
            TensorsRun();
            TorchRun();

            // Interleaved rounds; each side's figure is its best round.
            double tensorsMs = double.MaxValue, torchMs = double.MaxValue;
            for (int r = 0; r < rounds; r++)
            {
                if ((r & 1) == 0) { tensorsMs = Math.Min(tensorsMs, TensorsRun()); torchMs = Math.Min(torchMs, TorchRun()); }
                else { torchMs = Math.Min(torchMs, TorchRun()); tensorsMs = Math.Min(tensorsMs, TensorsRun()); }
            }

            using TorchTensor reference = torch.matmul(ta.to_type(torch.ScalarType.Float64), tb.to_type(torch.ScalarType.Float64));
            double scale = reference.abs().max().item<double>();
            using TorchTensor tensorsOut = HalfTensor(ToUInt16(backend.DownloadBytes(c, m * n * sizeof(ushort))), m, n);
            using TorchTensor torchOut = torch.matmul(ta, tb);
            double tensorsErr = (tensorsOut.to_type(torch.ScalarType.Float64) - reference).abs().max().item<double>() / scale;
            double torchErr = (torchOut.to_type(torch.ScalarType.Float64) - reference).abs().max().item<double>() / scale;

            var verdict = backend.Fp16AccumulateVerdict(m, n, k, halfOutput: true);
            string route = verdict is { Routed: true } v ? $"fp16-acc ({v.MedianSpeedup:F2}x cuBLAS)" : "cuBLAS";
            double flops = 2.0 * m * n * k;
            double ratio = tensorsMs / torchMs;
            ratios.Add(ratio);
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "| {0} | {1}×{2}×{3} | {4} | {5:F3} | {6:F3} | {7:F1} | {8:F1} | {9:F3} | {10:E2} | {11:E2} |",
                model, m, n, k, route, tensorsMs, torchMs, flops / tensorsMs / 1e9, flops / torchMs / 1e9, ratio,
                tensorsErr, torchErr));
        }
        double geomean = Math.Exp(ratios.Average(Math.Log));
        Console.WriteLine();
        Console.WriteLine($"geometric mean time ratio (Tensors / PyTorch): {geomean.ToString("F3", CultureInfo.InvariantCulture)}");
    }

    // Milliseconds per call over reps back-to-back calls, synchronized at both ends.
    private static double Time(Action run, Action synchronize, int reps)
    {
        synchronize();
        var watch = Stopwatch.StartNew();
        for (int i = 0; i < reps; i++) run();
        synchronize();
        return watch.Elapsed.TotalMilliseconds / reps;
    }

    private static TorchTensor HalfTensor(ushort[] bits, int rows, int columns)
    {
        var values = new float[bits.Length];
        for (int i = 0; i < bits.Length; i++) values[i] = (float)BitConverter.UInt16BitsToHalf(bits[i]);
        // Every value is FP16-representable, so the conversion on the device is exact.
        using TorchTensor host = torch.tensor(values, new long[] { rows, columns });
        return host.to(torch.CUDA).to_type(torch.ScalarType.Float16);
    }

    private static ushort[] HalfValues(Random random, int count)
    {
        var values = new ushort[count];
        for (int i = 0; i < count; i++)
            values[i] = BitConverter.HalfToUInt16Bits((Half)(random.NextSingle() * 2f - 1f));
        return values;
    }

    private static byte[] Bytes(ushort[] values)
    {
        var bytes = new byte[values.Length * sizeof(ushort)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static ushort[] ToUInt16(byte[] bytes)
    {
        var values = new ushort[bytes.Length / sizeof(ushort)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
}
