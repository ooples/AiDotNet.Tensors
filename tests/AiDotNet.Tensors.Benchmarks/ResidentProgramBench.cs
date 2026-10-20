// Copyright (c) AiDotNet. All rights reserved.
// #890 phase 3 (CUDA) and #1448-#1452 (HIP, OpenCL, Vulkan, Metal, WebGPU): the resident program measured end to end.
//
// A declared three-stage chain - per-column affine + ReLU, row softmax, residual add - run two ways: as a resident
// program (CUDA and HIP: ONE persistent cooperative launch; OpenCL, Vulkan, Metal and WebGPU: one submission of the
// ordered stage dispatches) and as the same stages run one by one (RunStagesSeparately, the same device code). The
// PyTorch composition of the chain is timed by tools/parity/resident_chain_torch.py with the same protocol.
//
// Protocol (phase 0.5): interleave the two sides sample by sample, time 50 chains per region with one sync at the end,
// and take the median of the per-sample ratios (median(A)/median(B) let clock drift into the ratio); three runs, P95
// reported beside the median.
//
// Usage: --resident-program [cuda|hip|opencl|vulkan|metal|webgpu] (default cuda). WebGPU runs on Dawn and needs
// AIDOTNET_WEBGPU_NODE_DIR (a directory whose node_modules holds "webgpu").

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;

namespace AiDotNet.Tensors.Benchmarks;

internal static class ResidentProgramBench
{
    private const int Warmup = 20;
    private const int Samples = 51;
    private const int ChainsPerSample = 50;
    private const int Runs = 3;

    /// <summary>One backend's two ways of running a chain, and what to release afterwards.</summary>
    private sealed class Target : IDisposable
    {
        public Target(string description, IDirectGpuBackend backend, Func<ResidentStage[], IDisposable?> perShape,
            Action<ResidentStage[]> resident, Action<ResidentStage[]> separate, Action release)
        {
            Description = description; Backend = backend; PerShape = perShape; Resident = resident; Separate = separate;
            _release = release;
        }

        private readonly Action _release;
        public string Description { get; }
        public IDirectGpuBackend Backend { get; }

        /// <summary>Called once per shape before timing (CUDA and HIP allocate their barrier here).</summary>
        public Func<ResidentStage[], IDisposable?> PerShape { get; }
        public Action<ResidentStage[]> Resident { get; set; }
        public Action<ResidentStage[]> Separate { get; }
        public void Dispose() => _release();
    }

    internal static int Run(string[] args)
    {
        string name = (args.Length > 1 ? args[1] : "cuda").ToLowerInvariant();
        using var target = Create(name, out string? unavailable);
        if (target is null)
        {
            Console.Error.WriteLine($"--resident-program {name}: {unavailable}");
            return 2;
        }

        Console.WriteLine($"resident program vs the same stages run separately on {target.Description} (per chain, end to end)");
        Console.WriteLine("chain: affine+ReLU -> row softmax -> residual add, fp32");
        Console.WriteLine("shape        run  resident us   separate us   ratio (median of paired)   P95/med resident");
        var backend = target.Backend;
        foreach (var (rows, cols) in new[] { (64, 256), (128, 512), (256, 1024), (1024, 1024) })
        {
            var rng = new Random(890);
            float[] R(int len) => Enumerable.Range(0, len).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            using var x = backend.AllocateBuffer(R(rows * cols));
            using var scale = backend.AllocateBuffer(R(cols));
            using var bias = backend.AllocateBuffer(R(cols));
            using var residual = backend.AllocateBuffer(R(rows * cols));
            using var hidden = backend.AllocateBuffer(rows * cols);
            using var probs = backend.AllocateBuffer(rows * cols);
            using var output = backend.AllocateBuffer(rows * cols);
            var stages = new[]
            {
                new ResidentStage(ResidentOp.AffineRelu, rows, cols, x, scale, bias, hidden),
                new ResidentStage(ResidentOp.RowSoftmax, rows, cols, hidden, null, null, probs),
                new ResidentStage(ResidentOp.Add, rows, cols, probs, residual, null, output),
            };
            using var perShape = target.PerShape(stages);

            for (int i = 0; i < Warmup; i++) { target.Resident(stages); target.Separate(stages); }
            backend.Synchronize();

            for (int run = 1; run <= Runs; run++)
            {
                var resident = new List<double>();
                var separate = new List<double>();
                var ratios = new List<double>();
                for (int s = 0; s < Samples; s++)
                {
                    // Alternate which side goes first so neither always follows the other.
                    double r, p;
                    if ((s & 1) == 0) { r = Time(backend, target.Resident, stages); p = Time(backend, target.Separate, stages); }
                    else { p = Time(backend, target.Separate, stages); r = Time(backend, target.Resident, stages); }
                    resident.Add(r); separate.Add(p); ratios.Add(p / r);
                }
                Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                    "{0,-12} {1,3}  {2,11:F2}   {3,11:F2}   {4,24:F3}x   {5,16:F2}",
                    $"{rows}x{cols}", run, Median(resident), Median(separate), Median(ratios),
                    Percentile(resident, 0.95) / Median(resident)));
            }
        }
        return 0;
    }

    private static Target? Create(string name, out string? unavailable)
    {
        unavailable = null;
        switch (name)
        {
            case "cuda":
            {
                var b = new CudaBackend();
                if (!b.IsAvailable || !b.SupportsResidentProgram) { b.Dispose(); unavailable = "needs CUDA with cooperative launch."; return null; }
                return Barriered("CUDA (one cooperative launch)", b, b.AllocateResidentBarrier,
                    (st, bar) => b.RunResidentProgram(st, bar), st => b.RunStagesSeparately(st), b.Dispose);
            }
            case "hip":
            {
                var b = new HipBackend();
                if (!b.IsAvailable || !b.SupportsResidentProgram) { b.Dispose(); unavailable = "needs HIP with cooperative launch."; return null; }
                return Barriered("HIP (one cooperative launch)", b, b.AllocateResidentBarrier,
                    (st, bar) => b.RunResidentProgram(st, bar), st => b.RunStagesSeparately(st), b.Dispose);
            }
            case "opencl":
            {
                var b = new OpenClBackend();
                if (!b.IsAvailable || !b.SupportsResidentProgram) { b.Dispose(); unavailable = "needs OpenCL."; return null; }
                return new Target("OpenCL (one flush vs one per stage)", b, _ => null,
                    st => b.RunResidentProgram(st), st => b.RunStagesSeparately(st), b.Dispose);
            }
            case "vulkan":
            {
                var b = VulkanBackend.Instance;
                if (!b.Initialize() || !b.IsGlslCompilerAvailable || !b.SupportsResidentProgram)
                {
                    unavailable = "needs Vulkan with a GLSL compiler.";
                    return null;
                }
                // The backend is a process-wide singleton: not disposed here.
                return new Target("Vulkan (one submission vs one per stage)", b, _ => null,
                    st => b.RunResidentProgram(st), st => b.RunStagesSeparately(st), () => { });
            }
            case "metal":
            {
                if (!OperatingSystem.IsMacOS() && !OperatingSystem.IsIOS()) { unavailable = "Metal runs only on macOS/iOS."; return null; }
                var b = new MetalBackend();
                if (!b.IsAvailable || !b.SupportsResidentProgram) { b.Dispose(); unavailable = "needs a Metal device."; return null; }
                return new Target("Metal (one command buffer vs one per stage)", b, _ => null,
                    st => b.RunResidentProgram(st), st => b.RunStagesSeparately(st), b.Dispose);
            }
            case "webgpu":
            {
                var host = DawnWebGpuHost.TryCreate();
                if (host is null) { unavailable = "set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu."; return null; }
                WebGpuNativeBindings.Host = host;
                var b = new WebGpuBackend();
                void Release() { b.Dispose(); WebGpuNativeBindings.Host = null; host.Dispose(); }
                if (!b.InitializeAsync().GetAwaiter().GetResult() || !b.SupportsResidentProgram)
                {
                    Release();
                    unavailable = "Dawn has no WebGPU adapter, or the stage shader did not build.";
                    return null;
                }
                return new Target("WebGPU on Dawn (one submission vs one per stage)", b, _ => null,
                    st => b.RunResidentProgram(st), st => b.RunStagesSeparately(st), Release);
            }
            default:
                unavailable = "unknown backend; use cuda, hip, opencl, vulkan, metal or webgpu.";
                return null;
        }
    }

    // CUDA and HIP synchronize on a barrier buffer: one per shape, held while that shape is timed.
    private static Target Barriered(string description, IDirectGpuBackend backend, Func<IGpuBuffer> allocate,
        Action<ResidentStage[], IGpuBuffer> resident, Action<ResidentStage[]> separate, Action release)
    {
        IGpuBuffer? barrier = null;
        Target? target = null;
        target = new Target(description, backend,
            _ => { barrier = allocate(); return barrier; },
            st => resident(st, barrier ?? throw new InvalidOperationException("No barrier allocated.")),
            separate, release);
        return target;
    }

    private static double Time(IDirectGpuBackend backend, Action<ResidentStage[]> run, ResidentStage[] stages)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < ChainsPerSample; i++) run(stages);
        backend.Synchronize();
        return sw.Elapsed.TotalMilliseconds * 1000.0 / ChainsPerSample;
    }

    private static double Median(List<double> v) { var o = v.OrderBy(x => x).ToList(); return o[o.Count / 2]; }

    private static double Percentile(List<double> v, double q)
    {
        var o = v.OrderBy(x => x).ToList();
        return o[Math.Min(o.Count - 1, (int)Math.Ceiling(q * o.Count) - 1)];
    }
}
