// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Measures this device's tensor-core issue rate for FP16 MMA with FP32 accumulation against FP16 accumulation
/// (#1549). GeForce parts issue the FP32-accumulate form at half (Ampere, Ada) or less (an eighth on the RTX 5090) of
/// the FP16-accumulate rate; datacenter and workstation parts, and GeForce cards whose speed-select registers were
/// rewritten by an unlock module (Zagitta/gpu-unlock, gpu-unlock-ada, KohakUnlock), issue both at the same rate. The
/// probe keeps eight independent <c>mma.sync.m16n8k16</c> chains per warp in registers (no memory traffic) on every SM,
/// so the time ratio of the two loops is the issue-rate ratio.
/// </summary>
internal static class PtxMmaRateProbe
{
    internal const string Fp32AccumulateEntry = "mma_rate_f32acc";
    internal const string Fp16AccumulateEntry = "mma_rate_f16acc";
    private const int Chains = 8;
    private const int ThreadsPerBlock = 128;
    private const int BlocksPerMultiprocessor = 8;

    /// <summary>The rate class of a device whose FP32-accumulate tensor-core math issues at the FP16 rate.</summary>
    internal const string FullRateClass = "fp32acc-full";

    /// <summary>Where a <see cref="Result"/> came from.</summary>
    internal enum RateSource
    {
        /// <summary>Timed by <see cref="Measure"/>; the rates are TFLOP/s.</summary>
        Measured,
        /// <summary>The driver's issue-rate modifiers (<see cref="NvRmIssueRateQuery"/>); the rates are fractions of full.</summary>
        DriverQuery,
        /// <summary>Set by a test.</summary>
        Forced,
    }

    /// <summary>
    /// One rate reading: each accumulate form's rate and their ratio (FP16 over FP32). Measured readings are TFLOP/s;
    /// a driver reading is in units of the full issue rate (1 = full, 0.5 = half).
    /// </summary>
    internal readonly record struct Result(double Fp32AccumulateTflops, double Fp16AccumulateTflops,
        RateSource Source = RateSource.Measured)
    {
        internal double Ratio => Fp16AccumulateTflops / Fp32AccumulateTflops;

        /// <summary>The two rates for a message, in the units of <see cref="Source"/>.</summary>
        internal string Describe() => Source == RateSource.DriverQuery
            ? $"driver speed selects: FP32 accumulation at {Fp32AccumulateTflops:0.###} and FP16 accumulation at " +
              $"{Fp16AccumulateTflops:0.###} of the full issue rate"
            : $"{Fp32AccumulateTflops:F0} against {Fp16AccumulateTflops:F0} TFLOP/s";

        /// <summary>FP32 accumulation issues at (close to) the FP16 rate: FP16 accumulation cannot double anything.</summary>
        internal bool Fp32AccumulateIsFullRate => Ratio < 1.3;

        /// <summary>Stable class for cache keys: "fp32acc-full" or "fp32acc-1/N" with N the rounded ratio.</summary>
        internal string RateClass => Fp32AccumulateIsFullRate
            ? FullRateClass
            : "fp32acc-1/" + Math.Round(Ratio).ToString(CultureInfo.InvariantCulture);
    }

    internal static string EmitPtx(int ccMajor, int ccMinor)
    {
        var ptx = new StringBuilder();
        ptx.AppendLine(".version 7.1");
        ptx.Append(".target ").AppendLine(PtxIsa71Target.For(ccMajor, ccMinor));
        ptx.AppendLine(".address_size 64");
        foreach (bool fp16 in new[] { false, true })
        {
            ptx.AppendLine();
            ptx.Append(".visible .entry ").Append(fp16 ? Fp16AccumulateEntry : Fp32AccumulateEntry)
                .AppendLine("(.param .u64 out, .param .u32 iterations)");
            ptx.AppendLine("{");
            ptx.AppendLine("    .reg .pred %p;");
            ptx.AppendLine("    .reg .b32 %a<4>, %b<2>, %i, %thread, %block, %width, %gid;");
            ptx.AppendLine("    .reg .b64 %rd<3>;");
            ptx.AppendLine(fp16 ? $"    .reg .b32 %c<{Chains * 2}>;" : $"    .reg .f32 %c<{Chains * 4}>;");
            ptx.AppendLine("    .reg .f32 %sum, %x, %y;");
            ptx.AppendLine("    .reg .b16 %h0, %h1;");
            // Operands of 2^-8 (FP16 0x1C00): each MMA adds 16 * 2^-16 to an accumulator, far below any overflow.
            for (int i = 0; i < 4; i++) ptx.AppendLine($"    mov.b32 %a{i}, 0x1C001C00;");
            for (int i = 0; i < 2; i++) ptx.AppendLine($"    mov.b32 %b{i}, 0x1C001C00;");
            int accumulators = fp16 ? Chains * 2 : Chains * 4;
            for (int i = 0; i < accumulators; i++)
                ptx.AppendLine(fp16 ? $"    mov.b32 %c{i}, 0;" : $"    mov.f32 %c{i}, 0f00000000;");
            ptx.AppendLine("    ld.param.u32 %i, [iterations];");
            ptx.AppendLine("LOOP:");
            for (int chain = 0; chain < Chains; chain++)
            {
                if (fp16)
                {
                    string c = $"{{%c{2 * chain},%c{2 * chain + 1}}}";
                    ptx.AppendLine($"    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {c}, {{%a0,%a1,%a2,%a3}}, {{%b0,%b1}}, {c};");
                }
                else
                {
                    string c = $"{{%c{4 * chain},%c{4 * chain + 1},%c{4 * chain + 2},%c{4 * chain + 3}}}";
                    ptx.AppendLine($"    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {c}, {{%a0,%a1,%a2,%a3}}, {{%b0,%b1}}, {c};");
                }
            }
            ptx.AppendLine("    sub.u32 %i, %i, 1;");
            ptx.AppendLine("    setp.ne.u32 %p, %i, 0;");
            ptx.AppendLine("    @%p bra LOOP;");
            // Fold every accumulator into one value stored per thread, so no MMA is dead code.
            ptx.AppendLine("    mov.f32 %sum, 0f00000000;");
            for (int i = 0; i < accumulators; i++)
            {
                if (fp16)
                {
                    ptx.AppendLine($"    mov.b32 {{%h0,%h1}}, %c{i};");
                    ptx.AppendLine("    cvt.f32.f16 %x, %h0;");
                    ptx.AppendLine("    cvt.f32.f16 %y, %h1;");
                    ptx.AppendLine("    add.f32 %sum, %sum, %x;");
                    ptx.AppendLine("    add.f32 %sum, %sum, %y;");
                }
                else
                {
                    ptx.AppendLine($"    add.f32 %sum, %sum, %c{i};");
                }
            }
            ptx.AppendLine("    mov.u32 %thread, %tid.x;");
            ptx.AppendLine("    mov.u32 %block, %ctaid.x;");
            ptx.AppendLine("    mov.u32 %width, %ntid.x;");
            ptx.AppendLine("    mad.lo.u32 %gid, %block, %width, %thread;");
            ptx.AppendLine("    ld.param.u64 %rd0, [out];");
            ptx.AppendLine("    mul.wide.u32 %rd1, %gid, 4;");
            ptx.AppendLine("    add.u64 %rd2, %rd0, %rd1;");
            ptx.AppendLine("    st.global.f32 [%rd2], %sum;");
            ptx.AppendLine("    ret;");
            ptx.AppendLine("}");
        }
        return ptx.ToString();
    }

    /// <summary>Runs both loops on the runtime's stream: a warm-up of each, then <see cref="MeasureRounds"/> rounds that
    /// time each form once in alternating order, keeping each form's best.</summary>
    internal static unsafe Result Measure(DirectPtxRuntime runtime, int iterations = 4096)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (runtime.ComputeCapabilityMajor < 8)
            throw new NotSupportedException("The m16n8k16 FP16 MMA probe needs compute capability 8.0+.");
        int multiprocessors = PtxGemmEvolution.MultiprocessorCount(runtime);
        uint blocks = (uint)(multiprocessors * BlocksPerMultiprocessor);
        using var module = runtime.LoadModule(EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor));
        using var output = runtime.AllocateBytes((nuint)(blocks * ThreadsPerBlock * sizeof(float)));
        double warps = blocks * (ThreadsPerBlock / 32.0);
        double flops = warps * iterations * Chains * (16.0 * 8 * 16 * 2);

        IntPtr fp32Function = module.GetFunction(Fp32AccumulateEntry);
        IntPtr fp16Function = module.GetFunction(Fp16AccumulateEntry);
        ulong pointer = (ulong)output.Pointer;
        uint count = (uint)iterations;
        void** arguments = stackalloc void*[2];
        arguments[0] = &pointer;
        arguments[1] = &count;

        double Seconds(IntPtr function)
        {
            var watch = Stopwatch.StartNew();
            module.Launch(function, blocks, 1, 1, ThreadsPerBlock, 1, 1, 0, arguments);
            runtime.Synchronize();
            return watch.Elapsed.TotalSeconds;
        }

        // A warm-up of each, then single timed launches of the two forms alternated round by round (AB, BA, AB, ...),
        // keeping each form's best. Alternating launch by launch, rather than timing one form's launches back to back,
        // means neither clock ramp nor a burst of other work on the device (another stream or process sharing the SMs)
        // can cover every sample of one form and none of the other: the old ABBA order timed both FP16 runs
        // contiguously, and work overlapping that window read FP16 accumulation at 26 against 140 TFLOP/s on an RTX
        // 3090, classing it full-rate.
        Seconds(fp32Function);
        Seconds(fp16Function);
        double bestFp32 = double.MaxValue, bestFp16 = double.MaxValue;
        for (int round = 0; round < MeasureRounds; round++)
        {
            bool fp32First = (round & 1) == 0;
            double first = Seconds(fp32First ? fp32Function : fp16Function);
            double second = Seconds(fp32First ? fp16Function : fp32Function);
            bestFp32 = Math.Min(bestFp32, fp32First ? first : second);
            bestFp16 = Math.Min(bestFp16, fp32First ? second : first);
        }
        return new Result(flops / bestFp32 / 1e12, flops / bestFp16 / 1e12);
    }

    // Timed launches per form; each round times both forms once, in alternating order.
    private const int MeasureRounds = 8;
}