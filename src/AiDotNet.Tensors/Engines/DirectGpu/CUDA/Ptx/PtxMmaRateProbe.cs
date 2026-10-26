// Copyright (c) AiDotNet. All rights reserved.

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

    /// <summary>One measurement: TFLOP/s of each accumulate form and their ratio (FP16 over FP32).</summary>
    internal readonly record struct Result(double Fp32AccumulateTflops, double Fp16AccumulateTflops)
    {
        internal double Ratio => Fp16AccumulateTflops / Fp32AccumulateTflops;

        /// <summary>FP32 accumulation issues at (close to) the FP16 rate: FP16 accumulation cannot double anything.</summary>
        internal bool Fp32AccumulateIsFullRate => Ratio < 1.3;

        /// <summary>Stable class for cache keys: "fp32acc-full" or "fp32acc-1/N" with N the rounded ratio.</summary>
        internal string RateClass => Fp32AccumulateIsFullRate
            ? "fp32acc-full"
            : "fp32acc-1/" + Math.Round(Ratio).ToString(CultureInfo.InvariantCulture);
    }

    internal static string EmitPtx(int ccMajor, int ccMinor)
    {
        var ptx = new StringBuilder();
        ptx.AppendLine(".version 7.1");
        ptx.Append(".target sm_").Append(ccMajor.ToString(CultureInfo.InvariantCulture))
            .AppendLine(ccMinor.ToString(CultureInfo.InvariantCulture));
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

    /// <summary>Runs both loops (best of three timed launches each, after a warm-up) on the runtime's stream.</summary>
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

        double Tflops(string entry)
        {
            IntPtr function = module.GetFunction(entry);
            ulong pointer = (ulong)output.Pointer;
            uint count = (uint)iterations;
            void** arguments = stackalloc void*[2];
            arguments[0] = &pointer;
            arguments[1] = &count;
            module.Launch(function, blocks, 1, 1, ThreadsPerBlock, 1, 1, 0, arguments);
            runtime.Synchronize();
            double best = double.MaxValue;
            for (int round = 0; round < 3; round++)
            {
                var watch = Stopwatch.StartNew();
                module.Launch(function, blocks, 1, 1, ThreadsPerBlock, 1, 1, 0, arguments);
                runtime.Synchronize();
                best = Math.Min(best, watch.Elapsed.TotalSeconds);
            }
            return flops / best / 1e12;
        }

        double fp32 = Tflops(Fp32AccumulateEntry);
        double fp16 = Tflops(Fp16AccumulateEntry);
        return new Result(fp32, fp16);
    }
}