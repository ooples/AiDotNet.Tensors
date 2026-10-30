// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

/// <summary>
/// MSL for <see cref="IGpuBatchExecution"/> on Metal (#1142), translated mechanically from
/// <see cref="BatchExecutionCudaKernels"/> so Metal runs the same kernels HIP and CUDA run. Every kernel parameter
/// binds to <c>[[buffer(i)]]</c> in declaration order: pointers as device memory, scalars as <c>constant</c>
/// references (set with <c>SetBytes</c>), and a 64-bit seed as <c>ulong</c>.
/// </summary>
internal static class MetalBatchExecutionKernels
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(2);

    public static string Source { get; } = Translate(BatchExecutionCudaKernels.GetSource());

    /// <summary>The FP32 PyTorch-parity kernels (#1125), from the same dialect the WebGPU translator reads.</summary>
    public static string TorchParitySource { get; } = Translate(TorchParityF32Kernels.GetSource());

    private static string Translate(string cuda)
    {
        string body = cuda;
        // The grid-index helper reads CUDA builtins; MSL hands thread ids to the kernel as parameters instead.
        body = Regex.Replace(body, @"__device__ __forceinline__ int bx_gid\(\)[^\n]*\n", "", RegexOptions.None, RegexTimeout);
        body = Regex.Replace(body, @"extern ""C"" __global__ __launch_bounds__\(256\) void (\w+)\(([^)]*)\)",
            m => Signature(m.Groups[1].Value, m.Groups[2].Value), RegexOptions.None, RegexTimeout);
        body = body.Replace("__device__ __forceinline__", "inline")
            .Replace("unsigned long long", "ulong")
            .Replace("(long long)", "(long)")
            .Replace("__shared__ float", "threadgroup float")
            .Replace("__syncthreads()", "threadgroup_barrier(mem_flags::mem_threadgroup)")
            .Replace("bx_gid()", "(int)bx_tid")
            .Replace("threadIdx.x", "(int)bx_lid")
            .Replace("blockDim.x", "(int)bx_bdim")
            .Replace("__float_as_uint(", "as_type<uint>(")
            .Replace("__uint_as_float(", "as_type<float>(")
            .Replace("__int_as_float(", "as_type<float>(")
            .Replace("__float_as_int(", "as_type<int>(")
            .Replace("__device__ const float ", "constant float ")
            .Replace("make_float2(", "float2(")
            .Replace("make_uint2(", "uint2(");
        // atomicMin on a uint buffer, and atomicAddFloat as a compare-and-swap loop over the float's bits.
        body = Regex.Replace(body, @"atomicMin\(&(\w+)\[([^\]]+)\], ", "atomic_fetch_min_explicit((device atomic_uint*)&$1[$2], ", RegexOptions.None, RegexTimeout);
        body = Regex.Replace(body, @"(atomic_fetch_min_explicit\([^;]*)\);", "$1, memory_order_relaxed);", RegexOptions.None, RegexTimeout);
        body = Regex.Replace(body, @"atomicAddFloat\(&(\w+)\[([^\]]+)\], ([^;]+)\);", m =>
            $"{{ device atomic_uint* tp_at = (device atomic_uint*)&{m.Groups[1].Value}[{m.Groups[2].Value}]; uint tp_old = atomic_load_explicit(tp_at, memory_order_relaxed); "
            + $"while (!atomic_compare_exchange_weak_explicit(tp_at, &tp_old, as_type<uint>(as_type<float>(tp_old) + ({m.Groups[3].Value})), memory_order_relaxed, memory_order_relaxed)) {{ }} }}",
            RegexOptions.None, RegexTimeout);
        body = Regex.Replace(body, @"(?<![\w*&])const float\* (\w+) =", "device const float* $1 =", RegexOptions.None, RegexTimeout);
        body = Regex.Replace(body, @"(?<!const )(?<![\w*&])float\* (\w+) =", "device float* $1 =", RegexOptions.None, RegexTimeout);
        body = Regex.Replace(body, @"\b(exp|log|sqrt|fmax|fmin|fabs|tanh|rsqrt|sin|cos|tan|floor|ceil|trunc|rint|pow|atan2|atan|ldexp|fma)f\(", "$1(", RegexOptions.None, RegexTimeout);
        return "#include <metal_stdlib>\nusing namespace metal;\n" + body;
    }

    private static string Signature(string name, string parameters)
    {
        var sb = new StringBuilder("kernel void ").Append(name).Append('(');
        string[] parts = parameters.Split(',');
        for (int i = 0; i < parts.Length; i++)
        {
            string p = Regex.Replace(parts[i].Trim(), @"\s+", " ", RegexOptions.None, RegexTimeout);
            string declaration =
                p.StartsWith("const float* ", StringComparison.Ordinal) ? "device " + p
                : p.StartsWith("const unsigned int* ", StringComparison.Ordinal) ? "device const uint* " + p.Substring("const unsigned int* ".Length)
                : p.StartsWith("unsigned int* ", StringComparison.Ordinal) ? "device uint* " + p.Substring("unsigned int* ".Length)
                : p.StartsWith("float* ", StringComparison.Ordinal) ? "device " + p
                : p.StartsWith("const int* ", StringComparison.Ordinal) || p.StartsWith("int* ", StringComparison.Ordinal) ? "device " + p
                : p.StartsWith("unsigned long long ", StringComparison.Ordinal) ? "constant ulong& " + p.Substring("unsigned long long ".Length)
                : p.StartsWith("int ", StringComparison.Ordinal) ? "constant int& " + p.Substring(4)
                : p.StartsWith("unsigned int ", StringComparison.Ordinal) ? "constant uint& " + p.Substring("unsigned int ".Length)
                : p.StartsWith("float ", StringComparison.Ordinal) ? "constant float& " + p.Substring(6)
                : throw new InvalidOperationException($"Untranslatable kernel parameter '{p}' in {name}.");
            sb.Append(declaration).Append(" [[buffer(").Append(i).Append(")]], ");
        }
        sb.Append("uint bx_tid [[thread_position_in_grid]], uint bx_lid [[thread_position_in_threadgroup]], ")
          .Append("uint bx_bdim [[threads_per_threadgroup]])");
        return sb.ToString();
    }
}
