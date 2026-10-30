// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The pointwise family on declared kernels (#890 phase 4a): a declared body that calls a library function derived
// from the hand-written kernel's own source. The IR, the emitted kernel and the derivation rules are pinned here on
// the CPU; the device half runs routed operators bit for bit against the hand-written kernels.

using System;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

[Collection("DirectGpuSerial")]
public class DeclaredElementwiseTests
{
    private static readonly CodegenLibraryFunction Twice = new("aidn_twice", "test_module", 1, v => 2 * v[0]);
    private static readonly CodegenLibraryFunction Blend = new("aidn_blend", "test_module", 2);

    [Fact]
    public void ACallAndAScalarEvaluateAndReportWhatTheyRead()
    {
        // twice($0) * %s0 + $1
        var body = CodegenExpr.Add(CodegenExpr.Multiply(CodegenExpr.Call(Twice, CodegenExpr.Operand(0)), CodegenExpr.Scalar(0)),
            CodegenExpr.Operand(1));
        Assert.Equal(new[] { 0, 1 }, body.Operands());
        Assert.Equal(1, body.ScalarCount());
        Assert.Single(body.Calls());
        Assert.Equal(2 * 3.0 * 0.5 + 4.0, body.Evaluate(i => i == 0 ? 3.0 : 4.0, s => 0.5));
        // A function with no fp64 evaluator cannot be interpreted -- refused, not guessed.
        Assert.Throws<NotSupportedException>(() =>
            CodegenExpr.Call(Blend, CodegenExpr.Operand(0), CodegenExpr.Operand(1)).Evaluate(_ => 1.0));
        Assert.Throws<ArgumentException>(() => CodegenExpr.Call(Blend, CodegenExpr.Operand(0)));
    }

    /// <summary>
    /// The runtime-length kernel: each call by the PTX ABI and declared extern at module scope, ahead of the entry;
    /// the 128-bit path behind a uniform branch on the host's flag; the length, the broadcast width and the scalars
    /// as parameters.
    /// </summary>
    [Fact]
    public void AnElementwiseKernelCallsItsLibraryFunctionsByTheAbi()
    {
        var spec = new CodegenElementwiseSpec("blend_scaled", 2,
            CodegenExpr.Multiply(CodegenExpr.Call(Blend, CodegenExpr.Operand(0), CodegenExpr.Operand(1)), CodegenExpr.Scalar(0)),
            ColumnBroadcastInput: 1);
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.EmitElementwise(spec, "declared_blend", 7, 5);

        Assert.Equal(new[] { Blend }, emitter.CalledFunctions);
        int extern_ = ptx.IndexOf(".extern .func (.param .b32 func_retval0) aidn_blend(", StringComparison.Ordinal);
        int entry = ptx.IndexOf(".visible .entry declared_blend(", StringComparison.Ordinal);
        Assert.True(extern_ >= 0 && extern_ < entry, "the call is declared ahead of the entry");
        Assert.Contains("call.uni (aidn_ret), aidn_blend, (aidn_arg0, aidn_arg1);", ptx, StringComparison.Ordinal);
        Assert.Contains(".param .u32 n", ptx, StringComparison.Ordinal);
        Assert.Contains(".param .u32 vectorized", ptx, StringComparison.Ordinal);
        Assert.Contains(".param .u32 columns", ptx, StringComparison.Ordinal);
        Assert.Contains(".param .f32 s0", ptx, StringComparison.Ordinal);
        Assert.Contains("ld.global.nc.v4.f32", ptx, StringComparison.Ordinal);
        Assert.Contains("SCALAR:", ptx, StringComparison.Ordinal);
        Assert.Contains("rem.u32", ptx, StringComparison.Ordinal);   // the broadcast column
        // Four calls on the 128-bit path, one on the scalar path.
        int calls = 0;
        for (int at = ptx.IndexOf("call.uni", StringComparison.Ordinal); at >= 0; at = ptx.IndexOf("call.uni", at + 1, StringComparison.Ordinal)) calls++;
        Assert.Equal(5, calls);
    }

    private static CudaMathModule Module(string source) => new("test_module_" + Guid.NewGuid().ToString("N"), source, FastMath: true);

    private const string Kernels = @"
extern ""C"" __global__ __launch_bounds__(256) void good(const float* __restrict__ A, float* __restrict__ B, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float x = A[idx];
    if (x != x) { B[idx] = x; return; }
    B[idx] = x > 0.0f ? x : alpha * x;
}
extern ""C"" __global__ void reads_fixed(const float* __restrict__ x, const float* __restrict__ mean, float* __restrict__ y, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    y[idx] = x[idx] - mean[0];
}
extern ""C"" __global__ void reads_size(const float* __restrict__ x, float* __restrict__ y, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    y[idx] = x[idx] / size;
}
extern ""C"" __global__ void two_outputs(const float* __restrict__ x, float* __restrict__ y, float* __restrict__ z, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    y[idx] = x[idx]; z[idx] = x[idx];
}
extern ""C"" __global__ void bare_exit(const float* __restrict__ x, float* __restrict__ y, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (x[idx] < 0.0f) return;
    y[idx] = x[idx];
}
";

    /// <summary>
    /// The derivation: the plain elementwise form becomes a function of the element (inputs, then scalars, each store
    /// a return); what is not a function of the element is refused, with the reason.
    /// </summary>
    [Fact]
    public void OnlyTheElementwiseFormIsDerived()
    {
        var module = Module(Kernels);
        var good = CudaMathLibrary.Resolve("good", module) ?? throw new InvalidOperationException(CudaMathLibrary.RefusalOf("good", module));
        Assert.Equal(1, good.Inputs);
        Assert.Equal(new[] { "alpha" }, good.Scalars);
        Assert.Equal("iosn", good.Parameters);
        Assert.Contains("float aidn_good(float aidn_v0, float alpha)", good.Wrapper, StringComparison.Ordinal);
        // The early store with its bare return gone (whitespace normalised: a store leaves "return  x").
        Assert.Contains("{ return x; }", System.Text.RegularExpressions.Regex.Replace(good.Wrapper, @"\s+", " "), StringComparison.Ordinal);
        Assert.DoesNotContain("idx", good.Wrapper, StringComparison.Ordinal);

        foreach (var (kernel, why) in new[]
                 {
                     ("reads_fixed", "mean"), ("reads_size", "length"), ("two_outputs", "two outputs"), ("bare_exit", "without storing"),
                 })
        {
            Assert.Null(CudaMathLibrary.Resolve(kernel, module));
            Assert.Contains(why, CudaMathLibrary.RefusalOf(kernel, module) ?? "", StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// A comment naming an input is not a use of it: where_select's comment about "the condition buffer" made the
    /// deriver refuse it (#1441), so every declared program that selects lost its only exact select.
    /// </summary>
    [Fact]
    public void CommentsDoNotCountAsUses()
    {
        const string source = @"
extern ""C"" __global__ void pick(const float* __restrict__ condition, const float* __restrict__ a,
    const float* __restrict__ b, float* __restrict__ output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // The condition buffer holds 1 or 0; output is written once. /* size, a and b too */
    /* a block comment naming condition and output */
    output[idx] = condition[idx] != 0.0f ? a[idx] : b[idx];
}
";
        var module = Module(source);
        var pick = CudaMathLibrary.Resolve("pick", module);
        Assert.True(pick is not null, CudaMathLibrary.RefusalOf("pick", module));
        Assert.Equal(3, pick!.Inputs);
    }

    /// <summary>
    /// On a device: routed operators -- an activation, a binary op, a scalar-parameter op, a precise trig function, a
    /// special function, the fused pair and the bias broadcast -- are bit-identical to the hand-written kernels, on
    /// random bit patterns (NaN, infinities, subnormals) at a 128-bit length and an odd one.
    /// </summary>
    [SkippableTheory]
    [InlineData("relu")]
    [InlineData("max_vectors")]
    [InlineData("abs_vector")]
    [InlineData("leaky_relu")]
    [InlineData("sin_vector")]
    [InlineData("parity210_lgamma_approx")]
    public void RoutedOperatorsAreBitIdenticalToTheHandWrittenKernels(string kernel)
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable && backend.DeclaredKernelsEnabled, "The declared route needs the sm_75+ conveyor.");
        var function = backend.ResolveElementwise(kernel) ?? throw new InvalidOperationException(backend.ElementwiseRefusal(kernel));
        var scalars = new float[function.Scalars.Length];
        for (int i = 0; i < scalars.Length; i++) scalars[i] = 0.3f * (i + 1);
        foreach (int n in new[] { 1 << 16, (1 << 16) + 3 })
        {
            var rng = new Random(kernel.Length * 977 + n);
            var inputs = new IGpuBuffer[function.Inputs];
            try
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    var data = new float[n];
                    var bytes = new byte[4];
                    for (int e = 0; e < n; e++) { rng.NextBytes(bytes); data[e] = BitConverter.ToSingle(bytes, 0); }
                    inputs[i] = backend.AllocateBuffer(data);
                }
                using var declared = backend.AllocateBuffer(n);
                using var hand = backend.AllocateBuffer(n);
                Assert.True(backend.TryDeclaredElementwise(kernel, inputs, declared, scalars, n), backend.LastElementwiseDecline);
                Assert.True(backend.LaunchHandWrittenElementwise(kernel, inputs, hand, scalars, n));
                backend.Synchronize();
                float[] a = backend.DownloadBuffer(declared), b = backend.DownloadBuffer(hand);
                for (int e = 0; e < n; e++)
                    Assert.True((float.IsNaN(a[e]) && float.IsNaN(b[e])) ||
                        TestHelpers.MathCompat.SingleToInt32Bits(a[e]) == TestHelpers.MathCompat.SingleToInt32Bits(b[e]),
                        $"{kernel} n={n} [{e}]: declared {a[e]} hand-written {b[e]}");
            }
            finally
            {
                foreach (var buffer in inputs) buffer?.Dispose();
            }
        }
    }
}
