using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// Validates the FP16-native row Softmax + LayerNorm kernels (CudaBackend.Fp16Softmax / Fp16LayerNorm) on real
/// hardware. These keep the transformer's per-layer norm + attention softmax HALF-resident (read FP16 directly,
/// reduce in FP32, write FP16) — the missing kernels that forced every layer to re-inflate the Half activations
/// to FP32 and negated the FP16 storage win. Verified in isolation against CPU FP32 references FIRST.
/// </summary>
public class Fp16NativeOpsTests
{
    private static (CudaBackend b, bool ok) Setup()
    {
        var eng = AiDotNetEngine.Current;
        if (eng is not DirectGpuTensorEngine) return (null!, false);
        var warm = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        _ = eng.TensorMatMul(warm, warm);
        var b = ((DirectGpuTensorEngine)eng).TestBackend as CudaBackend;
        return (b!, b is not null && b.IsAvailable);
    }

    [SkippableFact]
    public void Fp16Softmax_matches_fp32_reference()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var (b, ok) = Setup();
        Skip.IfNot(ok, "CudaBackend not available");
        Skip.IfNot(b.SupportsFp16NativeOps, "fp16 native kernels not compiled");

        const int rows = 40, cols = 70; // cols < and > a warp, non-power-of-2
        var rng = new Random(5);
        var x = new float[rows * cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 6 - 3);

        // CPU reference reads the SAME FP16-quantized input the kernel does (so the diff is the kernel, not rounding).
        var xq = x.Select(v => (float)(Half)v).ToArray();
        var cpu = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            double m = double.NegativeInfinity;
            for (int c = 0; c < cols; c++) m = Math.Max(m, xq[r * cols + c]);
            double s = 0; for (int c = 0; c < cols; c++) s += Math.Exp(xq[r * cols + c] - m);
            for (int c = 0; c < cols; c++) cpu[r * cols + c] = (float)(Math.Exp(xq[r * cols + c] - m) / s);
        }

        using var dx = b.AllocateBuffer(x);
        using var hx = b.AllocateByteBuffer(rows * cols * 2);
        using var hout = b.AllocateByteBuffer(rows * cols * 2);
        b.ConvertToFp16Native(dx, hx, rows * cols);
        try { b.Fp16Softmax(hx, hout, rows, cols); }
        catch (NotSupportedException ex) { Skip.If(true, $"not supported: {ex.Message}"); return; }
        catch (InvalidOperationException ex) when (ex.Message.IndexOf("kernel not found", StringComparison.OrdinalIgnoreCase) >= 0)
        { Skip.If(true, $"not supported: {ex.Message}"); return; }
        using var fout = b.AllocateBuffer(rows * cols);
        b.ConvertToFp32Native(hout, fout, rows * cols);
        var got = b.DownloadBuffer(fout);

        double num = 0, den = 0;
        for (int i = 0; i < cpu.Length; i++) { double e = got[i] - cpu[i]; num += e * e; den += (double)cpu[i] * cpu[i]; }
        double rel = Math.Sqrt(num / den);
        Assert.True(rel < 2e-2, $"Fp16Softmax exceeded FP16 tolerance: relFro={rel:E3}");
    }

    [SkippableFact]
    public void Fp16LayerNorm_matches_fp32_reference()
    {
        Skip.IfNot(CudaNativeBindings.IsAvailable, "CUDA driver not available");
        var (b, ok) = Setup();
        Skip.IfNot(ok, "CudaBackend not available");
        Skip.IfNot(b.SupportsFp16NativeOps, "fp16 native kernels not compiled");

        const int rows = 32, cols = 96;
        const float eps = 1e-5f;
        var rng = new Random(9);
        var x = new float[rows * cols];
        var gamma = new float[cols];
        var beta = new float[cols];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        for (int c = 0; c < cols; c++) { gamma[c] = (float)(rng.NextDouble() + 0.5); beta[c] = (float)(rng.NextDouble() - 0.5); }

        // CPU ref: population variance, eps inside rsqrt — the kernel's (and engine's) convention. FP16-quantized in.
        var xq = x.Select(v => (float)(Half)v).ToArray();
        var gq = gamma.Select(v => (float)(Half)v).ToArray();
        var bq = beta.Select(v => (float)(Half)v).ToArray();
        var cpu = new float[rows * cols];
        for (int r = 0; r < rows; r++)
        {
            double mean = 0; for (int c = 0; c < cols; c++) mean += xq[r * cols + c]; mean /= cols;
            double var = 0; for (int c = 0; c < cols; c++) { double d = xq[r * cols + c] - mean; var += d * d; } var /= cols;
            double invstd = 1.0 / Math.Sqrt(var + eps);
            for (int c = 0; c < cols; c++) cpu[r * cols + c] = (float)((xq[r * cols + c] - mean) * invstd * gq[c] + bq[c]);
        }

        using var dx = b.AllocateBuffer(x);
        using var dg = b.AllocateBuffer(gamma);
        using var db = b.AllocateBuffer(beta);
        using var hx = b.AllocateByteBuffer(rows * cols * 2);
        using var hg = b.AllocateByteBuffer(cols * 2);
        using var hb = b.AllocateByteBuffer(cols * 2);
        using var hout = b.AllocateByteBuffer(rows * cols * 2);
        using var dmean = b.AllocateBuffer(rows);
        using var dvar = b.AllocateBuffer(rows);
        b.ConvertToFp16Native(dx, hx, rows * cols);
        b.ConvertToFp16Native(dg, hg, cols);
        b.ConvertToFp16Native(db, hb, cols);
        try { b.Fp16LayerNorm(hx, hg, hb, hout, dmean, dvar, rows, cols, eps); }
        catch (NotSupportedException ex) { Skip.If(true, $"not supported: {ex.Message}"); return; }
        catch (InvalidOperationException ex) when (ex.Message.IndexOf("kernel not found", StringComparison.OrdinalIgnoreCase) >= 0)
        { Skip.If(true, $"not supported: {ex.Message}"); return; }
        using var fout = b.AllocateBuffer(rows * cols);
        b.ConvertToFp32Native(hout, fout, rows * cols);
        var got = b.DownloadBuffer(fout);

        double num = 0, den = 0;
        for (int i = 0; i < cpu.Length; i++) { double e = got[i] - cpu[i]; num += e * e; den += (double)cpu[i] * cpu[i]; }
        double rel = Math.Sqrt(num / den);
        Assert.True(rel < 3e-2, $"Fp16LayerNorm exceeded FP16 tolerance: relFro={rel:E3}");
    }
}
