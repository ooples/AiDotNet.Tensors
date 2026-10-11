using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Device paths for the PyTorch-parity random ops (#1112 tier 7): a counter-based Philox generator per element, so a
/// seed reproduces the same tensor on the device. The draws differ from the CPU's System.Random stream (as torch's CPU
/// and CUDA generators differ); the distributions and their algorithms are the CPU's. Per-element parameter errors are
/// detected on the device and raised as the CPU's exceptions after one 8-byte readback of the validation flag. Under a
/// tape or graph recording these defer to the CPU, which records the backward.
/// </summary>
public partial class DirectGpuTensorEngine
{
    private const ulong NoParityError = ulong.MaxValue;

    // A 64-bit device seed: SplitMix64 of the user's seed (so nearby seeds give unrelated streams), or a fresh secure
    // random value when none is given (the CPU then uses a secure generator too).
    private static ulong DeviceSeed(int? seed)
    {
        ulong z;
        if (seed.HasValue) z = (ulong)(uint)seed.Value + 0x9E3779B97F4A7C15UL;
        else
        {
            var bytes = new byte[8];
            RandomHelper.CreateSecureRandom().NextBytes(bytes);
            z = BitConverter.ToUInt64(bytes, 0);
        }
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private bool TryRandomBackend<T>(string opName, out IDirectGpuBackend backend, out ITorchParityElementwiseBackend kernels)
    {
        kernels = null!;
        if (IsTapeActive<T>(opName) || !TryGetBackend(out backend) || DirectGpuEngine.ShouldFallbackForPrecision<T>())
        {
            backend = null!;
            return false;
        }
        if (backend is not ITorchParityElementwiseBackend k) return false;
        kernels = k;
        return true;
    }

    private static ulong ReadParityFlag(IDirectGpuBackend backend, IGpuBuffer flag)
    {
        var words = backend.DownloadBuffer(flag);
        var bytes = new byte[8];
        Buffer.BlockCopy(words, 0, bytes, 0, 8);
        return BitConverter.ToUInt64(bytes, 0);
    }

    // Shape-only distributions (see ITorchParityElementwiseBackend.TorchParitySample for the kinds).
    private Tensor<T>? TrySampleOnDevice<T>(string opName, int[] shape, int kind, double a, double b, int? seed)
    {
        int n = 1;
        foreach (int d in shape) { if (d < 0) return null; n = checked(n * d); }
        if (n == 0) return null;
        if (!TryRandomBackend<T>(opName, out var backend, out var kernels)) return null;
        try
        {
            ulong deviceSeed = DeviceSeed(seed);
            return DispatchDeferredGpuOp<T>(backend, n, (int[])shape.Clone(), output => kernels.TorchParitySample(output, n, kind, a, b, deviceSeed));
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
    }

    // randperm: one random key per position, sorted on the device; the permutation is the key order's indices.
    private Tensor<T>? TryRandpermOnDevice<T>(int n, int? seed)
    {
        // Positions travel as float indices, exact only up to 2^24.
        if (n <= 0 || n > (1 << 24) || typeof(T) != typeof(float) || !TryRandomBackend<T>("TensorRandperm", out var backend, out var kernels)) return null;
        try
        {
            ulong deviceSeed = DeviceSeed(seed);
            using var keys = AllocateOutputBuffer(backend, n);
            kernels.TorchParityRandomKeys(keys.Buffer, n, deviceSeed);
            var (sorted, indices) = BitonicSortRowsToBuffers(backend, keys.Buffer, 1, n, descending: false);
            using (sorted)
            using (indices)
                return DispatchDeferredGpuOp<T>(backend, n, new[] { n }, output => backend.Copy(indices.Buffer, output, n));
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback("TensorRandperm", ex);
            return null;
        }
    }

    private Tensor<T>? TryRreluOnDevice<T>(Tensor<T> tensor, double lower, double upper, bool training, int? seed)
    {
        if (tensor is null || tensor.Length == 0 || lower > upper || !TryRandomBackend<T>("TensorRrelu", out var backend, out var kernels))
            return null;
        try
        {
            int n = tensor.Length;
            ulong deviceSeed = training ? DeviceSeed(seed) : 0;
            var source = tensor.IsContiguous ? tensor : tensor.Contiguous();
            using var x = GetOrAllocateBuffer(backend, source);
            return DispatchDeferredGpuOp<T>(backend, n, (int[])tensor._shape.Clone(),
                output => kernels.TorchParityRrelu(x.Buffer, output, n, lower, upper, training, deviceSeed));
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback("TensorRrelu", ex);
            return null;
        }
    }

    // Bernoulli / Poisson (one parameter tensor) and Binomial (count, probability): validated on the device.
    private Tensor<T>? TryValidatedSampleOnDevice<T>(Tensor<T> first, Tensor<T>? second, int? seed, string opName)
    {
        if (first is null || first.Length == 0) return null;
        if (opName == "TensorBinomial" && (second is null || !first._shape.AsSpan().SequenceEqual(second._shape))) return null;
        if (!TryRandomBackend<T>(opName, out var backend, out var kernels)) return null;
        Tensor<T>? result = null;
        ulong flagValue;
        try
        {
            int n = first.Length;
            ulong deviceSeed = DeviceSeed(seed);
            using var a = GetOrAllocateBuffer(backend, first.IsContiguous ? first : first.Contiguous());
            using var b = second is null ? default : GetOrAllocateBuffer(backend, second.IsContiguous ? second : second.Contiguous());
            using var flag = AllocateOutputBuffer(backend, 2);
            kernels.TorchParityFlagReset(flag.Buffer);
            result = DispatchDeferredGpuOp<T>(backend, n, (int[])first._shape.Clone(), output =>
            {
                if (opName == "TensorBernoulli") kernels.TorchParityBernoulli(a.Buffer, output, n, deviceSeed, flag.Buffer);
                else if (opName == "TensorPoisson") kernels.TorchParityPoisson(a.Buffer, output, n, deviceSeed, flag.Buffer);
                else kernels.TorchParityBinomial(a.Buffer, b.Buffer, output, n, deviceSeed, flag.Buffer);
            });
            flagValue = ReadParityFlag(backend, flag.Buffer);
        }
        catch (Exception ex)
        {
            result?.Dispose();
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
        if (flagValue == NoParityError) return result;
        result?.Dispose();
        long index = (long)((flagValue >> 3) & ((1UL << 58) - 1));
        int code = (int)(flagValue & 7);
        throw opName switch
        {
            "TensorBernoulli" => new ArgumentOutOfRangeException("probabilities", "probabilities must lie in [0, 1]."),
            "TensorPoisson" => new ArgumentOutOfRangeException("rate", "Poisson rates must be non-negative."),
            _ when code == 1 => new ArgumentOutOfRangeException("count",
                $"count must be a whole number of trials within the long range, got {MathHelper.GetNumericOperations<T>().ToDouble(first[(int)index])}."),
            _ => new ArgumentOutOfRangeException("p", "need count ≥ 0 and 0 ≤ p ≤ 1."),
        };
    }

    private Tensor<T>? TryMultinomialOnDevice<T>(Tensor<T> probabilities, int numSamples, bool replacement, int? seed)
    {
        if (probabilities is null || probabilities.Rank < 1 || probabilities.Rank > 2 || numSamples <= 0) return null;
        int categories = probabilities._shape[probabilities.Rank - 1];
        if (categories == 0 || (!replacement && numSamples > categories)) return null;
        if (categories > (1 << 24)) return null;   // picks travel as float indices, exact only up to 2^24
        int rows = probabilities.Length / categories;
        if (!TryRandomBackend<T>("TensorMultinomial", out var backend, out var kernels)) return null;
        int total = checked(rows * numSamples);
        var shape = probabilities.Rank == 1 ? new[] { numSamples } : new[] { rows, numSamples };
        Tensor<T>? result = null;
        ulong flagValue;
        try
        {
            ulong deviceSeed = DeviceSeed(seed);
            using var w = GetOrAllocateBuffer(backend, probabilities.IsContiguous ? probabilities : probabilities.Contiguous());
            using var scratch = AllocateOutputBuffer(backend, checked(2 * probabilities.Length));   // double prefix sums
            using var flag = AllocateOutputBuffer(backend, 2);
            kernels.TorchParityFlagReset(flag.Buffer);
            result = DispatchDeferredGpuOp<T>(backend, total, shape,
                output => kernels.TorchParityMultinomial(w.Buffer, scratch.Buffer, output, rows, categories, numSamples, replacement, deviceSeed, flag.Buffer));
            flagValue = ReadParityFlag(backend, flag.Buffer);
        }
        catch (Exception ex)
        {
            result?.Dispose();
            GpuLaunchProbe.OnFallback("TensorMultinomial", ex);
            return null;
        }
        if (flagValue == NoParityError) return result;
        result?.Dispose();
        long index = (long)((flagValue >> 3) & ((1UL << 58) - 1));
        if ((flagValue & 7) == 1) throw new ArgumentException("weights must be finite and non-negative.", nameof(probabilities));
        throw new ArgumentException($"row {index / categories} has no positive weight left to sample.", nameof(probabilities));
    }

    // Dropouts: a device keep mask drawn per element (spatialDims < 0) or per channel, folded into out = x * scale +
    // offset with the CPU's constants; the multiply / add run as device ops.
    private Tensor<T>? TryDropoutOnDevice<T>(Tensor<T> tensor, double p, bool training, int? seed, bool alpha, int spatialDims)
    {
        if (tensor is null || !(p >= 0 && p <= 1)) return null;   // the CPU raises the argument errors
        if (!training || p == 0) return tensor;                    // identity, exactly as the CPU
        string opName = alpha ? (spatialDims < 0 ? "TensorAlphaDropout" : "TensorFeatureAlphaDropout") : "TensorChannelDropout";
        if (tensor.Length == 0 || !TryRandomBackend<T>(opName, out var backend, out var kernels)) return null;
        try
        {
            int n = tensor.Length, block = 1;
            for (int k = tensor.Rank - Math.Max(0, spatialDims); k < tensor.Rank && spatialDims >= 0; k++) block *= tensor._shape[k];
            const double seluSaturation = -1.7580993408473766;
            double a, b = 0;
            if (alpha)
            {
                a = 1 / Math.Sqrt((1 - p) * (1 + p * seluSaturation * seluSaturation));
                b = -a * seluSaturation * p;
            }
            else a = p == 1 ? 0 : 1 / (1 - p);
            ulong deviceSeed = DeviceSeed(seed);
            OwnedBuffer scaleBuffer = default, offsetBuffer = default;
            bool ownsScale = false, ownsOffset = false;
            try
            {
                scaleBuffer = AllocateOutputBuffer(backend, n);
                ownsScale = true;
                if (alpha)
                {
                    offsetBuffer = AllocateOutputBuffer(backend, n);
                    ownsOffset = true;
                }
                // Channel dropout writes only the scale (the kernel leaves its offset argument untouched).
                kernels.TorchParityDropoutAffine(scaleBuffer.Buffer, alpha ? offsetBuffer.Buffer : scaleBuffer.Buffer, n, Math.Max(1, block), p, a,
                    seluSaturation, b, alpha, deviceSeed);
                var shape = (int[])tensor._shape.Clone();
                var scale = new Tensor<T>(FinishGpuOp<T>(backend, scaleBuffer, n), shape);
                ownsScale = false;
                Tensor<T>? offset = null;
                if (alpha)
                {
                    offset = new Tensor<T>(FinishGpuOp<T>(backend, offsetBuffer, n), (int[])shape.Clone());
                    ownsOffset = false;
                }
                var scaled = TensorMultiply(tensor, scale);
                return offset is null ? scaled : TensorAdd(scaled, offset);
            }
            finally
            {
                if (ownsScale) scaleBuffer.Dispose();
                if (ownsOffset) offsetBuffer.Dispose();
            }
        }
        catch (Exception ex)
        {
            GpuLaunchProbe.OnFallback(opName, ex);
            return null;
        }
    }
}
