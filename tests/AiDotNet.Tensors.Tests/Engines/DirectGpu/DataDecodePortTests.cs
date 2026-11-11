using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The data-decode kernels (<see cref="IDataDecodeBackend"/>, <see cref="DataDecodeKernelTemplate"/>) on each backend,
/// against the CPU engine bit for bit: every float16 pattern and random bytes of every other source type in both byte
/// orders (CpuEngine.TensorFromBytes), the exclusive scan, the CSV field bounds (CpuEngine.SplitFieldBounds), and the
/// per-field parse (NumericTextFastPath, itself checked against float.Parse in DataOpsTests).
/// </summary>
[Collection("DirectGpuSerial")]
public class DataDecodePortTests
{
    [SkippableFact]
    public void OpenCl_data_decode_matches_the_cpu()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        DataDecodeChecks.Run(backend);
    }

    [SkippableFact]
    public void Vulkan_data_decode_matches_the_cpu()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        DataDecodeChecks.Run(backend);
    }

    [SkippableFact]
    public void Metal_data_decode_matches_the_cpu()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.OSX), "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        DataDecodeChecks.Run(backend);
    }

    [SkippableFact]
    public void Cuda_data_decode_matches_the_cpu()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        DataDecodeChecks.Run(backend);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_data_decode_matches_the_cpu()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        DataDecodeChecks.Run(backend);
    }
#endif

    /// <summary>Every GLSL kernel compiles under shaderc (the hardware tests skip without a device).</summary>
    [SkippableFact]
    public void Vulkan_data_decode_shaders_compile()
    {
        using var compiler = new VulkanGlslCompiler();
        Skip.IfNot(compiler.IsAvailable, "libshaderc is unavailable.");
        foreach (var name in DataDecodeKernelTemplate.GetKernelNames())
            Assert.True(compiler.CompileToSpirv(DataDecodeKernelTemplate.GetGlslSource(name)) is not null,
                $"{name}: {compiler.LastError}");
    }

    /// <summary>The HIP module compiles as CUDA C++ under NVRTC (CudaBackend compiles the same source).</summary>
    [SkippableFact]
    public void Hip_data_decode_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        string source = DataDecodeKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip);
        IntPtr program = IntPtr.Zero;
        Assert.Equal(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcResult.Success,
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, "data_decode",
                0, IntPtr.Zero, IntPtr.Zero));
        try
        {
            var result = AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcCompileProgram(program, 1,
                new[] { "--gpu-architecture=compute_75" });
            Assert.True(result == AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcResult.Success,
                $"the HIP dialect did not compile ({result})");
        }
        finally
        {
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
    }
}

#if NET6_0_OR_GREATER
/// <summary>The same checks on HIP, sharing one backend (its construction is slow).</summary>
[Collection("DirectGpuSerial")]
public class DataDecodeHipPortTests : IClassFixture<HipBackendTestFixture>
{
    private readonly HipBackendTestFixture _fixture;

    public DataDecodeHipPortTests(HipBackendTestFixture fixture) => _fixture = fixture;

    [SkippableFact]
    public void Hip_data_decode_matches_the_cpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "HIP is not available.");
        DataDecodeChecks.Run(_fixture.Backend);
    }
}
#endif

internal static class DataDecodeChecks
{
    private static readonly CpuEngine Cpu = new();

    internal static void Run(IDirectGpuBackend backend)
    {
        var decode = Assert.IsAssignableFrom<IDataDecodeBackend>(backend);
        CheckDecode(backend, decode);
        CheckScan(backend, decode);
        CheckSplit(backend, decode);
        CheckParse(backend, decode);
    }

    private static int[] Pack(byte[] bytes)
    {
        var words = new int[Math.Max(1, (bytes.Length + 3) / 4)];
        Buffer.BlockCopy(bytes, 0, words, 0, bytes.Length);
        return words;
    }

    private static int[] DownloadInts(IDirectGpuBackend backend, IGpuBuffer buffer, int count)
    {
        backend.Synchronize();
        var raw = backend.DownloadBuffer(buffer);
        var ints = new int[count];
        Buffer.BlockCopy(raw, 0, ints, 0, count * sizeof(int));
        return ints;
    }

    private static void CheckDecode(IDirectGpuBackend backend, IDataDecodeBackend decode)
    {
        var rng = RandomHelper.CreateSeededRandom(11);
        foreach (NumericStorageType type in Enum.GetValues(typeof(NumericStorageType)))
        {
            int width = type.GetByteWidth();
            byte[] bytes;
            if (type == NumericStorageType.Float16)
            {
                bytes = new byte[65536 * 2];   // every pattern
                for (int h = 0; h < 65536; h++) { bytes[2 * h] = (byte)h; bytes[2 * h + 1] = (byte)(h >> 8); }
            }
            else
            {
                bytes = new byte[width * 20000 + width];   // an odd count, so the last word is partial for narrow types
                rng.NextBytes(bytes);
            }
            int count = bytes.Length / width;
            foreach (bool littleEndian in new[] { true, false })
            {
                var expected = Cpu.TensorFromBytes<float>(bytes, type, new[] { count }, littleEndian).ToArray();
                using var words = backend.AllocateIntBuffer(Pack(bytes));
                using var output = backend.AllocateBuffer(count);
                Assert.True(decode.TryDecodeBytes(words, output, count, type, littleEndian), $"{type} refused");
                var actual = DownloadInts(backend, output, count);
                for (int i = 0; i < count; i++)
                {
                    int e = BitConverter.ToInt32(BitConverter.GetBytes(expected[i]), 0);
                    if (float.IsNaN(expected[i]))
                        Assert.True(float.IsNaN(BitConverter.ToSingle(BitConverter.GetBytes(actual[i]), 0)),
                            $"{type} le={littleEndian} [{i}]: expected NaN, got {actual[i]:X8}");
                    else
                        Assert.True(e == actual[i], $"{type} le={littleEndian} [{i}]: expected {e:X8}, got {actual[i]:X8}");
                }
            }
        }
    }

    private static void CheckScan(IDirectGpuBackend backend, IDataDecodeBackend decode)
    {
        var rng = RandomHelper.CreateSeededRandom(12);
        foreach (int n in new[] { 1, 255, 256, 257, 1000, 70000 })
        {
            var values = Enumerable.Range(0, n).Select(_ => rng.Next(0, 4)).ToArray();
            using var buffer = backend.AllocateIntBuffer(values);
            using var total = backend.AllocateIntBuffer(1);
            Assert.True(decode.TryExclusiveScan(buffer, total, n), $"scan {n} refused");
            var prefix = DownloadInts(backend, buffer, n);
            int sum = 0;
            for (int i = 0; i < n; i++)
            {
                Assert.True(sum == prefix[i], $"scan {n} [{i}]: expected {sum}, got {prefix[i]}");
                sum += values[i];
            }
            Assert.Equal(sum, DownloadInts(backend, total, 1)[0]);
        }
    }

    private static IEnumerable<(string Text, byte Delimiter)> SplitCases()
    {
        yield return ("1,2,3\r\n4,,6\n", (byte)',');
        yield return ("a,b\r\n\nc\r\n", (byte)',');
        yield return (",", (byte)',');
        yield return ("x\ty\r", (byte)'\t');
        yield return ("1\r2\r\n3\n", (byte)'\r');   // the delimiter is CR: it is never dropped
        var rng = RandomHelper.CreateSeededRandom(13);
        var sb = new StringBuilder();
        // Long enough for many 64-byte segments and fields that straddle them.
        for (int row = 0; row < 3000; row++)
        {
            for (int c = 0; c < rng.Next(1, 9); c++)
            {
                if (c > 0) sb.Append(',');
                if (rng.Next(10) > 0) sb.Append((rng.NextDouble() * 1000 - 500).ToString("R", CultureInfo.InvariantCulture));
            }
            sb.Append(rng.Next(3) == 0 ? "\r\n" : "\n");
        }
        yield return (sb.ToString(), (byte)',');
        yield return (sb.ToString().TrimEnd('\n', '\r'), (byte)',');
    }

    private static void CheckSplit(IDirectGpuBackend backend, IDataDecodeBackend decode)
    {
        foreach (var (text, delimiter) in SplitCases())
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            var expected = CpuEngine.SplitFieldBounds(bytes, delimiter, out int fields);
            int length = bytes.Length;
            bool trailingNewline = length > 0 && bytes[length - 1] == (byte)'\n';
            if (trailingNewline) length--;
            int segments = (length + DataDecodeKernelTemplate.SegmentBytes - 1) / DataDecodeKernelTemplate.SegmentBytes;
            using var words = backend.AllocateIntBuffer(Pack(bytes));
            using var offsets = backend.AllocateIntBuffer(segments);
            using var total = backend.AllocateIntBuffer(1);
            Assert.True(decode.TryCountSeparators(words, offsets, length, delimiter), "count refused");
            Assert.True(decode.TryExclusiveScan(offsets, total, segments), "scan refused");
            Assert.Equal(fields, DownloadInts(backend, total, 1)[0] + 1);
            using var bounds = backend.AllocateIntBuffer(2 * fields);
            Assert.True(decode.TryScatterFieldBounds(words, offsets, bounds, length, delimiter, trailingNewline), "scatter refused");
            Assert.Equal(expected, DownloadInts(backend, bounds, 2 * fields));
        }
    }

    private static void CheckParse(IDirectGpuBackend backend, IDataDecodeBackend decode)
    {
        var corpus = AiDotNet.Tensors.Tests.Engines.DataOpsTests.NumericCorpus(20_000, seed: 21);
        var bytes = Encoding.UTF8.GetBytes(string.Join(",", corpus));
        var hostBounds = CpuEngine.SplitFieldBounds(bytes, (byte)',', out int fields);
        Assert.Equal(corpus.Count, fields);
        const float missing = -1234.5f;
        uint missingBits = BitConverter.ToUInt32(BitConverter.GetBytes(missing), 0);
        foreach (var styles in new[] { NumberStyles.Float, NumberStyles.Integer, NumberStyles.AllowDecimalPoint })
        {
            int mask = NumericTextFastPath.StyleMask(styles, CultureInfo.InvariantCulture);
            Assert.True(mask >= 0, $"{styles} has no fast path");
            using var words = backend.AllocateIntBuffer(Pack(bytes));
            using var bounds = backend.AllocateIntBuffer(hostBounds);
            using var output = backend.AllocateBuffer(fields);
            using var status = backend.AllocateIntBuffer(fields);
            Assert.True(decode.TryParseFields(words, bounds, output, status, fields, mask, missing), "parse refused");
            var bits = DownloadInts(backend, output, fields);
            var states = DownloadInts(backend, status, fields);
            for (int f = 0; f < fields; f++)
            {
                int start = hostBounds[2 * f];
                var field = new ReadOnlySpan<byte>(bytes, start, hostBounds[2 * f + 1] - start);
                int cpu = NumericTextFastPath.TryParseSingleBits(field, mask, out uint cpuBits);
                int expectedState = cpu == NumericTextFastPath.Undecided ? 1 : 0;
                uint expectedBits = cpu == NumericTextFastPath.Parsed ? cpuBits : missingBits;
                Assert.True(expectedState == states[f] && expectedBits == (uint)bits[f],
                    $"{styles} '{corpus[f]}': expected state {expectedState} bits {expectedBits:X8}, got {states[f]} {(uint)bits[f]:X8}");
            }
        }
    }
}
