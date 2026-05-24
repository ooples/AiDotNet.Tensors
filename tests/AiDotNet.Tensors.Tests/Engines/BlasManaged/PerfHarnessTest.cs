using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue A (#369) task A.5: <see cref="PerfHarness"/> measures BlasManaged vs.
/// native OpenBLAS for each shape and writes a JSON envelope including a hardware
/// fingerprint and git SHA. The harness is what the project owner runs in A.6 on
/// the authoritative self-hosted runner.
///
/// <para>
/// These tests verify the harness's mechanical correctness (returns a result,
/// produces JSON, captures the envelope). The perf semantics — what counts as
/// "BlasManaged wins" — are asserted by <c>PerfBarTest</c> in task A.7.
/// </para>
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class PerfHarnessTest
{
    [Fact]
    public void Run_Single_Shape_Produces_Result()
    {
        var shape = new Shape("Test_64x64x64_FP32", 64, 64, 64, false, false, DType.Single, 0, "test");
        var result = PerfHarness.RunShape(shape);

        Assert.Equal(shape.Name, result.ShapeName);
        Assert.Equal(shape.M, result.M);
        Assert.Equal(shape.N, result.N);
        Assert.Equal(shape.K, result.K);
        Assert.True(result.BlasManagedMedianMs > 0, "BlasManaged median should be positive");
        Assert.True(result.BlasManagedP95Ms >= result.BlasManagedMedianMs, "p95 must be >= median");
        // Native may be unavailable on this host (libopenblas not loaded).
        // The harness must tolerate that and report NativeAvailable=false.
        Assert.True(
            result.NativeAvailable
                ? result.NativeMedianMs > 0
                : result.NativeMedianMs == -1,
            $"NativeAvailable={result.NativeAvailable} should imply MedianMs sign");
    }

    [Fact]
    public void Run_Single_Shape_FP64_Produces_Result()
    {
        var shape = new Shape("Test_64x64x64_FP64", 64, 64, 64, false, false, DType.Double, 0, "test");
        var result = PerfHarness.RunShape(shape);

        Assert.Equal("Double", result.Dtype);
        Assert.True(result.BlasManagedMedianMs > 0);
    }

    [Fact]
    public void RunAll_Writes_Json_With_Envelope_And_All_Shapes()
    {
        var tmpPath = Path.Combine(Path.GetTempPath(), $"perfharness-{Guid.NewGuid():N}.json");
        try
        {
            var shapes = new[]
            {
                new Shape("Tiny_32x32x32_FP32", 32, 32, 32, false, false, DType.Single, 0, "test"),
                new Shape("Tiny_64x64x64_FP32", 64, 64, 64, false, false, DType.Single, 0, "test"),
            };
            PerfHarness.RunAll(shapes, tmpPath);

            Assert.True(File.Exists(tmpPath), $"Expected JSON output at {tmpPath}");
            var json = File.ReadAllText(tmpPath);

            // Envelope fields present.
            Assert.Contains("GitSha", json);
            Assert.Contains("HardwareFingerprint", json);
            Assert.Contains("TimestampUtc", json);

            // Both shapes appear.
            Assert.Contains("Tiny_32x32x32_FP32", json);
            Assert.Contains("Tiny_64x64x64_FP32", json);

            // Round-trip parse to verify schema.
            var parsed = JsonSerializer.Deserialize<HarnessOutput>(json,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            Assert.NotNull(parsed);
            Assert.Equal(2, parsed!.Shapes.Count);
            Assert.All(parsed.Shapes, r =>
            {
                Assert.True(r.BlasManagedMedianMs > 0);
                Assert.True(r.BlasManagedP95Ms >= r.BlasManagedMedianMs);
            });
        }
        finally
        {
            if (File.Exists(tmpPath)) File.Delete(tmpPath);
        }
    }

    [Fact]
    public void Ratio_Is_Positive_When_Native_Available_Else_Zero()
    {
        var shape = new Shape("Test_64x64x64_FP32", 64, 64, 64, false, false, DType.Single, 0, "test");
        var result = PerfHarness.RunShape(shape);

        if (result.NativeAvailable)
            Assert.True(result.RatioBmOverNative > 0, "Native available => ratio must be positive");
        else
            Assert.Equal(0.0, result.RatioBmOverNative);
    }
}
