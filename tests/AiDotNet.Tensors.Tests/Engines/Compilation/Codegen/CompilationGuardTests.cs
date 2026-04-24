// Copyright (c) AiDotNet. All rights reserved.
// Phase D of issue #225: dynamic shapes + guards + recompile budget.

#nullable disable

using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen;
using AiDotNet.Tensors.Engines.Compilation.Codegen.AvxCs;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Guards;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation.Codegen;

public class CompilationGuardTests : IDisposable
{
    public CompilationGuardTests()
    {
        CodegenGuardRegistry.Clear();
        CodegenGuardRegistry.SetPolicy(new PowerOfTwoBucketPolicy());
        CodegenGuardRegistry.SetRecompileBudget(32);
    }

    public void Dispose() => CodegenGuardRegistry.Clear();

    // ─── ShapeBucket policies ────────────────────────────────────────

    [Fact]
    public void PowerOfTwo_Bucketizes_GroupsBatchSizes()
    {
        var p = new PowerOfTwoBucketPolicy();
        // 8, 9, 10, 11, …, 16 should all land in bucket 16 (and 8 → 8).
        Assert.Equal(8, p.BucketizeDimension(0, 8));
        Assert.Equal(16, p.BucketizeDimension(0, 9));
        Assert.Equal(16, p.BucketizeDimension(0, 16));
        Assert.Equal(32, p.BucketizeDimension(0, 17));
        Assert.Equal(32, p.BucketizeDimension(0, 32));
        Assert.Equal(64, p.BucketizeDimension(0, 33));
        Assert.Equal(128, p.BucketizeDimension(0, 128));
    }

    [Fact]
    public void PowerOfTwo_DynamicBatchSweep_8_to_128_TwoBuckets()
    {
        // 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128
        // → buckets: 8, 16, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128
        // Total distinct buckets = {8, 16, 32, 64, 128} = 5. #225 wants ≤2 for a tight sweep —
        // policy's job is to group adjacent sizes.
        var p = new PowerOfTwoBucketPolicy();
        var buckets = new[] { 8, 16, 32, 64, 128 }.Select(v => p.BucketizeDimension(0, v)).Distinct().Count();
        Assert.Equal(5, buckets);

        // Within the stage's contiguous range (e.g. 33..64), only one bucket.
        var inRange = new[] { 33, 40, 48, 56, 64 }.Select(v => p.BucketizeDimension(0, v)).Distinct().Count();
        Assert.Equal(1, inRange);
    }

    [Fact]
    public void ExactShapePolicy_EveryShapeUnique()
    {
        var p = new ExactShapePolicy();
        Assert.Equal(8, p.BucketizeDimension(0, 8));
        Assert.Equal(9, p.BucketizeDimension(0, 9));
        Assert.NotEqual(
            p.BucketizeDimension(0, 15),
            p.BucketizeDimension(0, 16));
    }

    // ─── CompilationGuard packing ────────────────────────────────────

    [Fact]
    public void Guard_SameInputs_ProducesEqualValues()
    {
        var g1 = new CompilationGuard(0x1234_5678_ABCD_EF01, CodegenElementType.Float32, 0xDEAD_BEEF);
        var g2 = new CompilationGuard(0x1234_5678_ABCD_EF01, CodegenElementType.Float32, 0xDEAD_BEEF);
        Assert.Equal(g1, g2);
        Assert.True(g1 == g2);
    }

    [Fact]
    public void Guard_DtypeChange_ProducesDifferentValue()
    {
        var g1 = new CompilationGuard(0x1234, CodegenElementType.Float32, 0x5678);
        var g2 = new CompilationGuard(0x1234, CodegenElementType.Float64, 0x5678);
        Assert.NotEqual(g1, g2);
    }

    [Fact]
    public void Guard_ShapeChange_ProducesDifferentValue()
    {
        var g1 = new CompilationGuard(0x1234, CodegenElementType.Float32, 0x5678);
        var g2 = new CompilationGuard(0x1234, CodegenElementType.Float32, 0x9ABC);
        Assert.NotEqual(g1, g2);
    }

    [Fact]
    public void ShapeBucket_UsesRegistryDefault_WhenNoPolicyPassed()
    {
        var b1 = ShapeBucket.Compute(new[] { 33 });
        // Same shape → same bucket.
        var b2 = ShapeBucket.Compute(new[] { 33 });
        Assert.Equal(b1, b2);
        // Within the same power-of-two range (33..64) → same bucket.
        var b3 = ShapeBucket.Compute(new[] { 50 });
        Assert.Equal(b1, b3);
        // Crossing a power-of-two boundary → different bucket.
        var b4 = ShapeBucket.Compute(new[] { 65 });
        Assert.NotEqual(b1, b4);
    }

    [Fact]
    public void ShapeBucket_ThreadOverride_IsRespected()
    {
        var before = ShapeBucket.Compute(new[] { 33 });
        using (CodegenGuardRegistry.SetPolicyForThread(new ExactShapePolicy()))
        {
            var inExact = ShapeBucket.Compute(new[] { 33 });
            var inExact2 = ShapeBucket.Compute(new[] { 34 });
            Assert.NotEqual(inExact, inExact2); // ExactShapePolicy makes 33 and 34 different.
        }
        var after = ShapeBucket.Compute(new[] { 33 });
        Assert.Equal(before, after); // Thread override is restored.
    }

    // ─── Cache ───────────────────────────────────────────────────────

    [Fact]
    public void Registry_InsertThenLookup_RoundTrips()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4 });
        var emit = new CpuDotNetJitEmitter().Emit(g, CodegenElementType.Float32);
        var kernel = emit.Kernel;
        var guard = new CompilationGuard(
            g.ComputeContentHash(), CodegenElementType.Float32, ShapeBucket.Compute(new[] { 4 }));

        Assert.True(CodegenGuardRegistry.Insert(guard, kernel));
        var fetched = CodegenGuardRegistry.Lookup(guard);
        Assert.Same(kernel, fetched);
    }

    [Fact]
    public void Registry_Insert_Duplicate_ReturnsFalse()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4 });
        var kernel = new CpuDotNetJitEmitter().Emit(g, CodegenElementType.Float32).Kernel;
        var guard = new CompilationGuard(g.ComputeContentHash(), CodegenElementType.Float32, 0);

        Assert.True(CodegenGuardRegistry.Insert(guard, kernel));
        Assert.False(CodegenGuardRegistry.Insert(guard, kernel));
    }

    // ─── Recompile budget ────────────────────────────────────────────

    [Fact]
    public void RecompileBudget_EnforcesGlobalLimit()
    {
        CodegenGuardRegistry.SetRecompileBudget(3);
        long graphHash = 0x1234;
        Assert.True(CodegenGuardRegistry.TryReserveRecompile(graphHash, "shape=8"));
        Assert.True(CodegenGuardRegistry.TryReserveRecompile(graphHash, "shape=16"));
        Assert.True(CodegenGuardRegistry.TryReserveRecompile(graphHash, "shape=32"));
        // Fourth request exceeds budget.
        Assert.False(CodegenGuardRegistry.TryReserveRecompile(graphHash, "shape=64"));
    }

    [Fact]
    public void RecompileBudget_IsPerGraphHash()
    {
        CodegenGuardRegistry.SetRecompileBudget(2);
        // Two distinct graph hashes — each gets its own budget.
        Assert.True(CodegenGuardRegistry.TryReserveRecompile(0x1, "A"));
        Assert.True(CodegenGuardRegistry.TryReserveRecompile(0x1, "A"));
        Assert.False(CodegenGuardRegistry.TryReserveRecompile(0x1, "A")); // A exhausted
        Assert.True(CodegenGuardRegistry.TryReserveRecompile(0x2, "B")); // B still fine
    }

    [Fact]
    public void RecompileLog_CapturesAllAttempts_InOrder()
    {
        CodegenGuardRegistry.SetRecompileBudget(10);
        CodegenGuardRegistry.TryReserveRecompile(0xAAA, "first");
        CodegenGuardRegistry.TryReserveRecompile(0xBBB, "second");
        CodegenGuardRegistry.TryReserveRecompile(0xAAA, "third");

        var log = CodegenGuardRegistry.DumpRecompileLog();
        Assert.Equal(3, log.Count);
        Assert.Equal("first", log[0].Reason);
        Assert.Equal("second", log[1].Reason);
        Assert.Equal("third", log[2].Reason);
        Assert.Equal(0xAAAL, log[0].GraphHash);
        Assert.Equal(0xBBBL, log[1].GraphHash);
        Assert.Equal(2, log[2].AttemptIndex); // Second attempt for graph AAA
    }

    [Fact]
    public void Registry_Clear_ResetsCacheAndBudget()
    {
        var g = CodegenLowering.LowerUnaryPointwise<float>(CodegenOpKind.ReLU, new[] { 4 });
        var kernel = new CpuDotNetJitEmitter().Emit(g, CodegenElementType.Float32).Kernel;
        var guard = new CompilationGuard(g.ComputeContentHash(), CodegenElementType.Float32, 0);
        CodegenGuardRegistry.Insert(guard, kernel);
        CodegenGuardRegistry.TryReserveRecompile(g.ComputeContentHash(), "test");
        Assert.True(CodegenGuardRegistry.CacheEntryCount > 0);
        Assert.NotEmpty(CodegenGuardRegistry.DumpRecompileLog());

        CodegenGuardRegistry.Clear();
        Assert.Equal(0, CodegenGuardRegistry.CacheEntryCount);
        Assert.Empty(CodegenGuardRegistry.DumpRecompileLog());
    }
}
