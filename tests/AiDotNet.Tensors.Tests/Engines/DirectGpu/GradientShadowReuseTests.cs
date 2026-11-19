using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class GradientShadowReuseTests
{
    [Fact]
    public void RedirectRemappingPreservesSwappedTargetsAndSecondConsumerProtection()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        var sourceA = backend.AllocateBuffer(16);
        var sourceB = backend.AllocateBuffer(16);
        var targetA = backend.AllocateBuffer(16);
        var targetB = backend.AllocateBuffer(16);
        long slotA = 11L << 32, slotB = 12L << 32;
        var table = new FirstWriteRedirectTable();
        table.BeginRun();
        table.NoteContribution(sourceA, 11); table.Offer(slotA, sourceA, targetA);
        table.NoteContribution(sourceB, 12); table.Offer(slotB, sourceB, targetB);
        table.EndRun(true);
        table.RemapTargets(new Dictionary<IGpuBuffer, IGpuBuffer> { [targetA] = targetB, [targetB] = targetA });
        Assert.True(table.TryGetTarget(slotA, out var actualA)); Assert.Same(targetB, actualA);
        Assert.True(table.TryGetTarget(slotB, out var actualB)); Assert.Same(targetA, actualB);
        table.BeginRun();
        table.NoteContribution(targetB, 11); table.NoteContribution(targetB, 11);
        table.NoteContribution(targetA, 12);
        table.EndRun(true);
        Assert.False(table.TryGetTarget(slotA, out _));
        Assert.True(table.TryGetTarget(slotB, out _));
        table.BeginRun();
        table.NoteContribution(sourceA, 11); table.Offer(slotA, sourceA, targetA);
        table.EndRun(true);
        Assert.False(table.TryGetTarget(slotA, out _));
    }

    [Fact]
    public void NestedPointerScopesSuppressAndRestoreTheirCaller()
    {
        long previous = CompiledPointerTableScope.Current;
        try
        {
            using (CompiledPointerTableScope.Enter(101))
            {
                Assert.Equal(101, CompiledPointerTableScope.Current);
                using (CompiledPointerTableScope.Enter(0)) Assert.Equal(0, CompiledPointerTableScope.Current);
                Assert.Equal(101, CompiledPointerTableScope.Current);
                using (CompiledPointerTableScope.Enter(202)) Assert.Equal(202, CompiledPointerTableScope.Current);
                Assert.Equal(101, CompiledPointerTableScope.Current);
            }
            Assert.Equal(previous, CompiledPointerTableScope.Current);
        }
        finally { CompiledPointerTableScope.Current = previous; }
    }

    [SkippableFact]
    public void ChangingGradientPointersReuseMutableStorageAndLeaveLegacyTableImmutable()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "Actual CUDA required.");
        var backend = (CudaBackend)gpu.GetBackend()!;
        using var a = backend.AllocateBuffer(new float[32]);
        using var b = backend.AllocateBuffer(new float[32]);
        using var c = backend.AllocateBuffer(new float[32]);
        using var result = backend.AllocateBuffer(2);
        backend.Fill(a, 1f, 32); backend.Fill(b, 2f, 32); backend.Fill(c, 3f, 32);
        var sizes = new[] { 32 };
        var list = new IGpuBuffer[] { a };
        var field = typeof(CudaBackend).GetField("_gradientTable", BindingFlags.NonPublic | BindingFlags.Instance)!;
        object Member(object instance, string name) => instance.GetType().GetField(name)!.GetValue(instance)!;
        using var ordinary = CompiledPointerTableScope.Enter(0);
        backend.MultiTensorSumOfSquares(list, sizes, result);
        var legacy = field.GetValue(backend)!;
        var legacyPointers = Member(legacy, "Pointers");
        using (CompiledPointerTableScope.Enter(1000000001))
        {
            backend.MultiTensorSumOfSquares(list, sizes, result);
            var mutable = field.GetValue(backend)!;
            var pointers = Member(mutable, "Pointers");
            for (int i = 0; i < 12; i++)
            {
                int value = i % 3 + 1;
                list[0] = value == 1 ? a : value == 2 ? b : c;
                backend.MultiTensorSumOfSquares(list, sizes, result);
                Assert.Same(mutable, field.GetValue(backend));
                Assert.Same(pointers, Member(mutable, "Pointers"));
                var packed = backend.DownloadBuffer(result);
                long bits = ((long)BitConverter.SingleToInt32Bits(packed[1]) << 32)
                    | (uint)BitConverter.SingleToInt32Bits(packed[0]);
                Assert.Equal(32.0 * value * value, BitConverter.Int64BitsToDouble(bits));
            }
            Assert.True(backend.MutableGradientTableRefreshes >= 8);
        }
        list[0] = a;
        backend.MultiTensorSumOfSquares(list, sizes, result);
        Assert.Same(legacy, field.GetValue(backend));
        Assert.Same(legacyPointers, Member(legacy, "Pointers"));
        var retained = (System.Collections.ICollection)typeof(CudaBackend)
            .GetField("_retainedGradientTables", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(backend)!;
        Assert.True(retained.Count <= 8);
    }
}
