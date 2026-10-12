using System;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// #1221: compiling saved the thread's options with <c>TensorCodecOptions.Current</c> and put them back with
/// <c>SetCurrent</c>, which installed a thread-local determinism override of true that the thread never had, so the
/// thread then ignored <c>BlasProvider.SetDeterministicMode(false)</c>. Each case runs on a fresh thread, where no
/// earlier test can have left state behind.
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class TensorCodecOptionsScopeTests
{
    private static void OnFreshThread(Action body)
    {
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            try { body(); }
            catch (Exception e) { failure = e; }
        });
        thread.Start();
        thread.Join();
        if (failure is not null)
            throw new Exception("failed on the fresh thread", failure);
    }

    [Fact]
    public void CompileInference_LeavesNoDeterminismOverride()
    {
        bool processWide = BlasProvider.IsDeterministicMode;
        try
        {
            OnFreshThread(() =>
            {
                Assert.Null(BlasProvider.GetThreadLocalDeterministicMode());
                var x = new Tensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
                var y = new Tensor<float>(new float[] { 5, 6, 7, 8 }, new[] { 2, 2 });
                using (var plan = TensorCodec.CompileInference<float>(() => AiDotNetEngine.Current.TensorAdd(x, y)))
                {
                }

                Assert.Null(BlasProvider.GetThreadLocalDeterministicMode());
                BlasProvider.SetDeterministicMode(false);
                Assert.False(BlasProvider.IsDeterministicMode); // the process-wide switch reaches this thread again
            });
        }
        finally
        {
            BlasProvider.SetDeterministicMode(processWide);
        }
    }

    [Fact]
    public void Push_RestoresNoneAndAnExistingOverrideExactly()
    {
        OnFreshThread(() =>
        {
            // Nothing installed before: nothing installed after.
            using (TensorCodecOptions.Push(new TensorCodecOptions { Deterministic = false }))
                Assert.False(BlasProvider.GetThreadLocalDeterministicMode());
            Assert.Null(BlasProvider.GetThreadLocalDeterministicMode());

            // An existing override and options instance come back as they were.
            var outer = new TensorCodecOptions { Deterministic = false, EnableDataflowFusion = false };
            TensorCodecOptions.SetCurrent(outer);
            try
            {
                using (TensorCodecOptions.Push(new TensorCodecOptions { Deterministic = true }))
                    Assert.True(BlasProvider.GetThreadLocalDeterministicMode());
                Assert.Same(outer, TensorCodecOptions.Current);
                Assert.False(BlasProvider.GetThreadLocalDeterministicMode());
            }
            finally
            {
                TensorCodecOptions.SetCurrent(null);
            }
        });
    }
}
