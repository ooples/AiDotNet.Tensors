using System.Threading;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

// AutoTensorCache is process-wide; keep these off the parallel runner so another test's rentals cannot interleave.
[Collection("DeferredMaterializerRegistrySerial")]
public class AutoTensorCacheBudgetTests
{
    /// <summary>
    /// The per-shape copy limits bound each shape, not the total: before the aggregate cap, every distinct shape a
    /// model produced kept its copies for the life of the thread.
    /// </summary>
    [Fact]
    public void Return_StopsPoolingAtTheByteBudget()
    {
        long prior = AutoTensorCache.MaxPooledBytes;
        try
        {
            AutoTensorCache.Clear();
            AutoTensorCache.MaxPooledBytes = 3 * 1000 * sizeof(float);
            var shapes = new[] { new[] { 1000, 1 }, new[] { 1, 1000 }, new[] { 10, 100 }, new[] { 100, 10 } };
            var returned = new Tensor<float>[shapes.Length];
            for (int i = 0; i < shapes.Length; i++)
            {
                returned[i] = new Tensor<float>(shapes[i]);
                AutoTensorCache.Return(returned[i]);
            }

            // The first three fit the budget and come back; the fourth was refused.
            for (int i = 0; i < 3; i++) Assert.Same(returned[i], AutoTensorCache.RentOrAllocate<float>(shapes[i]));
            Assert.NotSame(returned[3], AutoTensorCache.RentOrAllocate<float>(shapes[3]));

            // Renting gave the budget back.
            AutoTensorCache.Return(returned[3]);
            Assert.Same(returned[3], AutoTensorCache.RentOrAllocate<float>(shapes[3]));
        }
        finally
        {
            AutoTensorCache.MaxPooledBytes = prior;
            AutoTensorCache.Clear();
        }
    }

    /// <summary>
    /// The budget is shared by every thread: the pools are per thread, and a parallel kernel's workers each filled
    /// their own, so a per-thread budget multiplied by the worker count.
    /// </summary>
    [Fact]
    public void Budget_IsSharedAcrossThreads()
    {
        long prior = AutoTensorCache.MaxPooledBytes;
        try
        {
            AutoTensorCache.Clear();
            AutoTensorCache.MaxPooledBytes = 1000 * sizeof(float);
            var worker = new Thread(() => AutoTensorCache.Return(new Tensor<float>(new[] { 25, 40 })));
            worker.Start();
            worker.Join();

            var mine = new Tensor<float>(new[] { 40, 25 });
            AutoTensorCache.Return(mine);
            Assert.NotSame(mine, AutoTensorCache.RentOrAllocate<float>(new[] { 40, 25 }));
        }
        finally
        {
            AutoTensorCache.MaxPooledBytes = prior;
            AutoTensorCache.Clear();
        }
    }

    /// <summary>
    /// Clear empties every thread's pools but runs on one thread: this thread's shape counts must not keep it refusing
    /// buffers as if its emptied pools were still full.
    /// </summary>
    [Fact]
    public void Clear_FromAnotherThread_ResetsThisThreadsShapeCounts()
    {
        long prior = AutoTensorCache.MaxPooledBytes;
        try
        {
            AutoTensorCache.Clear();
            AutoTensorCache.MaxPooledBytes = 1000 * sizeof(float);
            var shape = new[] { 20, 50 };
            AutoTensorCache.Return(new Tensor<float>(shape)); // fills this thread's budget

            var clearer = new Thread(AutoTensorCache.Clear);
            clearer.Start();
            clearer.Join();

            var after = new Tensor<float>(shape);
            AutoTensorCache.Return(after);
            Assert.Same(after, AutoTensorCache.RentOrAllocate<float>(shape));
        }
        finally
        {
            AutoTensorCache.MaxPooledBytes = prior;
            AutoTensorCache.Clear();
        }
    }
}
