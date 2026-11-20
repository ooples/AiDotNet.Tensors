using System;
using System.Buffers;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

public class ConvGemmPackingTests
{
    private sealed class CountingPool<T> : ArrayPool<T>
    {
        public int Rents, Returns;
        public override T[] Rent(int minimumLength) { Rents++; return new T[minimumLength]; }
        public override void Return(T[] array, bool clearArray = false) { Returns++; }
    }

    [Theory]
    [InlineData(12, 16)]
    [InlineData(48, 17)]
    [InlineData(12, 65)]
    public void Spatial_panels_share_filter_stripes_and_replace_changing_input(int rows, int columns)
    {
        const int k = 21, firstRow = 6;
        var a = new float[(rows + firstRow) * k];
        var b = new float[k * columns];
        var output = new float[rows * columns];
        var control = new float[output.Length];
        var filterPool = new CountingPool<float>();
        var workerPool = new CountingPool<byte>();
        for (int pass = 0; pass < 2; pass++)
        {
            for (int i = 0; i < a.Length; i++) a[i] = ((i + pass) % 29 - 14) * .123f;
            using var filter = new ConvGemmPacking<float>(a, rows + firstRow, k, pool: filterPool);
            using var worker = new ConvGemmPacking<float>.Worker(filter, pool: workerPool);
            for (int panel = 0; panel < 3; panel++)
            {
                for (int i = 0; i < b.Length; i++) b[i] = ((i + 3 * panel) % 11 - 5) * .211f;
                Array.Clear(control, 0, control.Length);
                Array.Clear(output, 0, output.Length);
                Multiply(a, firstRow, b, control, rows, columns, k, null);
                Multiply(a, firstRow, b, output, rows, columns, k, worker);
                Assert.Equal(control, output);
                Assert.Equal(pass + 1, filterPool.Rents);
                Assert.Equal(pass + 1, workerPool.Rents);
            }
        }
        Assert.Equal(filterPool.Rents, filterPool.Returns);
        Assert.Equal(workerPool.Rents, workerPool.Returns);
    }

    [Fact]
    public void Joined_workers_keep_disjoint_input_staging_and_reuse_their_own_arrays()
    {
        var pool = new CountingPool<byte>();
        using var filter = new ConvGemmPacking<float>(new float[12], 6, 2, workers: 2);
        using (var left = new ConvGemmPacking<float>.Worker(filter, 0, pool))
        using (var right = new ConvGemmPacking<float>.Worker(filter, 1, pool))
        {
            var a = left.TryUseB(64);
            var b = right.TryUseB(64);
            Assert.NotNull(a);
            Assert.NotNull(b);
            Assert.NotSame(a, b);
            left.ReleaseB();
            Assert.Same(a, left.TryUseB(32));
            right.ReleaseB();
            Assert.Same(b, right.TryUseB(32));
            left.ReleaseB();
            right.ReleaseB();
        }
        Assert.Equal(2, pool.Rents);
        Assert.Equal(2, pool.Returns);
    }

    [Fact]
    public void Joined_operation_returns_filter_and_worker_storage_on_exception()
    {
        const int rows = 12, columns = 16, k = 21;
        var a = new float[rows * k];
        var filters = new CountingPool<float>();
        var columnsPool = new CountingPool<byte>();
        void RunFailingOperation()
        {
            using var filter = new ConvGemmPacking<float>(a, rows, k, pool: filters);
            using var worker = new ConvGemmPacking<float>.Worker(filter, pool: columnsPool);
            Multiply(a, 0, new float[k * columns], new float[rows * columns], rows, columns, k, worker);
            throw new InvalidOperationException("The next convolution task failed.");
        }
        Assert.Throws<InvalidOperationException>(RunFailingOperation);
        Assert.Equal(1, filters.Rents);
        Assert.Equal(filters.Rents, filters.Returns);
        Assert.Equal(1, columnsPool.Rents);
        Assert.Equal(columnsPool.Rents, columnsPool.Returns);
    }

    [Fact]
    public void Nested_and_failed_operations_preserve_private_ownership()
    {
        const int rows = 12, cols = 16, k = 21;
        var a = new float[rows * k];
        var b = new float[k * cols];
        var c = new float[rows * cols];
        var workerPool = new CountingPool<byte>();
        using var filter = new ConvGemmPacking<float>(new float[a.Length], rows, k);
        using (var worker = new ConvGemmPacking<float>.Worker(filter, pool: workerPool))
        {
            // The packed filter belongs to a different matrix, so the negative source span
            // forces a checked exception in the normal live packer after borrowing the B buffer.
            Assert.Throws<ArgumentOutOfRangeException>(() => Multiply(a, 0, b, c, rows, cols, k, worker, true));
            var loan = worker.TryUseB(16);
            Assert.NotNull(loan);
            Assert.Null(worker.TryUseB(16));
            worker.ReleaseB();
            Multiply(a, 0, b, c, rows, cols, k, worker);
            Assert.Equal(1, workerPool.Rents);
            Assert.Equal(0, workerPool.Returns);
        }
        Assert.Equal(1, workerPool.Returns);
    }

    private static unsafe void Multiply(float[] a, int firstRow, float[] b, float[] c,
        int rows, int cols, int k, ConvGemmPacking<float>.Worker? worker, bool invalidLength = false)
    {
        fixed (float* aPtr = a, bPtr = b, cPtr = c)
            PackBothStrategy.RunSingleRegionSerialUnsafe(aPtr + firstRow * k, invalidLength ? -1 : rows * k, k,
                bPtr, b.Length, cols, cPtr, c.Length, cols, rows, cols, k, 12, 8, 6, 16, sizeof(float), worker);
    }
}
