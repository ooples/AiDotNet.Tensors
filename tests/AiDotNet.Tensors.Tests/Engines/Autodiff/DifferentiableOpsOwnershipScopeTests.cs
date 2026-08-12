using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

public sealed class DifferentiableOpsOwnershipScopeTests
{
    [Fact]
    public async Task TensorViewRecordingSuppression_CopiedLeaseReleasesDepthOnlyOnce()
    {
        await Task.Yield();

        Assert.False(DifferentiableOps.IsTensorViewRecordingSuppressed);
        var outer = DifferentiableOps.SuppressTensorViewRecording();
        var inner = DifferentiableOps.SuppressTensorViewRecording();
        var innerCopy = inner;

        try
        {
            Assert.True(DifferentiableOps.IsTensorViewRecordingSuppressed);
            inner.Dispose();
            Assert.True(DifferentiableOps.IsTensorViewRecordingSuppressed);

            innerCopy.Dispose();
            Assert.True(DifferentiableOps.IsTensorViewRecordingSuppressed);

            outer.Dispose();
            Assert.False(DifferentiableOps.IsTensorViewRecordingSuppressed);

            outer.Dispose();
            Assert.False(DifferentiableOps.IsTensorViewRecordingSuppressed);
        }
        finally
        {
            innerCopy.Dispose();
            inner.Dispose();
            outer.Dispose();
        }
    }

    [Fact]
    public async Task NestedBackwardStep_RestoresOuterDonationOwnership()
    {
        await Task.Yield();

        var engine = new CpuEngine();
        var outerGradients = new Dictionary<Tensor<double>, Tensor<double>>();
        var firstDestination = new Tensor<double>([2]);
        var secondDestination = new Tensor<double>([2]);
        var sharedContribution = new Tensor<double>(new[] { 2.0, -3.0 }, [2]);
        var previousOuterOwners = DifferentiableOps.BeginBackwardStep<double>();

        try
        {
            DifferentiableOps.AccumulateGrad(
                outerGradients,
                firstDestination,
                sharedContribution,
                engine);

            var previousInnerOwners = DifferentiableOps.BeginBackwardStep<double>();
            try
            {
                var innerGradients = new Dictionary<Tensor<double>, Tensor<double>>();
                DifferentiableOps.AccumulateGrad(
                    innerGradients,
                    new Tensor<double>([2]),
                    new Tensor<double>(new[] { 5.0, 7.0 }, [2]),
                    engine);
            }
            finally
            {
                DifferentiableOps.EndBackwardStep(previousInnerOwners);
            }

            DifferentiableOps.AccumulateGrad(
                outerGradients,
                secondDestination,
                sharedContribution,
                engine);

            Assert.Same(sharedContribution, outerGradients[firstDestination]);
            Assert.NotSame(outerGradients[firstDestination], outerGradients[secondDestination]);
            outerGradients[firstDestination][0] = 41.0;
            Assert.Equal(2.0, outerGradients[secondDestination][0]);
        }
        finally
        {
            DifferentiableOps.EndBackwardStep(previousOuterOwners);
        }
    }

    [Fact]
    public async Task AccumulateGrad_OverlappingStorageUsesIndependentResult()
    {
        await Task.Yield();

        var engine = new CpuEngine();
        var backing = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, [4]);
        var existing = backing.Reshape(2, 2);
        var overlappingContribution = backing.Reshape(2, 2);
        var destination = new Tensor<double>([2, 2]);
        var gradients = new Dictionary<Tensor<double>, Tensor<double>>
        {
            [destination] = existing,
        };
        destination.Grad = existing;
        var previousOwners = DifferentiableOps.BeginBackwardStep<double>();

        try
        {
            DifferentiableOps.AccumulateGrad(
                gradients,
                destination,
                overlappingContribution,
                engine);

            Assert.NotSame(existing, gradients[destination]);
            Assert.Equal(new[] { 2.0, 4.0, 6.0, 8.0 }, gradients[destination].ToArray());
            Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0 }, backing.ToArray());
        }
        finally
        {
            DifferentiableOps.EndBackwardStep(previousOwners);
        }
    }

    [Fact]
    public async Task AccumulateGrad_DisjointSharedStorageKeepsInPlaceFastPath()
    {
        await Task.Yield();

        var engine = new CpuEngine();
        var backing = new Tensor<double>(new[] { 1.0, 2.0, 3.0, 4.0 }, [4]);
        var existing = backing.Slice(axis: 0, start: 0, end: 2);
        var disjointContribution = backing.Slice(axis: 0, start: 2, end: 4);
        var destination = new Tensor<double>([2]);
        var gradients = new Dictionary<Tensor<double>, Tensor<double>>
        {
            [destination] = existing,
        };
        destination.Grad = existing;
        var previousOwners = DifferentiableOps.BeginBackwardStep<double>();

        try
        {
            DifferentiableOps.AccumulateGrad(
                gradients,
                destination,
                disjointContribution,
                engine);

            Assert.Same(existing, gradients[destination]);
            Assert.Equal(new[] { 4.0, 6.0 }, gradients[destination].ToArray());
            Assert.Equal(new[] { 3.0, 4.0 }, disjointContribution.ToArray());
        }
        finally
        {
            DifferentiableOps.EndBackwardStep(previousOwners);
        }
    }
}
