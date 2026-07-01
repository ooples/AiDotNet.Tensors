using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public sealed class CompiledTrainingPlanCleanupTests
{
    [Fact]
    public void BuildStepEvictionProtectSet_IncludesGradientSeedDestAndBackingKeys()
    {
        var loss = new Tensor<float>(new[] { 1 });
        var grad0 = new Tensor<float>(new[] { 2 });
        var grad1 = new Tensor<float>(new[] { 3 });
        var seed = new Tensor<float>(new[] { 1 });
        var dest = new Tensor<float>(new[] { 1 });
        var plan = CreatePlan(loss, new[] { grad0, grad1 }, seed, dest);

        var protect = (HashSet<object>)Invoke(plan, "BuildStepEvictionProtectSet")!;

        AssertContainsTensorKeys(protect, grad0);
        AssertContainsTensorKeys(protect, grad1);
        AssertContainsTensorKeys(protect, seed);
        AssertContainsTensorKeys(protect, dest);
        Assert.DoesNotContain(loss.DataVector, protect);
    }

    [Fact]
    public void MaterializeStepLoss_MaterializesBackingArrayAndVectorKeys()
    {
        var loss = new Tensor<float>(new float[] { 1.0f }, new[] { 1 });
        var plan = CreatePlan(loss, Array.Empty<Tensor<float>>(), new Tensor<float>(new[] { 1 }), null);
        var backing = loss.DataVector.GetBackingArrayUnsafe()
            ?? throw new InvalidOperationException("Expected CPU tensor to expose a backing array.");
        int backingCalls = 0;
        int vectorCalls = 0;

        DeferredArrayMaterializer.Register(backing, _ => backingCalls++);
        DeferredArrayMaterializer.Register(loss.DataVector, _ => vectorCalls++);
        try
        {
            Invoke(plan, "MaterializeStepLoss");

            Assert.Equal(1, backingCalls);
            Assert.Equal(1, vectorCalls);
        }
        finally
        {
            DeferredArrayMaterializer.Remove(backing);
            DeferredArrayMaterializer.Remove(loss.DataVector);
        }
    }

    private static void AssertContainsTensorKeys(HashSet<object> protect, Tensor<float> tensor)
    {
        Assert.Contains(tensor.DataVector, protect);
        var backing = tensor.DataVector.GetBackingArrayUnsafe();
        if (backing is not null)
            Assert.Contains(backing, protect);
    }

    private static CompiledTrainingPlan<float> CreatePlan(
        Tensor<float> loss,
        Tensor<float>[] gradients,
        Tensor<float> lossGradSeed,
        Tensor<float>? lossGradDest)
    {
#pragma warning disable SYSLIB0050
        var plan = (CompiledTrainingPlan<float>)FormatterServices.GetUninitializedObject(typeof(CompiledTrainingPlan<float>));
#pragma warning restore SYSLIB0050
        SetField(plan, "_lossOutput", loss);
        SetField(plan, "_gradients", gradients);
        SetField(plan, "_lossGradSeed", lossGradSeed);
        SetField(plan, "_lossGradDest", lossGradDest);
        return plan;
    }

    private static object? Invoke(CompiledTrainingPlan<float> plan, string methodName)
    {
        var method = typeof(CompiledTrainingPlan<float>).GetMethod(methodName, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Method not found: {methodName}");
        return method.Invoke(plan, null);
    }

    private static void SetField(CompiledTrainingPlan<float> plan, string fieldName, object? value)
    {
        var field = typeof(CompiledTrainingPlan<float>).GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Field not found: {fieldName}");
        field.SetValue(plan, value);
    }
}
