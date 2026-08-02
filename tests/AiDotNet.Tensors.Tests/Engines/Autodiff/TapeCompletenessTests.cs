using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Reflection-based tests ensuring every IEngine method that produces a tensor is classified
/// in OpRegistry. Fails CI when a new method is added without classification.
/// </summary>
/// <remarks>
/// A method can produce a tensor two ways: by RETURNING one, or by emitting one through an
/// <c>out</c>/<c>ref</c> parameter. Both are covered — historically only the first was, which let
/// STFT, FFT, IFFT, FFT2D, IFFT2D, FFTND, GroupNormInto and ProjectGaussians3DTo2D sit in no
/// registry set at all.
/// </remarks>
public class TapeCompletenessTests
{
    [Fact]
    public void AllTensorReturningMethods_AreClassified()
    {
        // Get all methods on IEngine that return Tensor<T> (generic)
        var engineType = typeof(IEngine);
        var methods = engineType.GetMethods(BindingFlags.Public | BindingFlags.Instance);

        var unclassified = new List<string>();

        foreach (var method in methods)
        {
            // Check if return type is Tensor<T> or a tuple containing Tensor<T>
            if (!ReturnsTensor(method.ReturnType))
                continue;

            // Skip property getters
            if (method.IsSpecialName)
                continue;

            var name = method.Name;

            // Strip generic arity suffix if present (e.g., TensorAdd`1 -> TensorAdd)
            if (name.Contains('`'))
                name = name.Substring(0, name.IndexOf('`'));

            if (!OpRegistry.IsClassified(name))
            {
                unclassified.Add($"{name} (returns {method.ReturnType.Name})");
            }
        }

        if (unclassified.Count > 0)
        {
            var message = $"The following {unclassified.Count} IEngine Tensor-returning methods are not classified in OpRegistry.\n" +
                          "Add each method to DifferentiableOps, NonDifferentiableOps, or DelegatorOps:\n" +
                          string.Join("\n", unclassified.Select(u => $"  - {u}"));
            Assert.Fail(message);
        }
    }

    [Fact]
    public void OpRegistry_HasNoDuplicates()
    {
        var all = new List<string>();
        all.AddRange(OpRegistry.DifferentiableOps);
        all.AddRange(OpRegistry.NonDifferentiableOps);
        all.AddRange(OpRegistry.DelegatorOps);

        var duplicates = all.GroupBy(x => x)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        if (duplicates.Count > 0)
        {
            Assert.Fail($"OpRegistry has {duplicates.Count} ops in multiple categories: {string.Join(", ", duplicates)}");
        }
    }

    private static bool ReturnsTensor(Type returnType)
    {
        if (returnType.IsGenericType)
        {
            var genericDef = returnType.GetGenericTypeDefinition();
            if (genericDef == typeof(Tensor<>))
                return true;
        }

        // Check for Tensor<> in open generic context (interface methods)
        if (returnType.Name.StartsWith("Tensor`"))
            return true;

        // Check for ValueTuple containing Tensor<> (e.g., (Tensor<T>, Tensor<T>))
        if (returnType.IsGenericType && returnType.FullName is not null && returnType.FullName.StartsWith("System.ValueTuple"))
        {
            foreach (var arg in returnType.GetGenericArguments())
            {
                if (ReturnsTensor(arg))
                    return true;
            }
        }

        // Check for Tensor<>[] array returns
        if (returnType.IsArray && returnType.GetElementType() is Type elemType)
        {
            return ReturnsTensor(elemType);
        }

        return false;
    }

    /// <summary>
    /// Companion to <see cref="AllTensorReturningMethods_AreClassified"/> for ops that emit their
    /// tensors through <c>out</c>/<c>ref</c> parameters instead of returning them.
    /// </summary>
    /// <remarks>
    /// Without this, such an op can silently record nothing on the tape and no test notices —
    /// which is how STFT, FFT, IFFT, FFT2D, IFFT2D, FFTND, GroupNormInto and
    /// ProjectGaussians3DTo2D all ended up unclassified.
    /// </remarks>
    [Fact]
    public void AllTensorOutParameterMethods_AreClassified()
    {
        var methods = typeof(IEngine).GetMethods(BindingFlags.Public | BindingFlags.Instance);
        var unclassified = new List<string>();

        foreach (var method in methods)
        {
            if (method.IsSpecialName)
                continue;

            bool emitsTensorByRef = method.GetParameters().Any(p =>
                (p.IsOut || p.ParameterType.IsByRef) && ReturnsTensor(
                    p.ParameterType.IsByRef ? p.ParameterType.GetElementType()! : p.ParameterType));

            if (!emitsTensorByRef)
                continue;

            var name = method.Name;
            if (name.Contains('`'))
                name = name.Substring(0, name.IndexOf('`'));

            if (!OpRegistry.IsClassified(name))
                unclassified.Add($"  - {name}");
        }

        if (unclassified.Count > 0)
        {
            var message =
                $"The following {unclassified.Distinct().Count()} IEngine method(s) emit a Tensor through an " +
                "out/ref parameter but are not classified in OpRegistry.\n" +
                "Add each to DifferentiableOps, NonDifferentiableOps, or DelegatorOps:\n" +
                string.Join("\n", unclassified.Distinct());
            Assert.Fail(message);
        }
    }
}
