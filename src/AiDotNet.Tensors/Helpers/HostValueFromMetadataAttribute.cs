// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Marks an engine operation that returns a host value computed from a tensor's metadata only (its element type, rank
/// or shape), never from its data. Shapes are static in a graph trace, so such a value is the same on every replay and
/// the operation needs no <c>BeginHostScalarRead</c> boundary (analyzer rule TENS0040).
/// </summary>
[AttributeUsage(AttributeTargets.Method, Inherited = false)]
internal sealed class HostValueFromMetadataAttribute : Attribute
{
}
