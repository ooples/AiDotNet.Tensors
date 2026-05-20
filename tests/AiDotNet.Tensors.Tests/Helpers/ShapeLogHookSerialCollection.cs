// Copyright (c) AiDotNet. All rights reserved.

using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// xunit collection marker for tests that mutate the process-wide static
/// <c>BlasProvider.ShapeLogHook</c>. Members of this collection run serially
/// with respect to each other (xunit default: tests in the same collection
/// never run in parallel). Use this to guard hook-ownership assertions
/// like <c>Assert.Null(BlasProvider.ShapeLogHook)</c> after dispose, which
/// would otherwise race against any sibling test class that installs its
/// own hook.
///
/// <para>PR #412 CodeRabbit review fix — applied to
/// <see cref="ShapeInstrumenterTests"/>. Add this attribute to any future
/// test class that touches <c>BlasProvider.ShapeLogHook</c>.</para>
/// </summary>
[CollectionDefinition("ShapeLogHookSerial", DisableParallelization = true)]
public sealed class ShapeLogHookSerialCollection
{
    // Marker only — no fixture needed.
}
