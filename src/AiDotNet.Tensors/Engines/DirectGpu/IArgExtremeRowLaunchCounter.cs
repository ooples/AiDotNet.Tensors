// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A backend whose argmax / argmin reduce a row at least <see cref="ArgExtremeRows.MinWidth"/> wide with one work group
/// per row instead of one work item per row, and count those launches (so a test can tell the row kernel ran).
/// </summary>
internal interface IArgExtremeRowLaunchCounter
{
    /// <summary>Work-group-per-row argmax / argmin launches so far.</summary>
    long ArgExtremeRowLaunches { get; }
}

/// <summary>Shared limits of the work-group-per-row argmax / argmin kernels.</summary>
internal static class ArgExtremeRows
{
    /// <summary>The row width from which the work-group-per-row kernel runs: narrower rows keep one work item per row,
    /// which is already coalesced across rows and pays no local-memory reduction.</summary>
    internal const int MinWidth = 512;
}
