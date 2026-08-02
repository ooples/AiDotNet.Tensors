using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// Owns the completeness decision and atomic publication boundary for one
/// measured autotune search. An incomplete full search may report diagnostics,
/// but it must never replace the last complete winner artifact. A probe can
/// publish an explicitly diagnostic artifact without claiming completeness.
/// </summary>
internal sealed class CodegenAutotuneEvidenceGate
{
    private readonly List<string> _inconclusivePromotableCandidates = new();
    private readonly List<string> _failures = new();

    internal IReadOnlyList<string> InconclusivePromotableCandidates =>
        _inconclusivePromotableCandidates;

    internal IReadOnlyList<string> Failures => _failures;

    internal bool IsComplete =>
        _inconclusivePromotableCandidates.Count == 0 && _failures.Count == 0;

    internal void RecordInconclusivePromotableCandidate(string candidateName)
    {
        if (!_inconclusivePromotableCandidates.Contains(candidateName))
            _inconclusivePromotableCandidates.Add(candidateName);
    }

    internal void RecordFailure(string failure)
    {
        if (!_failures.Contains(failure))
            _failures.Add(failure);
    }

    internal void CommitArtifact(
        string outputPath,
        string contents,
        bool requireCompleteSearch = true)
    {
        if (_failures.Count != 0 ||
            (requireCompleteSearch && _inconclusivePromotableCandidates.Count != 0))
            throw new InvalidOperationException(
                DescribeIncompleteSearch(requireCompleteSearch));

        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        int processId;
#if NET5_0_OR_GREATER
        processId = Environment.ProcessId;
#else
        using (Process process = Process.GetCurrentProcess())
            processId = process.Id;
#endif
        string temporaryOutput = outputPath + ".tmp-" +
            processId.ToString(CultureInfo.InvariantCulture) + "-" +
            Guid.NewGuid().ToString("N");
        try
        {
            File.WriteAllText(temporaryOutput, contents);
#if NET5_0_OR_GREATER
            File.Move(temporaryOutput, outputPath, overwrite: true);
#else
            if (File.Exists(outputPath))
                File.Replace(temporaryOutput, outputPath, null);
            else
                File.Move(temporaryOutput, outputPath);
#endif
        }
        finally
        {
            if (File.Exists(temporaryOutput))
                File.Delete(temporaryOutput);
        }
    }

    private string DescribeIncompleteSearch(bool requireCompleteSearch)
    {
        var reasons = new List<string>();
        if (_failures.Count != 0)
            reasons.Add(
                _failures.Count.ToString(CultureInfo.InvariantCulture) +
                " selected kernel(s) failed autotuning: " + string.Join("; ", _failures));
        if (requireCompleteSearch && _inconclusivePromotableCandidates.Count != 0)
            reasons.Add(
                "promotable candidate timing did not stabilize: " +
                string.Join(", ", _inconclusivePromotableCandidates) +
                "; the selected search is incomplete");
        return string.Join("; ", reasons) +
            "; no winner artifact written; the previous artifact was preserved";
    }
}
