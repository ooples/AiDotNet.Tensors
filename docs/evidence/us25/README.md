# US-25 Tensors verification

Production revision: `a33e6ef4604c47237f9fd83f467946ab29f61ac2`.
Date: September 14, 2026. No paid model calls or GPU performance claims.

Archive: [verification.zip](verification.zip), 1,058,509 bytes, SHA-256 `f60fb327601b1a057942ed4afac0410fcbd6338e26a5116ff7f8d97196bcd1ee`.

| Final local gate | Passed / total | Skipped | Autotune line / branch coverage |
| --- | --- | --- | --- |
| Windows .NET 10, normal package dependency | 223 / 223 | 0 | 84.70% / 68.49% |
| Windows .NET 8, normal package dependency | 223 / 223 | 0 | 84.70% / 68.49% |
| Windows .NET Framework 4.7.1, normal package dependency | 223 / 223 | 0 | 84.90% / 69.37% |

`verification.zip` retains TRX, coverage XML and console logs, including failures. Final directories are `net10.0-final`, `net8.0-final` and `net471-corrected`; earlier directories are historical attempts, not additional final-gate counts. The Framework source was validated before its identical contents were committed as `a33e6ef4`; the other two targets were then verified on that commit.

The downloaded public NuGet `AiDotNet.Evolution/0.1.0-preview.1` package SHA-512 was `14BC94F0CC2459F62B309423D80A601511C1123CB7B645C1D9C463CC1799095773DFFD338355CA60387C82B462FAA9EFEEA04D1CF1CB0EBCA7760A290E0BB548`. Its three framework DLLs matched the restored cache byte-for-byte. No shared package-cache files were changed.

## Hosted evidence

[Initial source-integration run 34891977626](https://github.com/ooples/AiDotNet.Tensors/actions/runs/34891977626) at `000dc747` passed 223/223 on each Linux target, exercising native artifact durability and rollback. Its Windows job failed with the nullable compatibility error fixed in `a33e6ef4`. The archive retains both successful Linux TRX/coverage receipts; the failed run is **not** claimed as a passing overall CI gate.

The [source-integration workflow](../../../.github/workflows/evolution-lifecycle.yml) pins Evolution source `255feb24369702a32ea9db7a3f8a0b7a847d2762`, references the actual Tensors project and runs on Linux .NET 10/.NET 8 and Windows Framework. Check the [PR's current-head checks](https://github.com/ooples/AiDotNet.Tensors/pull/1030/checks) for the post-fix hosted result. Normal package-based full-suite CI is unchanged and remains a separate merge check.

## Adversarial failures retained

- `eba4c597`: first .NET 10 run passed 221/223; corrupt/incompatible rollback artifacts escaped an `IOException` filter because `InvalidDataException` requires an explicit catch. `000dc747` fixes both paths. The already-failed run's whole-library coverage collector was stopped after the test host exited and collection continued consuming CPU; resulting transport/blame errors are retained. Scoped autotune coverage completed normally thereafter.
- `000dc747`: Framework compile failed with CS8604 after extracting the directory-barrier loop. `a33e6ef4` makes the null guard explicit; all three local targets pass with that source.

The tests cover applicability changes, content tampering/duplicate JSON, immutable policy authorization, stale promotion, drift/coalescing, budget inflation/exhaustion, abandoned capacity retention, all contributing regression windows, healthy-window reset, invalid prior fallback and evidence-loss deactivation. These are correctness/lifecycle proofs, not statistical superiority benchmarks. See [runtime and durability boundaries](../../evolution-artifact-lifecycle.md).

The AiDotNet program/AutoML companion and cross-story prerequisites remain outside this Tensors PR; Evolution issue #43 stays open.
