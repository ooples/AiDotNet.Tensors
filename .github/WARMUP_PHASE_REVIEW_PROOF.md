# Linear warmup phase and checkpoint proof

## Scope and compatibility

Base: `5d22aa7f4aec3f0a2d0b14953d3d398333f376dd` (main). The baseline
`LrSchedule.cs` Git blob is `ac6f84342ac475dece1ae8b071bce80adecdb1ef`, identical
to the file shipped at tag `v0.130.2`. Published-package tests use NuGet
`AiDotNet.Tensors` **0.130.2**, not a locally substituted package.

The endpoint belongs to the decay phase. With peak 1, initial rate 0.2, four
warmup updates and endpoint 0.9, the second update must use **0.4**, not **0.9**.
The fix retains the built-in fused schedule; it does not select an eager fallback
or a custom adapter that cannot be checkpointed.

| Serialized kind | Meaning |
| --- | --- |
| 9 | Existing fused checkpoint: preserve the historical warmup floor and zero-warmup first-step peak. |
| 10 | New phased schedule: warmup is not floored at the later decay endpoint. |
| 11 | Legacy eager checkpoint bridge: preserve the saved initial rate even when warmup is zero, followed by the historical floor. |

Kind 11 is an internal friend-assembly bridge for AiDotNet's checkpoint restore,
not a public recipe option. Existing kind 9 math is unchanged. The binary
serializer stores these kinds explicitly; unknown kinds fail closed. The checks
cover required array lengths, warmup/horizon constraints and decay-mode values
for new kinds. **Finite-double rate validation was already absent from the
factory/checkpoint contract and is not fixed or claimed by this change.**

## Actual before and after

All executions were CPU-only on Windows; no GPU numerical run or package-release
success is claimed. The selected cohort contains 40 existing `LrScheduleTests`
and 33 new phase/checkpoint controls. No skipped tests or widened tolerances.

| Actual assembly | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Published 0.130.2, final cohort, net10 | 55 | 18 | 0 |
| First phased fix before legacy-eager representation, net10 | 67 | 6 | 0 |
| Final source, net10 | 73 | 0 | 0 |
| Final source, net8 | 73 | 0 | 0 |
| Final source, net471 | 73 | 0 | 0 |
| Independent final source replay, net10 | 73 | 0 | 0 |

The 18 published-package failures are 12 numerical/validation controls plus six
new legacy-eager wire-format controls that the older reader correctly rejects.
`PUBLISHED_TENSORS` omits only calls to the newly added internal factory during
the published-package build; the same 73 cases still execute. Each legacy wire
case fails before reaching its after-only factory assertions on the old package.

The compiled-plan controls execute an actual squared-loss SGD plan, warm it with
kind 9, reconfigure it to kind 10, then restore both binary optimizer states.
Parameter updates must follow each representation's analytical rates. This
detects stale schedule reuse, not just matching metadata. The compiled plan
resolves its captured schedule per update; CUDA graph replay runs optimizer
scalar evaluation outside the captured graph. No GPU kernel or capture policy
was changed.

All final three-framework builds completed with zero errors and zero warnings.
Net10 execution took 133 ms, net8 201 ms and net471 761 ms (test execution times,
excluding discovery/startup). The first 67-case net10 snapshot was independently
reviewed and replayed before the separate six-case legacy-eager addition. A
second independent review covered the final two-file production diff and the
six added cases; its exact-hash final net10 replay passed all 73 cases in 131 ms.

TRX files retained locally:

- `artifacts/warmup-before-final/results/published-final-before.trx`
- `artifacts/warmup-after/results/legacy-eager-representation-before.trx`
- `artifacts/warmup-legacy-after/results/warmup-final-net10.trx`
- `artifacts/warmup-legacy-after/results/warmup-legacy-after-net8.trx`
- `artifacts/warmup-legacy-after/results/warmup-legacy-after-net471.trx`
- `artifacts/warmup-legacy-after/results/warmup-final-root-independent.trx`

## Reproduction

Run from this worktree in PowerShell with the required SDK/framework installed.
The published negative control must produce exactly 18 failures and 55 passes.
No unpublished package is placed in the global NuGet cache.

```powershell
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$project = 'tests/AiDotNet.Tensors.WarmupReview/AiDotNet.Tensors.WarmupReview.csproj'
$common = @('-c', 'Release', '-m:1', '-p:UseSharedCompilation=false',
    '-p:CopyLocalRuntimeTargetAssets=false', '-p:_GetChildProjectCopyToOutputDirectoryItems=false',
    '-p:GeneratePackageOnBuild=false', '-v:quiet', '--nologo')
dotnet build $project @common -f net10.0 --artifacts-path artifacts/reproduce-before -p:UsePublishedTensors=true
if ($LASTEXITCODE -ne 0) { throw 'Published-package test build failed.' }
dotnet vstest artifacts/reproduce-before/bin/AiDotNet.Tensors.WarmupReview/release_net10.0/AiDotNet.Tensors.Tests.dll `
    '/Logger:trx;LogFileName=before.trx' /ResultsDirectory:artifacts/reproduce-before/results
if ($LASTEXITCODE -ne 1) { throw 'Expected the published-package regression failures.' }
[xml]$before = Get-Content artifacts/reproduce-before/results/before.trx
$counts = $before.TestRun.ResultSummary.Counters
if ($counts.passed -ne '55' -or $counts.failed -ne '18' -or $counts.total -ne '73') {
    throw 'Unexpected published-package baseline; inspect every failure.'
}
foreach ($tfm in @('net10.0', 'net8.0', 'net471')) {
    dotnet build $project @common -f $tfm --artifacts-path artifacts/reproduce-after
    if ($LASTEXITCODE -ne 0) { throw "Source build failed for $tfm." }
    dotnet vstest "artifacts/reproduce-after/bin/AiDotNet.Tensors.WarmupReview/release_$tfm/AiDotNet.Tensors.Tests.dll" `
        "/Logger:trx;LogFileName=after-$tfm.trx" /ResultsDirectory:artifacts/reproduce-after/results
    if ($LASTEXITCODE -ne 0) { throw "Runtime test failure for $tfm." }
    [xml]$after = Get-Content "artifacts/reproduce-after/results/after-$tfm.trx"
    $counts = $after.TestRun.ResultSummary.Counters
    if ($counts.passed -ne '73' -or $counts.failed -ne '0' -or $counts.executed -ne '73') {
        throw "Incomplete successful cohort for $tfm."
    }
}
```

## Frozen SHA-256

| TFM | Core `AiDotNet.Tensors.dll` | Test `AiDotNet.Tensors.Tests.dll` |
| --- | --- | --- |
| net10 | `DFB17C5146B89075EF2C364792D44338E7AA1124632138446B1EC359ABCB3A12` | `FCD7328D5A2835E7B06ADCF29DBF632AECD6B86CB543D583BF9B52C52E1FDE4C` |
| net8 | `95185139FC9CAFE0C37EC3CB01C2E2E9C098B8ECF92C3CE82146CC4E80D52D78` | `85C6AF79DB6E0F1B787EB2E588A1F10C8D308B1800A2F53C73E1B62A0BA24472` |
| net471 | `0758BC303EB286FCFE021F8E9FB1E771FEEE215973BD7D3DFF8AB74438488357` | `C72ECD740EF79E47EF03EB8A14FA2074577FE5EDEE849CC5B13600835CDAEFCB` |

The companion is source/runtime proof, not published integration proof. Finch's
eager scheduler, versioned state restore and fused mapping require their own
AiDotNet validation and a stable Tensors release before the downstream package
update can be called complete. No merge, release, tag or remote policy change
was performed for this evidence.
