# Rectangular attention validation

This runner compiles the actual Tensors library and links 34 new regression cases plus
58 existing standard, tiled, generic, precision and backward attention cases. It does
not replace the attention implementation with a stub or disable fused dispatch.

## Defect and preserved contract

Ordinary noncausal cross-attention with query offset zero permits more queries than
keys. The old unconditional `queryOffset + queryLength > keyLength` check incorrectly
rejected that geometry and could overflow for a large positive offset. The same check
existed in the standard fused facade, direct tiled implementation and generic
forward/backward validator. All three now use a subtraction-safe bound only for
causal attention or an explicitly nonzero query offset. Negative offsets remain invalid.

The existing explicit-offset fitting-window contract is preserved even for noncausal
attention. Existing positive dimension, rank, batch/head, bias and key/value checks
remain in place. Kernel arithmetic, dispatch thresholds and fused implementations are
unchanged; no timing or GPU-throughput improvement is claimed.

## Actual before and after

Baseline: fresh `origin/main` commit
`5d22aa7f4aec3f0a2d0b14953d3d398333f376dd`.
Baseline net10 core SHA-256:
`EC526D0060E5C65E4A14B7F7619339A58F8C86922BD86E70DF83169DA5DDA3AB`.

The final 34 regression cases were run against that unchanged original core, not just
an early test draft: **17 passed, 17 failed** in
`artifacts/rectangular-attention-review/rectangular-final34-original-head-before.trx`.
The unchanged final test DLL SHA-256 was
`2D0B90F5A8F0AAE897E4AA010C6FFAF4E6F554005E640462632D4E43506F6326`.
Its failures expose rectangular-query rejection and positive-offset integer overflow
across the three actual implementations. The original core was copied only into this
owned runner output for that control, and the corrected core was restored afterward;
the global NuGet cache was never altered.

| Target framework | Core build errors/warnings | Passed | Failed | Skipped | After TRX |
| --- | --- | --- | --- | --- | --- |
| net10.0 | 0 / 0 | 92 | 0 | 0 | `rectangular-complete-net10.0.trx` |
| net8.0 | 0 / 0 | 92 | 0 | 0 | `rectangular-complete-net8.0.trx` |
| net471 | 0 / 0 | 92 | 0 | 0 | `rectangular-complete-net471.trx` |

All TRXs are under `artifacts/rectangular-attention-review`. The focused runner has
one existing unused-variable warning in the unchanged `FlashAttention2Tests`; this is
not presented as a warning-free test build. Tests explicitly select the CPU engine
and run without collection parallelism. These are real CPU value/gradient tests, not
physical GPU execution evidence.

After core SHA-256 values:

- net10.0: `CF366B1B208EA6342AF377406CAF67DA4768F54EDD17C6C3BA18B908C59D771F`
- net8.0: `F0124656CD4869A224315545FA7BEACFA280A71FE342E9ECF934A7F774C4BEAC`
- net471: `3D53048B9C8EA70A6FEF5218675CB9560D1FF604EEB6180855DB1FA0E81542B8`

After test DLL SHA-256 values:

- net10.0: `2D0B90F5A8F0AAE897E4AA010C6FFAF4E6F554005E640462632D4E43506F6326`
- net8.0: `9791BC7448C0766A1BBD24680B15F6AF21F61F135538891489FF5BACBB3FEA60`
- net471: `B32A575C783F5158EB2E92658F96FC1E7D7B344E4249BB7DF7CF52CBDB8A8FC7`

An independent reviewer read all three production changes and the complete new test
matrix, then replayed the same net10 core/test binaries: **92 passed, zero failed or
skipped**, in `results/rectangular-root-independent.trx` under the artifact directory.

The consuming AiDotNet PR2136 runner also passed all ten direct framework/dependency
attention controls against this exact local net10 Tensors DLL. Its TRX records the
actually loaded path and `CF366B1B...` hash. The larger MGIE run was **44 passed / one
failed**: the remaining failure is a separate injected-component clone-configuration
defect, not claimed fixed here. Consumer core SHA-256:
`9A33FF8B6AD3D9DA392C98EA9FAAEE94D0BAA835494946F8C346A1E2E85D05E9`;
consumer evidence: `artifacts/pr2136-mgie-review/pr2136-mgie-local-tensors-net10.trx`
in the AiDotNet PR2136 worktree.

The new cases exercise rank-three and rank-four inputs, multiple batches/heads,
query lengths 3, 77 and 129 versus two keys, independently calculated stable-softmax
values and analytic dQ/dK/dV, unchanged input values, explicit weights, tiled and
generic paths, exact-end causal windows, negative offsets and `int.MaxValue` offsets.
Earlier partial runs are retained as diagnostics; the complete results above supersede
them and include the existing offset-contract controls that caught an overbroad first
predicate draft.

## Reproduce after validation

Run in this repository with the .NET SDK and Windows .NET Framework 4.7.1 runtime
available. Package restore is explicit; every build is checked before any no-build test.

```powershell
$ErrorActionPreference = 'Stop'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$runner = 'tests/AiDotNet.Tensors.RectangularAttentionReview/AiDotNet.Tensors.RectangularAttentionReview.csproj'
dotnet restore $runner
if ($LASTEXITCODE -ne 0) { throw 'Restore failed; do not test stale binaries.' }

foreach ($tfm in @('net10.0', 'net8.0', 'net471')) {
    dotnet build src/AiDotNet.Tensors/AiDotNet.Tensors.csproj -c Release -f $tfm --no-restore `
        -p:GeneratePackageOnBuild=false -p:CopyLocalRuntimeTargetAssets=false `
        -p:CopyLocalLockFileAssemblies=false -m:1 -v:q
    if ($LASTEXITCODE -ne 0) { throw "Core build failed: $tfm" }

    dotnet build $runner -c Release -f $tfm --no-restore -p:BuildProjectReferences=false `
        -p:GeneratePackageOnBuild=false -p:CopyLocalRuntimeTargetAssets=false -m:1 -v:q
    if ($LASTEXITCODE -ne 0) { throw "Runner build failed: $tfm" }

    dotnet test $runner -c Release -f $tfm --no-build --no-restore `
        --logger "trx;LogFileName=rectangular-reproduction-$tfm.trx" `
        --results-directory artifacts/rectangular-attention-review
    if ($LASTEXITCODE -ne 0) { throw "Tests failed: $tfm" }
}
```

AiDotNet's corresponding MGIE replay uses this local corrected assembly as an explicit
unpublished dependency. A successful local replay does not mean a stable NuGet package
has been released or that the consuming PR can merge before that dependency is available.
