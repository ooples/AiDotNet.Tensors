# Deterministic fixture isolation review proof

The fixture now saves the thread-local override, clears it, and only then reads
the unmasked process-global deterministic setting. Teardown restores the two
independent values. This change does not modify production GEMM or GPU code.

## Actual before and after

The same six regression cases construct and dispose the real
`DeterministicStrategySelectionTests` fixture, covering both global values and
all three nullable thread-local values. Against the original fixture at
`fb7d68228110c5e24130b5888bc0a5be89500306`, four passed and two failed: the
disagreeing overrides were incorrectly written back to the global setting.

After the fixture correction, all six passed. The focused project also runs
the existing thirteen deterministic-cache/GEMM tests against an actual project
reference to AiDotNet.Tensors, not a replacement implementation.

| Runtime | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| .NET 10 | 19 | 0 | 0 |
| .NET 8 | 19 | 0 | 0 |
| .NET Framework 4.7.1 | 19 | 0 | 0 |

All three builds completed with zero errors and zero warnings. This is a
focused CPU correctness/isolation check, not full CI or a GPU performance claim.

## Reproduce

Run from the repository root, substituting `net8.0` or `net471` for the other
frameworks (the .NET Framework run requires Windows):

```powershell
dotnet build tests/AiDotNet.Tensors.DeterministicReview/AiDotNet.Tensors.DeterministicReview.csproj -c Release -f net10.0 -m:1 -p:UseSharedCompilation=false -p:CopyLocalRuntimeTargetAssets=false -p:GeneratePackageOnBuild=false -v:quiet
dotnet test tests/AiDotNet.Tensors.DeterministicReview/AiDotNet.Tensors.DeterministicReview.csproj -c Release -f net10.0 --no-build --no-restore --logger 'trx;LogFileName=scope-and-gemm-after-net10.trx' --results-directory artifacts/pr1034-review -v:quiet
```

Local evidence is retained under `artifacts/pr1034-review`: `scope-before.trx`
and `scope-and-gemm-after-net10.trx`, `scope-and-gemm-after-net8.0.trx`, and
`scope-and-gemm-after-net471.trx`. The frozen before runner is under `before/`.

The .NET 10 production DLL was identical before and after:
`D0358327AA1CF076956A14423CE6371268DA4493B0EA685C45184D7200F53121`.
Before test DLL: `61B4D478ACDB2B7B984A45A5D7A131266FBA294D1461C33F07C07CE5618CD549`.
After test DLL: `422072CA9E2B23FCBCDCD93158243A9131716D742D473CFD05884D31AE422664`.
