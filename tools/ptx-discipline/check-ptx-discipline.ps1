<#
.SYNOPSIS
    Direct-PTX discipline guard (Windows entry point).

.DESCRIPTION
    Thin wrapper around check-ptx-discipline.sh so there is a single source of
    truth for the rules (the bash script) and no risk of two implementations
    drifting apart. Requires Git Bash (bundled with Git for Windows), which the
    repository's other *.sh tooling already assumes.

    Run from anywhere:
        pwsh tools/ptx-discipline/check-ptx-discipline.ps1

    Exit code 0 = all checks pass; non-zero = a violation was found.
#>
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $here 'check-ptx-discipline.sh'

$bash = (Get-Command bash -ErrorAction SilentlyContinue)?.Source
if (-not $bash) {
    $candidate = 'C:\Program Files\Git\bin\bash.exe'
    if (Test-Path $candidate) { $bash = $candidate }
}
if (-not $bash) {
    Write-Error 'bash was not found. Install Git for Windows (Git Bash) to run the direct-PTX discipline guard.'
    exit 2
}

& $bash $script
exit $LASTEXITCODE
