<#
.SYNOPSIS
  Generates an RSA-2048 signing keypair for AiDotNet enterprise licensing and
  prints the PUBLIC key in the exact "{modulusBase64}:{exponentBase64}" form the
  client embeds (RSAParameters layout — matches SignedEntitlement /
  LicenseResponseVerifier import code byte-for-byte).

.DESCRIPTION
  RUN THIS IN A SECURE ENVIRONMENT (a clean operator workstation or, better, an
  HSM-backed host). The PRIVATE key it writes is the entire trust anchor for that
  signing purpose — anyone who obtains it can mint valid entitlements / sign
  responses. Move it into your HSM or secret manager and DELETE the local copy.
  NEVER commit it. The matching .gitignore in this folder blocks the default
  output names, but do not rely on that alone.

  Uses .NET's own RSA so the exported public key is guaranteed to import cleanly
  via RSA.ImportParameters on both net471 and net10.0.

.PARAMETER Purpose
  'entitlement' (paste into SignedEntitlement.EntitlementPublicKey) or
  'response'    (paste into LicenseResponseVerifier.ResponseSigningPublicKey).
  These MUST be two DIFFERENT keypairs (compromise isolation).

.PARAMETER OutDir
  Directory to write the private key to. Defaults to the SCRIPT's directory
  (build/keys/), which is covered by .gitignore — so a run with no -OutDir
  argument cannot accidentally drop the private PEM into a parent folder that
  the operator might subsequently `git add`. Override with an explicit path
  outside the repo (preferably on encrypted storage) for production use.

.EXAMPLE
  pwsh ./generate-signing-keys.ps1 -Purpose entitlement -OutDir C:\secure
  pwsh ./generate-signing-keys.ps1 -Purpose response    -OutDir C:\secure
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('entitlement', 'response')]
    [string]$Purpose,
    [string]$OutDir = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'

# Ensure the (possibly operator-supplied) output directory exists before we
# try to write the private PEM into it. Idempotent when it already exists.
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$rsa = [System.Security.Cryptography.RSA]::Create()
if ($rsa.KeySize -ne 2048) { $rsa.KeySize = 2048 }

$pub = $rsa.ExportParameters($false)
$embed = [Convert]::ToBase64String($pub.Modulus) + ':' + [Convert]::ToBase64String($pub.Exponent)

# Export the PRIVATE key as PKCS#8 PEM for transfer into your HSM/secret store.
$privPath = Join-Path $OutDir "aidotnet-$Purpose-signing-PRIVATE.pem"
# Refuse to clobber an existing private key — a silent overwrite on an
# accidental rerun would destroy the only local copy of the active signing key
# (and effectively rotate it while the matching public key is still embedded
# and in production use). The operator must move the prior PEM into their
# HSM/secret store and delete it from disk before regenerating.
if (Test-Path -LiteralPath $privPath) {
    throw "Refusing to overwrite existing private key at $privPath. Move/delete it first if you intend to regenerate."
}
$pkcs8 = $rsa.ExportPkcs8PrivateKey()
$pem = "-----BEGIN PRIVATE KEY-----`n" +
       ([Convert]::ToBase64String($pkcs8, 'InsertLineBreaks')) +
       "`n-----END PRIVATE KEY-----`n"
[System.IO.File]::WriteAllText($privPath, $pem)

$constName = if ($Purpose -eq 'entitlement') {
    'SignedEntitlement.EntitlementPublicKey'
} else {
    'LicenseResponseVerifier.ResponseSigningPublicKey'
}

Write-Host ""
Write-Host "=== AiDotNet $Purpose signing keypair generated ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "PUBLIC KEY (embed this — paste as the value of $constName):" -ForegroundColor Green
Write-Host ""
Write-Host "  `"$embed`""
Write-Host ""
Write-Host "PRIVATE KEY written to: $privPath" -ForegroundColor Yellow
Write-Host ""
Write-Host "  !! MOVE this into your HSM / secret manager and DELETE the local file." -ForegroundColor Red
Write-Host "  !! NEVER commit it. Anyone with it can forge $Purpose signatures." -ForegroundColor Red
Write-Host ""
Write-Host "Next:" -ForegroundColor Cyan
if ($Purpose -eq 'entitlement') {
    Write-Host "  - Replace the placeholder in src/AiDotNet.Tensors/Licensing/SignedEntitlement.cs"
    Write-Host "  - Sign customer entitlements with the private key (see build/LicenseValidator/README.md)."
} else {
    Write-Host "  - Replace the placeholder in src/AiDotNet.Tensors/Licensing/LicenseResponseVerifier.cs"
    Write-Host "  - Load the private key into the Supabase Edge Function (see"
    Write-Host "    docs/enterprise/license-server-response-signing.md)."
}
Write-Host ""

$rsa.Dispose()
