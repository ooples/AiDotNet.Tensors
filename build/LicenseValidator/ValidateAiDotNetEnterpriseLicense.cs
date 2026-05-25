using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;
using System.Runtime.CompilerServices;
using Org.BouncyCastle.Asn1;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Security;

[assembly: InternalsVisibleTo("AiDotNet.Tensors.LicenseValidator.Tests")]

namespace AiDotNet.Tensors.LicenseValidator;

/// <summary>
/// MSBuild task that validates an Ooples AiDotNet Enterprise license file before allowing the build
/// to define <c>AIDOTNET_DISABLE_TELEMETRY</c> or <c>AIDOTNET_DISABLE_LICENSE_GUARD</c> compile flags.
/// </summary>
/// <remarks>
/// <para>
/// Wired up via <see href="../AiDotNet.Tensors.Enterprise.targets"/>. Replaces the Phase-0 placeholder
/// marker-string check with the real RSA-2048 PKCS#1 v1.5 SHA-256 signature verification originally
/// drafted inline in the targets file (rejected by some MSBuild versions when expressed as a
/// <c>RoslynCodeTaskFactory</c> inline task — hence this precompiled assembly).
/// </para>
/// <para>
/// License-file schema (JSON):
/// </para>
/// <code>
/// {
///   "marker":    "AiDotNet-Enterprise-License-v1",
///   "tenant":    "ACME Corp.",
///   "issued":    "YYYY-MM-DD",
///   "expires":   "YYYY-MM-DD",
///   "scope":     ["AIDOTNET_DISABLE_TELEMETRY", "AIDOTNET_DISABLE_LICENSE_GUARD"],
///   "keyHash":   "&lt;lowercase-hex SHA-256 of the customer's secret license key&gt;",
///   "payload":   "&lt;base64 of canonical JSON without signature field&gt;",
///   "signature": "&lt;base64 RSA-2048 PKCS#1 v1.5 SHA-256 signature over payload&gt;"
/// }
/// </code>
/// <para>
/// Validation order: file exists → JSON parses → marker matches → expiry date is in the future →
/// every requested DISABLE_* flag is in the granted scope → RSA signature verifies against the
/// embedded Ooples enterprise-license public key. Any failure logs a structured MSBuild error
/// with a stable diagnostic code (AIDOTNET004 through AIDOTNET009) and aborts the build.
/// </para>
/// <para>
/// <b>Public-key replacement</b>: the constant <see cref="EnterpriseLicensePublicKeyPkcs1Base64"/>
/// currently ships as a marker placeholder (<c>PHASE4_PLACEHOLDER_*</c>). Production builds MUST
/// replace it with the real Ooples enterprise-license public key generated per the procedure in
/// <see href="README.md"/>. While the placeholder is in place, every real license is rejected with
/// AIDOTNET008 — the safe-by-default posture during rollout.
/// </para>
/// </remarks>
public sealed class ValidateAiDotNetEnterpriseLicense : Task
{
    /// <summary>
    /// Placeholder marker that callers (and the task itself) can detect to know the public key has
    /// not been replaced with the production one. The character sequence is intentionally long
    /// enough that no real PKCS#1-encoded RSA-2048 key would ever start with it.
    /// </summary>
    public const string PublicKeyPlaceholderMarker = "PHASE4_PLACEHOLDER_REPLACE_WITH_REAL_OOPLES_KEY";

    /// <summary>
    /// Base64-encoded RSA-2048 public key (PKCS#1 format, no PEM BEGIN/END wrapping).
    /// </summary>
    /// <remarks>
    /// To replace for production: generate the keypair per the README, then overwrite this constant
    /// with the contents of <c>enterprise-license-signing.pub</c> stripped of the
    /// <c>-----BEGIN/END PUBLIC KEY-----</c> markers and any newlines. The private key MUST stay in
    /// the Ooples HSM / password manager and never be committed.
    /// </remarks>
    public const string EnterpriseLicensePublicKeyPkcs1Base64 = PublicKeyPlaceholderMarker;

    /// <summary>
    /// Test-only override of the embedded public key. Production code paths read
    /// <see cref="EnterpriseLicensePublicKeyPkcs1Base64"/> directly (immutable const); the unit
    /// tests in <c>AiDotNet.Tensors.LicenseValidator.Tests</c> set this field to swap in the
    /// public half of a freshly-generated test keypair so they can drive an end-to-end
    /// sign-then-verify round trip without touching the production constant.
    /// </summary>
    internal static string? s_overridePublicKey;

    /// <summary>Returns the public key used for signature verification — override if set, else const.</summary>
    private static string ResolvePublicKey() => s_overridePublicKey ?? EnterpriseLicensePublicKeyPkcs1Base64;

    /// <summary>Absolute path to the customer's signed license file.</summary>
    [Required]
    public string LicensePath { get; set; } = string.Empty;

    /// <summary>
    /// Semicolon- or comma-delimited list of <c>DISABLE_*</c> compile flags this build requests.
    /// Every flag in the list must appear in the license's <c>scope</c> array.
    /// </summary>
    [Required]
    public string RequestedFlags { get; set; } = string.Empty;

    /// <summary>
    /// The customer's per-tenant secret license key (the <c>AiDotNetEnterpriseLicenseKey</c>
    /// MSBuild property / <c>AIDOTNET_ENTERPRISE_LICENSE_KEY</c> env var). It is bound to the
    /// signed license: the signed payload carries <c>keyHash = SHA-256(key)</c> (lowercase hex),
    /// and validation fails unless <c>SHA-256(LicenseKey)</c> matches it. This makes a leaked or
    /// redistributed (still-validly-signed) license file useless without the matching secret key.
    /// </summary>
    [Required]
    public string LicenseKey { get; set; } = string.Empty;

    /// <summary>Output: the tenant name from the validated license, surfaced for audit logging.</summary>
    [Output]
    public string Tenant { get; set; } = string.Empty;

    /// <inheritdoc/>
    public override bool Execute()
    {
        try
        {
            if (string.IsNullOrWhiteSpace(LicensePath))
            {
                Log.LogError(null, "AIDOTNET002", null, null, 0, 0, 0, 0,
                    "AiDotNet Enterprise license validation task invoked without a LicensePath value.");
                return false;
            }

            if (!File.Exists(LicensePath))
            {
                Log.LogError(null, "AIDOTNET003", null, LicensePath, 0, 0, 0, 0,
                    "AiDotNet Enterprise license file not found at: {0}.", LicensePath);
                return false;
            }

            string raw = File.ReadAllText(LicensePath);
            JsonDocument doc;
            try
            {
                doc = JsonDocument.Parse(raw);
            }
            catch (JsonException ex)
            {
                Log.LogError(null, "AIDOTNET004", null, LicensePath, 0, 0, 0, 0,
                    "AiDotNet Enterprise license at {0} is not valid JSON: {1}", LicensePath, ex.Message);
                return false;
            }

            using (doc)
            {
                var root = doc.RootElement;
                if (root.ValueKind != JsonValueKind.Object)
                {
                    Log.LogError(null, "AIDOTNET004", null, LicensePath, 0, 0, 0, 0,
                        "AiDotNet Enterprise license at {0} is not a JSON object.", LicensePath);
                    return false;
                }

                // Extract the signed payload + detached signature from the envelope.
                // SECURITY: only the bytes inside "payload" are RSA-signed. The outer
                // envelope is untrusted input — every privilege-bearing field (marker,
                // expires, scope, tenant) MUST be read from the *signed* payload below,
                // never from the envelope. Reading them from the envelope let an attacker
                // keep a valid (payload, signature) pair and re-wrap it with escalated
                // scope / extended expiry. The envelope's only trusted role is carrying
                // the payload + signature.
                if (!TryGetString(root, "payload", out var payloadB64)
                    || !TryGetString(root, "signature", out var signatureB64))
                {
                    Log.LogError(null, "AIDOTNET007", null, LicensePath, 0, 0, 0, 0,
                        "AiDotNet Enterprise license at {0} missing 'payload' or 'signature' field.",
                        LicensePath);
                    return false;
                }

                var pubKeyBase64 = ResolvePublicKey();
                if (string.Equals(pubKeyBase64, PublicKeyPlaceholderMarker, StringComparison.Ordinal))
                {
                    Log.LogError(null, "AIDOTNET008", null, LicensePath, 0, 0, 0, 0,
                        "AiDotNet Enterprise license verification refused: this build of the license validator ships with the placeholder public key (PHASE4_PLACEHOLDER_*). Production builds must replace EnterpriseLicensePublicKeyPkcs1Base64 in ValidateAiDotNetEnterpriseLicense.cs with the real Ooples key generated per build/LicenseValidator/README.md.");
                    return false;
                }

                byte[] payloadBytes, signatureBytes, pubKeyBytes;
                try
                {
                    payloadBytes = Convert.FromBase64String(payloadB64!);
                    signatureBytes = Convert.FromBase64String(signatureB64!);
                    pubKeyBytes = Convert.FromBase64String(pubKeyBase64);
                }
                catch (FormatException ex)
                {
                    Log.LogError(null, "AIDOTNET007", null, LicensePath, 0, 0, 0, 0,
                        "AiDotNet Enterprise license at {0} has invalid base64 in payload/signature/embedded-pubkey: {1}",
                        LicensePath, ex.Message);
                    return false;
                }

                RSA rsa;
                try
                {
                    rsa = ImportPkcs1RsaPublicKey(pubKeyBytes);
                }
                catch (Exception ex)
                {
                    Log.LogError(null, "AIDOTNET008", null, LicensePath, 0, 0, 0, 0,
                        "AiDotNet Enterprise license verification failed: embedded public key could not be imported ({0}). The targets file may be using a malformed key; contact admin@aidotnet.dev to verify the correct public key is shipped in this build/LicenseValidator/AiDotNet.Tensors.LicenseValidator.dll.",
                        ex.Message);
                    return false;
                }

                bool ok;
                using (rsa)
                {
                    ok = rsa.VerifyData(payloadBytes, signatureBytes,
                        HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
                }
                if (!ok)
                {
                    Log.LogError(null, "AIDOTNET008", null, LicensePath, 0, 0, 0, 0,
                        "AiDotNet Enterprise license at {0} has an invalid RSA signature. The file may be corrupt, modified, or signed with a different key. Contact admin@aidotnet.dev.",
                        LicensePath);
                    return false;
                }

                // Signature verified → the payload bytes are authentic. Parse the SIGNED
                // payload as the sole source of truth for the gating fields.
                JsonDocument payloadDoc;
                try
                {
                    payloadDoc = JsonDocument.Parse(new ReadOnlyMemory<byte>(payloadBytes));
                }
                catch (JsonException ex)
                {
                    Log.LogError(null, "AIDOTNET004", null, LicensePath, 0, 0, 0, 0,
                        "AiDotNet Enterprise license at {0} has a signed payload that is not valid JSON: {1}", LicensePath, ex.Message);
                    return false;
                }
                using (payloadDoc)
                {
                    var signed = payloadDoc.RootElement;
                    if (signed.ValueKind != JsonValueKind.Object)
                    {
                        Log.LogError(null, "AIDOTNET004", null, LicensePath, 0, 0, 0, 0,
                            "AiDotNet Enterprise license at {0} signed payload is not a JSON object.", LicensePath);
                        return false;
                    }

                    if (!TryGetString(signed, "marker", out var marker)
                        || !string.Equals(marker, "AiDotNet-Enterprise-License-v1", StringComparison.Ordinal))
                    {
                        Log.LogError(null, "AIDOTNET004", null, LicensePath, 0, 0, 0, 0,
                            "AiDotNet Enterprise license at {0} signed payload missing or wrong 'marker' field (expected 'AiDotNet-Enterprise-License-v1').",
                            LicensePath);
                        return false;
                    }

                    if (!TryGetString(signed, "expires", out var expiresRaw)
                        || !DateTime.TryParse(expiresRaw, CultureInfo.InvariantCulture,
                            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
                            out var expires))
                    {
                        Log.LogError(null, "AIDOTNET005", null, LicensePath, 0, 0, 0, 0,
                            "AiDotNet Enterprise license at {0} signed payload has missing or unparseable 'expires' field.",
                            LicensePath);
                        return false;
                    }
                    if (expires < DateTime.UtcNow.Date)
                    {
                        Log.LogError(null, "AIDOTNET005", null, LicensePath, 0, 0, 0, 0,
                            "AiDotNet Enterprise license at {0} expired on {1:yyyy-MM-dd} UTC. Contact admin@aidotnet.dev to renew.",
                            LicensePath, expires);
                        return false;
                    }

                    if (!signed.TryGetProperty("scope", out var scopeEl) || scopeEl.ValueKind != JsonValueKind.Array)
                    {
                        Log.LogError(null, "AIDOTNET006", null, LicensePath, 0, 0, 0, 0,
                            "AiDotNet Enterprise license at {0} signed payload missing 'scope' array.", LicensePath);
                        return false;
                    }
                    var grantedScope = new HashSet<string>(StringComparer.Ordinal);
                    foreach (var s in scopeEl.EnumerateArray())
                    {
                        if (s.ValueKind == JsonValueKind.String)
                        {
                            var sv = s.GetString();
                            if (!string.IsNullOrWhiteSpace(sv))
                                grantedScope.Add(sv!);
                        }
                    }
                    var requested = (RequestedFlags ?? string.Empty)
                        .Split(new[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var rawFlag in requested)
                    {
                        var flag = rawFlag.Trim();
                        if (flag.Length == 0) continue;
                        if (!grantedScope.Contains(flag))
                        {
                            Log.LogError(null, "AIDOTNET006", null, LicensePath, 0, 0, 0, 0,
                                "AiDotNet Enterprise license at {0} does not include '{1}' in its granted scope. Granted: [{2}].",
                                LicensePath, flag, string.Join(", ", grantedScope));
                            return false;
                        }
                    }

                    // Bind the customer's secret license KEY to the signed license. The
                    // signed payload carries keyHash = SHA-256(key) (hex); validation fails
                    // unless SHA-256(LicenseKey) matches. A valid signed license FILE alone
                    // is insufficient — without the matching key a leaked/redistributed file
                    // can't be used, which is what makes requiring the key meaningful.
                    if (!TryGetString(signed, "keyHash", out var keyHashExpected)
                        || string.IsNullOrWhiteSpace(keyHashExpected))
                    {
                        Log.LogError(null, "AIDOTNET004", null, LicensePath, 0, 0, 0, 0,
                            "AiDotNet Enterprise license at {0} signed payload is missing the 'keyHash' field required to bind the license key. Re-issue the license with a keyHash (see build/LicenseValidator/README.md).",
                            LicensePath);
                        return false;
                    }
                    var keyHashActual = ComputeKeyHash(LicenseKey ?? string.Empty);
                    if (!string.Equals(keyHashActual, keyHashExpected!.Trim(), StringComparison.OrdinalIgnoreCase))
                    {
                        Log.LogError(null, "AIDOTNET011", null, LicensePath, 0, 0, 0, 0,
                            "AiDotNet Enterprise license key does not match this license file. The AiDotNetEnterpriseLicenseKey property (or AIDOTNET_ENTERPRISE_LICENSE_KEY env var) is not the key this license was issued for. Contact admin@aidotnet.dev.");
                        return false;
                    }

                    Tenant = TryGetString(signed, "tenant", out var tenant) && !string.IsNullOrWhiteSpace(tenant)
                        ? tenant!
                        : "(unspecified)";
                    Log.LogMessage(MessageImportance.High,
                        "AiDotNet Enterprise license verified: tenant='{0}', expires={1:yyyy-MM-dd}, scope=[{2}]",
                        Tenant, expires, string.Join(", ", grantedScope));
                    return true;
                }
            }
        }
        catch (Exception ex)
        {
            Log.LogError(null, "AIDOTNET009", null, LicensePath, 0, 0, 0, 0,
                "AiDotNet Enterprise license validation failed unexpectedly: {0}", ex.Message);
            return false;
        }
    }

    /// <summary>
    /// Computes the lowercase-hex SHA-256 of the license key, matching the <c>keyHash</c>
    /// baked into the signed payload. The license issuer computes the same value
    /// (<c>printf '%s' "$KEY" | openssl dgst -sha256 -hex</c>); see README.md.
    /// </summary>
    private static string ComputeKeyHash(string key)
    {
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(key));
        var sb = new System.Text.StringBuilder(bytes.Length * 2);
        foreach (var b in bytes)
            sb.Append(b.ToString("x2", CultureInfo.InvariantCulture));
        return sb.ToString();
    }

    private static bool TryGetString(JsonElement obj, string name, out string? value)
    {
        if (obj.TryGetProperty(name, out var el) && el.ValueKind == JsonValueKind.String)
        {
            value = el.GetString();
            return true;
        }
        value = null;
        return false;
    }

    /// <summary>
    /// Imports an RSA-2048 PKCS#1 public key (the raw <c>RSAPublicKey</c> ASN.1 SEQUENCE, no
    /// SubjectPublicKeyInfo / PEM wrapping) into a managed <see cref="RSA"/> instance.
    /// </summary>
    /// <remarks>
    /// Uses BouncyCastle because <see cref="RSA.ImportRSAPublicKey(System.ReadOnlySpan{byte}, out int)"/>
    /// is netstandard2.1+ only and this validator must run inside MSBuild's task host (which can
    /// load netstandard2.0 task assemblies on both .NET MSBuild and Full-Framework msbuild.exe).
    /// BouncyCastle's <c>RsaPublicKeyStructure</c> parses the same ASN.1 layout that
    /// <c>openssl rsa -RSAPublicKey_out</c> produces and that <c>ImportRSAPublicKey</c> consumes
    /// on newer runtimes — so the on-disk key format is identical.
    /// </remarks>
    private static RSA ImportPkcs1RsaPublicKey(byte[] pkcs1Bytes)
    {
        // PKCS#1 RSAPublicKey ::= SEQUENCE { modulus INTEGER, publicExponent INTEGER }
        // Parse the two-element ASN.1 SEQUENCE directly via BouncyCastle's low-level
        // Asn1Sequence; this avoids depending on a specific BouncyCastle version's
        // higher-level RsaPublicKey* wrapper, which has been renamed across releases.
        var asn1 = Asn1Object.FromByteArray(pkcs1Bytes);
        if (asn1 is not Asn1Sequence seq || seq.Count != 2)
            throw new ArgumentException("PKCS#1 RSAPublicKey must be a 2-element SEQUENCE (modulus + publicExponent).");
        var modulus = DerInteger.GetInstance(seq[0]).PositiveValue;
        var exponent = DerInteger.GetInstance(seq[1]).PositiveValue;
        var bcKey = new RsaKeyParameters(isPrivate: false, modulus, exponent);
        var parms = DotNetUtilities.ToRSAParameters(bcKey);
        var rsa = RSA.Create();
        rsa.ImportParameters(parms);
        return rsa;
    }
}
