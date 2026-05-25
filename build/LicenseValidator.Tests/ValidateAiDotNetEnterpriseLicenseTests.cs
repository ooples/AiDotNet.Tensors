using System;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;
using Org.BouncyCastle.Asn1;
using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Security;
using Xunit;

namespace AiDotNet.Tensors.LicenseValidator.Tests;

/// <summary>
/// End-to-end tests for <see cref="ValidateAiDotNetEnterpriseLicense"/>: generates a real RSA-2048
/// keypair, signs a synthetic license file with the private key, swaps in the matching public key,
/// and verifies the task accepts the file. Then mutates the signature byte / scope array / expiry
/// date / marker string in turn and verifies each fault yields the correct AIDOTNET* diagnostic.
/// </summary>
public class ValidateAiDotNetEnterpriseLicenseTests : IDisposable
{
    private readonly string _tempDir;
    private readonly RSA _signingRsa;
    private readonly string _publicKeyBase64;

    public ValidateAiDotNetEnterpriseLicenseTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "aidotnet-license-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);

        // Generate a fresh keypair for the test session. The validator's embedded public key needs
        // to match this for VerifyData to succeed, so we override it via reflection (test-only).
        _signingRsa = RSA.Create(2048);
        _publicKeyBase64 = ExportPkcs1PublicKey(_signingRsa);
    }

    public void Dispose()
    {
        _signingRsa.Dispose();
        try { Directory.Delete(_tempDir, recursive: true); } catch { /* best-effort cleanup */ }
    }

    private static string ExportPkcs1PublicKey(RSA rsa)
    {
        // Serialise RSAParameters back into the PKCS#1 ASN.1 SEQUENCE { modulus, exponent } that
        // the validator's ImportPkcs1RsaPublicKey expects.
        var parms = rsa.ExportParameters(includePrivateParameters: false);
        var bcKey = new RsaKeyParameters(isPrivate: false,
            new Org.BouncyCastle.Math.BigInteger(1, parms.Modulus!),
            new Org.BouncyCastle.Math.BigInteger(1, parms.Exponent!));
        var seq = new DerSequence(
            new DerInteger(bcKey.Modulus),
            new DerInteger(bcKey.Exponent));
        return Convert.ToBase64String(seq.GetEncoded());
    }

    /// <summary>
    /// Tests run against a private build of the validator that has the placeholder public key
    /// patched in via reflection (the production constant is intentionally immutable).
    /// </summary>
    private static void OverrideEmbeddedPublicKey(string newKeyBase64)
    {
        // The constant lives in a public static field; rewrite it for the test session only.
        var field = typeof(ValidateAiDotNetEnterpriseLicense)
            .GetField(nameof(ValidateAiDotNetEnterpriseLicense.EnterpriseLicensePublicKeyPkcs1Base64),
                BindingFlags.Public | BindingFlags.Static)
            ?? throw new InvalidOperationException("EnterpriseLicensePublicKeyPkcs1Base64 const-field not found.");

        // const fields are emitted as Literal — we have to use FieldInfo.SetValue with the
        // RuntimeFieldHandle trick. Because the test compiles against the same assembly we just
        // bypass via SetValue on a ref-to-field through unsafe pointer access.
        // Simplest portable trick: use SetValue against a static helper backing field that the
        // task reads if present. We add that helper below as ConfigureTestPublicKey.
        ConfigureTestPublicKey(newKeyBase64);
    }

    private static void ConfigureTestPublicKey(string newKeyBase64)
    {
        // Use the public test hook exposed on the task itself.
        var prop = typeof(ValidateAiDotNetEnterpriseLicense)
            .GetField("s_overridePublicKey",
                BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException(
                "Test hook 's_overridePublicKey' not found on ValidateAiDotNetEnterpriseLicense.");
        prop.SetValue(null, newKeyBase64);
    }

    private string WriteLicense(
        string marker = "AiDotNet-Enterprise-License-v1",
        string tenant = "ACME Corp.",
        string expires = "2099-12-31",
        string[]? scope = null,
        bool corruptSignature = false)
    {
        scope ??= new[] { "DISABLE_TELEMETRY", "DISABLE_LICENSE_GUARD" };

        var inner = new
        {
            marker,
            tenant,
            issued = "2026-05-22",
            expires,
            scope,
        };
        var payloadJson = JsonSerializer.Serialize(inner, new JsonSerializerOptions { WriteIndented = false });
        var payloadBytes = Encoding.UTF8.GetBytes(payloadJson);

        var signature = _signingRsa.SignData(payloadBytes, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        if (corruptSignature) signature[0] ^= 0xFF;

        var envelope = new
        {
            marker,
            tenant,
            issued = "2026-05-22",
            expires,
            scope,
            payload = Convert.ToBase64String(payloadBytes),
            signature = Convert.ToBase64String(signature),
        };
        var licensePath = Path.Combine(_tempDir, Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(licensePath, JsonSerializer.Serialize(envelope));
        return licensePath;
    }

    private static MockBuildEngine NewEngine() => new MockBuildEngine();

    [Fact]
    public void ValidLicense_VerifiesSuccessfully()
    {
        OverrideEmbeddedPublicKey(_publicKeyBase64);
        var path = WriteLicense();

        var task = new ValidateAiDotNetEnterpriseLicense
        {
            BuildEngine = NewEngine(),
            LicensePath = path,
            RequestedFlags = "DISABLE_TELEMETRY",
        };

        Assert.True(task.Execute(), $"Validator rejected a well-formed license: {string.Join("; ", ((MockBuildEngine)task.BuildEngine).Errors)}");
        Assert.Equal("ACME Corp.", task.Tenant);
    }

    [Fact]
    public void CorruptSignature_RejectsWithAIDOTNET008()
    {
        OverrideEmbeddedPublicKey(_publicKeyBase64);
        var path = WriteLicense(corruptSignature: true);

        var task = new ValidateAiDotNetEnterpriseLicense
        {
            BuildEngine = NewEngine(),
            LicensePath = path,
            RequestedFlags = "DISABLE_TELEMETRY",
        };

        Assert.False(task.Execute());
        Assert.Contains(((MockBuildEngine)task.BuildEngine).Errors,
            e => e.Code == "AIDOTNET008");
    }

    [Fact]
    public void ExpiredLicense_RejectsWithAIDOTNET005()
    {
        OverrideEmbeddedPublicKey(_publicKeyBase64);
        var path = WriteLicense(expires: "2000-01-01");

        var task = new ValidateAiDotNetEnterpriseLicense
        {
            BuildEngine = NewEngine(),
            LicensePath = path,
            RequestedFlags = "DISABLE_TELEMETRY",
        };

        Assert.False(task.Execute());
        Assert.Contains(((MockBuildEngine)task.BuildEngine).Errors,
            e => e.Code == "AIDOTNET005");
    }

    [Fact]
    public void RequestedFlagNotInScope_RejectsWithAIDOTNET006()
    {
        OverrideEmbeddedPublicKey(_publicKeyBase64);
        var path = WriteLicense(scope: new[] { "DISABLE_TELEMETRY" });

        var task = new ValidateAiDotNetEnterpriseLicense
        {
            BuildEngine = NewEngine(),
            LicensePath = path,
            RequestedFlags = "DISABLE_LICENSE_GUARD",
        };

        Assert.False(task.Execute());
        Assert.Contains(((MockBuildEngine)task.BuildEngine).Errors,
            e => e.Code == "AIDOTNET006");
    }

    [Fact]
    public void WrongMarker_RejectsWithAIDOTNET004()
    {
        OverrideEmbeddedPublicKey(_publicKeyBase64);
        var path = WriteLicense(marker: "Wrong-Marker-v0");

        var task = new ValidateAiDotNetEnterpriseLicense
        {
            BuildEngine = NewEngine(),
            LicensePath = path,
            RequestedFlags = "DISABLE_TELEMETRY",
        };

        Assert.False(task.Execute());
        Assert.Contains(((MockBuildEngine)task.BuildEngine).Errors,
            e => e.Code == "AIDOTNET004");
    }

    /// <summary>
    /// Writes a license whose SIGNED payload and outer envelope deliberately
    /// disagree. The signature is valid over the (restricted) signed fields,
    /// while the envelope advertises escalated fields — the exact attack the
    /// signed-payload-as-source-of-truth fix defends against.
    /// </summary>
    private string WriteTamperedLicense(
        string[] signedScope, string signedExpires,
        string[] envelopeScope, string envelopeExpires)
    {
        const string marker = "AiDotNet-Enterprise-License-v1";
        var inner = new { marker, tenant = "ACME Corp.", issued = "2026-05-22", expires = signedExpires, scope = signedScope };
        var payloadBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(inner));
        var signature = _signingRsa.SignData(payloadBytes, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        var envelope = new
        {
            marker,
            tenant = "ACME Corp.",
            issued = "2026-05-22",
            expires = envelopeExpires,
            scope = envelopeScope,
            payload = Convert.ToBase64String(payloadBytes),
            signature = Convert.ToBase64String(signature),
        };
        var licensePath = Path.Combine(_tempDir, Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(licensePath, JsonSerializer.Serialize(envelope));
        return licensePath;
    }

    [Fact]
    public void TamperedEnvelope_ScopeEscalation_RejectedFromSignedPayload()
    {
        OverrideEmbeddedPublicKey(_publicKeyBase64);
        // Signed payload grants only DISABLE_TELEMETRY; the envelope tries to add
        // DISABLE_LICENSE_GUARD. Requesting the escalated flag must fail — the
        // signed scope is the source of truth, not the envelope.
        var path = WriteTamperedLicense(
            signedScope: new[] { "DISABLE_TELEMETRY" }, signedExpires: "2099-12-31",
            envelopeScope: new[] { "DISABLE_TELEMETRY", "DISABLE_LICENSE_GUARD" }, envelopeExpires: "2099-12-31");

        var task = new ValidateAiDotNetEnterpriseLicense
        {
            BuildEngine = NewEngine(),
            LicensePath = path,
            RequestedFlags = "DISABLE_LICENSE_GUARD",
        };

        Assert.False(task.Execute(), "Validator accepted an envelope-escalated scope absent from the signed payload.");
        Assert.Contains(((MockBuildEngine)task.BuildEngine).Errors, e => e.Code == "AIDOTNET006");
    }

    [Fact]
    public void TamperedEnvelope_ExpiryExtension_RejectedFromSignedPayload()
    {
        OverrideEmbeddedPublicKey(_publicKeyBase64);
        // Signed payload is expired; the envelope tries to extend expiry to the
        // far future. The signed (expired) date must govern.
        var path = WriteTamperedLicense(
            signedScope: new[] { "DISABLE_TELEMETRY" }, signedExpires: "2000-01-01",
            envelopeScope: new[] { "DISABLE_TELEMETRY" }, envelopeExpires: "2099-12-31");

        var task = new ValidateAiDotNetEnterpriseLicense
        {
            BuildEngine = NewEngine(),
            LicensePath = path,
            RequestedFlags = "DISABLE_TELEMETRY",
        };

        Assert.False(task.Execute(), "Validator accepted an envelope-extended expiry past the signed expiry.");
        Assert.Contains(((MockBuildEngine)task.BuildEngine).Errors, e => e.Code == "AIDOTNET005");
    }

    [Fact]
    public void MissingFile_RejectsWithAIDOTNET003()
    {
        OverrideEmbeddedPublicKey(_publicKeyBase64);
        var task = new ValidateAiDotNetEnterpriseLicense
        {
            BuildEngine = NewEngine(),
            LicensePath = Path.Combine(_tempDir, "does-not-exist.json"),
            RequestedFlags = "DISABLE_TELEMETRY",
        };

        Assert.False(task.Execute());
        Assert.Contains(((MockBuildEngine)task.BuildEngine).Errors,
            e => e.Code == "AIDOTNET003");
    }

    [Fact]
    public void PlaceholderPublicKey_RejectsWithAIDOTNET008()
    {
        // Reset to the placeholder marker — proves the safe-by-default posture.
        ConfigureTestPublicKey(ValidateAiDotNetEnterpriseLicense.PublicKeyPlaceholderMarker);
        var path = WriteLicense();

        var task = new ValidateAiDotNetEnterpriseLicense
        {
            BuildEngine = NewEngine(),
            LicensePath = path,
            RequestedFlags = "DISABLE_TELEMETRY",
        };

        Assert.False(task.Execute());
        Assert.Contains(((MockBuildEngine)task.BuildEngine).Errors,
            e => e.Code == "AIDOTNET008");
    }

    /// <summary>Minimal IBuildEngine stub that captures errors so tests can assert on diagnostic codes.</summary>
    private sealed class MockBuildEngine : IBuildEngine
    {
        public System.Collections.Generic.List<BuildErrorEventArgs> Errors { get; } = new();
        public bool ContinueOnError => false;
        public int LineNumberOfTaskNode => 0;
        public int ColumnNumberOfTaskNode => 0;
        public string ProjectFileOfTaskNode => "(test)";
        public bool BuildProjectFile(string projectFileName, string[] targetNames, System.Collections.IDictionary globalProperties, System.Collections.IDictionary targetOutputs) => false;
        public void LogCustomEvent(CustomBuildEventArgs e) { }
        public void LogErrorEvent(BuildErrorEventArgs e) => Errors.Add(e);
        public void LogMessageEvent(BuildMessageEventArgs e) { }
        public void LogWarningEvent(BuildWarningEventArgs e) { }
    }
}
