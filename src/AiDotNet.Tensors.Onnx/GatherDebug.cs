namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Temporary diagnostic for the BERT × 100 state-bleed investigation.
/// Toggle Enabled from tests.
/// </summary>
public static class GatherDebug
{
    public static bool Enabled { get; set; } = false;
    public static readonly List<string> Messages = new();
    public static int StepCounter = 0;

    public static void Log(string msg)
    {
        lock (Messages) Messages.Add(msg);
    }
    public static void Clear() { lock (Messages) Messages.Clear(); }
    public static string[] Snapshot() { lock (Messages) return Messages.ToArray(); }
}
