#pragma warning disable AIDEVO005 // AiDotNet.Evolution.Ptx is experimental; this campaign driver is its consumer.
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Holds a PTX campaign to one dollar budget across every model call. <see cref="ClaudeCodeChatClientOptions.MaxBudgetUsd"/>
/// caps a single <c>claude -p</c> process, and the client starts a new one per attempt, so a per-call cap alone let a
/// campaign spend it again on every call and every retry.
/// </summary>
/// <remarks>
/// <c>claude -p</c> reports tokens but no spend, so each call reserves its full cap before it starts, and the reserved
/// caps together stay within the budget. That bounds what the campaign starts, not what it spends: Claude Code checks
/// its cap after a response, so the response that crosses it still completes, and a call can spend somewhat past its
/// own cap. <see cref="CommittedUsd"/> is the sum of reserved caps, not an upper bound on actual spend. Each call runs one
/// attempt (retries off); the evolution loop's next request is the retry, and it is charged like any other call. Once
/// the remaining budget is under the per-call cap's floor, calls are refused with an
/// <see cref="EvolutionResourceBudgetException"/>, so no further process is started.
/// </remarks>
internal sealed class CampaignUsdBudgetChatClient : IProgramChatClient
{
    // Below this a call could not do useful work, so it is refused rather than started with a token cap.
    private const decimal MinimumCallUsd = 0.01m;

    private readonly string _model;
    private readonly decimal _perCallUsd;
    private readonly object _gate = new();
    private decimal _remainingUsd;
    private decimal _committedUsd;
    private int _refusedCalls;

    public CampaignUsdBudgetChatClient(string model, decimal campaignUsd, decimal perCallUsd)
    {
        if (string.IsNullOrWhiteSpace(model)) throw new ArgumentException("A model is required.", nameof(model));
        if (campaignUsd <= 0) throw new ArgumentOutOfRangeException(nameof(campaignUsd), "The campaign budget must be positive.");
        if (perCallUsd <= 0) throw new ArgumentOutOfRangeException(nameof(perCallUsd), "The per-call cap must be positive.");
        _model = model;
        _remainingUsd = campaignUsd;
        _perCallUsd = Math.Min(perCallUsd, campaignUsd);
    }

    public string ModelId => _model;

    /// <summary>Dollars reserved so far: the sum of every started call's cap. Not an upper bound on actual spend.</summary>
    public decimal CommittedUsd { get { lock (_gate) return _committedUsd; } }

    /// <summary>Calls refused because the remaining budget could not cover one.</summary>
    public int RefusedCalls { get { lock (_gate) return _refusedCalls; } }

    public Task<ProgramChatResponse> GetResponseAsync(
        IReadOnlyList<ProgramChatMessage> messages, ProgramChatOptions options, CancellationToken cancellationToken)
    {
        decimal cap;
        lock (_gate)
        {
            cap = Math.Min(_perCallUsd, _remainingUsd);
            if (cap < MinimumCallUsd)
            {
                _refusedCalls++;
                throw new EvolutionResourceBudgetException("ptx-campaign-usd");
            }

            _remainingUsd -= cap;
            _committedUsd += cap;
        }

        var client = new ClaudeCodeChatClient(new ClaudeCodeChatClientOptions
        {
            Model = _model,
            MaxBudgetUsd = cap,
            MaxRetries = 0,
        });
        return client.GetResponseAsync(messages, options, cancellationToken);
    }
}
