namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The state update a Gated DeltaNet scan or decode step applies, per head, with forget gate <c>a_t</c>, write strength
/// <c>b_t</c> and matrix state <c>S</c> (initially zero); the output is <c>o_t = S_t q_t</c>.
/// </summary>
public enum GatedDeltaRule
{
    /// <summary>
    /// The scan's original rule: the op scales the keys by <c>1/sqrt(headDim)</c> and the prediction reads the state
    /// before this position's decay, <c>S_t = a_t S_{t-1} + b_t (v_t - S_{t-1} k_t) k_tᵀ</c>, that is
    /// <c>S_t = S_{t-1} (a_t I - b_t k_t k_tᵀ) + b_t v_t k_tᵀ</c>.
    /// </summary>
    ScaledKeys = 0,

    /// <summary>
    /// The gated delta rule of Yang, Kautz and Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule"
    /// (ICLR 2025, arXiv:2412.06464): the decay applies to the whole transition and the keys are used as given,
    /// <c>S_t = a_t S_{t-1} (I - b_t k_t k_tᵀ) + b_t v_t k_tᵀ</c>, so the prediction reads the decayed state,
    /// <c>S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_tᵀ</c>. The caller normalizes the keys and scales the
    /// queries: the paper L2-normalizes both, and its reference implementation scales the queries by
    /// <c>1/sqrt(headDim)</c>. With unit keys, <c>0 &lt;= a_t &lt;= 1</c> and <c>0 &lt;= b_t &lt;= 1</c> the transition
    /// <c>a_t (I - b_t k_t k_tᵀ)</c> is non-expansive (spectral norm at most 1); it is not a strict contraction in
    /// general: at <c>a_t = 1</c> it keeps every direction orthogonal to <c>k_t</c>, and at <c>b_t = 1</c> it is the
    /// orthogonal projection off <c>k_t</c>.
    /// </summary>
    Standard = 1,
}
