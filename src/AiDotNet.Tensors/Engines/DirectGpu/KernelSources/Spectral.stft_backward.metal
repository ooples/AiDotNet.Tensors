// STFT analysis adjoint (backward of IEngine.STFT): frames, waveform (overlap-add + reflection fold), window.
kernel void parity210_stft_backward_frames(
    device const float* grad [[buffer(0)]],
    device const float* magnitude [[buffer(1)]],
    device const float* phase [[buffer(2)]],
    device float* frameGrad [[buffer(3)]],
    constant int& batch [[buffer(4)]],
    constant int& nFft [[buffer(5)]],
    constant int& numFrames [[buffer(6)]],
    constant int& numFreqs [[buffer(7)]],
    constant int& phaseTarget [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= batch*numFrames*nFft) return;
    int i = idx % nFft; int tmp = idx / nFft; int frame = tmp % numFrames; int b = tmp / numFrames;
    int specOff = b*numFreqs*numFrames + frame;
    // Twiddle e^(+2*pi*i*m/nFft) from the integer phase m = k*i mod nFft, quarter turns exact, as in the forward.
    float tw = 2.0f * M_PI_F / (float)nFft; float acc = 0.0f; int m = 0;
    for (int k = 0; k < numFreqs; k++) {
        int bin = specOff + k*numFrames; float g = grad[bin]; float p = phase[bin];
        float cp = cos(p); float sp = sin(p); float sr; float si;
        if (phaseTarget != 0) { float mg = magnitude[bin]; float r = mg > 1e-12f ? g / mg : 0.0f; sr = -r*sp; si = r*cp; }
        else { sr = g*cp; si = g*sp; }
        float c; float s;
        if (m == 0) { c = 1.0f; s = 0.0f; }
        else if (2 * m == nFft) { c = -1.0f; s = 0.0f; }
        else if (4 * m == nFft) { c = 0.0f; s = 1.0f; }
        else if (4 * m == 3 * nFft) { c = 0.0f; s = -1.0f; }
        else { float a = tw * (float)(2 * m > nFft ? m - nFft : m); c = cos(a); s = sin(a); }
        acc += sr*c - si*s;
        m += i; if (m >= nFft) m -= nFft;
    }
    frameGrad[idx] = acc;
}

static float parity210_stft_overlap(device const float* frameGrad, device const float* window,
    int b, int q, int nFft, int hop, int numFrames) {
    int fHi = q / hop; if (fHi > numFrames - 1) fHi = numFrames - 1;
    int fLo = q - nFft + 1 > 0 ? (q - nFft + hop) / hop : 0;
    float acc = 0.0f;
    for (int f = fLo; f <= fHi; f++) { int i = q - f*hop; acc += frameGrad[(b*numFrames + f)*nFft + i] * window[i]; }
    return acc;
}

kernel void parity210_stft_backward_signal(
    device const float* frameGrad [[buffer(0)]],
    device const float* window [[buffer(1)]],
    device float* gradSignal [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& sigLen [[buffer(4)]],
    constant int& pad [[buffer(5)]],
    constant int& nFft [[buffer(6)]],
    constant int& hop [[buffer(7)]],
    constant int& numFrames [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= batch*sigLen) return;
    int s = idx % sigLen; int b = idx / sigLen;
    float acc = parity210_stft_overlap(frameGrad, window, b, pad + s, nFft, hop, numFrames);
    if (s >= 1 && s <= pad) acc += parity210_stft_overlap(frameGrad, window, b, pad - s, nFft, hop, numFrames);
    if (s >= sigLen - 1 - pad && s <= sigLen - 2) acc += parity210_stft_overlap(frameGrad, window, b, 2*sigLen - 2 + pad - s, nFft, hop, numFrames);
    gradSignal[idx] = acc;
}

kernel void parity210_stft_backward_window(
    device const float* frameGrad [[buffer(0)]],
    device const float* signal [[buffer(1)]],
    device float* gradWindow [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& sigLen [[buffer(4)]],
    constant int& pad [[buffer(5)]],
    constant int& nFft [[buffer(6)]],
    constant int& hop [[buffer(7)]],
    constant int& numFrames [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    int idx = (int)gid; if (idx >= nFft) return;
    int i = idx; float acc = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int f = 0; f < numFrames; f++) {
            int j = f*hop + i;
            int src = j < pad ? pad - j : (j < pad + sigLen ? j - pad : sigLen - 2 - (j - pad - sigLen));
            acc += frameGrad[(b*numFrames + f)*nFft + i] * signal[b*sigLen + src];
        }
    }
    gradWindow[i] = acc;
}
