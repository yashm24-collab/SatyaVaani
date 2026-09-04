"""SatyaVaani core -- features, scoring interface, verdict bands, EER.

numpy only, so it runs before anyone installs torch. The model behind
`Scorer.score()` is the only thing that changes on day 2.

Run this file to self-check:  python satyavaani.py
"""
import math
import wave
import numpy as np

SR = 16000
N_FFT = 512
HOP = 160          # 10 ms
WIN = 400          # 25 ms
N_MELS = 64
WINDOW_S = 2.0     # inference window
HOP_S = 0.5        # how often we re-score

MIN_SPEECH_S = 0.6     # REQ-6: below this we refuse to answer
MIN_RMS = 0.005        # REQ-6: below this it is effectively silence


# ---------------------------------------------------------------- audio io

def load_wav(path):
    """16-bit PCM mono wav -> float32 in [-1, 1]. stdlib only.

    ponytail: `wave` covers the demo clips. train.py uses torchaudio, which
    reads the FLAC that ASVspoof actually ships.
    """
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
        return x, w.getframerate()


def trim_silence(x, frame=400, thresh_ratio=0.08):
    """REQ-7. Strip leading/trailing silence.

    Not cosmetic. ASVspoof's bonafide and spoof sets differ systematically in
    their silence padding, so a model trained on untrimmed audio learns the
    padding instead of the synthesis artefacts and collapses on real input.
    """
    if len(x) < frame:
        return x
    n = len(x) // frame
    energy = np.abs(x[: n * frame].reshape(n, frame)).mean(axis=1)
    if energy.max() <= 0:
        return x
    loud = np.where(energy > energy.max() * thresh_ratio)[0]
    if len(loud) == 0:
        return x
    return x[loud[0] * frame : (loud[-1] + 1) * frame]


# ---------------------------------------------------------------- features

def _mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=20.0, fmax=None):
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        lo, mid, hi = bins[m], bins[m + 1], bins[m + 2]
        if mid > lo:
            fb[m, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        if hi > mid:
            fb[m, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return fb


_FB = _mel_filterbank()


def melspec(x, sr=SR):
    """log-mel spectrogram, shape (n_mels, frames). Per-utterance normalised."""
    if sr != SR:
        # ponytail: linear resample is fine for a demo mic path.
        # Use torchaudio.functional.resample if quality ever matters.
        idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)

    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    x = np.ascontiguousarray(x, dtype=np.float32)     # as_strided needs this

    n = 1 + (len(x) - WIN) // HOP
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, WIN), strides=(x.strides[0] * HOP, x.strides[0])
    ) * np.hanning(WIN).astype(np.float32)

    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel = np.log(_FB @ spec.T + 1e-8)
    return (mel - mel.mean()) / (mel.std() + 1e-8)


FRAMES = int(WINDOW_S * SR / HOP)      # 200 frames for a 2 s window


def fix_frames(mel, frames=FRAMES):
    """Crop or pad to a fixed width so batches stack. Same at train and infer."""
    n = mel.shape[1]
    if n == frames:
        return mel
    if n > frames:
        off = (n - frames) // 2
        return mel[:, off : off + frames]
    return np.pad(mel, ((0, 0), (0, frames - n)), mode="edge")


# ---------------------------------------------------------------- scoring

class HeuristicScorer:
    """PLACEHOLDER. Not a detector. Do not present this as a result.

    Exists so the streaming path, the UI and the demo rig are testable on day 1
    while two people train the real model. It leans on the fact that many
    vocoders leave less high-frequency energy and a flatter spectrum over time
    than a real microphone recording -- which is a real tendency and a useless
    defence, because it does not survive a decent generator.

    ponytail: swap for TorchScorer the moment train.py produces a checkpoint.
    """

    is_placeholder = True

    def score(self, x, sr=SR):
        mel = melspec(x, sr)
        hf = mel[N_MELS // 2 :].mean()          # high-band energy
        flat = mel.std(axis=1).mean()           # spectral movement over time
        z = 1.4 * hf + 1.1 * (flat - 1.0)
        return float(1.0 / (1.0 + math.exp(-z)))


class TorchScorer:
    """Real scorer. Loads the checkpoint train.py writes."""

    is_placeholder = False

    def __init__(self, ckpt="satyavaani.pt"):
        import torch                     # imported here so numpy-only use works
        from train import SpoofCNN
        self.torch = torch
        self.net = SpoofCNN()
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.net.eval()

    def score(self, x, sr=SR):
        mel = fix_frames(melspec(x, sr))[None, None]   # (1, 1, n_mels, FRAMES)
        with self.torch.no_grad():
            logit = self.net(self.torch.from_numpy(mel))
        return float(self.torch.sigmoid(logit).item())

    def explain(self, x, sr=SR):
        """REQ-4. (N_MELS, FRAMES) heatmap in [0, 1], or None."""
        from train import gradcam
        return gradcam(self.net, fix_frames(melspec(x, sr)))


def get_scorer(ckpt="satyavaani.pt"):
    """Real model if a checkpoint exists, placeholder otherwise."""
    import os
    if os.path.exists(ckpt):
        try:
            return TorchScorer(ckpt)
        except Exception as e:
            print(f"[satyavaani] checkpoint found but unusable ({e}); using placeholder")
    return HeuristicScorer()


# ---------------------------------------------------------------- verdict

# PRD 03: we output an action, never "fake". Absence of evidence is not proof
# of authenticity, so the low band says "no indicators", not "genuine".
BANDS = (
    (0.66, "low",    "No indicators detected"),
    (0.33, "medium", "Verify through another channel before acting"),
    (0.00, "high",   "Strong synthetic indicators -- do not act on this call"),
)

INSUFFICIENT = ("none", "Insufficient audio -- keep speaking")


def band(score):
    """score is P(bonafide). Higher = more likely a real human."""
    for lo, name, action in BANDS:
        if score >= lo:
            return name, action
    return BANDS[-1][1:]


def usable(x, sr=SR):
    """REQ-6. Refuse rather than guess on too-short or too-quiet audio."""
    if len(x) < MIN_SPEECH_S * sr:
        return False
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) >= MIN_RMS


def verdict(x, scorer, sr=SR):
    """The whole decision, one call. Returns (score|None, band, action)."""
    x = trim_silence(x)
    if not usable(x, sr):
        return None, *INSUFFICIENT
    s = scorer.score(x, sr)
    return (s, *band(s))


# ---------------------------------------------------------------- metrics

def eer(scores, labels):
    """labels: 1 = bonafide, 0 = spoof. scores: higher = more bonafide.

    ponytail: O(n*t) threshold sweep. Fine for a few thousand eval clips;
    switch to the sorted-cumsum trick only if eval ever gets slow.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if not (labels == 1).any() or not (labels == 0).any():
        raise ValueError("need both classes present")

    best, gap = 0.5, np.inf
    for t in np.unique(scores):
        far = float(np.mean(scores[labels == 0] >= t))   # spoof accepted
        frr = float(np.mean(scores[labels == 1] < t))    # bonafide rejected
        if abs(far - frr) < gap:
            gap, best = abs(far - frr), (far + frr) / 2
    return best


def false_alarm_rate(scores, labels, threshold):
    """Genuine speech wrongly flagged. The constraint that keeps it usable."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    return float(np.mean(scores[labels == 1] < threshold))


def threshold_for_far(scores, labels, target_far=0.05):
    """Threshold that holds false alarms on genuine speech at `target_far`.

    The operating threshold is a product decision, not a tuning detail: it
    spends user annoyance to buy detection. Pick the annoyance budget first,
    then report what detection that budget bought. Do not hardcode 0.5.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    return float(np.quantile(scores[labels == 1], target_far))


def detection_rate(scores, labels, threshold):
    """Fraction of spoofs caught at this threshold. The number the budget buys."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    return float(np.mean(scores[labels == 0] < threshold))


# ---------------------------------------------------------------- self-check

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # eer: perfectly separable -> ~0
    s = np.r_[rng.uniform(0.0, 0.3, 200), rng.uniform(0.7, 1.0, 200)]
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    assert eer(s, y) < 0.02, eer(s, y)

    # eer: indistinguishable -> ~0.5
    s2 = rng.uniform(0, 1, 400)
    assert 0.4 < eer(s2, y) < 0.6, eer(s2, y)

    # eer is symmetric to a constant shift
    assert abs(eer(s, y) - eer(s + 5.0, y)) < 1e-9

    # trim_silence strips padding but keeps the speech
    sig = np.r_[np.zeros(4000, np.float32),
                rng.normal(0, 0.3, 8000).astype(np.float32),
                np.zeros(4000, np.float32)]
    t = trim_silence(sig)
    assert 6000 < len(t) < 10000, len(t)

    # melspec shape: 2 s -> ~200 frames
    m = melspec(rng.normal(0, 0.1, 2 * SR).astype(np.float32))
    assert m.shape[0] == N_MELS and 150 < m.shape[1] < 250, m.shape

    # bands map to the right actions, boundaries included
    assert band(0.90)[0] == "low"
    assert band(0.50)[0] == "medium"
    assert band(0.10)[0] == "high"
    assert band(0.66)[0] == "low" and band(0.33)[0] == "medium"

    # REQ-6 guards: too short, and long but silent
    assert not usable(rng.normal(0, 0.3, 1000).astype(np.float32))
    assert not usable(np.zeros(SR, np.float32))
    assert usable(rng.normal(0, 0.3, 2 * SR).astype(np.float32))

    # verdict refuses instead of guessing
    sc = HeuristicScorer()
    assert verdict(np.zeros(SR, np.float32), sc)[0] is None

    # placeholder still returns a usable probability
    v = verdict(rng.normal(0, 0.3, 2 * SR).astype(np.float32), sc)
    assert v[0] is not None and 0.0 <= v[0] <= 1.0, v

    print("satyavaani self-check ok")
