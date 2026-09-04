"""Gate a Colab checkpoint before it touches the demo.

    python verify_checkpoint.py                # checks ./satyavaani.pt
    python verify_checkpoint.py --ckpt other.pt
    python verify_checkpoint.py --wav myvoice.wav

A checkpoint trained on a GPU in a different process can fail locally in ways
training never reveals: shape drift, a feature-pipeline mismatch, or a model
that collapsed to one class and still looked fine on a loss curve. Every one of
those shows up as a confident, wrong demo.

This does NOT tell you the model is accurate. It tells you it is not obviously
broken. Accuracy comes from train.report() in Colab.
"""
import argparse
import sys

import numpy as np

import satyavaani as sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="satyavaani.pt")
    ap.add_argument("--wav", help="optional: a real recording of your own voice")
    a = ap.parse_args()

    print(f"checkpoint: {a.ckpt}")
    scorer = sv.get_scorer(a.ckpt)
    if scorer.is_placeholder:
        sys.exit("FAIL: fell back to the placeholder. File missing or unloadable.")
    print("loaded ok, not the placeholder")

    rng = np.random.default_rng(0)
    n = int(sv.WINDOW_S * sv.SR)
    t = np.arange(n) / sv.SR

    probes = {
        "white noise":  0.25 * rng.normal(size=n),
        "pure tone":    0.35 * np.sin(2 * np.pi * 220 * t),
        "harmonics":    0.30 * sum(np.sin(2 * np.pi * 150 * k * t) / k
                                   for k in range(1, 9)),
        "noisy speechlike": 0.30 * np.sin(2 * np.pi * 160 * t) *
                            (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
                            + 0.05 * rng.normal(size=n),
    }

    print("\nprobe                score  band")
    scores = []
    for name, x in probes.items():
        s, band, _ = sv.verdict(x.astype(np.float32), scorer)
        scores.append(s)
        print(f"  {name:<18} {s:.3f}  {band}")

    fails = []

    # 1. must not be stuck. A collapsed model returns near-identical scores.
    spread = max(scores) - min(scores)
    print(f"\nscore spread across probes: {spread:.3f}")
    if spread < 0.02:
        fails.append("model is stuck - near-identical score on every input. "
                     "Likely collapsed to one class, or features mismatch.")

    # 2. must not be saturated at an extreme
    if all(s > 0.98 for s in scores):
        fails.append("every probe reads bonafide - model says yes to everything")
    if all(s < 0.02 for s in scores):
        fails.append("every probe reads spoof - model says no to everything")

    # 3. REQ-6 refusal must survive the swap
    if sv.verdict(np.zeros(sv.SR, np.float32), scorer)[0] is not None:
        fails.append("silence did not trigger the REQ-6 refusal")

    # 4. evidence must be a real map
    cam = scorer.explain(probes["harmonics"].astype(np.float32))
    print(f"evidence map: shape {cam.shape}, std {cam.std():.3f}")
    if cam.shape != (sv.N_MELS, sv.FRAMES):
        fails.append(f"evidence shape wrong: {cam.shape}")
    if cam.std() < 0.01:
        fails.append("evidence map is flat - explains nothing on stage")

    # 5. determinism: same input twice must give the same score
    x = probes["pure tone"].astype(np.float32)
    if abs(scorer.score(x) - scorer.score(x)) > 1e-6:
        fails.append("scores are non-deterministic - dropout/BN left in train mode")

    # 6. optional: your own voice should not read as a strong spoof
    if a.wav:
        x, sr = sv.load_wav(a.wav)
        s, band, action = sv.verdict(x[: int(sv.WINDOW_S * sr)], scorer, sr)
        print(f"\nyour recording: {s if s is None else round(s, 3)}  {band}")
        print(f"  -> {action}")
        if s is not None and s < 0.33:
            fails.append("YOUR OWN VOICE reads as a strong spoof. Do not demo "
                         "this checkpoint - it will flag the presenter.")

    print()
    if fails:
        for f in fails:
            print("FAIL:", f)
        sys.exit(1)
    print("checkpoint usable (not proof of accuracy - see report() in Colab)")


if __name__ == "__main__":
    main()
