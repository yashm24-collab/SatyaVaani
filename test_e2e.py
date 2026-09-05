"""End-to-end plumbing test:  data -> train -> eval -> save -> swap -> verdict.

    python test_e2e.py

WHAT THIS PROVES: the wiring. Manifest parsing, the training loop, the eval
asserts, checkpoint save/load, `get_scorer()` picking up the checkpoint, and
`TorchScorer` producing a verdict through the same path `server.py` uses.

WHAT THIS DOES NOT PROVE: anything about detecting real synthetic speech. The
fake "spoof" class here is band-limited on purpose, which is trivially
separable. A CNN will hit ~0 EER on it and that number means nothing. Real
numbers come from ASVspoof plus person 5's generators.

This exists so day 4 is not the first time anyone runs the swap.
"""
import csv
import os
import shutil
import tempfile
import wave

import numpy as np

import satyavaani as sv

TMP = os.path.join(tempfile.gettempdir(), "satyavaani_e2e")
CKPT = os.path.join(TMP, "e2e.pt")
DUR = 2.0


def _write(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sv.SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _lowpass(x, cutoff):
    """Hard spectral cut. Stands in for a vocoder that band-limits its output."""
    X = np.fft.rfft(x)
    X[int(cutoff / (sv.SR / 2) * len(X)):] = 0
    return np.fft.irfft(X, n=len(x))


def _harmonics(n, rng, f0, jitter):
    t = np.arange(n) / sv.SR
    out = np.zeros(n)
    for k in range(1, 12):
        amp = 1.0 / k
        if jitter:                                  # natural amplitude drift
            amp *= 1 + 0.35 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t + rng.random() * 6)
        out += amp * np.sin(2 * np.pi * f0 * k * t + rng.random() * 6)
    return out / np.abs(out).max()


def make_clip(kind, rng):
    n = int(DUR * sv.SR)
    f0 = rng.uniform(95, 175)
    if kind == "bonafide":
        x = _harmonics(n, rng, f0, jitter=True)
        x += 0.06 * rng.normal(size=n)              # broadband floor
    elif kind == "spoof":
        x = _lowpass(_harmonics(n, rng, f0, jitter=False), 6500)
    else:                                           # "unseen" generator
        x = _lowpass(_harmonics(n, rng, f0, jitter=False), 5000)
        x += 0.01 * rng.normal(size=n)
    env = 0.4 * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.4, 1.2) *
                                    np.arange(n) / sv.SR))
    return (x * env).astype(np.float32)


def build_dataset():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    rng = np.random.default_rng(7)
    rows = []

    def emit(kind, label, split, count, source):
        for i in range(count):
            p = os.path.join(TMP, f"{split}_{kind}_{i}.wav")
            _write(p, make_clip(kind, rng))
            rows.append((p, label, split, source))

    emit("bonafide", 1, "train",  40, "real")
    emit("spoof",    0, "train",  40, "gen_a")
    emit("bonafide", 1, "seen",   20, "real")
    emit("spoof",    0, "seen",   20, "gen_a")
    emit("bonafide", 1, "unseen", 20, "real")
    emit("unseen",   0, "unseen", 20, "gen_b")   # never trained on

    manifest = os.path.join(TMP, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split", "source"])
        w.writerows(rows)
    return manifest


def main():
    import torch
    import train

    print("1/6 building synthetic dataset")
    manifest = build_dataset()
    splits = train.load_manifest(manifest)
    assert len(splits["train"]) == 80 and len(splits["unseen"]) == 40, "manifest"

    # the leak guard must actually fire -- a guard nobody tested is decoration
    leaked = [{"path": "a", "label": "0", "split": "train",  "source": "gen_a"},
              {"path": "b", "label": "0", "split": "unseen", "source": "gen_a"}]
    try:
        train.check_manifest(leaked)
        raise AssertionError("leak guard did not fire")
    except AssertionError as e:
        assert "LEAK" in str(e), f"wrong failure: {e}"
    print("    leak guard fires correctly")

    print("2/6 training")
    net = train.fit(train.SpoofCNN(), splits["train"], epochs=6)

    print("3/6 eval (asserts live here)")
    # out=None: this trains on trivially separable synthetic audio, so its
    # EER of 0.000 is meaningless. Writing it to the metrics sidecar would put
    # that number straight into the UI's evaluation card.
    e_seen, thr, det = train.report(net, splits, out=None)

    print("4/6 saving checkpoint")
    torch.save(net.state_dict(), CKPT)
    assert os.path.exists(CKPT)

    print("5/6 swap: get_scorer picks up the checkpoint")
    scorer = sv.get_scorer(CKPT)
    assert not scorer.is_placeholder, "still on the placeholder - swap is broken"

    print("6/6 verdict through the real scorer")
    rng = np.random.default_rng(99)
    bona = make_clip("bonafide", rng)
    spoof = make_clip("spoof", rng)

    s_b, band_b, act_b = sv.verdict(bona, scorer)
    s_s, band_s, act_s = sv.verdict(spoof, scorer)
    print(f"    bonafide -> {s_b:.3f}  {band_b}")
    print(f"    spoof    -> {s_s:.3f}  {band_s}")
    assert s_b is not None and s_s is not None
    assert s_b > s_s, "model ranks spoof above bonafide - labels are flipped"

    # the refusal path must survive the swap too
    assert sv.verdict(np.zeros(sv.SR, np.float32), scorer)[0] is None, "REQ-6"

    # REQ-4: evidence must be a real map, not a flat rectangle
    cam = scorer.explain(spoof)
    assert cam.shape == (sv.N_MELS, sv.FRAMES), cam.shape
    assert cam.min() >= 0.0 and cam.max() <= 1.0, "cam out of range"
    assert cam.std() > 0.01, f"cam is flat ({cam.std():.4f}) - explains nothing"
    # it must actually differ between two different inputs
    cam_b = scorer.explain(bona)
    assert np.abs(cam - cam_b).mean() > 0.01, "same evidence for both classes"
    print(f"    evidence map ok  (std {cam.std():.3f}, "
          f"class delta {np.abs(cam - cam_b).mean():.3f})")

    # and the server helpers must not choke on a real window
    import server
    env = server.envelope(bona)
    assert len(env) == server.WAVE_POINTS and max(env) > 0, "envelope"
    heat = server.shrink(cam)
    assert len(heat) == server.HEAT_W * server.HEAT_H, "heat payload size"
    assert max(heat) > 0 and all(0 <= v <= 255 for v in heat), "heat range"

    print(f"\ne2e ok  |  seen EER {e_seen:.3f}  thr {thr:.3f}  detection {det:.3f}")
    print("NOTE: those numbers are from trivially separable synthetic audio.")
    print("They validate the pipeline, not the detector.")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
