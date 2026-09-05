"""Train the spoof detector. Owner: persons 1+2. Needs torch.

    python train.py --smoke                    # prove the loop works, no data
    python train.py --manifest manifest.csv    # real run

manifest.csv columns:  path,label,split
    label: 1 = bonafide, 0 = spoof
    split: train | seen | unseen
"train" fits. "seen" is held-out clips of attacks the model trained on.
"unseen" is a generator it has never met -- that is the number that matters.

Build one from the ASVspoof protocol file with `build_manifest()` below, then
append your own generated attacks (person 5).
"""
import argparse
import csv
import os
import sys

import numpy as np

from satyavaani import (SR, N_MELS, FRAMES, melspec, fix_frames, trim_silence,
                        load_wav, eer, false_alarm_rate, threshold_for_far,
                        detection_rate)

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("needs torch:  pip install torch torchaudio")


class SpoofCNN(nn.Module):
    """Small on purpose. ~25k params, trains on CPU, no excuse not to have a
    baseline by end of day 2.

    ponytail: if this plateaus above the EER floor, fine-tune a pretrained
    audio model instead of growing this one. Do not grow it first.
    """

    def __init__(self):
        super().__init__()
        def block(i, o):
            # GroupNorm, not BatchNorm. Inference scores ONE 2 s window at a
            # time, and BN's running stats disagree with its training-time
            # batch stats on short runs -- which showed up as perfect
            # separation and a 100% false-alarm rate at the same time.
            # GroupNorm has no running stats and behaves identically at
            # batch size 1.
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(min(8, o), o),
                nn.ReLU(), nn.MaxPool2d(2))
        self.body = nn.Sequential(block(1, 16), block(16, 32), block(32, 64),
                                  nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.body(x).flatten(1)).squeeze(1)


# ------------------------------------------------------------------ data

def read_audio(path):
    """stdlib for wav; torchaudio only for the FLAC that ASVspoof ships.

    ponytail: keeps torchaudio off the critical path for anyone testing with
    wavs. Install it when you actually point this at ASVspoof.
    """
    if str(path).lower().endswith(".wav"):
        x, sr = load_wav(path)
    else:
        import torchaudio
        wav, sr = torchaudio.load(path)
        x = wav.mean(0).numpy()
    if sr != SR:
        idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
        x = np.interp(idx, np.arange(len(x)), x)
    return np.ascontiguousarray(x, dtype=np.float32)


def featurise(path):
    return fix_frames(melspec(trim_silence(read_audio(path))))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Clips(torch.utils.data.Dataset):
    """Featurises on demand. Fine for a few hundred clips; too slow for ASVspoof."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        return torch.from_numpy(featurise(path))[None], torch.tensor(float(label))


class Cached(torch.utils.data.Dataset):
    """Features already in memory. Use this for anything dataset-sized.

    Featurising costs ~25 ms/clip. On ASVspoof's ~25k training clips that is
    ~10 min PER EPOCH of pure numpy, which dwarfs the actual training. Pay it
    once, then epochs cost seconds.
    """

    def __init__(self, feats, labels):
        self.x = torch.from_numpy(feats).unsqueeze(1)      # (N, 1, mels, frames)
        self.y = torch.from_numpy(labels.astype("float32"))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # Cast per item, not up front -- keeps the cache at float16 in RAM.
        return self.x[i].float(), self.y[i]


def precompute(rows, cache=None, workers=None):
    """Featurise once. Returns (feats, labels) and optionally caches to disk.

    ponytail: stores float16. Mel frames are normalised, so the precision is
    irrelevant and it halves both RAM and disk (410 MB -> 205 MB at 8k clips).
    Uncompressed on purpose -- zlib on float arrays is minutes of CPU for
    almost no saving.
    """
    import concurrent.futures as cf

    if cache and os.path.exists(cache):
        d = np.load(cache)
        print(f"loaded cached features {d['x'].shape} from {cache}")
        return d["x"], d["y"]

    paths = [p for p, _ in rows]
    labels = np.array([l for _, l in rows], dtype=np.int64)

    workers = workers or min(8, (os.cpu_count() or 2))
    with cf.ThreadPoolExecutor(workers) as ex:      # numpy releases the GIL
        feats = list(ex.map(featurise, paths))
    x = np.stack(feats).astype("float16")

    if cache:
        np.savez(cache, x=x, y=labels)
        print(f"cached features {x.shape} -> {cache} "
              f"({x.nbytes/1e6:.0f} MB)")
    return x, labels


def spec_augment(x, n_freq=2, n_time=2, f_max=8, t_max=24):
    """Mask random frequency bands and time spans. Training only.

    The headline metric is accuracy on a generator never seen in training, and
    that is a generalisation problem, not a capacity problem. Masking is the
    cheapest regulariser that targets exactly it: the model cannot lean on one
    narrow band of vocoder artefact if that band keeps disappearing.

    ponytail: one mask per batch, not per sample. Per-sample is better and
    needs a loop; shuffling varies it enough across epochs.
    """
    _, _, mels, frames = x.shape
    for _ in range(n_freq):
        f = int(torch.randint(0, f_max + 1, (1,)))
        if f:
            f0 = int(torch.randint(0, max(1, mels - f), (1,)))
            x[:, :, f0:f0 + f, :] = 0
    for _ in range(n_time):
        t = int(torch.randint(0, t_max + 1, (1,)))
        if t:
            t0 = int(torch.randint(0, max(1, frames - t), (1,)))
            x[:, :, :, t0:t0 + t] = 0
    return x


COLUMNS = ["path", "label", "split", "source"]


def check_manifest(rows):
    """Refuse the one mistake that silently destroys the headline number.

    If a generator appears in both `train` and `unseen`, the unseen score is
    not a generalisation result -- it is a memorisation result, and nobody
    downstream can tell. Fail loudly here rather than believe it on stage.
    """
    # Only spoof sources. Genuine speech belongs in every split -- you need
    # bonafide clips everywhere to compute an EER at all.
    by_split = {}
    for r in rows:
        if int(r["label"]) == 0:
            by_split.setdefault(r["split"], set()).add(r.get("source") or "?")

    leaked = by_split.get("train", set()) & by_split.get("unseen", set())
    leaked.discard("?")
    assert not leaked, (
        f"LEAK: generator {sorted(leaked)} appears in both train and unseen. "
        "The held-out generator must never be trained on.")

    for split in ("train", "seen", "unseen"):
        srcs = by_split.get(split, set())
        n = sum(1 for r in rows if r["split"] == split)
        print(f"  {split:7s} {n:6d} clips  from {sorted(srcs)}")


def load_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    check_manifest(rows)                      # runs every training run, always
    splits = {"train": [], "seen": [], "unseen": []}
    for r in rows:
        splits[r["split"]].append((r["path"], int(r["label"])))
    return splits


def build_manifest(protocol, audio_dir, out_csv, split="train", ext=".flac"):
    """ASVspoof protocol line:  SPK  UTT  -  ATTACK  bonafide|spoof

    Indexed from the end, so it survives the column differences between years.
    """
    n = 0
    with open(protocol) as f, open(out_csv, "w", newline="") as o:
        w = csv.writer(o)
        w.writerow(COLUMNS)
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            utt, label = parts[1], parts[-1]
            w.writerow([os.path.join(audio_dir, utt + ext),
                        1 if label == "bonafide" else 0, split, "asvspoof"])
            n += 1
    print(f"wrote {out_csv}: {n} rows")


def add_clips(manifest_csv, folder, label, split, source, pattern="*.wav"):
    """Append a folder of clips. Person 5: one call per generator.

        add_clips("manifest.csv", "attacks/xtts",   0, "train",  "xtts")
        add_clips("manifest.csv", "attacks/piper",  0, "train",  "piper")
        add_clips("manifest.csv", "attacks/tortoise", 0, "unseen", "tortoise")

    `source` names the generator. It is what `check_manifest` uses to prove the
    held-out one was never trained on, so give each generator its own name and
    never reuse one across train and unseen.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise SystemExit(f"no {pattern} under {folder}")
    new = not os.path.exists(manifest_csv)
    with open(manifest_csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        for p in paths:
            w.writerow([p, int(label), split, source])
    print(f"added {len(paths)} clips: source={source} split={split} label={label}")


# ------------------------------------------------------------ explainability

def gradcam(net, mel):
    """REQ-4. Which time-frequency regions pushed the score toward *spoof*.

    Returns (N_MELS, FRAMES) in [0, 1]. Grad-CAM over the last conv block:
    weight each channel by its mean gradient, sum, ReLU, upsample.

    We backprop -logit on purpose. The logit points at "bonafide", and the
    question a user has is "what made this look fake" -- so the gradient has to
    be taken toward the spoof direction, not away from it.
    """
    net.eval()
    x = torch.from_numpy(np.ascontiguousarray(mel, np.float32))[None, None]

    feats = {}
    last_conv = net.body[2]                       # third block, before pooling
    h = last_conv.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    logit = net(x)
    h.remove()

    a = feats["a"]                                # (1, C, H, W)
    grad = torch.autograd.grad(-logit.sum(), a)[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * a).sum(dim=1, keepdim=True))

    cam = torch.nn.functional.interpolate(
        cam, size=mel.shape, mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().numpy()

    span = cam.max() - cam.min()
    if span < 1e-8:
        return np.zeros_like(cam)                 # flat map: nothing to show
    return (cam - cam.min()) / span


# ------------------------------------------------------------------ eval

FAR_BUDGET = 0.05          # how often we accept annoying a real caller


def _dataset(rows_or_ds):
    """Accept raw rows or a ready Dataset, so cached and uncached paths share code."""
    return rows_or_ds if isinstance(rows_or_ds, torch.utils.data.Dataset) \
        else Clips(rows_or_ds)


@torch.no_grad()
def collect(net, rows, bs=64):
    net.eval().to(DEVICE)
    scores, labels = [], []
    dl = torch.utils.data.DataLoader(_dataset(rows), batch_size=bs)
    for xb, yb in dl:
        scores += torch.sigmoid(net(xb.to(DEVICE))).cpu().tolist()
        labels += yb.int().tolist()
    return np.array(scores), np.array(labels)


METRICS_JSON = "satyavaani.metrics.json"


def report(net, splits, far_budget=FAR_BUDGET, out=METRICS_JSON):
    """The one runnable check. Seen and unseen, always together.

    Writes the numbers to `out` so the UI can display measured values instead
    of a developer retyping them into HTML. A hand-typed accuracy on a screen
    is indistinguishable from a real one right up until a judge asks which
    generators produced it. `unseen` stays null until it has actually been
    measured -- the UI renders that as NOT MEASURED, never as a blank or a 0.
    """
    s_seen, y_seen = collect(net, splits["seen"])
    e_seen = eer(s_seen, y_seen)

    # Spend the false-alarm budget, then report what detection it bought.
    thr = threshold_for_far(s_seen, y_seen, far_budget)
    det = detection_rate(s_seen, y_seen, thr)
    far = false_alarm_rate(s_seen, y_seen, thr)

    print(f"\nEER seen            {e_seen:.3f}")
    print(f"threshold @ {far_budget:.0%} FA    {thr:.3f}")
    print(f"detection at thr    {det:.3f}   (false alarms {far:.3f})")

    m = {"eer_seen": e_seen, "far_budget": far_budget, "threshold": thr,
         "detection_seen": det, "false_alarm_seen": far,
         "eer_unseen": None, "detection_unseen": None,
         "n_seen": int(len(y_seen)), "n_unseen": 0}

    if splits["unseen"]:
        s_un, y_un = collect(net, splits["unseen"])
        e_un = eer(s_un, y_un)
        det_un = detection_rate(s_un, y_un, thr)     # same threshold, honest
        print(f"EER unseen          {e_un:.3f}   <-- the headline")
        print(f"detection unseen    {det_un:.3f}   (same threshold)")
        m.update(eer_unseen=e_un, detection_unseen=det_un, n_unseen=int(len(y_un)))
    else:
        print("EER unseen          MISSING - person 5 has not delivered yet")

    assert e_seen <= 0.10, f"seen EER {e_seen:.3f} above floor"
    assert det >= 0.50, f"detection {det:.3f} at the FA budget - not worth shipping"
    # No assert on unseen, on purpose: it is the number we are trying to move,
    # and a failing assert only tempts someone to quietly relax it.

    # Written only after the asserts pass. A failing run leaves no metrics file,
    # so the UI says "no evaluation on record" rather than showing numbers from
    # a model that was rejected.
    if out:
        import datetime
        import json
        m["measured_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        with open(out, "w") as f:
            json.dump(m, f, indent=2)
        print(f"wrote {out}")

    return e_seen, thr, det


# ------------------------------------------------------------------ train

def fit(net, rows, epochs=8, bs=32, lr=1e-3, pos_weight=None, augment=True):
    ds = _dataset(rows)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # ASVspoof is ~9:1 spoof:bonafide. Unweighted, the model learns to say
    # "spoof" and still looks good on accuracy while being useless.
    pw = None if pos_weight is None else torch.tensor([pos_weight], device=DEVICE)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    for ep in range(epochs):
        net.train()
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if augment:
                xb = spec_augment(xb)      # training only; never at eval
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        print(f"epoch {ep+1}/{epochs}  loss {total/len(ds):.4f}")
    return net


def smoke():
    """Prove the loop runs before the multi-GB download finishes."""
    net = SpoofCNN()
    x = torch.randn(16, 1, N_MELS, FRAMES)
    y = (torch.rand(16) > 0.5).float()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    before = lossf(net(x), y).item()
    for _ in range(60):
        opt.zero_grad()
        lossf(net(x), y).backward()
        opt.step()
    after = lossf(net(x), y).item()
    assert after < before, f"loss did not move: {before:.4f} -> {after:.4f}"
    assert net(x).shape == (16,), net(x).shape
    print(f"smoke ok: loss {before:.4f} -> {after:.4f}, "
          f"{sum(p.numel() for p in net.parameters())} params")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="satyavaani.pt")
    a = ap.parse_args()

    if a.smoke or not a.manifest:
        smoke()
        if not a.manifest:
            sys.exit(0)

    splits = load_manifest(a.manifest)
    print(f"train {len(splits['train'])}  seen {len(splits['seen'])}  "
          f"unseen {len(splits['unseen'])}")

    net = fit(SpoofCNN(), splits["train"], epochs=a.epochs)
    report(net, splits)
    torch.save(net.state_dict(), a.out)
    print(f"saved {a.out} -- app.py picks it up automatically")
