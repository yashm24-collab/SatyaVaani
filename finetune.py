"""Adapt a checkpoint to your own microphone. Runs locally, no Colab, no GPU.

    python finetune.py --holdout tortoise

Expects clips recorded with record.py, laid out like:

    clips/bonafide/<anyone>/*.wav      real people, your demo mic
    clips/spoof/<engine>/*.wav         clones PLAYED THROUGH THE DEMO SPEAKERS
                                       and recorded on that same mic

WHY: a model trained on ASVspoof alone learns that dataset's capture chain, not
synthesis artefacts -- it scores every real microphone recording as spoof. Both
classes have to come through the SAME chain so the only thing left to learn is
the synthesis. Clean TTS files against mic-recorded speech teaches "mic = real",
which fails the moment a clone is played through speakers on stage.

The --holdout engine is never trained on. That is the number that matters.
"""
import argparse
import csv
import glob
import os
import random
import sys

BONAFIDE_CUTS = (0.70, 0.85)      # train / seen / unseen
SPOOF_CUT = 0.80                  # train / seen  (trained engines never hit unseen)


def collect(clips_dir, holdout):
    """Folder layout -> manifest rows. Returns (rows, engines_found)."""
    rng = random.Random(0)
    rows = []

    bona = sorted(glob.glob(os.path.join(clips_dir, "bonafide", "**", "*.wav"),
                            recursive=True))
    if not bona:
        sys.exit(f"no bonafide clips under {clips_dir}/bonafide/ "
                 f"-- run: python record.py {clips_dir}/bonafide/<yourname> 20")
    rng.shuffle(bona)
    a, b = (int(len(bona) * c) for c in BONAFIDE_CUTS)
    for split, chunk in (("train", bona[:a]), ("seen", bona[a:b]), ("unseen", bona[b:])):
        rows += [[p, 1, split, "real_mic"] for p in chunk]

    engines = {}
    for d in sorted(glob.glob(os.path.join(clips_dir, "spoof", "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        ps = sorted(glob.glob(os.path.join(d, "*.wav")))
        if not ps:
            continue
        engines[name] = len(ps)
        rng.shuffle(ps)
        if name == holdout:
            rows += [[p, 0, "unseen", name] for p in ps]     # never trained on
        else:
            k = int(len(ps) * SPOOF_CUT)
            rows += [[p, 0, "train", name] for p in ps[:k]]
            rows += [[p, 0, "seen", name] for p in ps[k:]]
    return rows, engines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--holdout", required=True,
                    help="engine folder name to hold out of training entirely")
    ap.add_argument("--init", default="satyavaani_noise.pt",
                    help="checkpoint to adapt from")
    ap.add_argument("--out", default="satyavaani_matched.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    import torch
    import train

    rows, engines = collect(a.clips, a.holdout)
    if a.holdout not in engines:
        sys.exit(f"--holdout '{a.holdout}' not found. engines present: "
                 f"{sorted(engines) or 'none'}")
    print("engines:", ", ".join(f"{k}={v}" for k, v in sorted(engines.items())))

    with open("matched.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(train.COLUMNS)
        w.writerows(rows)
    splits = train.load_manifest("matched.csv")        # runs the leak guard

    net = train.SpoofCNN()
    if os.path.exists(a.init):
        net.load_state_dict(torch.load(a.init, map_location="cpu"))
        print(f"adapting from {a.init}")
    else:
        print(f"{a.init} not found -- training from scratch")

    net = train.fit(net, splits["train"], epochs=a.epochs, bs=32, lr=a.lr,
                    augment=False)
    train.report(net, splits)                          # asserts live here

    torch.save(net.state_dict(), a.out)
    print(f"\nsaved {a.out}")
    print(f"next:  python verify_checkpoint.py --ckpt {a.out} --wav myvoice.wav")


if __name__ == "__main__":
    main()
