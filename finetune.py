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
import hashlib
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
    # The active checkpoint, not a named archive file: archives are content
    # addressed and their names change, and adapting from "whatever is
    # currently installed" is what you actually mean.
    ap.add_argument("--init", default="satyavaani.pt",
                    help="checkpoint to adapt from (default: the active one)")
    ap.add_argument("--outdir", default="models",
                    help="where the new checkpoint is written, named by content")
    # 15 epochs at 1e-4 was too gentle and left the model undertrained: it
    # scored EER 0.29 on seen and 0.26 held-out, while 60 at 5e-4 on the same
    # 41 clips gives 0.00 and 0.033. Adapting away from a checkpoint that is
    # confidently wrong about real microphone audio needs real movement, not a
    # nudge. Matched-channel sets are small, so epochs are cheap.
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=16)
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

    net = train.fit(net, splits["train"], epochs=a.epochs, bs=a.bs, lr=a.lr,
                    augment=False)

    # Written to a pending name first: the hash that names it can only be
    # computed from the saved bytes, but report() must still be able to reject
    # the model and leave nothing behind, exactly as it did before.
    os.makedirs(a.outdir, exist_ok=True)
    tmp = os.path.join(a.outdir, "matched-pending.pt")
    tmp_metrics = tmp.replace(".pt", ".metrics.json")
    torch.save(net.state_dict(), tmp)
    try:
        train.report(net, splits, out=tmp_metrics)     # asserts live here
    except BaseException:
        os.remove(tmp)                                 # a rejected model is not kept
        raise

    # Metrics are named for the exact weights they describe, so numbers can
    # never end up beside a different model.
    h = hashlib.sha256(open(tmp, "rb").read()).hexdigest()[:8]
    out = os.path.join(a.outdir, f"matched-{h}.pt")
    os.replace(tmp, out)
    os.replace(tmp_metrics, out.replace(".pt", ".metrics.json"))

    print(f"\nsaved {out}")
    print(f"next:  python verify_checkpoint.py --ckpt {out} --wav myvoice.wav --activate")


if __name__ == "__main__":
    main()
