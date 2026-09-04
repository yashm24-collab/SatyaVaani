"""Colab training cells. Copy each block into its own Colab cell.

WHY CLONE INSTEAD OF PASTE: training must use the exact same feature code as
inference. If Colab has its own copy of melspec/trim_silence/fix_frames and it
drifts by one line, the checkpoint scores garbage locally and you lose a day
finding out why. Clone the repo, always.

FIRST: Runtime -> Change runtime type -> T4 GPU. Without it, training is 20x
slower and there is no reason to be on Colab at all.
"""

# =====================================================================
# CELL 1 -- setup
# =====================================================================
CELL_1 = r"""
!git clone https://github.com/yashm24-collab/SatyaVaani.git repo
%cd repo
!git pull                      # re-run this cell after any local push

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU -- Runtime > Change runtime type > T4 GPU"

import train, satyavaani as sv
print("device:", train.DEVICE)
train.smoke()                  # proves the loop before any data exists
"""

# =====================================================================
# CELL 2 -- get ASVspoof 2019 LA.  YOU FILL THIS IN.
# =====================================================================
CELL_2 = r"""
# Kaggle credentials.
#
# NEVER paste the token into a cell as a literal. Colab saves cell text AND
# output into the .ipynb, so a pasted token travels with every share, export
# and screenshot of that notebook. getpass keeps it out of both.
#
# Get one: kaggle.com -> avatar -> Settings -> API -> Create New Token
import os, getpass

os.environ["KAGGLE_API_TOKEN"] = getpass.getpass("Kaggle API token (KGAT_...): ").strip()
!pip install -q kaggle

# Older accounts issue kaggle.json (username + key) instead. If the download
# below 401s, use this route instead and upload the file:
#   from google.colab import files; files.upload()
#   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Verified to exist. NOTE the slug really is "asvpoof" -- the typo is theirs.
!kaggle datasets download -d anishsarkar22/asvpoof-2019-dataset-la -p /content/data --unzip

# Fallback mirror if that one is slow or gone:
# !kaggle datasets download -d dhruvtangri1998/asvspoof-dataset-2019 -p /content/data --unzip

!du -sh /content/data && ls /content/data
"""

# =====================================================================
# CELL 3 -- confirm the real paths BEFORE building a manifest
# =====================================================================
CELL_3 = r"""
import glob, os

# Mirrors nest things differently. Find the real root instead of guessing.
hits = glob.glob("/content/data/**/ASVspoof2019_LA_train", recursive=True)
print("candidate roots:", hits)
ROOT = os.path.dirname(hits[0]) if hits else "/content/data/LA"
print("using ROOT =", ROOT)

PROTO_TRN = f"{ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
PROTO_DEV = f"{ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
FLAC_TRN  = f"{ROOT}/ASVspoof2019_LA_train/flac"
FLAC_DEV  = f"{ROOT}/ASVspoof2019_LA_dev/flac"

for p in (PROTO_TRN, PROTO_DEV, FLAC_TRN, FLAC_DEV):
    print(("OK  " if os.path.exists(p) else "MISSING  ") + p)

print("\ntrain flac:", len(glob.glob(FLAC_TRN + "/*.flac")))
print("dev   flac:", len(glob.glob(FLAC_DEV + "/*.flac")))
print("\nfirst protocol line:", open(PROTO_TRN).readline().strip())
"""

# =====================================================================
# CELL 4 -- manifest.  train from trn, seen from dev.
# =====================================================================
CELL_4 = r"""
import csv, random, train

train.build_manifest(PROTO_TRN, FLAC_TRN, "manifest.csv", split="train")

# append dev as the "seen" split
rows = []
for line in open(PROTO_DEV):
    parts = line.split()
    if len(parts) < 3: continue
    rows.append([f"{FLAC_DEV}/{parts[1]}.flac",
                 1 if parts[-1] == "bonafide" else 0, "seen", "asvspoof"])
with open("manifest.csv", "a", newline="") as f:
    csv.writer(f).writerows(rows)

splits = train.load_manifest("manifest.csv")     # runs the leak guard
"""

# =====================================================================
# CELL 5 -- subsample, then featurise ONCE
# =====================================================================
CELL_5 = r"""
import random, numpy as np, train
random.seed(0)

# ponytail: full train set is ~25k clips. Subsampling to 8k cuts featurising
# and training ~3x and still lands a demo-ready number. Raise SUB (or set it
# to None) once the whole pipeline is proven end to end and you have time.
SUB = 8000

def stratified(rows, n):
    if n is None or n >= len(rows): return rows
    bona = [r for r in rows if r[1] == 1]
    spoof = [r for r in rows if r[1] == 0]
    k = max(1, n // 2)
    return random.sample(bona, min(k, len(bona))) + \
           random.sample(spoof, min(n - k, len(spoof)))

tr = stratified(splits["train"], SUB)
ev = stratified(splits["seen"], 3000)
print("train", len(tr), " seen", len(ev))

xtr, ytr = train.precompute(tr, cache="feat_train.npz")
xev, yev = train.precompute(ev, cache="feat_seen.npz")
print(xtr.shape, "bonafide frac", ytr.mean().round(3))
"""

# =====================================================================
# CELL 6 -- train
# =====================================================================
CELL_6 = r"""
import numpy as np, torch, train

ds_tr = train.Cached(xtr, ytr)
ds_ev = train.Cached(xev, yev)

# class balance: ASVspoof is heavily spoof-weighted
pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
print("pos_weight", round(pw, 2))

net = train.fit(train.SpoofCNN(), ds_tr, epochs=12, bs=64, lr=1e-3, pos_weight=pw)
"""

# =====================================================================
# CELL 7 -- eval.  seen and unseen, same threshold.
# =====================================================================
CELL_7 = r"""
import train
e_seen, thr, det = train.report(net, {"seen": ds_ev, "unseen": []})
print("\nthreshold to carry into the demo:", round(thr, 3))
"""

# =====================================================================
# CELL 8 -- export
# =====================================================================
CELL_8 = r"""
import torch
from google.colab import files
torch.save(net.state_dict(), "satyavaani.pt")
files.download("satyavaani.pt")
# drop it next to satyavaani.py, then LOCALLY:  python verify_checkpoint.py
"""

CELLS = [CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6, CELL_7, CELL_8]

if __name__ == "__main__":
    for i, c in enumerate(CELLS, 1):
        print(f"\n{'='*66}\nCELL {i}\n{'='*66}{c}")
