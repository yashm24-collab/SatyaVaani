# SatyaVaani

Real-time synthetic-speech detection. SIH26104.

## Run now (no torch needed)

```
python satyavaani.py           # self-check, must print "ok"
python mic.py                  # 5 s mic test, prints a level bar
python server.py               # live demo -> http://localhost:8000
python server.py demo_clip.wav # replay a file (demo fallback path)
```

`pip install sounddevice` for the mic. Everything else is numpy + stdlib.

The file path **loops**. A fallback clip that runs dry mid-demo freezes the
display on a stale verdict while the LED still reads LIVE, which is worse than
having no fallback because nobody notices it stopped. The `Latency` readout is
measured per window and turns red if compute exceeds the 0.5 s hop.

With torch installed, two more:

```
python train.py --smoke        # training loop moves loss, needs no data
python test_e2e.py             # full chain: train -> save -> swap -> verdict
```

`test_e2e.py` trains on synthetic audio so the **day-4 swap is not the first
time anyone runs it**. Its EER of 0.000 is meaningless — the fake spoof class is
band-limited and trivially separable. It proves wiring, not detection.

## Files

| File | Owner | What |
|---|---|---|
| `satyavaani.py` | shared, **frozen** | features, `verdict()`, bands, EER. Change only by agreement. |
| `mic.py` | person 3 | `Mic` / `FileMic` -> `.start() .get() .stop()` |
| `requirements.txt` | shared | numpy runs everything; torch only for persons 1+2 |
| `server.py` + `index.html` | person 4 | stdlib HTTP server + signal-analyser UI |
| `train.py` | persons 1+2 | CNN, training, seen/unseen eval, writes `satyavaani.pt` |

## The interface — frozen, do not renegotiate

```python
satyavaani.verdict(x, scorer, sr) -> (score | None, band, action)
mic.Mic().get()                   -> np.float32 window, or None
train.SpoofCNN                    -> writes satyavaani.pt
```

`server.py` calls `verdict()`. It does not know or care whether the scorer is
the placeholder or the trained model. Day 4 swap is zero lines in the UI.

## Placeholder scorer

Until `satyavaani.pt` exists, `get_scorer()` returns `HeuristicScorer`.

**It is not a detector.** It is a spectral heuristic that returns a plausible
number so persons 3, 4 and 6 can build and rehearse on day 1. The app shows a
red PLACEHOLDER banner while it is active. Never demo it as a result, and never
put a number from it on a slide.

## Training

```
python train.py --smoke                     # proves the loop, needs no data
python train.py --manifest manifest.csv
```

`manifest.csv`: `path,label,split,source` — label 1 bonafide / 0 spoof,
split `train` | `seen` | `unseen`, source names the generator.

```python
import train
train.build_manifest("ASVspoof2019.LA.cm.train.trn.txt", "flac/", "manifest.csv", "train")
train.add_clips("manifest.csv", "attacks/xtts",     0, "train",  "xtts")
train.add_clips("manifest.csv", "attacks/piper",    0, "train",  "piper")
train.add_clips("manifest.csv", "attacks/tortoise", 0, "unseen", "tortoise")
```

**`load_manifest` refuses to run if a spoof `source` appears in both `train` and
`unseen`.** That leak turns your headline number from a generalisation result
into a memorisation result, and nothing downstream can tell the difference. The
guard runs on every training run and cannot be forgotten.

(Genuine speech is exempt — you need bonafide clips in every split to compute an
EER at all.)

### How the threshold is chosen

Not 0.5. `report()` spends a **false-alarm budget** (default 5%) and reports what
detection that budget bought:

```
threshold @ 5% FA    0.859      <- derived from genuine speech, not guessed
detection at thr     1.000      <- what the budget buys
EER unseen           0.312      <- the headline
detection unseen     0.640      <- same threshold, no quiet re-tuning
```

The unseen split is scored at the **same** threshold. Re-tuning per split is how
a number stops meaning anything.

`report()` asserts seen EER <= 0.10 and detection >= 0.50 at the budget, and
refuses to save otherwise. There is deliberately no assert on the unseen number
— that is the number you are trying to move, and an assert only tempts someone
to quietly relax it.

Threshold is a **product decision** (PRD Q2): it spends user annoyance to buy
detection. Someone owns that number. Do not let it default silently.

## Training on Colab

`python colab.py` prints 8 cells to paste in. Runtime -> Change runtime type -> **T4 GPU** first.

Cell 1 clones this repo rather than pasting code, on purpose: training must use
the **exact** feature functions inference uses. A one-line drift in a pasted
copy of `melspec` produces a checkpoint that scores garbage locally, and you
lose a day finding out why.

Then, before it goes anywhere near the demo:

```
python verify_checkpoint.py                  # ./satyavaani.pt
python verify_checkpoint.py --wav myvoice.wav
```

Catches what training never shows you: a model collapsed to one class, a
feature mismatch, a flat evidence map, non-determinism, and -- with `--wav` --
a model that flags **your own voice** as synthetic. That last one would fail
live, in front of judges, with you as the input.

It proves the checkpoint is not broken. It does not prove accuracy; that is
`report()` in Colab.

## Day 1 checklist

- [ ] everyone: `python satyavaani.py` prints ok
- [ ] person 3: `python mic.py` shows a moving bar
- [ ] person 4: `python server.py demo_clip.wav` -> browser shows a moving trace
- [ ] persons 1+2: `python train.py --smoke` and `python test_e2e.py` pass, ASVspoof downloading
- [ ] person 5: one TTS tool installed, one clip generated
- [ ] person 6: loud room booked, fallback clip recorded
