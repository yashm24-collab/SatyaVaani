"""Record matched-channel clips on the demo hardware.

    python record.py clips/bonafide/yash 20      # 20 clips, 6 s each
    python record.py clips/spoof/xtts 20 --dur 8
    python record.py clips/spoof/edgetts --play fakes/   # play each, record it

Writes 16 kHz mono 16-bit wav -- the only format `satyavaani.load_wav` reads.
Windows Voice Recorder produces .m4a and will not load, so use this.

WHY THIS EXISTS: a model trained on ASVspoof bonafide flags every real
microphone recording as synthetic, because it learned the dataset's capture
chain rather than synthesis artefacts. The fix is clips of BOTH classes through
the SAME chain -- so record bonafide here, and record spoof by playing the
clone through the demo speakers back into this same mic.
"""
import argparse
import glob
import os
import sys
import wave

import numpy as np

from satyavaani import SR, MIN_RMS, load_wav

CLIP_RMS_WARN = 0.98      # peak at/above this many times means the gain is too hot
PLAY_EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".flac")
TAIL_S = 0.4              # keep recording past the end so the room decay lands


def read_any(path):
    """wav via stdlib; anything else needs soundfile, which is optional."""
    if path.lower().endswith(".wav"):
        return load_wav(path)
    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit(
            f"cannot read {os.path.basename(path)} -- only .wav works out of the box.\n"
            "  either generate wav (piper and XTTS already do), or:  pip install soundfile")
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def to_sr(x, sr):
    if sr == SR:
        return np.ascontiguousarray(x, dtype=np.float32)
    idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def check(x):
    """Reject a clip rather than let 100 bad ones through. Returns None if ok."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    clipped = int((np.abs(x) >= CLIP_RMS_WARN).sum())
    if rms < MIN_RMS:
        return f"too quiet (rms {rms:.4f}) -- nothing usable was captured"
    if clipped > len(x) * 0.001:
        return f"clipping ({clipped} samples) -- move back or lower input gain"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("count", type=int, nargs="?",
                    help="how many clips to record (omit when using --play)")
    ap.add_argument("--dur", type=float, default=6.0)
    ap.add_argument("--device", type=int)
    ap.add_argument("--play", metavar="DIR",
                    help="play every audio file in DIR through the speakers and "
                         "record each one back through the mic")
    a = ap.parse_args()

    import sounddevice as sd

    sources = []
    if a.play:
        sources = sorted(p for p in glob.glob(os.path.join(a.play, "*"))
                         if p.lower().endswith(PLAY_EXTS))
        if not sources:
            sys.exit(f"no audio files in {a.play} (looked for {', '.join(PLAY_EXTS)})")
        a.count = len(sources)
        print(f"{a.count} files to play back through the speakers.\n"
              "TURN THE VOLUME UP and do not talk over it. If clips come back "
              "rejected as too quiet, your mic is cancelling the speaker -- turn "
              "off microphone 'enhancements' / noise suppression in Windows sound "
              "settings.\n")
    elif a.count is None:
        sys.exit("give a count (python record.py DIR 20) or use --play DIR")

    os.makedirs(a.outdir, exist_ok=True)
    start = len([f for f in os.listdir(a.outdir) if f.endswith(".wav")])
    if start:
        print(f"{start} clips already in {a.outdir}, continuing from there\n")

    n = start
    idx = 0                   # attempts; in --play mode this is the source pointer
    while n < start + a.count:
        if a.play:
            if idx >= len(sources):
                break         # a rejected playback skips its file, never retries it
            src = sources[idx]
            y = to_sr(*read_any(src))
            print(f"[{idx + 1}/{a.count}] playing {os.path.basename(src)} "
                  f"({len(y)/SR:.1f}s)...")
            y = np.append(y, np.zeros(int(TAIL_S * SR), np.float32))
            x = sd.playrec(y[:, None], samplerate=SR, channels=1,
                           dtype="float32", device=a.device)
        else:
            input(f"[{n - start + 1}/{a.count}] press Enter, then speak "
                  f"for {a.dur:.0f}s... ")
            x = sd.rec(int(a.dur * SR), samplerate=SR, channels=1,
                       dtype="float32", device=a.device)
        sd.wait()
        x = x[:, 0]
        idx += 1

        why = check(x)
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        if why:
            print(f"    REJECTED: {why}\n")
            continue

        p = os.path.join(a.outdir, f"{n:04d}.wav")
        write_wav(p, x)
        bar = "#" * min(30, int(rms * 200))
        print(f"    saved {p}  rms {rms:.4f} {bar}\n")
        n += 1

    print(f"done: {n - start} clips in {a.outdir}")
    print("vary speaker, distance and room between batches -- one voice in one "
          "spot teaches the model that voice in that spot.")


if __name__ == "__main__":
    main()
