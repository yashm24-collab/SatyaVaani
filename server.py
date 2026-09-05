"""SatyaVaani demo server. Owner: person 4.

    python server.py                # live mic
    python server.py clip.wav       # replay a file (demo fallback)

Python keeps the audio and the model. The browser only draws. The seam is one
JSON blob per window, polled over plain HTTP.

ponytail: stdlib http.server + polling. A websocket would be the "right" answer
and costs a dependency plus an async story to buy nothing at 2 Hz.
"""
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np

import satyavaani as sv
from mic import Mic, FileMic

PORT = 8000
WAVE_POINTS = 128          # envelope resolution sent to the browser
HEAT_W, HEAT_H = 64, 24    # Grad-CAM sent downsampled; quantised to 0-255

_state = {
    "score": None, "band": "none", "action": "Waiting for audio",
    "placeholder": True, "wave": [0.0] * WAVE_POINTS, "heat": None,
    "ms": None, "behind": False, "seq": 0,
}
_lock = threading.Lock()

HOP_MS = sv.HOP_S * 1000       # if compute exceeds this, we cannot keep up


def envelope(x, n=WAVE_POINTS):
    """Peak envelope of the real window. The scope shows actual audio, not decor."""
    edges = np.linspace(0, len(x), n + 1).astype(int)
    return [float(np.abs(x[a:b]).max()) if b > a else 0.0
            for a, b in zip(edges[:-1], edges[1:])]


def shrink(cam, w=HEAT_W, h=HEAT_H):
    """(mels, frames) float -> flat row-major ints, small enough to poll."""
    ys = np.linspace(0, cam.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, cam.shape[1] - 1, w).astype(int)
    return (cam[np.ix_(ys, xs)] * 255).astype(np.uint8).ravel().tolist()


def run_audio(source, scorer):
    """Poll the mic, score, publish. One thread, no async."""
    while True:
        win = source.get()
        if win is None:
            time.sleep(0.05)
            continue
        t0 = time.perf_counter()
        score, band, action = sv.verdict(win, scorer)

        heat = None
        if score is not None and hasattr(scorer, "explain"):
            try:
                heat = shrink(scorer.explain(win))
            except Exception:
                heat = None            # evidence is optional; the verdict is not

        wave = envelope(win)
        # Measured, not asserted. A hardcoded latency on a readout is the kind
        # of number a judge asks about once. Grad-CAM is ~25 ms on the small
        # CNN; a pretrained backbone will cost far more, and `behind` says so.
        ms = (time.perf_counter() - t0) * 1000

        with _lock:
            _state.update(
                score=score, band=band, action=action,
                placeholder=getattr(scorer, "is_placeholder", False),
                wave=wave, heat=heat, ms=round(ms, 1),
                behind=ms > HOP_MS, seq=_state["seq"] + 1,
            )


METRICS_JSON = "satyavaani.metrics.json"


def read_metrics():
    """Measured eval numbers from train.report(), or None if never run.

    None is a real answer and the UI says so. The alternative -- a developer
    typing an accuracy into the page -- looks identical to a measured one and
    survives right up to the question "which generators produced that?".
    """
    p = Path(__file__).parent / METRICS_JSON
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/verdict"):
            with _lock:
                snapshot = dict(_state)
            self._json(snapshot)
            return
        if self.path.startswith("/metrics"):
            # `config` is the pipeline's own constants, so the UI draws the
            # real band cutoffs instead of a second copy that can drift.
            # `eval` re-read per request, so re-running report() shows up on
            # refresh without restarting the server.
            self._json({
                "config": {
                    "sr": sv.SR, "window_s": sv.WINDOW_S, "hop_s": sv.HOP_S,
                    "n_mels": sv.N_MELS, "frames": sv.FRAMES,
                    "min_speech_s": sv.MIN_SPEECH_S, "min_rms": sv.MIN_RMS,
                    "hop_budget_ms": HOP_MS,
                    "bands": [[lo, name] for lo, name, _ in sv.BANDS],
                },
                "eval": read_metrics(),
            })
            return
        # index.html, so SimpleHTTPRequestHandler serves "/" with no special
        # case here and GitHub Pages can host the same file unchanged.
        return super().do_GET()

    def log_message(self, *a):
        pass                      # a request line every 150 ms is not useful


def main():
    scorer = sv.get_scorer()
    source = FileMic(sys.argv[1]) if len(sys.argv) > 1 else Mic()
    try:
        source.start()
    except Exception as e:
        sys.exit(f"audio failed: {e}\ntry: pip install sounddevice")

    threading.Thread(target=run_audio, args=(source, scorer), daemon=True).start()

    if scorer.is_placeholder:
        print("!! PLACEHOLDER SCORER - not a detector. Train a model first.")
    print(f"http://localhost:{PORT}")

    import os
    os.chdir(Path(__file__).parent)
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        source.stop()


if __name__ == "__main__":
    main()
