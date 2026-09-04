"""Audio capture. Owner: person 3.

One class, two sources, same interface:

    m = Mic()               # or FileMic("clip.wav") when there is no mic
    m.start()
    win = m.get()           # newest 2 s window, or None if nothing new yet
    m.stop()

`get()` is non-blocking so the Tk loop can poll it. Run this file to check
your mic works before wiring anything up:  python mic.py
"""
import queue
import numpy as np

from satyavaani import SR, WINDOW_S, HOP_S, load_wav

WIN_N = int(WINDOW_S * SR)
HOP_N = int(HOP_S * SR)


class _Base:
    def __init__(self):
        self._buf = np.zeros(WIN_N, dtype=np.float32)
        self._q = queue.Queue(maxsize=4)
        self._primed = 0        # samples seen; suppress output until buffer full

    def _push(self, block):
        """Slide `block` into the ring buffer, emit a full window."""
        n = len(block)
        if n >= WIN_N:
            self._buf = block[-WIN_N:].copy()
        else:
            self._buf = np.roll(self._buf, -n)
            self._buf[-n:] = block
        self._primed += n
        if self._primed < WIN_N:
            return                      # do not emit a half-empty window
        try:
            self._q.put_nowait(self._buf.copy())
        except queue.Full:
            # UI stalled. Drop the OLDEST and keep the newest -- dropping the
            # new one instead would leave `get()` serving a stale window once
            # the UI recovers.
            try:
                self._q.get_nowait()
                self._q.put_nowait(self._buf.copy())
            except (queue.Empty, queue.Full):
                pass

    def get(self):
        """Newest window, discarding any backlog. None if nothing new."""
        item = None
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return item

    def start(self):
        raise NotImplementedError

    def stop(self):
        pass


class Mic(_Base):
    """Live microphone. Needs `pip install sounddevice`."""

    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self._stream = None
        self.dropped = 0        # callbacks that reported an overflow

    def start(self):
        import sounddevice as sd        # imported late so FileMic works without it
        self._stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype="float32",
            blocksize=HOP_N,            # one callback per hop
            device=self.device,
            callback=self._cb,
        )
        self._stream.start()

    def _cb(self, indata, frames, time_info, status):
        # ponytail: keep this callback trivial. Real work belongs on the UI
        # thread; anything slow here drops audio frames.
        if status:
            # Input overflow means audio was LOST, not delayed. Silently
            # swallowing it is how a demo degrades without anyone noticing.
            self.dropped += 1
        self._push(indata[:, 0].copy())

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class FileMic(_Base):
    """Replays a wav at the same hop rate. Demo fallback and offline testing.

    Person 6: this is the path that saves the demo when the venue mic dies.
    """

    def __init__(self, path, realtime=True, loop=True):
        super().__init__()
        x, sr = load_wav(path)
        if sr != SR:
            idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
            x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        self.x = x
        self.realtime = realtime
        # Loops by default. A fallback clip that runs out mid-demo freezes the
        # display on a stale verdict while the LED still reads LIVE -- which is
        # worse than no fallback, because nobody notices it stopped.
        self.loop = loop
        self._pos = 0
        self._timer = None

    def start(self):
        if self.realtime:
            import threading
            self._timer = threading.Event()
            threading.Thread(target=self._run, daemon=True).start()
        else:
            # One pass, always. Looping here would spin forever -- the realtime
            # thread is the only place a loop can be interrupted.
            loop, self.loop = self.loop, False
            while self._step():
                pass
            self.loop = loop

    def _run(self):
        while not self._timer.is_set() and self._step():
            self._timer.wait(HOP_S)

    def _step(self):
        if self._pos >= len(self.x):
            if not self.loop:
                return False
            self._pos = 0
        self._push(self.x[self._pos : self._pos + HOP_N])
        self._pos += HOP_N
        return True

    def stop(self):
        if self._timer is not None:
            self._timer.set()


def _self_check():
    """Ring buffer correctness. No audio device needed, so it always runs."""
    import queue as _q

    b = _Base()
    assert b.get() is None, "must not emit before the buffer is primed"

    # feed a ramp in hop-sized blocks; the window must hold the newest samples
    total = WIN_N * 3
    ramp = np.arange(total, dtype=np.float32)
    for i in range(0, total, HOP_N):
        b._push(ramp[i : i + HOP_N])
    w = b.get()
    assert w is not None and len(w) == WIN_N, "window size"
    assert np.array_equal(w, ramp[total - WIN_N :]), "window is not the newest audio"

    # a block bigger than the window keeps only its tail
    b2 = _Base()
    big = np.arange(WIN_N * 2, dtype=np.float32)
    b2._push(big)
    assert np.array_equal(b2.get(), big[-WIN_N:]), "oversized block"

    # priming boundary: exactly one sample short emits nothing
    b3 = _Base()
    b3._push(np.ones(WIN_N - 1, dtype=np.float32))
    assert b3.get() is None, "emitted a half-full window"
    b3._push(np.ones(1, dtype=np.float32))
    assert b3.get() is not None, "did not emit once primed"

    # a stalled reader must end up with the NEWEST window, not a stale one
    b4 = _Base()
    b4._primed = WIN_N
    for k in range(12):                       # more pushes than queue maxsize
        b4._push(np.full(HOP_N, float(k), dtype=np.float32))
    last = b4.get()
    assert last[-1] == 11.0, f"stale window survived a stall: {last[-1]}"

    # FileMic: non-realtime must terminate even though looping is the default,
    # and the looping path must actually wrap instead of running dry.
    import os
    import tempfile
    import wave as _wave

    tmp = os.path.join(tempfile.gettempdir(), "_miccheck.wav")
    ramp = (np.linspace(-0.5, 0.5, WIN_N * 2).astype(np.float32))
    with _wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((ramp * 32767).astype("<i2").tobytes())

    f = FileMic(tmp, realtime=False)          # must not hang
    f.start()
    assert f.get() is not None, "non-realtime FileMic produced nothing"

    f2 = FileMic(tmp, realtime=False, loop=True)
    f2.start()                                 # loop must be ignored here
    assert f2.get() is not None

    f3 = FileMic(tmp, realtime=True, loop=True)
    f3._pos = len(f3.x)                        # sitting at the end
    assert f3._step() is True, "loop did not wrap"
    assert f3._pos == HOP_N, "loop did not reset position"

    f4 = FileMic(tmp, realtime=True, loop=False)
    f4._pos = len(f4.x)
    assert f4._step() is False, "loop=False should stop at the end"

    os.remove(tmp)
    print("mic ring-buffer self-check ok")


def list_devices():
    import sounddevice as sd
    print("input devices (use the number with --device N):\n")
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        mark = " <- default" if i == default_in else ""
        print(f"  [{i:2d}] {d['name']}  "
              f"({d['max_input_channels']} ch, {int(d['default_samplerate'])} Hz){mark}")


if __name__ == "__main__":
    import sys
    import time

    _self_check()

    if "--list" in sys.argv:
        list_devices()
        raise SystemExit(0)

    if "--live" not in sys.argv:
        print("\nnext:  python mic.py --list     see your input devices")
        print("       python mic.py --live     full chain check on real audio")
        raise SystemExit(0)

    dev = None
    if "--device" in sys.argv:
        dev = int(sys.argv[sys.argv.index("--device") + 1])

    # Full chain, not just capture: mic -> window -> verdict. This is the one
    # command that answers "does the actual product path work on my laptop".
    import satyavaani as sv
    scorer = sv.get_scorer()
    print(f"\nscorer: {'PLACEHOLDER (expected before training)' if scorer.is_placeholder else 'trained model'}")
    print("speak normally for 8 s, then stay silent for the last 2 s...\n")

    m = Mic(device=dev)
    try:
        m.start()
    except Exception as e:
        raise SystemExit(
            f"mic failed: {e}\n"
            "  pip install sounddevice\n"
            "  python mic.py --list   then retry with --device N")

    t0, seen, refused = time.time(), 0, 0
    while time.time() - t0 < 10:
        w = m.get()
        if w is not None:
            seen += 1
            rms = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
            score, band, _ = sv.verdict(w, scorer)
            if score is None:
                refused += 1
            bar = "#" * min(30, int(rms * 300))
            shown = "  --  " if score is None else f"{score:.3f}"
            print(f"rms {rms:.4f} {bar:<30} {shown}  {band}")
        time.sleep(0.1)
    m.stop()

    print(f"\nwindows {seen}   refused {refused}   overflows {m.dropped}")
    ok = True
    if seen == 0:
        print("FAIL: no audio. Wrong device? try --list then --device N"); ok = False
    if m.dropped:
        print("WARN: overflow means audio was LOST. Close other apps.")
    if refused == 0:
        print("WARN: silence never triggered the REQ-6 refusal - mic gain may be high")
    print("live chain ok" if ok else "live chain FAILED")
