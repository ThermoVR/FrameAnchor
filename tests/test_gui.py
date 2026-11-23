"""Worker tested against the fake SteamVR; window code exercised with a stand-in for tkinter."""
import queue, time, types
from unittest import mock
import numpy as np
import frame_anchor as fa, frame_anchor_gui as g
from test_frame_anchor import FakeWorld, STATIONS


def drain(q):
    out = []
    while True:
        try: out.append(q.get_nowait())
        except queue.Empty: return out


def wait_for(q, pred, timeout=8.0, seen=None):
    seen = [] if seen is None else seen
    end = time.time() + timeout
    while time.time() < end:
        seen += drain(q)
        if any(pred(e) for e in seen): return seen
        time.sleep(0.02)
    raise AssertionError(f"timed out; saw {seen[-8:]}")


def make_world():
    w = FakeWorld(STATIONS)
    w.events = []
    w.poll_events = lambda: [w.events.pop(0)] if w.events else []
    w.shutdown = lambda: None
    return w


def test_worker_full_session(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "LOG_FILE", str(tmp_path / "log.txt"))
    w = make_world()
    available = {"up": False}
    def factory():
        if not available["up"]: raise RuntimeError("SteamVR not running")
        return w
    q = queue.Queue()
    wk = g.Worker(factory, str(tmp_path / "anchor.json"), q, tick=0.01, retry=0.05)
    wk.settings.settle_s = 0.2; wk.settings.cooldown_s = 0.1; wk.settings.anchor_avg_s = 0.1
    wk.start()
    try:
        wait_for(q, lambda e: e[0] == "status" and e[1] == "waiting")
        available["up"] = True
        wait_for(q, lambda e: e[0] == "status" and e[1] == "no-anchor" and e[4] == 2)
        wk.cmds.put(("lock",))
        wait_for(q, lambda e: e[0] == "status" and e[1] == "ok")
        assert (tmp_path / "anchor.json").exists()
        # Frame relocalizes: floor drops 15 cm
        d = np.eye(4); d[:3, 3] = [0.1, -0.15, 0]
        w.relocalize(d); w.spacecal_converges()
        seen = wait_for(q, lambda e: e[0] == "status" and e[1] == "corrected")
        assert np.allclose(w.standing_of_real(np.zeros(3)), 0, atol=1e-6)
        # test mode: nothing changes
        wk.cmds.put(("dry", True)); calls = w.set_calls
        w.relocalize(d); w.spacecal_converges()
        wait_for(q, lambda e: e[0] == "log" and "dry-run" in e[1])
        assert w.set_calls == calls
        wk.cmds.put(("dry", False))
        wait_for(q, lambda e: e[0] == "status" and e[1] == "corrected" and e[2] >= 3)
        # SteamVR quits, then comes back: saved lock is reused
        available["up"] = False
        w.events.append("Quit")
        wait_for(q, lambda e: e[0] == "log" and "SteamVR closed" in e[1])
        available["up"] = True
        wait_for(q, lambda e: e[0] == "log" and "saved lock" in e[1], timeout=10)
        # backend blows up -> reconnects instead of dying
        orig = w.hmd_ok
        w.hmd_ok = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        wait_for(q, lambda e: e[0] == "log" and "reconnecting" in e[1])
        w.hmd_ok = orig
        wait_for(q, lambda e: e[0] == "status" and e[1] in ("ok", "settling"))
        assert wk.is_alive()
    finally:
        wk.stop_flag.set(); wk.join(3)


def test_window_code_paths():
    tk = mock.MagicMock()
    tk.IntVar.return_value.get.return_value = 0
    wk = types.SimpleNamespace(cmds=queue.Queue(), stop_flag=mock.MagicMock())
    q = queue.Queue()
    app = g.App(tk, wk, q)
    for st in ["waiting", "no-anchor", "paused", "ok", "corrected", "settling", "settling (still moving)",
               "hmd-not-tracking", "no-anchor-devices-visible", "shape-mismatch rms=12.0cm",
               "shift-too-large 4.00m", "backing-off", "cooldown", "weird"]:
        q.put(("status", st, 2, "dy(height)=-11.8cm", 2))
        h, d, c = g.friendly(st, False); assert h and c.startswith("#")
    q.put(("log", "hello"))
    app.pump()
    app.on_lock(); app.on_pause(); app.on_pause(); app.on_dry(); app.on_forget(); app.close()
    assert [c[0] for c in drain(wk.cmds)] == ["lock", "pause", "pause", "dry", "forget"]


def test_fallback_to_console(monkeypatch, capsys):
    monkeypatch.setattr(g, "run_gui", lambda: (_ for _ in ()).throw(ImportError("no tkinter")))
    monkeypatch.setattr(fa, "main", lambda argv=None: 42)
    assert g.main() == 42
