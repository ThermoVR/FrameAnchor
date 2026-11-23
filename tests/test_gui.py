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
    for st in ["waiting", "no-anchor", "paused", "ok", "corrected", "settling", "settling (still moving)", "being-moved",
               "hmd-not-tracking", "no-anchor-devices-visible", "shape-mismatch rms=12.0cm",
               "shape-mismatch worst=6.2cm", "shift-too-large 4.00m", "backing-off", "cooldown",
               "confirm-needed dx=+50.0cm", "ok", "write-refused", "crashed", "weird"]:
        q.put(("status", st, 2, "dy(height)=-11.8cm", 2))
        h, d, c = g.friendly(st, False); assert h and c.startswith("#")
    for st in ["shape-mismatch worst=6.2cm", "confirm-needed dx=+50.0cm", "write-refused"]:
        assert g.friendly(st, False)[2] == g.ORANGE
    q.put(("log", "hello"))
    q.put(("paused", True))
    app.pump()
    assert app.paused
    for i in range(g.LOGBOX_MAX_LINES + 10):
        app.add_log(f"line {i}")
    assert app.log_lines <= g.LOGBOX_MAX_LINES
    app.set_paused(False)
    app.on_lock(); app.on_pause(); app.on_pause(); app.on_dry(); app.on_confirm(); app.on_undo()
    app.on_forget(); app.close()
    assert [c[0] for c in drain(wk.cmds)] == ["lock", "pause", "pause", "dry", "confirm", "undo", "forget"]
    # saying "No" in the confirmation dialogs sends nothing
    tk.messagebox.askyesno.return_value = False
    app.on_undo(); app.on_forget()
    assert drain(wk.cmds) == []


def test_worker_lock_resumes_and_undo_pauses(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "LOG_FILE", str(tmp_path / "log.txt"))
    w = make_world()
    z0 = w.Z.copy()
    q = queue.Queue()
    wk = g.Worker(lambda: w, str(tmp_path / "anchor.json"), q, tick=0.01, retry=0.05)
    wk.settings.settle_s = 0.2; wk.settings.cooldown_s = 0.1; wk.settings.anchor_avg_s = 0.1
    wk.start()
    try:
        wait_for(q, lambda e: e[0] == "status" and e[1] == "no-anchor")
        wk.cmds.put(("pause", True))
        wait_for(q, lambda e: e[0] == "paused" and e[1] is True)
        wk.cmds.put(("lock",))                     # Pause -> fix floor -> Lock: lock must resume
        wait_for(q, lambda e: e[0] == "paused" and e[1] is False)
        wait_for(q, lambda e: e[0] == "status" and e[1] == "ok")
        d = np.eye(4); d[:3, 3] = [0.1, -0.15, 0]
        w.relocalize(d); w.spacecal_converges()
        wait_for(q, lambda e: e[0] == "status" and e[1] == "corrected")
        assert not np.allclose(w.Z, z0)
        wk.cmds.put(("undo",))
        wait_for(q, lambda e: e[0] == "status" and e[1] == "paused")
        assert np.allclose(w.Z, z0)
    finally:
        wk.stop_flag.set(); wk.join(3)


def test_worker_reports_damaged_lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "LOG_FILE", str(tmp_path / "log.txt"))
    (tmp_path / "anchor.json").write_text('{"created": 1, "poses": []}')
    w = make_world()
    q = queue.Queue()
    wk = g.Worker(lambda: w, str(tmp_path / "anchor.json"), q, tick=0.01, retry=0.05)
    wk.start()
    try:
        wait_for(q, lambda e: e[0] == "log" and "damaged" in e[1])
        wait_for(q, lambda e: e[0] == "status" and e[1] == "no-anchor")
        assert wk.is_alive()
    finally:
        wk.stop_flag.set(); wk.join(3)


def test_log_file_rotates(tmp_path, monkeypatch):
    log = tmp_path / "frame_anchor_log.txt"
    monkeypatch.setattr(g, "LOG_FILE", str(log))
    monkeypatch.setattr(g, "LOG_MAX_BYTES", 100)
    wk = g.Worker(None, str(tmp_path / "a.json"), queue.Queue())
    for i in range(20):
        wk.log(f"message number {i}")
    assert (tmp_path / "frame_anchor_log.old.txt").exists() and log.stat().st_size < 200


def test_fallback_to_console(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(g, "LOG_FILE", str(tmp_path / "log.txt"))   # don't write into the real log
    monkeypatch.setattr(g.fa, "acquire_single_instance", lambda *a, **k: True)
    monkeypatch.setattr(g, "run_gui", lambda: (_ for _ in ()).throw(ImportError("no tkinter")))
    monkeypatch.setattr(fa, "main", lambda argv=None: 42)
    assert g.main() == 42


def test_remove_lock_keeps_undo(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "LOG_FILE", str(tmp_path / "log.txt"))
    w = make_world()
    z0 = w.Z.copy()
    q = queue.Queue()
    wk = g.Worker(lambda: w, str(tmp_path / "anchor.json"), q, tick=0.01, retry=0.05)
    wk.settings.settle_s = 0.2; wk.settings.cooldown_s = 0.1; wk.settings.anchor_avg_s = 0.1
    wk.start()
    try:
        wait_for(q, lambda e: e[0] == "status" and e[1] == "no-anchor")
        wk.cmds.put(("lock",))
        wait_for(q, lambda e: e[0] == "status" and e[1] == "ok")
        d = np.eye(4); d[:3, 3] = [0.1, -0.15, 0]
        w.relocalize(d); w.spacecal_converges()
        wait_for(q, lambda e: e[0] == "status" and e[1] == "corrected")
        wk.cmds.put(("forget",))
        wait_for(q, lambda e: e[0] == "log" and "Undo fixes" in e[1] and "still works" in e[1])
        assert not (tmp_path / "anchor.json").exists() and (tmp_path / "frame_anchor_undo.json").exists()
        # SteamVR restarts: undo must still work from the separate undo file
        wk.cmds.put(("pause", False)); w.events.append("Quit")
        wait_for(q, lambda e: e[0] == "log" and "SteamVR closed" in e[1])
        wait_for(q, lambda e: e[0] == "status" and e[1] == "no-anchor", timeout=10)
        wk.cmds.put(("undo",))
        wait_for(q, lambda e: e[0] == "log" and e[1].find("Undone") >= 0)
        assert np.allclose(w.Z, z0)
    finally:
        wk.stop_flag.set(); wk.join(3)


def test_late_put_it_back_click_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "LOG_FILE", str(tmp_path / "log.txt"))
    w = make_world()
    q = queue.Queue()
    wk = g.Worker(lambda: w, str(tmp_path / "anchor.json"), q, tick=0.01, retry=0.05)
    wk.start()
    try:
        wait_for(q, lambda e: e[0] == "status" and e[1] == "no-anchor")
        wk.cmds.put(("confirm",))
        wait_for(q, lambda e: e[0] == "log" and "too late" in e[1])
    finally:
        wk.stop_flag.set(); wk.join(3)


def test_second_copy_does_not_start(monkeypatch):
    started = []
    monkeypatch.setattr(g.fa, "acquire_single_instance", lambda *a, **k: False)
    monkeypatch.setattr(g, "already_running_notice", lambda: started.append("notice"))
    monkeypatch.setattr(g, "run_gui", lambda: started.append("gui") or 0)
    assert g.main() == 0 and started == ["notice"]
