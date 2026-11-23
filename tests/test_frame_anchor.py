"""Simulated tests: a fake SteamVR where the Frame's tracking space jumps (relocalization)."""
import math
import os
import numpy as np
import pytest

import frame_anchor as fa


def pose(x, y, z, yaw_deg=0.0, pitch_deg=0.0):
    m = fa.yaw_rotation(math.radians(yaw_deg))
    p = math.radians(pitch_deg)
    rx = np.eye(4)
    rx[1, 1], rx[1, 2], rx[2, 1], rx[2, 2] = math.cos(p), -math.sin(p), math.sin(p), math.cos(p)
    m = m @ rx
    m[:3, 3] = [x, y, z]
    return m


class FakeWorld:
    """
    real room --(D: Frame raw vs real room)--> raw --(inv Z)--> standing
    Space Calibrator maps lighthouse devices into raw using its *believed* D
    (sc_D), which lags behind the true D until continuous calibration converges.
    """

    def __init__(self, stations, noise=0.0, seed=0):
        self.stations = dict(stations)    # serial -> real-room pose (own copy: tests move stations)
        self.D = np.eye(4)
        self.sc_D = np.eye(4)
        self.Z = np.eye(4)
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.hmd_tracking = True
        self.set_calls = 0

    # backend API
    def hmd_ok(self):
        return self.hmd_tracking

    def get_anchor_poses(self, serials):
        out = {}
        zi = np.linalg.inv(self.Z)
        for k, p in self.stations.items():
            m = zi @ self.sc_D @ p
            if self.noise:
                m = m.copy()
                m[:3, 3] += self.rng.normal(0, self.noise, 3)
            out[k] = m
        return out

    def get_zero_pose(self):
        return self.Z.copy()

    # live standing origin, as SteamVR currently applies it (incl. OVRAS's unsaved preview)
    use_live_zero = True
    live_lags = False                  # True: the live value never shows our writes (broken reading)
    head = np.array([0.4, 1.7, 0.3])   # headset position in standing space

    def get_live_zero_pose(self, serials):
        if self.live_lags:
            return self._live_frozen.copy()
        return self.Z.copy()

    def hmd_position(self):
        return self.head.copy()

    def set_zero_pose(self, z):
        self.Z = z.copy()
        self.set_calls += 1

    # helpers
    def relocalize(self, d):
        self.D = d @ self.D

    def spacecal_converges(self):
        self.sc_D = self.D.copy()

    def standing_of_real(self, pt):
        p = np.append(pt, 1.0)
        return (np.linalg.inv(self.Z) @ self.D @ p)[:3]


STATIONS = {"LHB-A": pose(-2.0, 2.1, -1.5, 45, -30), "LHB-B": pose(2.0, 2.0, 1.5, -135, -30)}


def make(world, **kw):
    s = fa.Settings(**kw)
    a = fa.FrameAnchor(world, s, log=lambda *_: None)
    t = [0.0]
    a.capture_anchor(sleep=lambda dt: t.__setitem__(0, t[0] + dt), now=lambda: t[0])
    return a


def run(a, seconds, start=10.0, dt=0.05):
    t = start
    states = []
    while t < start + seconds:
        states.append(a.step(t))
        t += dt
    return states


def test_fit_recovers_known_transform():
    src = np.random.default_rng(1).normal(size=(6, 3))
    t_true = fa.yaw_rotation(math.radians(37))
    t_true[:3, 3] = [0.4, -0.2, 1.3]
    dst = (t_true[:3, :3] @ src.T).T + t_true[:3, 3]
    t, rms = fa.fit_yaw_translation(src, dst)
    assert np.allclose(t, t_true, atol=1e-9) and rms < 1e-9


def test_relocalization_floor_and_position_restored():
    w = FakeWorld(STATIONS)
    a = make(w)
    floor_before = w.standing_of_real(np.array([0.5, 0.0, 0.3]))
    # Frame relocalizes: space jumps 12 cm down, 30 cm sideways, turns 8 degrees
    d = fa.yaw_rotation(math.radians(8))
    d[:3, 3] = [0.30, -0.12, 0.05]
    w.relocalize(d)
    w.spacecal_converges()
    states = run(a, 3.0)
    assert "corrected" in states
    floor_after = w.standing_of_real(np.array([0.5, 0.0, 0.3]))
    assert np.allclose(floor_after, floor_before, atol=1e-6), (floor_before, floor_after)
    # stations back where they were anchored
    for k, p in w.get_anchor_poses(None).items():
        assert np.allclose(p, a.anchor.poses[k], atol=1e-6)


def test_steamvr_zero_pose_reset_also_fixed():
    """SteamVR itself resets the standing origin (StandingZeroPoseReset) to somewhere else."""
    w = FakeWorld(STATIONS)
    a = make(w)
    floor_before = w.standing_of_real(np.array([0.0, 0.0, 0.0]))
    w.Z = fa.yaw_rotation(math.radians(-20))
    w.Z[:3, 3] = [0.1, 0.25, -0.4]
    run(a, 3.0)
    assert np.allclose(w.standing_of_real(np.array([0.0, 0.0, 0.0])), floor_before, atol=1e-6)


def test_waits_for_spacecal_then_corrects():
    w = FakeWorld(STATIONS)
    a = make(w)
    d = np.eye(4)
    d[:3, 3] = [0, 0.2, 0]
    w.relocalize(d)
    # before Space Cal converges, stations haven't moved in standing space -> nothing to do
    assert set(run(a, 2.0)) == {"ok"}
    w.spacecal_converges()
    assert "corrected" in run(a, 3.0, start=20)
    assert abs(w.standing_of_real(np.zeros(3))[1]) < 1e-6


def test_small_noise_never_corrects():
    w = FakeWorld(STATIONS, noise=0.004)
    a = make(w)
    states = run(a, 30.0)
    assert "corrected" not in states and w.set_calls == 0


def test_brief_glitch_ignored():
    w = FakeWorld(STATIONS)
    a = make(w)
    d = np.eye(4)
    d[:3, 3] = [0, 0.3, 0]
    w.relocalize(d); w.spacecal_converges()
    run(a, 0.8)                       # shorter than settle time
    w.relocalize(np.linalg.inv(d)); w.spacecal_converges()
    run(a, 3.0, start=20)
    assert w.set_calls == 0


def test_no_correction_while_hmd_lost():
    w = FakeWorld(STATIONS)
    a = make(w)
    d = np.eye(4); d[:3, 3] = [0, 0.3, 0]
    w.relocalize(d); w.spacecal_converges()
    w.hmd_tracking = False
    assert set(run(a, 3.0)) == {"hmd-not-tracking"}
    w.hmd_tracking = True
    assert "corrected" in run(a, 3.0, start=20)


def test_moved_base_station_detected_not_applied():
    w = FakeWorld(STATIONS)
    a = make(w)
    w.stations["LHB-B"] = pose(0.5, 1.0, -2.0, 10, -30)  # someone moved a base station
    states = run(a, 3.0)
    assert any(s.startswith("shape-mismatch") for s in states) and w.set_calls == 0


def test_single_station_uses_orientation_for_yaw():
    w = FakeWorld({"LHB-A": STATIONS["LHB-A"]})
    a = make(w)
    d = fa.yaw_rotation(math.radians(15)); d[:3, 3] = [0.2, -0.1, 0]
    w.relocalize(d); w.spacecal_converges()
    run(a, 3.0)
    assert np.allclose(w.standing_of_real(np.array([1.0, 0, 1.0])), [1.0, 0, 1.0], atol=1e-6)


def test_dry_run_changes_nothing():
    w = FakeWorld(STATIONS)
    a = make(w, dry_run=True)
    d = np.eye(4); d[:3, 3] = [0, 0.3, 0]
    w.relocalize(d); w.spacecal_converges()
    assert "corrected" in run(a, 3.0)
    assert w.set_calls == 0


def test_anchor_roundtrip(tmp_path):
    anc = fa.Anchor({k: v for k, v in STATIONS.items()})
    p = tmp_path / "a.json"
    anc.save(str(p))
    back = fa.Anchor.load(str(p))
    assert all(np.allclose(back.poses[k], STATIONS[k]) for k in STATIONS)


def test_burst_guard_backs_off():
    w = FakeWorld(STATIONS)
    a = make(w, burst_count=3)
    t = 10.0
    seen = []
    for i in range(6):
        d = np.eye(4); d[:3, 3] = [0, 0.2, 0]
        w.relocalize(d); w.spacecal_converges()
        st = run(a, 4.0, start=t); seen += st; t += 4.0
    assert "backing-off" in seen
    assert w.set_calls == 3


def test_saved_anchor_needs_longer_settle_first_time():
    w = FakeWorld(STATIONS)
    a = fa.FrameAnchor(w, fa.Settings(), log=lambda *_: None)
    a.anchor = fa.Anchor({k: v.copy() for k, v in STATIONS.items()})   # loaded from disk
    d = np.eye(4); d[:3, 3] = [0.06, 0.02, 0]                          # small: no confirmation needed
    w.relocalize(d); w.spacecal_converges()                            # new session, different origin
    assert "corrected" not in run(a, 4.0)
    assert "corrected" in run(a, 3.0, start=14.0)
    assert np.allclose(w.standing_of_real(np.zeros(3)), 0, atol=1e-6)


# ---- tests for the fixes ---------------------------------------------------

def test_saved_anchor_big_jump_waits_for_confirmation():
    w = FakeWorld(STATIONS)
    a = fa.FrameAnchor(w, fa.Settings(), log=lambda *_: None)
    a.anchor = fa.Anchor({k: v.copy() for k, v in STATIONS.items()})
    d = np.eye(4); d[:3, 3] = [0.5, 0.1, 0]
    w.relocalize(d); w.spacecal_converges()
    states = run(a, 20.0)
    assert any(s.startswith("confirm-needed") for s in states) and w.set_calls == 0
    a.confirm()
    assert "corrected" in run(a, 3.0, start=40.0)
    assert np.allclose(w.standing_of_real(np.zeros(3)), 0, atol=1e-6)


def test_fresh_lock_big_jump_needs_no_confirmation():
    w = FakeWorld(STATIONS)
    a = make(w)
    d = np.eye(4); d[:3, 3] = [0.5, 0.1, 0]
    w.relocalize(d); w.spacecal_converges()
    assert "corrected" in run(a, 3.0)


def test_one_station_resolved_by_steamvr_is_not_applied():
    """vrserver.txt showed SteamVR moving ONE base station 125 mm / 2 deg. That's not a playspace jump."""
    w = FakeWorld(dict(STATIONS))
    a = make(w)
    m = fa.yaw_rotation(math.radians(2)) @ STATIONS["LHB-B"]
    m[:3, 3] += [0.125, 0, 0]
    w.stations["LHB-B"] = m
    states = run(a, 5.0)
    assert w.set_calls == 0 and any(s.startswith("shape-mismatch") for s in states)


def test_one_of_three_stations_nudged_is_not_applied():
    st = dict(STATIONS); st["LHB-C"] = pose(0.0, 2.2, -2.5, 180, -30)
    w = FakeWorld(st)
    a = make(w)
    w.stations["LHB-C"] = pose(0.06, 2.2, -2.5, 180, -30)
    run(a, 5.0)
    assert w.set_calls == 0


def test_real_jump_with_noise_still_corrected():
    w = FakeWorld(STATIONS, noise=0.004, seed=3)
    a = make(w)
    d = fa.yaw_rotation(math.radians(3)); d[:3, 3] = [0.06, -0.04, 0.02]
    w.relocalize(d); w.spacecal_converges()
    assert "corrected" in run(a, 6.0)


def test_half_turn_jump_with_noise_is_corrected():
    w = FakeWorld(STATIONS, noise=0.002)
    a = make(w)
    d = fa.yaw_rotation(math.radians(180)); d[:3, 3] = [0.2, 0, 0.1]
    w.relocalize(d); w.spacecal_converges()
    assert "corrected" in run(a, 6.0)


class RefusingWorld(FakeWorld):
    def set_zero_pose(self, z):
        raise fa.ChaperoneWriteError("SteamVR refused to save the new standing origin")


def test_refused_write_is_reported_not_crashed():
    w = RefusingWorld(STATIONS)
    logs = []
    s = fa.Settings()
    a = fa.FrameAnchor(w, s, log=logs.append)
    t = [0.0]
    a.capture_anchor(sleep=lambda dt: t.__setitem__(0, t[0] + dt), now=lambda: t[0])
    d = np.eye(4); d[:3, 3] = [0, 0.2, 0]
    w.relocalize(d); w.spacecal_converges()
    states = run(a, 5.0)
    assert "write-refused" in states and "corrected" not in states and a.corrections == 0
    assert any("refused" in m for m in logs)


def test_undo_restores_zero_pose_from_lock_and_pauses():
    w = FakeWorld(STATIONS)
    w.Z = fa.yaw_rotation(0.3); w.Z[:3, 3] = [0.1, 0.0, 0.2]
    z_at_lock = w.Z.copy()
    a = make(w)
    d = np.eye(4); d[:3, 3] = [0.3, -0.1, 0]
    w.relocalize(d); w.spacecal_converges()
    assert "corrected" in run(a, 3.0)
    assert not np.allclose(w.Z, z_at_lock)
    assert a.undo()
    assert np.allclose(w.Z, z_at_lock) and a.paused
    assert set(run(a, 3.0, start=30)) == {"paused"}


def test_lock_refused_while_headset_not_tracking():
    w = FakeWorld(STATIONS)
    w.hmd_tracking = False
    a = fa.FrameAnchor(w, fa.Settings(), log=lambda *_: None)
    assert a.capture_anchor(sleep=lambda dt: None) is False and a.anchor is None


def test_anchor_roundtrip_keeps_undo_point(tmp_path):
    z = fa.yaw_rotation(0.5); z[:3, 3] = [1, 2, 3]
    anc = fa.Anchor({k: v for k, v in STATIONS.items()}, zero_pose=z)
    p = tmp_path / "a.json"
    anc.save(str(p))
    back = fa.Anchor.load(str(p))
    assert np.allclose(back.zero_pose, z)


def test_damaged_anchor_files_are_ignored(tmp_path):
    p = tmp_path / "a.json"
    for content in ['{"created": 1, "poses": []}', '{"created": 1, "poses": {"x": [1, 2]}}',
                    '{"poses": {}}', '[1, 2]', '{"created": 1, "poses": {"x": "abc"}}', '{trunc', '']:
        p.write_text(content)
        assert fa.Anchor.load(str(p)) is None, content


def test_chaperone_backup_made_once(tmp_path, monkeypatch):
    cfg = tmp_path / "Steam config"; cfg.mkdir()
    (cfg / "chaperone_info.vrchap").write_text("original")
    monkeypatch.setattr(fa, "steamvr_config_dir", lambda: str(cfg))
    bdir = tmp_path / "backup"
    fa.backup_chaperone_once(str(bdir), log=lambda *_: None)
    (cfg / "chaperone_info.vrchap").write_text("changed")
    fa.backup_chaperone_once(str(bdir), log=lambda *_: None)
    assert (bdir / "chaperone_info (before FrameAnchor).vrchap").read_text() == "original"


def test_wrap_deg():
    assert fa.wrap_deg(359.9) == pytest.approx(-0.1) and fa.wrap_deg(-359.9) == pytest.approx(0.1)


def test_console_mode_waits_for_steamvr(monkeypatch, capsys):
    import io, sys
    def no_steamvr(*a, **k):
        raise RuntimeError("not running")
    monkeypatch.setattr(fa, "OpenVRBackend", no_steamvr)
    monkeypatch.setattr(sys, "stdin", io.StringIO("q\n"))
    assert fa.main([]) == 0
    assert "waiting for SteamVR" in capsys.readouterr().out


# ---- tests for the second round of fixes -----------------------------------

class NaNWorld(FakeWorld):
    bad = False
    def get_anchor_poses(self, s):
        out = super().get_anchor_poses(s)
        if self.bad:
            out["LHB-B"] = out["LHB-B"].copy(); out["LHB-B"][0, 3] = np.nan
        return out


def test_nan_pose_is_ignored_and_never_written():
    w = NaNWorld(dict(STATIONS)); a = make(w); w.bad = True
    d = np.eye(4); d[:3, 3] = [0, -0.2, 0]
    w.relocalize(d); w.spacecal_converges()
    run(a, 5.0)
    assert np.all(np.isfinite(w.Z)) and fa.is_rigid(w.Z)
    assert np.allclose(w.standing_of_real(np.zeros(3)), 0, atol=1e-6)   # fixed using the good station


class ZeroZWorld(FakeWorld):
    written = []
    def get_zero_pose(self): return np.zeros((4, 4))
    def set_zero_pose(self, z): self.written.append(z)


def test_invalid_standing_origin_from_steamvr_is_never_written_back():
    w = ZeroZWorld(dict(STATIONS)); w.written = []
    a = make(w)
    assert a.undo_pose is None                         # lock didn't keep an invalid undo point
    d = np.eye(4); d[:3, 3] = [0, -0.2, 0]
    w.relocalize(d); w.spacecal_converges()
    assert "write-refused" in run(a, 4.0) and w.written == []


def test_lock_file_with_non_rigid_matrices(tmp_path):
    import json
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"created": 1, "poses": {"x": (np.eye(4) * 5).tolist()}}))
    assert fa.Anchor.load(str(p)) is None
    good = {k: v.tolist() for k, v in STATIONS.items()}
    p.write_text(json.dumps({"created": 1, "poses": good, "zero_pose": np.zeros((4, 4)).tolist()}))
    back = fa.Anchor.load(str(p))
    assert back is not None and back.zero_pose is None   # lock kept, broken undo point dropped


def test_is_rigid():
    assert fa.is_rigid(np.eye(4)) and fa.is_rigid(STATIONS["LHB-A"])
    bad = np.eye(4); bad[0, 0] = np.nan
    assert not any(fa.is_rigid(m) for m in (bad, np.zeros((4, 4)), np.eye(4) * 2, np.eye(3), "x"))


def test_late_confirm_click_does_not_disable_safety():
    w = FakeWorld(dict(STATIONS))
    a = fa.FrameAnchor(w, fa.Settings(), log=lambda *_: None)
    a.anchor = fa.Anchor({k: v.copy() for k, v in STATIONS.items()})
    assert a.confirm() is False                       # nothing was being asked
    d = np.eye(4); d[:3, 3] = [0.8, 0, 0]
    w.relocalize(d); w.spacecal_converges()
    states = run(a, 10.0)
    assert "corrected" not in states and any(s.startswith("confirm-needed") for s in states)


def test_console_status_lines_not_spammed(tmp_path, monkeypatch, capsys):
    import sys, time as _t
    w = FakeWorld(dict(STATIONS), noise=0.002)
    w.poll_events = lambda: []; w.shutdown = lambda: None
    fa.Anchor({k: v.copy() for k, v in STATIONS.items()}).save(str(tmp_path / "a.json"))
    w.stations["LHB-B"] = pose(0.5, 1.0, -2.0, 10, -30)          # -> shape-mismatch with changing numbers
    monkeypatch.setattr(fa, "OpenVRBackend", lambda **k: w)
    monkeypatch.setattr(fa, "acquire_single_instance", lambda *a, **k: True)
    class SlowStdin:
        def __iter__(self):
            _t.sleep(2.0); yield "q\n"
    monkeypatch.setattr(sys, "stdin", SlowStdin())
    assert fa.main(["--anchor-file", str(tmp_path / "a.json")]) == 0
    out = capsys.readouterr().out
    assert out.count("Status:") <= 2, out


def test_single_instance_on_windows(tmp_path):
    import subprocess, sys, textwrap
    if sys.platform != "win32":
        pytest.skip("Windows only")
    name = "Local\\FrameAnchor-test-" + str(os.getpid())
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import sys, time; sys.path.insert(0, {os.path.dirname(fa.__file__)!r})
        import frame_anchor as fa
        print(fa.acquire_single_instance({name!r}), flush=True); time.sleep(5)""")],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "True"
        second = subprocess.run([sys.executable, "-c",
                                 f"import sys; sys.path.insert(0, {os.path.dirname(fa.__file__)!r}); "
                                 f"import frame_anchor as fa; print(fa.acquire_single_instance({name!r}))"],
                                capture_output=True, text=True, timeout=30)
        assert second.stdout.strip() == "False"
    finally:
        holder.kill()
    third = subprocess.run([sys.executable, "-c",
                            f"import sys; sys.path.insert(0, {os.path.dirname(fa.__file__)!r}); "
                            f"import frame_anchor as fa; print(fa.acquire_single_instance({name!r}))"],
                           capture_output=True, text=True, timeout=30)
    assert third.stdout.strip() == "True"            # released when the first copy exits
