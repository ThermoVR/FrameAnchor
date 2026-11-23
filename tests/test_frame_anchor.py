"""Simulated tests: a fake SteamVR where the Frame's tracking space jumps (relocalization)."""
import math
import numpy as np

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
        self.stations = stations          # serial -> real-room pose
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
    d = np.eye(4); d[:3, 3] = [0.5, 0.1, 0]
    w.relocalize(d); w.spacecal_converges()                            # new session, different origin
    assert "corrected" not in run(a, 4.0)
    assert "corrected" in run(a, 3.0, start=14.0)
    assert np.allclose(w.standing_of_real(np.zeros(3)), 0, atol=1e-6)
