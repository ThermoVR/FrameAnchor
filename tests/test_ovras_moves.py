"""Simulated OVR Advanced Settings playspace mover (drags, snap turns, its stale remembered origin)."""
import math

import numpy as np

import frame_anchor as fa
from test_frame_anchor import FakeWorld, STATIONS, make


def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def interp(m, f):
    """Fraction f of a yaw+translation transform."""
    out = fa.yaw_rotation(fa.yaw_of(m) * f)
    out[:3, 3] = m[:3, 3] * f
    return out


class Clock:
    def __init__(self, a):
        self.a, self.t, self.states = a, 100.0, []

    def run(self, seconds):
        for _ in range(int(round(seconds / 0.05))):
            self.t += 0.05
            self.states.append(self.a.step(self.t))
        return self.states[-1]


def ovras_drag(world, clock, base, offset, ticks=20):
    """OVRAS writes Z = base @ O, growing O a little each tick (what a controller drag does)."""
    for i in range(1, ticks + 1):
        world.Z = base @ interp(offset, i / ticks)
        clock.run(0.05)


PT = np.array([1.0, 0.0, -2.0])   # a point on the real floor


def moved(o, p):
    """Where standing point p ends up after OVRAS applies offset o."""
    return fa.apply_points(np.linalg.inv(o), p[None])[0]


def test_ovras_drag_is_kept_not_undone():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    before = w.standing_of_real(PT)
    o = fa.yaw_rotation(math.radians(20))
    o[:3, 3] = [0.8, 0.0, -0.5]
    ovras_drag(w, c, w.Z.copy(), o)
    assert "being-moved" in c.states
    assert c.run(5.0) == "ok"
    assert w.set_calls == 0                          # FrameAnchor didn't fight the drag
    assert np.allclose(w.standing_of_real(PT), moved(o, before), atol=1e-6)
    assert a.anchor_moved and a.moves_followed == 1


def test_jump_after_drag_is_fixed_to_the_dragged_position():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    ovras_drag(w, c, w.Z.copy(), translate(1.0, 0, 0.5))
    c.run(2.0)
    want = w.standing_of_real(PT)
    w.relocalize(translate(0.3, 0.1, -0.2) @ fa.yaw_rotation(math.radians(8)))
    w.spacecal_converges()
    c.run(5.0)
    assert w.set_calls == 1
    assert np.allclose(w.standing_of_real(PT), want, atol=1e-3)


def test_stale_ovras_drag_drops_fix_and_it_is_put_back():
    """OVRAS remembers the playspace from before FrameAnchor's fix and re-applies it when a drag starts."""
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    ovras_cache = w.Z.copy()
    p0 = w.standing_of_real(PT)
    w.relocalize(translate(0.25, 0.08, 0.1) @ fa.yaw_rotation(math.radians(6)))
    w.spacecal_converges()
    c.run(5.0)
    assert w.set_calls == 1                          # the jump was fixed
    o = translate(-0.6, 0.0, 0.9)
    ovras_drag(w, c, ovras_cache, o)                 # starts by putting back the old, unfixed origin
    c.run(6.0)
    assert w.set_calls == 2                          # the dropped fix is made again, once
    assert np.allclose(w.standing_of_real(PT), moved(o, p0), atol=2e-3)
    assert "backing-off" not in c.states


def test_stale_drag_with_turn_drops_fix_and_it_is_put_back():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    cache = w.Z.copy()
    p0 = w.standing_of_real(PT)
    w.relocalize(translate(-0.15, 0.05, 0.3) @ fa.yaw_rotation(math.radians(-10)))
    w.spacecal_converges()
    c.run(5.0)
    o = fa.yaw_rotation(math.radians(35))
    o[:3, 3] = [0.4, 0.0, 0.2]
    ovras_drag(w, c, cache, o, ticks=30)
    c.run(6.0)
    assert np.allclose(w.standing_of_real(PT), moved(o, p0), atol=3e-3)


def test_many_stale_drags_do_not_trigger_backoff():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    cache = w.Z.copy()
    w.relocalize(translate(0.2, 0.05, 0.0))
    w.spacecal_converges()
    c.run(5.0)
    for i in range(8):
        # OVRAS keeps its offset between drags; every drag re-applies its stale remembered origin.
        ovras_drag(w, c, cache, translate(0.05, 0, 0.1), ticks=10)
        cache = cache @ translate(0.05, 0, 0.1)
        c.run(3.5)
    assert "backing-off" not in c.states
    assert c.states[-1] in ("ok", "cooldown", "corrected")


def test_sudden_zero_pose_step_is_still_fixed():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    p0 = w.standing_of_real(PT)
    w.Z = w.Z @ translate(0.2, -0.1, 0.15) @ fa.yaw_rotation(math.radians(4))   # one-step change
    c.run(5.0)
    assert w.set_calls == 1
    assert a.moves_followed == 0
    assert np.allclose(w.standing_of_real(PT), p0, atol=1e-3)


def test_snap_turn_around_head_is_kept():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    h = w.head
    w.Z = w.Z @ translate(*h) @ fa.yaw_rotation(math.radians(45)) @ translate(*(-h))
    c.run(5.0)
    assert w.set_calls == 0
    assert a.moves_followed == 1


def test_snap_turn_around_playspace_centre_is_kept():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    w.Z = w.Z @ fa.yaw_rotation(math.radians(-30))
    c.run(5.0)
    assert w.set_calls == 0


def test_one_step_turn_far_from_head_is_treated_as_jump():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    w.Z = w.Z @ translate(3.0, 0, 3.0) @ fa.yaw_rotation(math.radians(30)) @ translate(-3.0, 0, -3.0)
    c.run(5.0)
    assert w.set_calls == 1
    assert a.moves_followed == 0


def test_follow_off_undoes_drag_like_before():
    w = FakeWorld(STATIONS)
    a = make(w, follow_moves=False)
    c = Clock(a)
    c.run(1.0)
    p0 = w.standing_of_real(PT)
    ovras_drag(w, c, w.Z.copy(), translate(0.5, 0, 0))
    c.run(5.0)
    assert "being-moved" not in c.states
    assert w.set_calls >= 1
    assert np.allclose(w.standing_of_real(PT), p0, atol=1e-3)


def test_drag_while_paused_is_kept_after_resume():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    a.paused = True
    ovras_drag(w, c, w.Z.copy(), translate(0.7, 0, 0.2))
    c.run(1.0)
    a.paused = False
    c.run(4.0)
    assert w.set_calls == 0


def test_own_fix_is_not_mistaken_for_a_move():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    w.relocalize(translate(0.3, 0, 0))
    w.spacecal_converges()
    c.run(6.0)
    assert w.set_calls == 1 and a.moves_followed == 0 and a.live_confirmed
    assert "being-moved" not in c.states


def test_unreliable_live_reading_switches_following_off():
    w = FakeWorld(STATIONS)
    w.live_lags = True
    w._live_frozen = w.Z.copy()
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    w.relocalize(translate(0.3, 0, 0))
    w.spacecal_converges()
    c.run(8.0)
    assert w.use_live_zero is False
    assert w.set_calls == 1                          # it still fixed the jump, once


def test_backend_live_zero_pose_from_raw_and_standing():
    """OpenVRBackend.get_live_zero_pose recovers Z from raw = Z @ standing (fake openvr)."""
    Z = translate(0.3, -0.2, 1.1) @ fa.yaw_rotation(math.radians(33))
    std = {"A": STATIONS["LHB-A"], "B": STATIONS["LHB-B"]}
    b = fa.OpenVRBackend.__new__(fa.OpenVRBackend)
    b.openvr = type("O", (), {"TrackingUniverseRawAndUncalibrated": "raw"})
    b.get_anchor_poses = lambda s, u=None: {k: (Z @ v if u == "raw" else v) for k, v in std.items()}
    assert np.allclose(b.get_live_zero_pose(None), Z, atol=1e-9)
    b.get_anchor_poses = lambda s, u=None: ({"A": Z @ std["A"], "B": translate(0.1, 0, 0) @ Z @ std["B"]}
                                            if u == "raw" else std)
    assert b.get_live_zero_pose(None) is None        # stations disagree -> don't trust it


def ovras_drag_profile(world, clock, base, offset, fractions):
    """Drag with a given speed profile: fractions = how far along the drag each tick is."""
    for f in fractions:
        world.Z = base @ interp(offset, f)
        clock.run(0.05)


def test_fast_jerky_drag_without_any_fix_is_kept_exactly():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    p0 = w.standing_of_real(PT)
    o = translate(1.5, 0, -1.0)
    ovras_drag_profile(w, c, w.Z.copy(), o, [0.02, 0.2, 0.45, 0.7, 0.85, 0.95, 1.0])
    c.run(5.0)
    assert w.set_calls == 0
    assert np.allclose(w.standing_of_real(PT), moved(o, p0), atol=1e-6)


def test_jerky_drag_by_up_to_date_ovras_after_a_fix_is_kept_exactly():
    """OVRAS refreshed its remembered origin after our fix, so nothing is dropped - don't invent a drop."""
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    w.relocalize(translate(0.2, 0.05, 0.1))
    w.spacecal_converges()
    c.run(5.0)
    assert w.set_calls == 1
    p1 = w.standing_of_real(PT)
    o = translate(-1.2, 0, 0.6)
    ovras_drag_profile(w, c, w.Z.copy(), o, [0.01, 0.25, 0.5, 0.75, 0.9, 1.0])
    c.run(5.0)
    assert w.set_calls == 1
    assert np.allclose(w.standing_of_real(PT), moved(o, p1), atol=1e-6)


def test_small_dropped_fix_during_fast_drag_is_put_back():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    cache = w.Z.copy()
    p0 = w.standing_of_real(PT)
    w.relocalize(translate(0.04, 0.0, 0.0))          # a small 4 cm jump, fixed
    w.spacecal_converges()
    c.run(5.0)
    assert w.set_calls == 1
    o = translate(2.0, 0, 0)                         # fast: 10 cm per tick
    ovras_drag(w, c, cache, o)
    c.run(5.0)
    assert np.allclose(w.standing_of_real(PT), moved(o, p0), atol=1e-6)


def test_very_slow_drag_is_kept():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    p0 = w.standing_of_real(PT)
    base = w.Z.copy()
    o = translate(0.12, 0, 0)
    for i in range(1, 61):                           # 2 mm every 0.6 s
        w.Z = base @ interp(o, i / 60)
        c.run(0.6)
    c.run(5.0)
    assert w.set_calls == 0
    assert np.allclose(w.standing_of_real(PT), moved(o, p0), atol=1e-6)


def test_reset_bind_back_to_origin_is_kept():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    home = w.Z.copy()                                # OVRAS origin (offsets 0,0,0)
    p0 = w.standing_of_real(PT)
    ovras_drag(w, c, home, translate(0.9, 0, -0.4))
    c.run(2.0)
    ovras_drag(w, c, home @ translate(0.9, 0, -0.4), fa.yaw_rotation(math.radians(25)))
    c.run(2.0)
    w.Z = home.copy()                                # reset bind: one instant jump back to 0,0
    c.run(5.0)
    assert w.set_calls == 0
    assert np.allclose(w.standing_of_real(PT), p0, atol=1e-6)


def test_reset_after_a_fix_keeps_the_fix():
    """OVRAS's origin is from before FrameAnchor's fix: reset must not bring the jump back."""
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    home = w.Z.copy()
    p0 = w.standing_of_real(PT)
    w.relocalize(translate(0.2, 0.06, -0.1) @ fa.yaw_rotation(math.radians(7)))
    w.spacecal_converges()
    c.run(5.0)
    ovras_drag(w, c, home, translate(-0.7, 0, 0.5))  # stale drag
    c.run(5.0)
    w.Z = home.copy()                                # reset to OVRAS's (stale) origin
    c.run(6.0)
    assert np.allclose(w.standing_of_real(PT), p0, atol=2e-3)


def test_jump_to_a_new_position_is_still_fixed_with_reset_support():
    w = FakeWorld(STATIONS)
    a = make(w)
    c = Clock(a)
    c.run(1.0)
    home = w.Z.copy()
    ovras_drag(w, c, home, translate(0.9, 0, -0.4))
    c.run(2.0)
    p1 = w.standing_of_real(PT)
    w.Z = w.Z @ translate(0.15, 0.05, 0.1)           # sudden step somewhere new = a jump
    c.run(5.0)
    assert w.set_calls == 1
    assert np.allclose(w.standing_of_real(PT), p1, atol=1e-3)
