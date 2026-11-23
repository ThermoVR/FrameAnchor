#!/usr/bin/env python3
"""
FrameAnchor - keeps a Steam Frame + Lighthouse (FBT) playspace locked to the real room.

The problem
-----------
When the Steam Frame relocalizes (cameras covered, headset taken off, tracking
lost and regained) its tracking origin can jump. SteamVR then treats this like a
recenter, so the whole playspace - floor included - ends up somewhere else.
Space Calibrator keeps trackers lined up *with the headset*, but it can't put the
floor back, because the headset itself is now wrong.

The idea
--------
Base stations never move. Once Space Calibrator has them lined up, their
positions in SteamVR's standing space are a perfect "where is the real room"
reference. FrameAnchor records where they are when everything is correct
(the anchor). If the whole space later jumps, it measures the jump from the
base stations and moves SteamVR's standing origin back so the base stations -
and therefore the floor and the room - are where they were.

It only corrects yaw (turning) and position (including height). Both tracking
systems use gravity for "up", so tilt is never touched.

Usage
-----
    python frame_anchor.py            # run, applies corrections
    python frame_anchor.py --dry-run  # only log what it would do

While running, type a letter and press Enter:
    a  re-anchor (do this after the floor/calibration look correct)
    p  pause / resume corrections
    s  print status
    q  quit

Unofficial, MIT licensed. Not affiliated with Valve.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

def data_dir() -> str:
    """Folder for the saved lock and log (always writable, survives moving the app)."""
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "FrameAnchor")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.path.dirname(os.path.abspath(__file__))
    return d


ANCHOR_FILE_DEFAULT = os.path.join(data_dir(), "frame_anchor.json")
AXIS_LEN = 0.25  # metres; length of the orientation "arms" added per anchor device


# ----------------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------------

def mat34_to_np(m) -> np.ndarray:
    """OpenVR HmdMatrix34_t -> 4x4 numpy matrix."""
    out = np.eye(4)
    for r in range(3):
        for c in range(4):
            out[r, c] = m.m[r][c]
    return out


def np_to_mat34(a: np.ndarray, target) -> None:
    """Write a 4x4 numpy matrix into an OpenVR HmdMatrix34_t in place."""
    for r in range(3):
        for c in range(4):
            target.m[r][c] = float(a[r, c])


def yaw_rotation(theta: float) -> np.ndarray:
    """4x4 rotation about +Y (OpenVR up axis)."""
    c, s = math.cos(theta), math.sin(theta)
    m = np.eye(4)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def yaw_of(t: np.ndarray) -> float:
    """Yaw angle (radians) of a yaw+translation transform."""
    return math.atan2(t[0, 2], t[0, 0])


def device_points(pose: np.ndarray) -> np.ndarray:
    """Position plus two short orientation arms, so a single base station still gives yaw."""
    p = pose[:3, 3]
    x_axis = pose[:3, 0]
    z_axis = pose[:3, 2]
    return np.array([p, p + AXIS_LEN * x_axis, p + AXIS_LEN * z_axis])


def fit_yaw_translation(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Best-fit transform T (rotation about +Y, then translation) with T @ src ~= dst.
    src, dst: (N, 3) point arrays. Returns (4x4 T, RMS residual in metres).
    """
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    a = src - cs
    b = dst - cd
    # maximise sum b . R a  for R = rotation about Y
    num = np.sum(b[:, 0] * a[:, 2] - b[:, 2] * a[:, 0])
    den = np.sum(b[:, 0] * a[:, 0] + b[:, 2] * a[:, 2])
    theta = math.atan2(num, den)
    t = yaw_rotation(theta)
    t[:3, 3] = cd - t[:3, :3] @ cs
    moved = (t[:3, :3] @ src.T).T + t[:3, 3]
    rms = float(np.sqrt(np.mean(np.sum((moved - dst) ** 2, axis=1))))
    return t, rms


def describe(t: np.ndarray) -> str:
    d = t[:3, 3]
    return (f"dx={d[0]*100:+.1f}cm  dy(height)={d[1]*100:+.1f}cm  dz={d[2]*100:+.1f}cm  "
            f"yaw={math.degrees(yaw_of(t)):+.2f}deg")


# ----------------------------------------------------------------------------
# OpenVR backend (kept separate so the logic can be tested without SteamVR)
# ----------------------------------------------------------------------------

class OpenVRBackend:
    def __init__(self):
        import openvr  # imported here so tests don't need SteamVR
        self.openvr = openvr
        self.vr = openvr.init(openvr.VRApplication_Background)
        self.setup = openvr.VRChaperoneSetup()
        self._pose_array = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        self._serial_cache: dict[int, str] = {}
        self._system_cache: dict[int, str] = {}

    def _is_lighthouse(self, idx: int) -> bool:
        """True for Lighthouse devices (or if SteamVR won't say what the device is)."""
        if idx not in self._system_cache:
            try:
                self._system_cache[idx] = self.vr.getStringTrackedDeviceProperty(
                    idx, self.openvr.Prop_TrackingSystemName_String).lower()
            except Exception:
                return True
        name = self._system_cache[idx]
        return (not name) or ("lighthouse" in name)

    def _serial(self, idx: int) -> str:
        if idx not in self._serial_cache:
            try:
                self._serial_cache[idx] = self.vr.getStringTrackedDeviceProperty(
                    idx, self.openvr.Prop_SerialNumber_String)
            except Exception:
                return f"device{idx}"
        return self._serial_cache[idx]

    def hmd_ok(self) -> bool:
        ovr = self.openvr
        poses = self.vr.getDeviceToAbsoluteTrackingPose(
            ovr.TrackingUniverseStanding, 0, self._pose_array)
        hmd = poses[ovr.k_unTrackedDeviceIndex_Hmd]
        return bool(hmd.bPoseIsValid) and hmd.eTrackingResult == ovr.TrackingResult_Running_OK

    def get_anchor_poses(self, serials: set[str] | None) -> dict[str, np.ndarray]:
        """Standing-space poses of base stations (or of the given serials)."""
        ovr = self.openvr
        poses = self.vr.getDeviceToAbsoluteTrackingPose(
            ovr.TrackingUniverseStanding, 0, self._pose_array)
        out = {}
        for i in range(ovr.k_unMaxTrackedDeviceCount):
            cls = self.vr.getTrackedDeviceClass(i)
            if cls == ovr.TrackedDeviceClass_Invalid:
                continue
            p = poses[i]
            if not (p.bPoseIsValid and p.bDeviceIsConnected):
                continue
            serial = self._serial(i)
            if serials:
                if serial not in serials:
                    continue
            elif cls != ovr.TrackedDeviceClass_TrackingReference or not self._is_lighthouse(i):
                continue
            out[serial] = mat34_to_np(p.mDeviceToAbsoluteTracking)
        return out

    def get_zero_pose(self) -> np.ndarray:
        self.setup.revertWorkingCopy()
        ok, m = self.setup.getWorkingStandingZeroPoseToRawTrackingPose()
        if not ok:
            raise RuntimeError("SteamVR refused to give the standing zero pose")
        return mat34_to_np(m)

    def set_zero_pose(self, z: np.ndarray) -> None:
        m = self.openvr.HmdMatrix34_t()
        np_to_mat34(z, m)
        self.setup.setWorkingStandingZeroPoseToRawTrackingPose(m)
        self.setup.commitWorkingCopy(self.openvr.EChaperoneConfigFile_Live)

    def poll_events(self) -> list[str]:
        ovr = self.openvr
        names = {
            ovr.VREvent_StandingZeroPoseReset: "StandingZeroPoseReset (Frame relocalized / recentered)",
            ovr.VREvent_ChaperoneUniverseHasChanged: "ChaperoneUniverseHasChanged",
            ovr.VREvent_Quit: "Quit",
        }
        out = []
        ev = ovr.VREvent_t()
        while self.vr.pollNextEvent(ev):
            if ev.eventType in names:
                out.append(names[ev.eventType])
                if ev.eventType == ovr.VREvent_Quit:
                    self.vr.acknowledgeQuit_Exiting()
        return out

    def shutdown(self):
        self.openvr.shutdown()


# ----------------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------------

@dataclass
class Settings:
    threshold_m: float = 0.03        # ignore shifts smaller than this...
    threshold_deg: float = 1.5       # ...and turns smaller than this
    settle_s: float = 1.5            # a shift must be steady this long before fixing
    stable_m: float = 0.015          # "steady" = varies less than this...
    stable_deg: float = 0.75         # ...and this, during the settle window
    max_rms_m: float = 0.10          # shape mismatch above this = not calibrated / stations moved
    max_shift_m: float = 3.0         # refuse absurd corrections
    cooldown_s: float = 1.0          # wait after a correction before judging again
    anchor_avg_s: float = 1.0        # average poses this long when anchoring
    first_settle_s: float = 5.0      # be extra patient before the first fix after connecting
    burst_count: int = 6             # this many fixes...
    burst_window_s: float = 60.0     # ...within this time means something is fighting us
    burst_backoff_s: float = 60.0    # so stop trying for this long
    dry_run: bool = False


@dataclass
class Anchor:
    poses: dict[str, np.ndarray]
    created: float = field(default_factory=time.time)

    def save(self, path: str) -> None:
        data = {"created": self.created,
                "poses": {k: v.tolist() for k, v in self.poses.items()}}
        with open(path, "w") as f:
            json.dump(data, f, indent=1)

    @staticmethod
    def load(path: str) -> "Anchor | None":
        try:
            with open(path) as f:
                data = json.load(f)
            return Anchor({k: np.array(v) for k, v in data["poses"].items()}, data["created"])
        except (OSError, ValueError, KeyError):
            return None


def average_poses(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Average positions and orientation axes (re-orthonormalised) per serial."""
    out = {}
    serials = set.intersection(*(set(s) for s in samples)) if samples else set()
    for k in serials:
        mats = np.array([s[k] for s in samples])
        avg = mats.mean(axis=0)
        u, _, vt = np.linalg.svd(avg[:3, :3])
        avg[:3, :3] = u @ vt
        avg[3] = [0, 0, 0, 1]
        out[k] = avg
    return out


class FrameAnchor:
    def __init__(self, backend, settings: Settings, log=print, anchor_serials=None):
        self.b = backend
        self.s = settings
        self.log = log
        self.anchor_serials = set(anchor_serials) if anchor_serials else None
        self.anchor: Anchor | None = None
        self.paused = False
        self.window: collections.deque = collections.deque()
        self.cooldown_until = 0.0
        self.corrections = 0
        self.last_fix = ""
        self.last_seen: dict[str, np.ndarray] = {}
        self.fix_times: collections.deque = collections.deque()
        self.backoff_until = 0.0
        self.first_fix_done = False

    # -- anchoring -----------------------------------------------------------
    def capture_anchor(self, sleep=time.sleep, now=time.monotonic) -> bool:
        samples = []
        end = now() + self.s.anchor_avg_s
        while now() < end:
            p = self.b.get_anchor_poses(self.anchor_serials)
            if p:
                samples.append(p)
            sleep(0.05)
        poses = average_poses(samples)
        if not poses:
            self.log("!! No base stations visible - can't anchor. Are they on and is Space Calibrator running?")
            return False
        self.anchor = Anchor(poses)
        self.window.clear()
        self.fix_times.clear()
        self.backoff_until = 0.0
        self.first_fix_done = True  # the user just confirmed things are right
        heights = ", ".join(f"{k}: {v[1, 3]:.2f}m high" for k, v in poses.items())
        self.log(f"Anchored to {len(poses)} device(s): {heights}")
        if len(poses) == 1 and not self.anchor_serials:
            self.log("   (only one base station - yaw relies on its orientation, which is less precise)")
        return True

    # -- per-tick logic ------------------------------------------------------
    def measure(self) -> tuple[np.ndarray, float] | None:
        """Transform that would move the current space back onto the anchor."""
        cur = self.b.get_anchor_poses(self.anchor_serials)
        self.last_seen = cur
        common = sorted(set(cur) & set(self.anchor.poses))
        if not common:
            return None
        src = np.vstack([device_points(cur[k]) for k in common])
        dst = np.vstack([device_points(self.anchor.poses[k]) for k in common])
        return fit_yaw_translation(src, dst)

    def step(self, now: float) -> str:
        """One update. Returns a short status string (also useful for tests)."""
        if self.anchor is None:
            return "no-anchor"
        if self.paused:
            return "paused"
        if now < self.backoff_until:
            return "backing-off"
        if now < self.cooldown_until:
            return "cooldown"
        if not self.b.hmd_ok():
            self.window.clear()
            return "hmd-not-tracking"
        m = self.measure()
        if m is None:
            self.window.clear()
            return "no-anchor-devices-visible"
        t, rms = m
        shift = float(np.linalg.norm(t[:3, 3]))
        yaw_deg = abs(math.degrees(yaw_of(t)))
        if rms > self.s.max_rms_m:
            self.window.clear()
            return f"shape-mismatch rms={rms*100:.1f}cm"
        if shift < self.s.threshold_m and yaw_deg < self.s.threshold_deg:
            self.window.clear()
            return "ok"
        if shift > self.s.max_shift_m:
            self.window.clear()
            return f"shift-too-large {shift:.2f}m"

        self.window.append((now, t))
        need = self.s.settle_s if self.first_fix_done else max(self.s.settle_s, self.s.first_settle_s)
        while self.window and now - self.window[0][0] > need + 0.25:
            self.window.popleft()
        if now - self.window[0][0] < need:
            return "settling"
        ts = [w[1] for w in self.window]
        spread_m = max(np.linalg.norm(a[:3, 3] - ts[-1][:3, 3]) for a in ts)
        spread_deg = max(abs(math.degrees(yaw_of(a) - yaw_of(ts[-1]))) for a in ts)
        if spread_m > self.s.stable_m or spread_deg > self.s.stable_deg:
            return "settling (still moving)"

        self.apply(t)
        self.window.clear()
        self.first_fix_done = True
        self.cooldown_until = now + self.s.cooldown_s
        self.fix_times.append(now)
        while self.fix_times and now - self.fix_times[0] > self.s.burst_window_s:
            self.fix_times.popleft()
        if len(self.fix_times) >= self.s.burst_count:
            self.fix_times.clear()
            self.backoff_until = now + self.s.burst_backoff_s
            self.log("!! Had to fix the playspace many times in a row - something else keeps moving it "
                     "(OVR Advanced Settings space drag? another playspace tool?). Taking a 60 s break.")
        return "corrected"

    def apply(self, t: np.ndarray) -> None:
        self.corrections += 1
        self.last_fix = describe(t)
        if self.s.dry_run:
            self.log(f"[dry-run] would correct: {describe(t)}")
            return
        z = self.b.get_zero_pose()
        self.b.set_zero_pose(z @ np.linalg.inv(t))
        self.log(f"Corrected playspace: {describe(t)}")


# ----------------------------------------------------------------------------
# Console app
# ----------------------------------------------------------------------------

def stdin_reader(q: "queue.Queue[str]"):
    for line in sys.stdin:
        q.put(line.strip().lower())
    q.put("q")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Keep a Steam Frame + Lighthouse playspace locked to the room.")
    ap.add_argument("--dry-run", action="store_true", help="log corrections without applying them")
    ap.add_argument("--threshold-cm", type=float, default=3.0, help="smallest shift to fix (default 3)")
    ap.add_argument("--threshold-deg", type=float, default=1.5, help="smallest turn to fix (default 1.5)")
    ap.add_argument("--anchor-file", default=ANCHOR_FILE_DEFAULT)
    ap.add_argument("--anchor-serial", action="append",
                    help="anchor to this device serial instead of base stations (repeatable). "
                         "Only use devices that never move, e.g. a spare tracker on a shelf.")
    ap.add_argument("--fresh", action="store_true", help="ignore the saved anchor and make a new one at start")
    args = ap.parse_args(argv)

    settings = Settings(threshold_m=args.threshold_cm / 100, threshold_deg=args.threshold_deg,
                        dry_run=args.dry_run)

    def log(msg):
        print(time.strftime("[%H:%M:%S] ") + msg, flush=True)

    log("FrameAnchor starting - connecting to SteamVR...")
    try:
        backend = OpenVRBackend()
    except Exception as e:
        log(f"Could not connect to SteamVR: {e}\nStart SteamVR first, then run this again.")
        return 1

    fa = FrameAnchor(backend, settings, log, args.anchor_serial)
    saved = None if args.fresh else Anchor.load(args.anchor_file)
    if saved:
        fa.anchor = saved
        log(f"Loaded saved anchor from {time.ctime(saved.created)} ({len(saved.poses)} device(s)).")
        log("If the floor isn't right yet: fix it, recalibrate, then type 'a' + Enter to re-anchor.")
    else:
        log("No saved anchor yet. When your floor and trackers look CORRECT, type 'a' + Enter.")
    if settings.dry_run:
        log("DRY RUN: nothing will be changed.")
    log("Commands: a = anchor now, p = pause/resume, s = status, q = quit")

    cmds: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=stdin_reader, args=(cmds,), daemon=True).start()

    last_state = None
    try:
        while True:
            for ev in backend.poll_events():
                log(f"SteamVR event: {ev}")
                if ev == "Quit":
                    return 0
            try:
                while True:
                    c = cmds.get_nowait()
                    if c == "a":
                        if fa.capture_anchor():
                            fa.anchor.save(args.anchor_file)
                            log(f"Saved anchor to {args.anchor_file}")
                    elif c == "p":
                        fa.paused = not fa.paused
                        log("Paused." if fa.paused else "Resumed.")
                    elif c == "s":
                        log(f"State: {last_state}; corrections so far: {fa.corrections}")
                    elif c == "q":
                        return 0
            except queue.Empty:
                pass

            state = fa.step(time.monotonic())
            if state != last_state and state not in ("ok", "cooldown", "corrected"):
                if not (state.startswith("settling") and (last_state or "").startswith("settling")):
                    log(f"Status: {state}")
            last_state = state
            time.sleep(0.05)
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            backend.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
