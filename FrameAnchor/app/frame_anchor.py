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

If the base stations move relative to *each other* (someone bumped one, or
SteamVR re-solved a base station's position), that is not a playspace jump, so
FrameAnchor refuses to "fix" it and asks for a new lock instead.

Safety
------
- Before its first change, FrameAnchor copies SteamVR's chaperone_info.vrchap
  to %APPDATA%\\FrameAnchor\\backup (once; the oldest copy is kept).
- The lock remembers SteamVR's standing origin at the moment you locked, and
  "undo" puts it back.
- After connecting with a saved lock, a big correction (default >10 cm or >5 deg)
  is not applied until you confirm it.

Usage
-----
    python frame_anchor.py            # run, applies corrections
    python frame_anchor.py --dry-run  # only log what it would do

While running, type a letter and press Enter:
    a  lock (anchor) now - do this after the floor/calibration look correct.
       Also resumes if paused.
    p  pause / resume corrections (pause BEFORE recentering or fixing the floor)
    y  yes, apply a big correction that is waiting for confirmation
    u  undo: put the playspace back to how it was when you locked (then pauses)
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
import shutil
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
BACKUP_DIR_DEFAULT = os.path.join(data_dir(), "backup")
AXIS_LEN = 0.25  # metres; length of the orientation "arms" added per anchor device


class ChaperoneWriteError(RuntimeError):
    """SteamVR would not let us read or change the standing origin."""


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


def wrap_deg(a: float) -> float:
    """Wrap an angle in degrees into [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def is_rigid(m, tol: float = 1e-3) -> bool:
    """True for a finite 4x4 rotation+translation matrix - the only kind SteamVR should ever get from us."""
    try:
        a = np.asarray(m, dtype=float)
    except (TypeError, ValueError):
        return False
    if a.shape != (4, 4) or not np.all(np.isfinite(a)):
        return False
    r = a[:3, :3]
    return (np.allclose(a[3], [0, 0, 0, 1], atol=tol)
            and np.allclose(r @ r.T, np.eye(3), atol=tol)
            and abs(float(np.linalg.det(r)) - 1.0) < tol)


def state_name(state: str | None) -> str:
    """'shape-mismatch rms=4.2cm' -> 'shape-mismatch' (the part that doesn't change every tick)."""
    return (state or "").split(" ")[0]


# ----------------------------------------------------------------------------
# Only one FrameAnchor at a time (two would fight and over-correct)
# ----------------------------------------------------------------------------

_instance_handle = None


def acquire_single_instance(name: str = "Local\\FrameAnchor-single-instance") -> bool:
    """
    True if no other FrameAnchor is running (or we can't tell). The claim is held until
    this process exits. Calling it again in the same process returns True.
    """
    global _instance_handle
    if _instance_handle is not None or sys.platform != "win32":
        return True
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.CreateMutexW(None, False, name)
        err = ctypes.get_last_error()
    except Exception:
        return True
    if not h:
        return True
    if err == 183:  # ERROR_ALREADY_EXISTS
        k32.CloseHandle(h)
        return False
    _instance_handle = h
    return True


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


def apply_points(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (t[:3, :3] @ pts.T).T + t[:3, 3]


def describe(t: np.ndarray) -> str:
    d = t[:3, 3]
    return (f"dx={d[0]*100:+.1f}cm  dy(height)={d[1]*100:+.1f}cm  dz={d[2]*100:+.1f}cm  "
            f"yaw={math.degrees(yaw_of(t)):+.2f}deg")


# ----------------------------------------------------------------------------
# SteamVR config backup
# ----------------------------------------------------------------------------

def steamvr_config_dir() -> str | None:
    """SteamVR's config folder (where chaperone_info.vrchap lives), or None."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        try:
            with open(os.path.join(local, "openvr", "openvrpaths.vrpath"), encoding="utf-8") as f:
                for d in json.load(f).get("config") or []:
                    if os.path.isdir(d):
                        return d
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
            steam = winreg.QueryValueEx(k, "SteamPath")[0]
        d = os.path.join(os.path.normpath(steam), "config")
        if os.path.isdir(d):
            return d
    except (ImportError, OSError):
        pass
    return None


def backup_chaperone_once(backup_dir: str, log=print) -> None:
    """Copy chaperone_info.vrchap to backup_dir, unless an earlier backup already exists."""
    dest = os.path.join(backup_dir, "chaperone_info (before FrameAnchor).vrchap")
    if os.path.exists(dest):
        return
    cfg = steamvr_config_dir()
    src = os.path.join(cfg, "chaperone_info.vrchap") if cfg else None
    if not src or not os.path.isfile(src):
        log("(Couldn't find SteamVR's chaperone_info.vrchap to back up - continuing anyway.)")
        return
    try:
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(src, dest)
        log(f"Backed up SteamVR's playspace file to {dest}")
    except OSError as e:
        log(f"(Couldn't back up SteamVR's playspace file: {e} - continuing anyway.)")


# ----------------------------------------------------------------------------
# OpenVR backend (kept separate so the logic can be tested without SteamVR)
# ----------------------------------------------------------------------------

class OpenVRBackend:
    def __init__(self, backup_dir: str = BACKUP_DIR_DEFAULT, log=print):
        import openvr  # imported here so tests don't need SteamVR
        self.openvr = openvr
        self.vr = openvr.init(openvr.VRApplication_Background)
        self.setup = openvr.VRChaperoneSetup()
        self._pose_array = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        self._serial_cache: dict[int, str] = {}
        self._system_cache: dict[int, str] = {}
        self.backup_dir = backup_dir
        self.log = log
        self._backed_up = False

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
            pose = mat34_to_np(p.mDeviceToAbsoluteTracking)
            if is_rigid(pose):          # skip garbage (NaN etc.) instead of trusting it
                out[serial] = pose
        return out

    def get_zero_pose(self) -> np.ndarray:
        self.setup.revertWorkingCopy()
        ok, m = self.setup.getWorkingStandingZeroPoseToRawTrackingPose()
        if not ok:
            raise ChaperoneWriteError("SteamVR wouldn't give out the standing origin "
                                      "(is the room / playspace set up?)")
        z = mat34_to_np(m)
        if not is_rigid(z):
            raise ChaperoneWriteError("SteamVR gave an invalid standing origin, so nothing was changed")
        return z

    def set_zero_pose(self, z: np.ndarray) -> None:
        if not is_rigid(z):
            raise ChaperoneWriteError("FrameAnchor refused to write an invalid standing origin")
        if not self._backed_up:
            backup_chaperone_once(self.backup_dir, self.log)
            self._backed_up = True
        m = self.openvr.HmdMatrix34_t()
        np_to_mat34(z, m)
        self.setup.setWorkingStandingZeroPoseToRawTrackingPose(m)
        if not self.setup.commitWorkingCopy(self.openvr.EChaperoneConfigFile_Live):
            self.setup.revertWorkingCopy()
            raise ChaperoneWriteError("SteamVR refused to save the new standing origin")

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
    max_rms_m: float = 0.04          # shape mismatch above this = not calibrated / stations moved
    max_point_err_m: float = 0.05    # ...or any single reference point off by more than this
    max_err_ratio: float = 0.6       # ...or the misfit is a big part of the jump (one station moved)
    max_shift_m: float = 3.0         # refuse absurd corrections
    cooldown_s: float = 1.0          # wait after a correction before judging again
    anchor_avg_s: float = 1.0        # average poses this long when anchoring
    first_settle_s: float = 5.0      # be extra patient before the first fix after connecting
    confirm_m: float = 0.10          # first fix after connecting bigger than this...
    confirm_deg: float = 5.0         # ...or this needs the user to confirm
    burst_count: int = 6             # this many fixes...
    burst_window_s: float = 60.0     # ...within this time means something is fighting us
    burst_backoff_s: float = 60.0    # so stop trying for this long
    refused_backoff_s: float = 30.0  # SteamVR refused a change: wait this long before retrying
    dry_run: bool = False


def _valid_pose(v) -> np.ndarray | None:
    try:
        a = np.array(v, dtype=float)
    except (TypeError, ValueError):
        return None
    return a if is_rigid(a) else None


UNDO_FILE_NAME = "frame_anchor_undo.json"


def save_undo_point(path: str, z: np.ndarray) -> None:
    """Keep the undo point when the lock itself is removed."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"saved": time.time(), "zero_pose": np.asarray(z).tolist()}, f, indent=1)
    os.replace(tmp, path)


def load_undo_point(path: str) -> np.ndarray | None:
    try:
        with open(path, encoding="utf-8") as f:
            return _valid_pose(json.load(f)["zero_pose"])
    except Exception:
        return None


@dataclass
class Anchor:
    poses: dict[str, np.ndarray]
    created: float = field(default_factory=time.time)
    zero_pose: np.ndarray | None = None   # SteamVR standing origin when locked (for undo)

    def save(self, path: str) -> None:
        data = {"created": self.created,
                "poses": {k: v.tolist() for k, v in self.poses.items()}}
        if self.zero_pose is not None:
            data["zero_pose"] = self.zero_pose.tolist()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "Anchor | None":
        """Load a saved lock. Returns None for a missing or damaged file (never raises)."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            poses = {}
            for k, v in data["poses"].items():
                p = _valid_pose(v)
                if p is None:
                    return None
                poses[str(k)] = p
            if not poses:
                return None
            zero = data.get("zero_pose")
            zero = _valid_pose(zero) if zero is not None else None
            return Anchor(poses, float(data["created"]), zero)
        except Exception:
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
        self.refused_until = 0.0
        self.first_fix_done = False
        self.undo_pose: np.ndarray | None = None   # standing origin at the last lock
        self.awaiting_confirm = False              # a big first fix is waiting for a yes

    # -- anchoring -----------------------------------------------------------
    def capture_anchor(self, sleep=time.sleep, now=time.monotonic) -> bool:
        if not self.b.hmd_ok():
            self.log("!! The headset isn't tracking right now - put it on (or uncover it) "
                     "and lock again once things look right.")
            return False
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
        try:
            zero = self.b.get_zero_pose()
            if not is_rigid(zero):
                raise ChaperoneWriteError("it wasn't a valid position")
        except Exception as e:
            zero = None
            self.log(f"(Couldn't read SteamVR's standing origin, so Undo won't be available: {e})")
        self.anchor = Anchor(poses, zero_pose=zero)
        self.undo_pose = zero
        self.awaiting_confirm = False
        self.window.clear()
        self.fix_times.clear()
        self.backoff_until = 0.0
        self.refused_until = 0.0
        self.first_fix_done = True  # the user just confirmed things are right
        heights = ", ".join(f"{k}: {v[1, 3]:.2f}m high" for k, v in poses.items())
        self.log(f"Anchored to {len(poses)} device(s): {heights}")
        if len(poses) == 1 and not self.anchor_serials:
            self.log("   (only one base station - yaw relies on its orientation, which is less precise)")
        return True

    def confirm(self) -> bool:
        """
        User says: yes, put the playspace back to the saved lock even though it's far off.
        Only counts while that question is actually being asked (a late click does nothing).
        """
        if not self.awaiting_confirm:
            return False
        self.first_fix_done = True
        return True

    def undo(self) -> bool:
        """Put SteamVR's standing origin back to how it was at the last lock, then pause."""
        if self.undo_pose is None or not is_rigid(self.undo_pose):
            self.log("!! Nothing to undo to - there's no saved starting point from a lock yet.")
            return False
        self.b.set_zero_pose(self.undo_pose)
        self.paused = True
        self.window.clear()
        self.log("Undone: the playspace is back to how it was when you clicked Lock. "
                 "FrameAnchor is now paused.")
        return True

    # -- per-tick logic ------------------------------------------------------
    def measure(self) -> tuple[np.ndarray, float, float, float] | None:
        """
        Transform that would move the current space back onto the anchor.
        Returns (T, rms, worst point error, mean point movement) or None.
        """
        cur = {k: v for k, v in self.b.get_anchor_poses(self.anchor_serials).items() if is_rigid(v)}
        self.last_seen = cur
        common = sorted(set(cur) & set(self.anchor.poses))
        if not common:
            return None
        src = np.vstack([device_points(cur[k]) for k in common])
        dst = np.vstack([device_points(self.anchor.poses[k]) for k in common])
        t, rms = fit_yaw_translation(src, dst)
        moved = apply_points(t, src)
        worst = float(np.max(np.linalg.norm(moved - dst, axis=1)))
        disp = float(np.mean(np.linalg.norm(moved - src, axis=1)))
        return t, rms, worst, disp

    def step(self, now: float) -> str:
        """One update. Returns a short status string (also useful for tests)."""
        state = self._step(now)
        self.awaiting_confirm = state.startswith("confirm-needed")
        return state

    def _step(self, now: float) -> str:
        if self.anchor is None:
            return "no-anchor"
        if self.paused:
            return "paused"
        if now < self.refused_until:
            return "write-refused"
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
        t, rms, worst, disp = m
        shift = float(np.linalg.norm(t[:3, 3]))
        yaw_deg = abs(math.degrees(yaw_of(t)))
        if rms > self.s.max_rms_m:
            self.window.clear()
            return f"shape-mismatch rms={rms*100:.1f}cm"
        if shift < self.s.threshold_m and yaw_deg < self.s.threshold_deg:
            self.window.clear()
            return "ok"
        if worst > self.s.max_point_err_m or worst > self.s.max_err_ratio * disp:
            # The base stations didn't all move together: one was bumped or SteamVR
            # re-solved its position. Not a playspace jump - don't "fix" it.
            self.window.clear()
            return f"shape-mismatch worst={worst*100:.1f}cm"
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
        spread_deg = max(abs(wrap_deg(math.degrees(yaw_of(a) - yaw_of(ts[-1])))) for a in ts)
        if spread_m > self.s.stable_m or spread_deg > self.s.stable_deg:
            return "settling (still moving)"

        if not self.first_fix_done and (shift > self.s.confirm_m or yaw_deg > self.s.confirm_deg):
            # Saved lock from an earlier session and the playspace is far from it.
            # Maybe the user redid room setup on purpose - ask instead of guessing.
            return "confirm-needed " + describe(t)

        try:
            self.apply(t)
        except ChaperoneWriteError as e:
            self.window.clear()
            self.refused_until = now + self.s.refused_backoff_s
            self.log(f"!! {e}. Will try again in {self.s.refused_backoff_s:.0f} s.")
            return "write-refused"
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
        if self.s.dry_run:
            self.corrections += 1
            self.last_fix = describe(t)
            self.log(f"[dry-run] would correct: {describe(t)}")
            return
        z = self.b.get_zero_pose()
        if not is_rigid(z):
            raise ChaperoneWriteError("SteamVR gave an invalid standing origin, so nothing was changed")
        new = z @ np.linalg.inv(t)
        if not is_rigid(new):
            raise ChaperoneWriteError("the computed fix was invalid, so it wasn't applied")
        self.b.set_zero_pose(new)
        self.corrections += 1
        self.last_fix = describe(t)
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

    if not acquire_single_instance():
        log("FrameAnchor is already running (look for its window in the taskbar). "
            "Two copies would fight each other, so this one is closing.")
        return 0

    cmds: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=stdin_reader, args=(cmds,), daemon=True).start()

    log("FrameAnchor starting - waiting for SteamVR (start it whenever you like; Ctrl+C or q + Enter quits)...")
    backend = None
    try:
        while backend is None:
            try:
                backend = OpenVRBackend(log=log)
            except Exception:
                try:
                    if cmds.get(timeout=2.0) == "q":
                        return 0
                except queue.Empty:
                    pass
        log("Connected to SteamVR.")

        fa = FrameAnchor(backend, settings, log, args.anchor_serial)
        saved = None if args.fresh else Anchor.load(args.anchor_file)
        undo_file = os.path.join(os.path.dirname(os.path.abspath(args.anchor_file)), UNDO_FILE_NAME)
        fa.undo_pose = saved.zero_pose if saved and saved.zero_pose is not None else load_undo_point(undo_file)
        if saved:
            fa.anchor = saved
            log(f"Loaded saved lock from {time.ctime(saved.created)} ({len(saved.poses)} device(s)).")
            log("To recenter or fix the floor on purpose: type 'p' + Enter FIRST, fix it, then 'a' + Enter.")
        else:
            log("No saved lock yet. When your floor and trackers look CORRECT, type 'a' + Enter.")
        if settings.dry_run:
            log("DRY RUN: nothing will be changed.")
        log("Commands: a = lock now, p = pause/resume, y = apply a big fix, u = undo, s = status, q = quit")

        last_state = None
        while True:
            for ev in backend.poll_events():
                log(f"SteamVR event: {ev}")
                if ev == "Quit":
                    log("SteamVR closed. Start FrameAnchor again next time you play.")
                    return 0
            try:
                while True:
                    c = cmds.get_nowait()
                    if c == "a":
                        if fa.capture_anchor():
                            try:
                                fa.anchor.save(args.anchor_file)
                                log(f"Saved lock to {args.anchor_file}")
                            except OSError as e:
                                log(f"(Couldn't save the lock: {e})")
                            if fa.paused:
                                fa.paused = False
                                log("Resumed.")
                    elif c == "p":
                        fa.paused = not fa.paused
                        log("Paused." if fa.paused else "Resumed.")
                    elif c == "y":
                        if fa.confirm():
                            log("OK - putting the playspace back to your lock.")
                        else:
                            log("Nothing is waiting for a yes right now.")
                    elif c == "u":
                        try:
                            fa.undo()
                        except ChaperoneWriteError as e:
                            log(f"!! Undo failed: {e}")
                    elif c == "s":
                        log(f"State: {last_state}; corrections so far: {fa.corrections}")
                    elif c == "q":
                        return 0
            except queue.Empty:
                pass

            state = fa.step(time.monotonic())
            name = state_name(state)
            # Only print when the kind of state changes - the numbers in it change every tick.
            if name != state_name(last_state) and name not in ("ok", "cooldown", "corrected"):
                log(f"Status: {state}")
                if name == "confirm-needed":
                    log("The playspace is far from your saved lock. If you changed it on purpose, "
                        "type 'a' + Enter to lock the new one. Otherwise type 'y' + Enter to put it back.")
            last_state = state
            time.sleep(0.05)
    except KeyboardInterrupt:
        return 0
    finally:
        if backend is not None:
            try:
                backend.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
