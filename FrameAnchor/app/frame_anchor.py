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

Moves you make on purpose with OVR Advanced Settings' playspace mover are kept:
FrameAnchor works out SteamVR's live standing origin (read-only, from the base
stations in raw vs standing space). If it changes gradually over several ticks
(a drag) or by a clean turn around your head (a snap turn), the lock follows
it. If it changes in one sudden step, that's a jump and gets fixed. OVR Advanced
Settings re-applies its own remembered position when a drag starts, which can drop
an earlier fix; that shows up as one big step at the start of the drag, so the
drag is kept and the dropped fix is made again afterwards.

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


def rotation_deg(r: np.ndarray) -> float:
    """Total rotation angle (degrees) of a 3x3 rotation matrix."""
    c = (float(np.trace(r[:3, :3])) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def poses_close(a: np.ndarray, b: np.ndarray, m: float, deg: float) -> bool:
    """True if two 4x4 poses differ by less than m metres and deg degrees."""
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])) < m
            and rotation_deg(a[:3, :3] @ b[:3, :3].T) < deg)


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
        self.use_live_zero = True     # switched off if the live reading ever proves unreliable

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

    def hmd_position(self) -> np.ndarray | None:
        """Headset position in standing space (used to recognise snap turns), or None."""
        ovr = self.openvr
        poses = self.vr.getDeviceToAbsoluteTrackingPose(
            ovr.TrackingUniverseStanding, 0, self._pose_array)
        hmd = poses[ovr.k_unTrackedDeviceIndex_Hmd]
        if not hmd.bPoseIsValid:
            return None
        m = mat34_to_np(hmd.mDeviceToAbsoluteTracking)
        return m[:3, 3].copy() if is_rigid(m) else None

    def get_live_zero_pose(self, serials: set[str] | None) -> np.ndarray | None:
        """
        SteamVR's standing origin *as currently in effect*, worked out read-only from where the
        base stations are in raw vs standing tracking space (raw = Z @ standing).

        Unlike reading the chaperone working copy, this includes a move that another app
        (OVR Advanced Settings' playspace mover) is previewing but hasn't saved, and it never
        throws away that app's unsaved move. None if it can't be worked out reliably.
        """
        std = self.get_anchor_poses(serials)
        raw = self.get_anchor_poses(serials, self.openvr.TrackingUniverseRawAndUncalibrated)
        zs = [raw[k] @ np.linalg.inv(std[k]) for k in sorted(set(std) & set(raw))]
        zs = [z for z in zs if is_rigid(z, tol=1e-2)]
        if not zs:
            return None
        if any(not poses_close(z, zs[0], 0.01, 0.5) for z in zs[1:]):
            return None     # the stations disagree (read mid-change?) - don't trust this tick
        z = average_poses([{"z": z} for z in zs]).get("z")
        return z if z is not None and is_rigid(z) else None

    def get_anchor_poses(self, serials: set[str] | None, universe=None) -> dict[str, np.ndarray]:
        """Standing-space (or `universe`) poses of base stations (or of the given serials)."""
        ovr = self.openvr
        if universe is None:
            universe = ovr.TrackingUniverseStanding
        poses = self.vr.getDeviceToAbsoluteTrackingPose(universe, 0, self._pose_array)
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
        """The standing origin to build a fix on: the live one if possible (keeps another app's
        unsaved playspace move), otherwise SteamVR's saved one."""
        if self.use_live_zero:
            z = self.get_live_zero_pose(None)
            if z is not None:
                return z
        return self.get_saved_zero_pose()

    def get_saved_zero_pose(self) -> np.ndarray:
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
    # Playspace moves by other apps (OVR Advanced Settings playspace mover / turning):
    follow_moves: bool = True        # keep a move made gradually (a drag) instead of undoing it
    move_detect_m: float = 0.002     # standing origin changed by more than this...
    move_detect_deg: float = 0.1     # ...or this between two ticks = someone is moving it
    move_still_s: float = 0.5        # the move is over once it's been still this long
    clobber_min_m: float = 0.02      # a dropped fix smaller than this is ignored
    clobber_tol: float = 0.5         # match tolerance for a dropped fix, as a fraction of a drag step
    snap_min_deg: float = 10.0       # a one-step pure turn at least this big...
    snap_axis_m: float = 0.75        # ...around the headset (or the playspace centre) = a snap turn
    snap_max_dy_m: float = 0.01      # (and no height change)
    expect_s: float = 2.0            # our own write must show up within this long
    reapply_grace_s: float = 5.0     # fixes this soon after a followed move don't count as "fighting"
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
        # Watching SteamVR's live standing origin for moves made by other apps:
        self.z_last: np.ndarray | None = None      # live standing origin at the last tick
        self.moving = False                        # another app is moving the playspace right now
        self.move_z0: np.ndarray | None = None     # standing origin just before that move began
        self.moves: list[np.ndarray] = []          # standing origin after each step of the move
        self.move_last_t = 0.0
        self.z_expect: np.ndarray | None = None    # what our own last write should look like
        self.z_expect_t = 0.0
        self.live_confirmed = False                # one of our writes showed up exactly as expected
        self.followed_t = -1e9                     # when we last kept another app's move
        self.sudden_t = -1e9                       # when the standing origin last jumped in one step
        self.anchor_moved = False                  # the lock followed a move; caller should save it
        self.moves_followed = 0
        self.fix_writes: list[np.ndarray] = []     # raw-space changes our writes made this session
        self.writer_synced = 0                     # other apps already know fix_writes[:writer_synced]
        self.fix_seqs: list[int] = []              # sequence number of each entry in fix_writes
        self.fix_seq_next = 0
        # Positions the playspace rested at (for recognising resets), each with fix_seq_next at
        # that moment: fixes numbered below it were already part of that position.
        self.rest_zs: list[tuple[np.ndarray, int]] = []

    # -- moves made by other apps (OVR Advanced Settings playspace mover) ------
    def _live_zero(self) -> np.ndarray | None:
        f = getattr(self.b, "get_live_zero_pose", None)
        if f is None or not getattr(self.b, "use_live_zero", True):
            return None
        return f(self.anchor_serials)

    def _expect(self, z: np.ndarray, now: float, before: np.ndarray | None = None) -> None:
        """We just wrote z (over `before`): when it shows up, it's ours, not a move by another app."""
        self.z_expect = z.copy()
        self.z_expect_t = now
        if before is not None and is_rigid(before, tol=1e-2):
            self.fix_writes.append(z @ np.linalg.inv(before))   # raw-space change we made
            self.fix_seqs.append(self.fix_seq_next)
            self.fix_seq_next += 1
            if len(self.fix_writes) > 50:
                del self.fix_writes[0], self.fix_seqs[0]
                self.writer_synced = max(0, self.writer_synced - 1)

    def track_moves(self, now: float) -> bool:
        """Watch SteamVR's live standing origin. Returns True while another app is moving it."""
        z = self._live_zero()
        if z is None:
            return self.moving
        if self.z_expect is not None:
            if poses_close(z, self.z_expect, 0.005, 0.3):
                self.z_expect = None
                self.z_last = z
                self.live_confirmed = True
                self._remember_rest(z)
                return self.moving
            if now - self.z_expect_t > self.s.expect_s:
                self.z_expect = None
                if not self.live_confirmed and hasattr(self.b, "use_live_zero"):
                    self.b.use_live_zero = False
                    self.moving, self.moves, self.z_last = False, [], None
                    self.log("!! Couldn't confirm SteamVR's live playspace position after a fix, so "
                             "FrameAnchor has stopped keeping other apps' playspace moves this session "
                             "(it still fixes jumps). Please send the log to whoever gave you FrameAnchor.")
                    return False
        if self.z_last is None:
            self.z_last = z
            self._remember_rest(z)
            return False
        if poses_close(z, self.z_last, self.s.move_detect_m, self.s.move_detect_deg):
            if self.moving and now - self.move_last_t >= self.s.move_still_s:
                self._finish_move(now)
            return self.moving
        if not self.moving:
            self.moving = True
            self.move_z0 = self.z_last
            self.moves = []
        self.moves.append(z)
        self.z_last = z
        self.move_last_t = now
        return True

    def _step_size(self, z_prev: np.ndarray, z_new: np.ndarray) -> float:
        """How far one step of a move shifted the room around you (mean, metres)."""
        return self._transform_size(np.linalg.inv(z_new) @ z_prev)

    def _transform_size(self, d: np.ndarray) -> float:
        """Mean distance a standing-space transform moves the base stations (metres)."""
        if self.anchor and self.anchor.poses:
            pts = np.vstack([device_points(p) for p in self.anchor.poses.values()])
        else:
            pts = np.zeros((1, 3))
        return float(np.mean(np.linalg.norm(apply_points(d, pts) - pts, axis=1)))

    def _is_snap_turn(self, z0: np.ndarray, z1: np.ndarray) -> bool:
        return self._snap_score(z0, z1) is not None

    def _snap_score(self, z0: np.ndarray, z1: np.ndarray) -> float | None:
        """
        One-step pure turn around the headset (or the playspace centre) = a snap turn.
        Returns None if it isn't one, else how far it is from a perfect one (smaller = cleaner).
        """
        d = np.linalg.inv(z1) @ z0
        yaw = abs(math.degrees(yaw_of(d)))
        tilt = rotation_deg(d[:3, :3]) - yaw
        if yaw < self.s.snap_min_deg or tilt > 0.5:
            return None
        if abs(d[1, 3]) > self.s.snap_max_dy_m:
            return None
        a = np.array([[1 - d[0, 0], -d[0, 2]], [-d[2, 0], 1 - d[2, 2]]])
        try:
            axis = np.linalg.solve(a, np.array([d[0, 3], d[2, 3]]))   # the point the turn is around
        except np.linalg.LinAlgError:
            return None
        centres = [np.zeros(2)]
        f = getattr(self.b, "hmd_position", None)
        hmd = f() if f else None
        if hmd is not None:
            centres.append(np.array([hmd[0], hmd[2]]))
        dist = min(float(np.linalg.norm(axis - c)) for c in centres)
        if dist > self.s.snap_axis_m:
            return None
        return dist + abs(float(d[1, 3])) + 0.01 * max(tilt, 0.0)

    def _dropped_fix(self, z0: np.ndarray, zs: list[np.ndarray]) -> np.ndarray | None:
        """
        If a move started by throwing away FrameAnchor's latest fix(es), return those fixes
        (as a raw-space transform to put back on the left), else None.

        Candidates are "nothing dropped" and "the last 1, 2, ... fixes FrameAnchor really wrote
        this session were dropped". For each, the move's first step is worked out; the candidate
        under which that first step looks most like a normal drag step (small, or like the next
        step) wins. Ties go to "nothing dropped".
        """
        d2 = np.linalg.inv(zs[1]) @ zs[0] if len(zs) > 1 else None

        def score(k):
            d1 = np.linalg.inv(k @ zs[0]) @ z0
            s = self._transform_size(d1)
            if d2 is not None:
                s = min(s, self._transform_size(d1 @ np.linalg.inv(d2)))
            return s

        cands = self._drop_candidates()
        scores = {m: score(k) for m, k in cands}
        scores[0] -= 0.005                           # ties go to "nothing dropped"
        best = min(scores, key=scores.get)
        # Most likely: it dropped exactly the fixes it hasn't seen yet. Prefer that unless
        # another explanation is clearly better.
        prior = len(self.fix_writes) - self.writer_synced
        if prior in scores:
            typical = float(np.median([self._step_size(a, b) for a, b in zip(zs[:-1], zs[1:])]))
            if scores[prior] <= scores[best] + max(self.s.clobber_min_m, 0.25 * typical):
                best = prior
        return self._note_dropped(best, dict(cands)[best])

    def _drop_candidates(self) -> list[tuple[int, np.ndarray]]:
        """(m, transform that puts back FrameAnchor's last m fixes), m = 0 meaning none."""
        out = [(0, np.eye(4))]
        k = np.eye(4)
        for i, m in enumerate(reversed(self.fix_writes), start=1):   # latest fix, latest two, ...
            k = k @ m
            if self._transform_size(np.linalg.inv(k)) >= self.s.clobber_min_m:
                out.append((i, k.copy()))
        return out

    def _remember_rest(self, z: np.ndarray | None) -> None:
        """A position the playspace rested at, for recognising resets."""
        if z is None:
            return
        self.rest_zs = [r for r in self.rest_zs if not poses_close(z, r[0], 0.003, 0.15)]
        self.rest_zs.append((z.copy(), self.fix_seq_next))
        del self.rest_zs[:-30]

    def _match_rest(self, z: np.ndarray) -> tuple[np.ndarray, int] | None:
        for r in self.rest_zs:
            if poses_close(z, r[0], 0.003, 0.15):
                return r
        return None

    def _note_dropped(self, m: int, k: np.ndarray) -> np.ndarray | None:
        """The other app's move dropped our last m fixes: they're gone (the fix that puts them
        back is a new one), and it knows about all the others."""
        if m:
            del self.fix_writes[len(self.fix_writes) - m:], self.fix_seqs[len(self.fix_seqs) - m:]
        self.writer_synced = len(self.fix_writes)
        return k if m else None

    def _finish_move(self, now: float) -> None:
        zs, z0 = self.moves, self.move_z0
        self.moving, self.moves = False, []
        if self.anchor is None or z0 is None or not zs:
            return
        target, note = None, ""
        if self.s.follow_moves:
            rest = self._match_rest(zs[0]) if len(zs) == 1 else None
            if rest is not None:
                # Back exactly onto a position the playspace rested at before: a "reset playspace"
                # (OVR Advanced Settings Reset / reset-offsets bind). A Frame jump never lands
                # there to the millimetre. Keep it - and the fixes made since it rested there,
                # which the reset dropped.
                m = sum(1 for q in self.fix_seqs if q >= rest[1])
                k = np.eye(4)
                for f in self.fix_writes[len(self.fix_writes) - m:][::-1]:
                    k = k @ f
                k = self._note_dropped(m, k)
                target = zs[0] if k is None else k @ zs[0]
                note = " (back to an earlier position - a playspace reset)"
            elif len(zs) == 1:
                # A one-step change is a jump, unless it's a snap turn or a tiny nudge - possibly
                # made from a remembered origin that drops some of FrameAnchor's fixes.
                best = None              # (score, z1, kind, m, k) - the cleanest explanation wins
                for m, k in self._drop_candidates():
                    z1 = k @ zs[0]
                    sc = self._snap_score(z0, z1)
                    if sc is not None and (best is None or sc < best[0]):
                        best = (sc, z1, " (a turn)", m, k)
                    size = self._step_size(z0, z1)
                    if size < self.s.threshold_m and (best is None or size < best[0]):
                        best = (size, z1, "", m, k)     # a tiny nudge (e.g. a very slow drag)
                if best is not None:
                    target, note = best[1], best[2]
                    if self._note_dropped(best[3], best[4]) is not None:
                        note += " (it had dropped an earlier fix - put back)"
            else:
                target = zs[-1]
                dropped = self._dropped_fix(z0, zs)
                if dropped is not None:
                    # OVR Advanced Settings started the drag from the position it remembered from
                    # before FrameAnchor's fix(es), dropping them. Keep the drag and put the fix back.
                    target = dropped @ zs[-1]
                    note = f" (it had dropped an earlier fix of {self._step_size(zs[-1], target)*100:.0f} cm)"
        if target is None or not is_rigid(target, tol=1e-2):
            self.sudden_t = now
            self.log("SteamVR's playspace position changed in one sudden step - treating it as a jump.")
            return
        self._remember_rest(zs[-1])         # where the other app left it (as it wrote it)
        d = np.linalg.inv(target) @ z0      # old standing coordinates -> new ones
        self.anchor.poses = {k: d @ v for k, v in self.anchor.poses.items()}
        self.anchor_moved = True
        self.followed_t = now
        self.moves_followed += 1
        self.window.clear()
        if note or self._step_size(z0, target) >= 0.01:     # don't fill the log with tiny nudges
            self.log(f"Kept your playspace move{note}: {describe(d)}")

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
        self.moving, self.moves = False, []
        try:
            self.z_last = self._live_zero()
        except Exception:
            self.z_last = None
        self._remember_rest(self.z_last)
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
        before = self.z_last
        self.b.set_zero_pose(self.undo_pose)
        self._expect(self.undo_pose, time.monotonic(), before=before)
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
        moving = self.track_moves(now)
        if self.anchor is None:
            return "no-anchor"
        if self.paused:
            return "paused"
        if moving and self.s.follow_moves:
            # Another app (OVR Advanced Settings playspace mover) is moving the playspace:
            # don't fight it. Once it stops, the lock follows the move.
            self.window.clear()
            return "being-moved"
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

        reapply = now - self.followed_t < self.s.reapply_grace_s
        if reapply:
            cause = "putting back a fix that a playspace move dropped"
        elif now - self.sudden_t < 30.0:
            cause = "cause: SteamVR's standing origin jumped"
        elif getattr(self.b, "use_live_zero", False) and self.z_last is not None:
            cause = "cause: base stations moved in raw tracking space"
        else:
            cause = ""
        try:
            self.apply(t, now, cause)
        except ChaperoneWriteError as e:
            self.window.clear()
            self.refused_until = now + self.s.refused_backoff_s
            self.log(f"!! {e}. Will try again in {self.s.refused_backoff_s:.0f} s.")
            return "write-refused"
        self.window.clear()
        self.first_fix_done = True
        self.cooldown_until = now + self.s.cooldown_s
        if reapply:
            return "corrected"   # one fix per playspace move is expected, not "something fighting us"
        self.fix_times.append(now)
        while self.fix_times and now - self.fix_times[0] > self.s.burst_window_s:
            self.fix_times.popleft()
        if len(self.fix_times) >= self.s.burst_count:
            self.fix_times.clear()
            self.backoff_until = now + self.s.burst_backoff_s
            self.log("!! Had to fix the playspace many times in a row - something else keeps moving it "
                     "(OVR Advanced Settings space drag? another playspace tool?). Taking a 60 s break.")
        return "corrected"

    def apply(self, t: np.ndarray, now: float | None = None, cause: str = "") -> None:
        why = f"  ({cause})" if cause else ""
        if self.s.dry_run:
            self.corrections += 1
            self.last_fix = describe(t)
            self.log(f"[dry-run] would correct: {describe(t)}{why}")
            return
        z = self.b.get_zero_pose()
        if not is_rigid(z):
            raise ChaperoneWriteError("SteamVR gave an invalid standing origin, so nothing was changed")
        new = z @ np.linalg.inv(t)
        if not is_rigid(new):
            raise ChaperoneWriteError("the computed fix was invalid, so it wasn't applied")
        self.b.set_zero_pose(new)
        self._expect(new, time.monotonic() if now is None else now, before=z)
        self.corrections += 1
        self.last_fix = describe(t)
        self.log(f"Corrected playspace: {describe(t)}{why}")


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
    ap.add_argument("--no-follow", action="store_true",
                    help="undo playspace moves made by other apps (OVR Advanced Settings) instead of keeping them")
    args = ap.parse_args(argv)

    settings = Settings(threshold_m=args.threshold_cm / 100, threshold_deg=args.threshold_deg,
                        dry_run=args.dry_run, follow_moves=not args.no_follow)

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
            if fa.anchor_moved and fa.anchor is not None:
                fa.anchor_moved = False
                try:
                    fa.anchor.save(args.anchor_file)
                except OSError as e:
                    log(f"(Couldn't save the moved lock: {e})")
            name = state_name(state)
            # Only print when the kind of state changes - the numbers in it change every tick.
            if name != state_name(last_state) and name not in ("ok", "cooldown", "corrected", "being-moved"):
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
