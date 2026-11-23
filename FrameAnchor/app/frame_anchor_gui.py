#!/usr/bin/env python3
"""FrameAnchor window. Run this file; it falls back to the text version if the window can't open."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback

import frame_anchor as fa

LOG_FILE = os.path.join(fa.data_dir(), "frame_anchor_log.txt")

GREEN, YELLOW, ORANGE, GREY, BLUE = "#2e9e5b", "#c99a1a", "#d2691e", "#6b7280", "#3b6fd4"


def friendly(state: str, dry_run: bool) -> tuple[str, str, str]:
    """core state -> (headline, detail, colour)"""
    if state == "waiting":
        return ("Waiting for SteamVR",
                "Start SteamVR and put your headset on. This connects by itself.", GREY)
    if state == "no-anchor":
        return ("Connected - not locked yet",
                "Get your floor and trackers looking right, then click the big button.", BLUE)
    if state == "paused":
        return ("Paused", "Nothing will be fixed until you click Resume.", GREY)
    if state in ("ok", "cooldown", "corrected"):
        if dry_run:
            return ("Watching (test mode)", "Jumps are written to the log below but not fixed.", GREEN)
        return ("Locked", "Watching for jumps. You can go play now.", GREEN)
    if state.startswith("settling"):
        return ("Jump detected", "Waiting a moment for tracking to settle, then fixing it.", YELLOW)
    if state == "hmd-not-tracking":
        return ("Headset lost tracking", "Waiting for the headset to track normally again.", YELLOW)
    if state == "no-anchor-devices-visible":
        return ("Can't see the base stations",
                "Check that the base stations are on and that a tracker is switched on.", ORANGE)
    if state.startswith("shape-mismatch"):
        return ("Base stations don't match the lock",
                "Normal for a few seconds while Space Calibrator calibrates. If it stays like this, "
                "a base station was moved: get everything looking right, then lock again.", ORANGE)
    if state.startswith("shift-too-large"):
        return ("Playspace is very far off",
                "Too far to fix safely. Get everything looking right, then lock again.", ORANGE)
    if state == "backing-off":
        return ("Taking a break",
                "Something else keeps moving the playspace (OVR Advanced Settings space drag?). "
                "Trying again in a minute.", ORANGE)
    return (state, "", GREY)


class Worker(threading.Thread):
    """Owns the SteamVR connection. Talks to the window through two queues."""

    def __init__(self, backend_factory, anchor_file, out_q, tick=0.05, retry=2.0):
        super().__init__(daemon=True)
        self.backend_factory = backend_factory
        self.anchor_file = anchor_file
        self.out = out_q
        self.cmds: "queue.Queue[tuple]" = queue.Queue()
        self.settings = fa.Settings()
        self.tick = tick
        self.retry = retry
        self.stop_flag = threading.Event()
        self.paused = False
        self.backend = None
        self.core = None

    # -- helpers
    def log(self, msg):
        line = time.strftime("[%H:%M:%S] ") + msg
        self.out.put(("log", line))
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d ") + line + "\n")
        except OSError:
            pass

    def status(self, state):
        seen = len(self.core.last_seen) if self.core else 0
        fixes = self.core.corrections if self.core else 0
        last = self.core.last_fix if self.core else ""
        self.out.put(("status", state, fixes, last, seen))

    def disconnect(self):
        if self.backend is not None:
            try:
                self.backend.shutdown()
            except Exception:
                pass
        self.backend = None
        self.core = None

    def connect(self) -> bool:
        try:
            self.backend = self.backend_factory()
        except Exception:
            self.backend = None
            return False
        self.core = fa.FrameAnchor(self.backend, self.settings, self.log)
        self.core.paused = self.paused
        self.log("Connected to SteamVR.")
        saved = fa.Anchor.load(self.anchor_file)
        if saved:
            self.core.anchor = saved
            self.log(f"Using your saved lock from {time.strftime('%d %b %H:%M', time.localtime(saved.created))}.")
        return True

    def handle(self, cmd):
        name = cmd[0]
        if name == "pause":
            self.paused = cmd[1]
            if self.core:
                self.core.paused = self.paused
            self.log("Paused." if self.paused else "Resumed.")
        elif name == "dry":
            self.settings.dry_run = cmd[1]
            self.log("Test mode ON - nothing will be changed." if cmd[1] else "Test mode OFF - jumps will be fixed.")
        elif name == "lock":
            if not self.core:
                self.log("Not connected to SteamVR yet - start SteamVR first.")
            elif self.core.capture_anchor():
                try:
                    self.core.anchor.save(self.anchor_file)
                except OSError as e:
                    self.log(f"(Couldn't save the lock to disk: {e} - it still works until you close this.)")
                self.log("Check: do those heights match how high your real base stations are above the "
                         "floor? If not, fix your floor first and lock again.")
        elif name == "forget":
            try:
                os.remove(self.anchor_file)
            except OSError:
                pass
            if self.core:
                self.core.anchor = None
            self.log("Lock removed.")

    # -- main loop
    def run(self):
        last_state = None
        last_push = 0.0
        waiting_logged = False
        while not self.stop_flag.is_set():
            try:
                while True:
                    self.handle(self.cmds.get_nowait())
            except queue.Empty:
                pass
            except Exception:
                self.log("Problem handling a button click:\n" + traceback.format_exc())

            if self.backend is None:
                if not self.connect():
                    if not waiting_logged:
                        self.log("Waiting for SteamVR to start...")
                        waiting_logged = True
                    self.status("waiting")
                    self.stop_flag.wait(self.retry)
                    continue
                waiting_logged = False
                last_state = None

            try:
                quit_seen = False
                for ev in self.backend.poll_events():
                    if ev == "Quit":
                        quit_seen = True
                    else:
                        self.log("SteamVR: " + ev)
                if quit_seen:
                    self.log("SteamVR closed.")
                    self.disconnect()
                    self.status("waiting")
                    self.stop_flag.wait(5.0)
                    continue
                if self.core.anchor is None:
                    self.core.last_seen = self.backend.get_anchor_poses(None)
                state = self.core.step(time.monotonic())
            except Exception:
                self.log("Lost connection to SteamVR, reconnecting...\n" + traceback.format_exc(limit=3))
                self.disconnect()
                self.stop_flag.wait(self.retry)
                continue

            if state != last_state:
                self.status(state)
                last_state = state
                last_push = time.monotonic()
            elif time.monotonic() - last_push > 1.0:
                self.status(state)   # keeps the base-station count fresh
                last_push = time.monotonic()
            self.stop_flag.wait(self.tick)
        self.disconnect()


HELP = ("How to use:  1) Start SteamVR, turn on trackers, let Space Calibrator line everything up.   "
        "2) When your floor and trackers look right, click the big button.   "
        "3) Leave this window open while you play.   "
        "If you recenter or change your floor on purpose, click the big button again afterwards.")


class App:
    def __init__(self, tk, worker, events):
        self.tk = tk
        self.worker = worker
        self.events = events
        self.paused = False
        self.state = "waiting"

        root = self.root = tk.Tk()
        root.title("FrameAnchor")
        root.geometry("560x560")
        root.minsize(480, 460)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.headline = tk.Label(root, text="Starting...", font=("Segoe UI", 18, "bold"),
                                 fg="white", bg=GREY, pady=12)
        self.headline.pack(fill="x")
        self.detail = tk.Label(root, text="", font=("Segoe UI", 10), wraplength=520, justify="left",
                               anchor="w", padx=12, pady=8)
        self.detail.pack(fill="x")

        self.lock_btn = tk.Button(root, text="My floor and trackers look right - LOCK IT",
                                  font=("Segoe UI", 12, "bold"), bg=BLUE, fg="white",
                                  activebackground="#2f59ab", activeforeground="white",
                                  pady=10, command=self.on_lock)
        self.lock_btn.pack(fill="x", padx=12, pady=(4, 8))

        row = tk.Frame(root)
        row.pack(fill="x", padx=12)
        self.pause_btn = tk.Button(row, text="Pause", width=12, command=self.on_pause)
        self.pause_btn.pack(side="left")
        self.forget_btn = tk.Button(row, text="Remove lock", width=12, command=self.on_forget)
        self.forget_btn.pack(side="left", padx=8)
        self.dry_var = tk.IntVar(value=0)
        self.dry_chk = tk.Checkbutton(row, text="Test mode (don't change anything)",
                                      variable=self.dry_var, command=self.on_dry)
        self.dry_chk.pack(side="left", padx=8)

        self.info = tk.Label(root, text="", font=("Segoe UI", 9), anchor="w", justify="left",
                             padx=12, pady=6, fg="#374151")
        self.info.pack(fill="x")

        box = tk.Frame(root)
        box.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        scroll = tk.Scrollbar(box)
        scroll.pack(side="right", fill="y")
        self.logbox = tk.Text(box, height=10, wrap="word", font=("Consolas", 9), state="disabled",
                              yscrollcommand=scroll.set)
        self.logbox.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.logbox.yview)

        self.help = tk.Label(root, text=HELP, font=("Segoe UI", 9), wraplength=530, justify="left",
                             anchor="w", padx=12, pady=6, fg="#374151")
        self.help.pack(fill="x")

        self.show_status("waiting", 0, "", 0)
        root.after(100, self.pump)

    # -- buttons
    def on_lock(self):
        self.worker.cmds.put(("lock",))

    def on_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")
        self.worker.cmds.put(("pause", self.paused))

    def on_forget(self):
        self.worker.cmds.put(("forget",))

    def on_dry(self):
        self.worker.cmds.put(("dry", bool(self.dry_var.get())))

    def close(self):
        self.worker.stop_flag.set()
        self.root.destroy()

    # -- updates from the worker
    def add_log(self, line):
        self.logbox.config(state="normal")
        self.logbox.insert("end", line + "\n")
        self.logbox.see("end")
        self.logbox.config(state="disabled")

    def show_status(self, state, fixes, last, seen):
        self.state = state
        head, detail, colour = friendly(state, bool(self.dry_var.get()))
        self.headline.config(text=head, bg=colour)
        self.detail.config(text=detail)
        locked = state not in ("waiting", "no-anchor")
        self.lock_btn.config(text="Lock again (after a recenter or floor change)" if locked
                             else "My floor and trackers look right - LOCK IT")
        word = "Jumps seen" if self.dry_var.get() else "Jumps fixed"
        text = f"Base stations visible: {seen}     {word} this session: {fixes}"
        if last:
            text += f"\nLast: {last}"
        self.info.config(text=text)

    def pump(self):
        try:
            for _ in range(200):
                ev = self.events.get_nowait()
                if ev[0] == "log":
                    self.add_log(ev[1])
                elif ev[0] == "status":
                    self.show_status(*ev[1:])
        except queue.Empty:
            pass
        self.root.after(100, self.pump)


def run_gui() -> int:
    import tkinter as tk
    events: "queue.Queue[tuple]" = queue.Queue()
    worker = Worker(fa.OpenVRBackend, fa.ANCHOR_FILE_DEFAULT, events)
    app = App(tk, worker, events)
    worker.start()
    try:
        app.root.mainloop()
    finally:
        worker.stop_flag.set()
        worker.join(timeout=3)
    return 0


def main() -> int:
    try:
        return run_gui()
    except Exception:
        err = traceback.format_exc()
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(err + "\n")
        except OSError:
            pass
        print(err)
        print("The window couldn't open, so FrameAnchor is running in this text window instead.")
        print("Type  a  then Enter to lock when your floor and trackers look right.\n")
        return fa.main([])


if __name__ == "__main__":
    sys.exit(main())
