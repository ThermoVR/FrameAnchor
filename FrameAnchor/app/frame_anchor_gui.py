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
LOG_MAX_BYTES = 2_000_000          # rotate to frame_anchor_log.old.txt above this
LOGBOX_MAX_LINES = 2000            # on-screen log keeps about this many lines

GREEN, YELLOW, ORANGE, GREY, BLUE = "#2e9e5b", "#c99a1a", "#d2691e", "#6b7280", "#3b6fd4"

LOCK_TEXT = "My floor and trackers look right - LOCK IT"
RELOCK_TEXT = "Lock again (after a recenter or floor change)"


def friendly(state: str, dry_run: bool) -> tuple[str, str, str]:
    """core state -> (headline, detail, colour)"""
    if state == "waiting":
        return ("Waiting for SteamVR",
                "Start SteamVR and put your headset on. This connects by itself.", GREY)
    if state == "no-anchor":
        return ("Connected - not locked yet",
                "Get your floor and trackers looking right, then click the big button.", BLUE)
    if state == "paused":
        return ("Paused",
                "Nothing will be fixed. If you're recentering or fixing your floor, do it now, then click "
                "the big button when it looks right - that locks and resumes. Or click Resume.", GREY)
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
                "Normal for a few seconds while Space Calibrator calibrates. If it stays like this, a base "
                "station was moved or SteamVR adjusted its position: get everything looking right, then "
                "lock again. Nothing is being changed meanwhile.", ORANGE)
    if state.startswith("shift-too-large"):
        return ("Playspace is very far off",
                "Too far to fix safely. Get everything looking right, then lock again.", ORANGE)
    if state.startswith("confirm-needed"):
        return ("Playspace moved since your lock",
                "It's quite far from the lock you saved last time. If you changed your floor or room "
                "setup on purpose, click the big button to lock the new one. If not, click "
                "\"Put it back\".", ORANGE)
    if state == "write-refused":
        return ("SteamVR didn't accept the fix",
                "FrameAnchor will try again in a moment. If this keeps happening, check that your room / "
                "playspace is set up in SteamVR.", ORANGE)
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
        self.undo_file = os.path.join(os.path.dirname(os.path.abspath(anchor_file)), fa.UNDO_FILE_NAME)
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
            try:
                if os.path.getsize(LOG_FILE) > LOG_MAX_BYTES:
                    os.replace(LOG_FILE, os.path.splitext(LOG_FILE)[0] + ".old.txt")
            except OSError:
                pass
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d ") + line + "\n")
        except OSError:
            pass

    def status(self, state):
        seen = len(self.core.last_seen) if self.core else 0
        fixes = self.core.corrections if self.core else 0
        last = self.core.last_fix if self.core else ""
        self.out.put(("status", state, fixes, last, seen))

    def set_paused(self, paused: bool):
        self.paused = paused
        if self.core:
            self.core.paused = paused
        self.out.put(("paused", paused))

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
        if saved and saved.zero_pose is not None:
            self.core.undo_pose = saved.zero_pose
        else:
            self.core.undo_pose = fa.load_undo_point(self.undo_file)   # kept after "Remove lock"
        if saved:
            self.core.anchor = saved
            self.log(f"Using your saved lock from {time.strftime('%d %b %H:%M', time.localtime(saved.created))}.")
        elif os.path.exists(self.anchor_file):
            self.log("Your saved lock file is damaged, so it was ignored. Click the big button to lock again.")
        return True

    def handle(self, cmd):
        name = cmd[0]
        if name == "pause":
            self.set_paused(cmd[1])
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
                    if self.core.anchor.zero_pose is not None:
                        try:
                            os.remove(self.undo_file)     # the new lock carries the new undo point
                        except OSError:
                            pass
                except OSError as e:
                    self.log(f"(Couldn't save the lock to disk: {e} - it still works until you close this.)")
                self.log("Check: do those heights match how high your real base stations are above the "
                         "floor? If not, fix your floor first and lock again.")
                if self.paused:
                    self.set_paused(False)
                    self.log("Resumed - watching for jumps again.")
        elif name == "confirm":
            if self.core and self.core.confirm():
                self.log("OK - putting the playspace back to your lock.")
            else:
                self.log("(That click came too late - nothing is waiting for a yes right now.)")
        elif name == "undo":
            if not self.core:
                self.log("Not connected to SteamVR yet - start SteamVR first.")
            else:
                try:
                    if self.core.undo():
                        self.set_paused(True)
                except fa.ChaperoneWriteError as e:
                    self.log(f"!! Undo failed: {e}")
        elif name == "forget":
            # Keep the undo point: "Remove lock" must not take away the way back.
            undo = self.core.undo_pose if self.core else None
            if undo is None:
                saved = fa.Anchor.load(self.anchor_file)
                undo = saved.zero_pose if saved else None
            if undo is not None:
                try:
                    fa.save_undo_point(self.undo_file, undo)
                except OSError:
                    pass
            try:
                os.remove(self.anchor_file)
            except OSError:
                pass
            if self.core:
                self.core.anchor = None
            self.log("Lock removed." + (" \"Undo fixes\" still works." if undo is not None else ""))

    # -- main loop
    def run(self):
        try:
            self._run()
        except Exception:
            # Should never happen, but never die silently: tell the user.
            self.log("FrameAnchor hit an unexpected problem and stopped watching. Close it and start it "
                     "again. Details:\n" + traceback.format_exc())
            self.status("crashed")
        finally:
            self.disconnect()

    def _run(self):
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

            # Push right away when the kind of state changes; otherwise once a second
            # (the numbers inside some states change every tick).
            if fa.state_name(state) != fa.state_name(last_state) or time.monotonic() - last_push > 1.0:
                self.status(state)
                last_push = time.monotonic()
            last_state = state
            self.stop_flag.wait(self.tick)


HELP = ("How to use:  1) Start SteamVR, turn on trackers, let Space Calibrator line everything up.   "
        "2) When your floor and trackers look right, click the big button.   "
        "3) Leave this window open while you play.   "
        "Want to recenter or change your floor on purpose? Click Pause FIRST, make the change, then click "
        "the big button (it locks and resumes).")


class App:
    def __init__(self, tk, worker, events):
        self.tk = tk
        self.worker = worker
        self.events = events
        self.paused = False
        self.state = "waiting"
        self.log_lines = 0
        self.confirm_shown = False

        root = self.root = tk.Tk()
        root.title("FrameAnchor")
        root.geometry("580x620")
        root.minsize(500, 520)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.headline = tk.Label(root, text="Starting...", font=("Segoe UI", 18, "bold"),
                                 fg="white", bg=GREY, pady=12)
        self.headline.pack(fill="x")
        self.detail = tk.Label(root, text="", font=("Segoe UI", 10), wraplength=540, justify="left",
                               anchor="w", padx=12, pady=8)
        self.detail.pack(fill="x")

        self.lock_btn = tk.Button(root, text=LOCK_TEXT,
                                  font=("Segoe UI", 12, "bold"), bg=BLUE, fg="white",
                                  activebackground="#2f59ab", activeforeground="white",
                                  pady=10, command=self.on_lock)
        self.lock_btn.pack(fill="x", padx=12, pady=(4, 8))

        # Only shown when a big correction from a saved lock is waiting for a yes.
        self.confirm_btn = tk.Button(root, text="Put it back to my saved lock",
                                     font=("Segoe UI", 11, "bold"), bg=ORANGE, fg="white",
                                     activebackground="#a85418", activeforeground="white",
                                     pady=6, command=self.on_confirm)

        row = self.btn_row = tk.Frame(root)
        row.pack(fill="x", padx=12)
        self.pause_btn = tk.Button(row, text="Pause", width=12, command=self.on_pause)
        self.pause_btn.pack(side="left")
        self.undo_btn = tk.Button(row, text="Undo fixes", width=12, command=self.on_undo)
        self.undo_btn.pack(side="left", padx=8)
        self.forget_btn = tk.Button(row, text="Remove lock", width=12, command=self.on_forget)
        self.forget_btn.pack(side="left")

        self.dry_var = tk.IntVar(value=0)
        self.dry_chk = tk.Checkbutton(root, text="Test mode (only write down jumps, don't fix them)",
                                      variable=self.dry_var, command=self.on_dry, anchor="w")
        self.dry_chk.pack(fill="x", padx=8, pady=(4, 0))

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

        self.help = tk.Label(root, text=HELP, font=("Segoe UI", 9), wraplength=550, justify="left",
                             anchor="w", padx=12, pady=6, fg="#374151")
        self.help.pack(fill="x")

        self.show_status("waiting", 0, "", 0)
        root.after(100, self.pump)

    # -- buttons
    def on_lock(self):
        self.worker.cmds.put(("lock",))

    def on_confirm(self):
        self.worker.cmds.put(("confirm",))

    def on_pause(self):
        self.set_paused(not self.paused)
        self.worker.cmds.put(("pause", self.paused))

    def on_undo(self):
        if self.tk.messagebox.askyesno(
                "Undo FrameAnchor's fixes?",
                "Put the playspace back to exactly how it was when you last clicked the big Lock button?\n\n"
                "FrameAnchor will pause afterwards so it doesn't fix things again.", parent=self.root):
            self.worker.cmds.put(("undo",))

    def on_forget(self):
        if self.tk.messagebox.askyesno(
                "Remove lock?",
                "Forget the saved lock? FrameAnchor will stop fixing jumps until you lock again.\n\n"
                "(This doesn't undo fixes already made - use \"Undo fixes\" for that.)", parent=self.root):
            self.worker.cmds.put(("forget",))

    def on_dry(self):
        self.worker.cmds.put(("dry", bool(self.dry_var.get())))

    def close(self):
        self.worker.stop_flag.set()
        self.root.destroy()

    # -- updates from the worker
    def set_paused(self, paused):
        self.paused = paused
        self.pause_btn.config(text="Resume" if paused else "Pause")

    def add_log(self, line):
        self.logbox.config(state="normal")
        self.logbox.insert("end", line + "\n")
        self.log_lines += line.count("\n") + 1
        if self.log_lines > LOGBOX_MAX_LINES:
            cut = self.log_lines - LOGBOX_MAX_LINES // 2
            self.logbox.delete("1.0", f"{cut + 1}.0")
            self.log_lines -= cut
        self.logbox.see("end")
        self.logbox.config(state="disabled")

    def show_status(self, state, fixes, last, seen):
        self.state = state
        if state == "crashed":
            head, detail, colour = ("Stopped", "Something went wrong. Close FrameAnchor and start it again. "
                                    "The log below says what happened.", ORANGE)
        else:
            head, detail, colour = friendly(state, bool(self.dry_var.get()))
        self.headline.config(text=head, bg=colour)
        self.detail.config(text=detail)
        locked = state not in ("waiting", "no-anchor")
        self.lock_btn.config(text=RELOCK_TEXT if locked else LOCK_TEXT)
        want_confirm = state.startswith("confirm-needed")
        if want_confirm and not self.confirm_shown:
            self.confirm_btn.pack(fill="x", padx=12, pady=(0, 8), after=self.lock_btn)
            self.confirm_shown = True
        elif not want_confirm and self.confirm_shown:
            self.confirm_btn.pack_forget()
            self.confirm_shown = False
        word = "Jumps seen" if self.dry_var.get() else "Jumps fixed"
        text = f"Base stations visible: {seen}     {word} this session: {fixes}"
        if want_confirm:
            text += "\nDifference from saved lock: " + state[len("confirm-needed "):]
        elif last:
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
                elif ev[0] == "paused":
                    self.set_paused(ev[1])
        except queue.Empty:
            pass
        self.root.after(100, self.pump)


def run_gui() -> int:
    import tkinter as tk
    import tkinter.messagebox  # noqa: F401  (makes tk.messagebox available)
    events: "queue.Queue[tuple]" = queue.Queue()
    worker = Worker(None, fa.ANCHOR_FILE_DEFAULT, events)
    worker.backend_factory = lambda: fa.OpenVRBackend(log=worker.log)
    app = App(tk, worker, events)
    worker.start()
    try:
        app.root.mainloop()
    finally:
        worker.stop_flag.set()
        worker.join(timeout=3)
    return 0


def already_running_notice() -> None:
    msg = ("FrameAnchor is already running.\n\nLook for its window in the taskbar. "
           "(Two copies at once would fight each other, so this one won't start.)")
    print(msg)
    try:
        import tkinter as tk
        import tkinter.messagebox
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showinfo("FrameAnchor", msg, parent=root)
        root.destroy()
    except Exception:
        pass


def main() -> int:
    if not fa.acquire_single_instance():
        already_running_notice()
        return 0
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
