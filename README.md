# FrameAnchor

Keeps a **Steam Frame + Lighthouse (Vive tracker)** playspace locked to the real room, so the floor and playspace stop jumping when the Frame relocalizes.

> **Experimental.** It works on at least one real Steam Frame setup, but it's young. It changes SteamVR's saved standing position. It backs that up first and has an Undo button.

## Download

Get **FrameAnchor.zip** from the [Releases](../../releases) page, extract it, and double-click **Start FrameAnchor**. The included `READ ME FIRST.txt` has the full user guide. The first start needs internet, about a minute, to set up Python. After that it works offline.

## How it works

- With Space Calibrator lining the Lighthouse devices up with the headset, the base stations are a fixed reference for "where the real room is".
- **Lock:** when you lock, FrameAnchor records where the base stations appear in SteamVR's standing space.
- **Jump:** if they all shift together (a relocalization), it moves SteamVR's standing origin back using `IVRChaperoneSetup`.
- **Something else:** if they move relative to each other (a bumped base station, or SteamVR re-solving one), it refuses to "fix" anything.
- **Moves made on purpose:** moves with **OVR Advanced Settings** are kept, not undone. That covers playspace drags, snap and smooth turns, and reset-to-origin binds.
  - FrameAnchor reads SteamVR's live standing origin read-only, from the base stations' raw vs standing poses.
  - A change spread over several ticks (a drag), a clean turn around the headset, or a return to a position the playspace rested at before is kept.
  - A sudden change to somewhere new is treated as a jump.
  - If OVR Advanced Settings drops an earlier fix when a drag starts (it remembers an older origin), that fix is put back afterwards.
- **Safety:**
  - A one-time backup of `chaperone_info.vrchap`.
  - An undo point saved with every lock.
  - It asks before applying a big correction from a lock saved in an earlier session.
  - Only one copy runs at a time.
  - It refuses to write any non-rigid or NaN pose.

## Layout

| Path | What |
|---|---|
| `FrameAnchor/app/frame_anchor.py` | Core logic, OpenVR backend, text-mode fallback |
| `FrameAnchor/app/frame_anchor_gui.py` | tkinter window |
| `FrameAnchor/Start FrameAnchor.bat` | Launcher. Runs the app with the bundled `uv`, offline from the bundled wheels |
| `FrameAnchor/READ ME FIRST.txt` | End-user guide (ships in the zip) |
| `tests/` | pytest suite against a simulated SteamVR, including a simulated OVR Advanced Settings |
| `build_release.ps1` | Builds `dist/FrameAnchor.zip` |

The bundled binaries (`bin/uv.exe`, `wheels/*.whl`) aren't in git. `build_release.ps1` downloads uv 0.13.8 from Astral's GitHub release and numpy/openvr from PyPI, and checks each against a pinned SHA256.

## Develop

```powershell
# build the release zip (downloads + verifies binaries, runs the tests, then packages)
powershell -ExecutionPolicy Bypass -File build_release.ps1
```

To run only the tests, put `FrameAnchor/app/*.py` and `tests/*.py` in one folder and run `pytest` with `numpy` and `openvr` installed.

## License

MIT. See [LICENSE](LICENSE). Unofficial; not affiliated with Valve.
