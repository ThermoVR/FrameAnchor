@echo off
title FrameAnchor
set "HERE=%~dp0"
rem Ignore Python settings left behind by other software - they break the bundled Python.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
if not exist "%HERE%bin\uv.exe" goto notextracted
if not exist "%HERE%app\frame_anchor_gui.py" goto notextracted
echo.
echo   Starting FrameAnchor...
echo.
echo   The very first start sets itself up. That needs internet and takes
echo   about a minute. After that it starts right away.
echo.
echo   Keep this black window open while you play.
echo   Closing it closes FrameAnchor.
echo.
"%HERE%bin\uv.exe" run --no-project --managed-python --python 3.12 --no-index --find-links "%HERE%wheels" --with openvr --with numpy "%HERE%app\frame_anchor_gui.py"
if errorlevel 1 goto failed
exit /b 0

:notextracted
echo.
echo   FrameAnchor can't find its files.
echo.
echo   You probably opened it from inside the ZIP. Do this instead:
echo     1. Right-click FrameAnchor.zip and choose "Extract All..."
echo     2. Open the new FrameAnchor folder
echo     3. Double-click "Start FrameAnchor" again
echo.
pause
exit /b 1

:failed
echo.
echo   FrameAnchor stopped with an error.
echo.
echo   - If this was the first start, check your internet connection and try again.
echo   - If it keeps happening, take a screenshot of this window and send it
echo     to whoever gave you FrameAnchor.
echo.
pause
exit /b 1
