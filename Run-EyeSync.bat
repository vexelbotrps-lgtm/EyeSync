@echo off
setlocal
set EYESYNC_STANDALONE=1
cd /d "%~dp0EyeSync"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Python 3.15.2 environment is missing. See BRAHMASTRA_README.md.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:3] == (3, 15, 2), sys.version"
if errorlevel 1 (
  echo EyeSync requires exactly Python 3.15.2.
  pause
  exit /b 1
)
start "EyeSync" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0EyeSync\live_inference.py" --always-active --mouse software --stall-timeout 45 --site-origin http://localhost:8000 --site-origin https://vexelbotrps-lgtm.github.io
