@echo off
REM EyeSync uninstaller: stops EyeSync, removes it from Windows login and
REM removes the Desktop / Start menu shortcuts. Then, if you agree, deletes
REM this folder - including your settings, calibration and API key.

setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo.
echo  ===== Remove EyeSync =====
echo.
"%PY%" autostart.py stop
"%PY%" autostart.py uninstall
echo.
choice /m "Also delete the EyeSync folder (%~dp0) with your settings and calibration"
if errorlevel 2 (
    echo EyeSync no longer starts. The folder was kept.
    pause
    exit /b 0
)
echo EyeSync is removed. This window closes and the folder is deleted.
timeout /t 3 >nul
cd /d "%TEMP%"
start "" /min cmd /c "timeout /t 2 /nobreak >nul & rmdir /s /q "%~dp0.""
exit /b 0
