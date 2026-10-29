@echo off
REM EyeSync installer - extract the zip, then double-click this file.
REM Safe to run again, and it is also how EyeSync is UPDATED: run the
REM install.bat of a newer zip and your settings, calibration and API key stay.
REM   1. copies EyeSync to %LocalAppData%\Programs\EyeSync (the extracted
REM      folder can be deleted afterwards)
REM   2. installs Python 3.15 from python.org if no suitable Python is installed
REM   3. creates a private Python environment in .venv and installs the packages
REM   4. checks the face model and runs the self-tests
REM   5. adds "EyeSync" to the Desktop and Start menu, waits for the website
REM      at every login, and opens the EyeSync website

setlocal
cd /d "%~dp0"
set "SITE=https://aniruddh9140.github.io/face-control-website/"
set "DEST=%LocalAppData%\Programs\EyeSync"
echo.
echo  ===== EyeSync setup =====
echo.

if not exist "live_inference.py" (
    echo  The EyeSync files are missing next to install.bat. If you opened the zip
    echo  without extracting it: close this window, right-click the zip, choose
    echo  "Extract All...", then double-click install.bat in the extracted folder.
    pause
    exit /b 1
)

REM An update: stop the EyeSync that is running now. (No ( ) blocks around
REM paths from here on: a user name like "Sam (Work)" would break them.)
if exist "%DEST%\.venv\Scripts\python.exe" "%DEST%\.venv\Scripts\python.exe" "%DEST%\autostart.py" stop >nul 2>nul

if /i "%~dp0"=="%DEST%\" goto in_place
echo [install] Copying EyeSync to "%DEST%" ...
robocopy "%~dp0." "%DEST%" /E /NFL /NDL /NJH /NJS /NP /XD .venv __pycache__ dist /XF eyesync_settings.json eyesync_api_key.txt *.log *.log.* >nul
if errorlevel 8 goto failed
cd /d "%DEST%"
:in_place
REM Files from a downloaded zip are marked "from the internet"; they are ours.
powershell -NoProfile -Command "Get-ChildItem -Recurse -LiteralPath $env:DEST | Unblock-File" >nul 2>nul

call :find_python
if not defined BASE (
    echo [install] Python not found - downloading Python 3.15 from python.org ...
    powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest 'https://www.python.org/ftp/python/3.15.0/python-3.15.0-amd64.exe' -OutFile (Join-Path $env:TEMP 'python-eyesync.exe')"
    if exist "%TEMP%\python-eyesync.exe" (
        echo [install] Installing Python 3.15 - about a minute ...
        "%TEMP%\python-eyesync.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
        del "%TEMP%\python-eyesync.exe" >nul 2>nul
    )
    call :find_python
)
if not defined BASE (
    echo [install] Could not install Python automatically. Install Python 3.15
    echo           from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
    echo           then run install.bat again.
    pause
    exit /b 1
)
echo [install] Using Python: %BASE%

REM An environment whose Python was uninstalled or replaced can't run: rebuild it.
if exist ".venv\Scripts\python.exe" .venv\Scripts\python.exe -c "import sys" >nul 2>nul || rmdir /s /q .venv
if not exist ".venv\Scripts\python.exe" (
    echo [install] Creating the Python environment ...
    %BASE% -m venv .venv || goto failed
)
set "PY=.venv\Scripts\python.exe"

echo [install] Installing packages - a few minutes the first time ...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt || goto failed

echo [install] Checking the face model ...
"%PY%" -c "import face_engine as fe; fe._ensure_face_landmarker_model()" || goto failed

echo [install] Running the self-tests - about a minute ...
"%PY%" -m unittest discover -s tests || goto failed

echo [install] Adding EyeSync to the Desktop and Start menu, and to Windows login ...
"%PY%" autostart.py shortcuts
"%PY%" autostart.py install || goto failed
"%PY%" autostart.py start --standby

echo.
echo  EyeSync is installed in "%DEST%"
echo  (you can delete the downloaded zip and the extracted folder).
echo.
echo  It stays hidden until you use the EyeSync website: the website opens now -
echo  press Connect on its Dashboard. The cursor follows your head; blink to
echo  click. Tip: press Calibrate (30 seconds) to fit EyeSync to you.
echo.
start "" "%SITE%"
pause
exit /b 0

:failed
echo.
echo [install] A step failed - see the messages above, then run install.bat again.
pause
exit /b 1

REM Sets BASE to a working Python, or leaves it undefined. The versions all of
REM EyeSync's packages are published for come first, newest first (3.15 is
REM tested); any other Python 3.10+ only if none of them is installed.
:find_python
set "BASE="
set "CHECK=import sys; sys.exit(sys.version_info < (3, 10))"
for %%V in (3.15 3.14 3.13 3.12) do if not defined BASE py -%%V -c "%CHECK%" >nul 2>nul && set "BASE=py -%%V"
if defined BASE exit /b 0
for %%V in (315 314 313 312) do if not defined BASE if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" set BASE="%LocalAppData%\Programs\Python\Python%%V\python.exe"
if defined BASE exit /b 0
py -3 -c "%CHECK%" >nul 2>nul && (set "BASE=py -3" & exit /b 0)
python -c "%CHECK%" >nul 2>nul && (set "BASE=python" & exit /b 0)
exit /b 0
