$installedExe = Join-Path $env:LOCALAPPDATA 'Programs\EyeSync\EyeSync.exe'
$installed = Get-CimInstance Win32_Process -Filter "Name = 'EyeSync.exe'" |
    Where-Object { $_.ExecutablePath -eq $installedExe }
foreach ($process in $installed) {
    Stop-Process -Id $process.ProcessId -ErrorAction Stop
}
