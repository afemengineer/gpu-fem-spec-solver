@echo off
setlocal

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0configure_windows_cuda.ps1"
exit /b %errorlevel%
