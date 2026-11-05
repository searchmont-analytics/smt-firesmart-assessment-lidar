@echo off
rem Double-click to run the FireSmart corridor pipeline (downloads, scores, builds the map).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_corridor.ps1" %*
pause
