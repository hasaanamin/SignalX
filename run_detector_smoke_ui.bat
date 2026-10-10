@echo off
setlocal
cd /d "%~dp0"
py -3.14 -m signal_processing.ui.hdf5_app --smoke-dataset "%~dp0detector_smoke314"
if errorlevel 1 (
  echo.
  echo Detector test bench exited with an error.
  pause
)
