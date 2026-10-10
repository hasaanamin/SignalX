@echo off
setlocal
cd /d "%~dp0"
py -3.14 -m signal_processing.ui.hdf5_app --dataset "%~dp0subset_test.h5"
if errorlevel 1 (
  echo.
  echo The UI exited with an error. Verify Python 3.14 and the optional UI dependencies.
  echo Install them with: py -3.14 -m pip install -e ".[ui]"
  pause
)
