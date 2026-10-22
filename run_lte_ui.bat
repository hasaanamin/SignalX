@echo off
setlocal
cd /d "%~dp0"
echo Launching SignalX LTE Signal Processing Studio UI...
py -3.14 -m signal_processing.ui.lte_app %*
if errorlevel 1 (
  echo.
  echo The UI exited with an error. Verify Python 3.14, PyQt6 and pyqtgraph.
  echo Install them with: py -3.14 -m pip install PyQt6 pyqtgraph
  pause
)
