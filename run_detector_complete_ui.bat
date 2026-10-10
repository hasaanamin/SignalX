@echo off
cd /d "%~dp0"
py -3.14 -m signal_processing.ui.hdf5_app --smoke-dataset "%~dp0detector_dataset_complete"
if errorlevel 1 pause
