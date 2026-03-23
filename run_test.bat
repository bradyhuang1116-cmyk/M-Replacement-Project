@echo off
cd /d "c:\Users\huang\Downloads\Downloads\Mitsubishi-Electric-Drawing-Replacement-Project"
C:\Users\huang\miniconda3\python.exe run_green_batch_v2.py > diagnostic_output\green_batch_v2_log.txt 2>&1
echo DONE (exit code: %ERRORLEVEL%)
