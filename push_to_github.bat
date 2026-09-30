@echo off
title Push to GitHub with Git LFS
powershell -ExecutionPolicy Bypass -File "%~dp0push_to_github.ps1"
pause
