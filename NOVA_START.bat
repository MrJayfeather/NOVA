@echo off
chcp 65001 >nul
title NOVA
cd /d "%~dp0"
set PYTHONUTF8=1
uv run python scripts/start_all.py
pause
