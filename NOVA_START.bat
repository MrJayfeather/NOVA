@echo off
chcp 65001 >nul
title NOVA
cd /d "%~dp0"
set PYTHONUTF8=1
rem локальная копия памяти NOVA — всегда свежая (третья, после бокса и GitHub)
if exist nova-memory\ (
  pushd nova-memory
  git pull --quiet
  popd
)
uv run python scripts/start_all.py
pause
