@echo off
chcp 65001 >nul
title NOVA — выключение
cd /d "%~dp0"
set PYTHONUTF8=1
echo Выключаю сервер NOVA (диск и модели сохраняются)...
uv run python scripts/vast.py down
pause
