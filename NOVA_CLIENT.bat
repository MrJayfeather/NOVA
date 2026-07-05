@echo off
chcp 65001 >nul
title NOVA — клиент
cd /d "%~dp0"
set PYTHONUTF8=1
echo [nova] starting client (server must be up)...
uv run python -m nova.client.main
pause
