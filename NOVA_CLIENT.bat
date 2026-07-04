@echo off
chcp 65001 >nul
title NOVA — клиент
cd /d "%~dp0"
set PYTHONUTF8=1
echo Открываю клиента (сервер должен уже работать)...
uv run python -m nova.client.main
pause
