@echo off
chcp 65001 >nul
title NOVA — пробуждение сервера
cd /d "%~dp0"
set PYTHONUTF8=1
echo Бужу видеокарту (клиент НЕ открываю)...
uv run python scripts/vast.py up
echo.
echo Сервер поднят. Модели греются ещё пару минут — потом жми NOVA_CLIENT.
pause
