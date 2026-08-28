@echo off
title Clonador de Voz Moderno (Flow Matching DiT) - PyQt GUI
cd /d "%~dp0"

echo ========================================================
echo   Iniciando o Clonador de Voz Moderno (PyQt Desktop GUI)
echo ========================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual 'venv' nao encontrado!
    echo Por favor, certifique-se de que o venv foi criado corretamente.
    echo.
    pause
    exit /b 1
)

echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo Abrindo aplicativo Desktop PyQt6...
venv\Scripts\python.exe gui_pyqt.py

if errorlevel 1 (
    echo.
    echo [AVISO] O aplicativo foi encerrado com erro.
    pause
)
