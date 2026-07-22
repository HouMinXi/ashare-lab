@echo off
setlocal EnableDelayedExpansion
set NSSM=H:\nssm\nssm.exe
if not exist "%NSSM%" (echo ERROR: nssm.exe not found at %NSSM% & exit /b 1)
set TARGET=%1
if "%TARGET%"=="" (echo Usage: gpu-switch.bat ollama^|llama-server^|training^|training-3080 & exit /b 1)

:: Lock file (Finding 3: prevent concurrent execution)
set LOCKDIR=%TEMP%\gpu-switch.lock
set LOCKFILE=%LOCKDIR%\pid
if exist "%LOCKDIR%" (
    set STALE=0
    for /f "tokens=1" %%p in ('type "%LOCKFILE%" 2^>nul') do (
        tasklist /fi "PID eq %%p" 2>nul | findstr /c:"%%p" >nul 2>&1
        if errorlevel 1 set STALE=1
    )
    if "!STALE!"=="0" (
        powershell -NoProfile -Command "if ((Get-Date)-(Get-Item '%LOCKDIR%').CreationTime).TotalMinutes -ge 5 { exit 0 } else { exit 1 }" >nul 2>&1
        if not errorlevel 1 set STALE=1
    )
    if "!STALE!"=="0" (
        echo ERROR: gpu-switch.bat already running ^(PID in %LOCKFILE%^) & exit /b 1
    )
    rd /s /q "%LOCKDIR%" 2>nul
)
mkdir "%LOCKDIR%"
for /f %%i in ('wmic process where "name='cmd.exe'" get ProcessId /value 2^>nul ^| findstr "ProcessId"') do set "%%i"
echo %ProcessId%>"%LOCKFILE%"

if not exist "H:\ollama\logs" mkdir "H:\ollama\logs"
:: Locale-independent ISO 8601 timestamp (Finding 1)
set "ts="
for /f "usebackq delims=" %%a in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-ddTHH:mm:ss.fffzzz'" ^| findstr /r "[0-9]"`) do set "ts=%%a"
echo [%ts%] gpu-switch: %TARGET% >> H:\ollama\logs\gpu-switch.log

if "%TARGET%"=="ollama" goto :ollama
if "%TARGET%"=="llama-server" goto :llama-server
if "%TARGET%"=="training" goto :training
if "%TARGET%"=="training-3080" goto :training-3080
echo Unknown target: %TARGET%
call :cleanup_lock
exit /b 1

:ollama
:: Finding 3: stop must succeed before start
%NSSM% stop llama-server >nul 2>&1
if errorlevel 1 (echo ERROR: failed to stop llama-server & call :cleanup_lock & exit /b 1)
%NSSM% status OllamaService 2>nul | findstr /I "SERVICE_RUNNING" >nul 2>&1
if not errorlevel 1 (echo OllamaService already running, skipping start & call :cleanup_lock & exit /b 0)
%NSSM% start OllamaService
if errorlevel 1 (echo ERROR: failed to start OllamaService & call :cleanup_lock & exit /b 1)
call :cleanup_lock
exit /b 0

:llama-server
:: Finding 3: stop must succeed before start
%NSSM% stop OllamaService >nul 2>&1
if errorlevel 1 (echo ERROR: failed to stop OllamaService & call :cleanup_lock & exit /b 1)
%NSSM% status llama-server 2>nul | findstr /I "SERVICE_RUNNING" >nul 2>&1
if not errorlevel 1 (echo llama-server already running, skipping start & call :cleanup_lock & exit /b 0)
%NSSM% start llama-server
if errorlevel 1 (echo ERROR: failed to start llama-server & call :cleanup_lock & exit /b 1)
call :cleanup_lock
exit /b 0

:training
:: Finding 3: stop must succeed before clearing GPU
%NSSM% stop OllamaService >nul 2>&1
if errorlevel 1 (echo ERROR: failed to stop OllamaService & call :cleanup_lock & exit /b 1)
echo GPU 1 cleared for training (llama-server on GPU 0 untouched)
call :cleanup_lock
exit /b 0

:training-3080
:: Finding 3: stop must succeed before clearing GPU
%NSSM% stop OllamaService >nul 2>&1
if errorlevel 1 (echo ERROR: failed to stop OllamaService & call :cleanup_lock & exit /b 1)
%NSSM% stop llama-server >nul 2>&1
if errorlevel 1 (echo ERROR: failed to stop llama-server & call :cleanup_lock & exit /b 1)
echo GPU 0 cleared for training-3080
call :cleanup_lock
exit /b 0

:cleanup_lock
rd /s /q "%LOCKDIR%" 2>nul
goto :eof
