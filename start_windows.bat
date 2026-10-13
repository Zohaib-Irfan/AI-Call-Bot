@echo off
REM ============================================================
REM  Starts the Restaurant Voice Agent on Windows.
REM  Prereqs (one-time): see DEPLOYMENT_GUIDE.txt section 4.
REM    1. Python 3.10-3.12 installed
REM    2. venv created + packages installed
REM    3. Qdrant running on localhost:6333
REM    4. Collection restored (python restore_qdrant.py)
REM ============================================================
cd /d "%~dp0voice_inference_service"

REM Load every KEY=value line from keys.env into the environment
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("keys.env") do set "%%a=%%b"

REM Point BM25 at the restaurant_data folder that ships in this package
set "VIS_BM25_DATA_DIR=%~dp0restaurant_data"

REM Use the venv if it exists next to this script, else system python
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" app.py
) else (
    python app.py
)
pause
