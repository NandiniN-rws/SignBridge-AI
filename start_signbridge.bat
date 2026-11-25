@echo off
cd /d "%~dp0"
if not exist backend\.venv\Scripts\activate.bat (echo Run setup.bat first.& pause & exit /b 1)
where node >nul 2>nul || (echo Node.js missing.& pause & exit /b 1)
start "SignBridge API" cmd /k "cd /d ""%~dp0backend"" && .venv\Scripts\activate && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
timeout /t 3 /nobreak >nul
start "SignBridge Web" cmd /k "cd /d ""%~dp0frontend"" && npm run dev"
timeout /t 2 /nobreak >nul
start http://localhost:5173
