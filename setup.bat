@echo off
cd /d "%~dp0"
where py >nul 2>nul || (echo Install Python 3.11 first.& pause & exit /b 1)
where node >nul 2>nul || (echo Install Node.js 20+ first.& pause & exit /b 1)
if not exist backend\.venv py -3.11 -m venv backend\.venv
call backend\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt || (echo Backend install failed.& pause & exit /b 1)
cd frontend
npm install || (echo Frontend install failed.& pause & exit /b 1)
cd ..
if not exist .env copy .env.example .env
echo Setup complete. Run start_signbridge.bat
pause
