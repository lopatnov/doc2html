@echo off
echo Creating conda environment "pdf2html" (python 3.11)...
call conda create -y -n pdf2html python=3.11
if errorlevel 1 goto :error

echo Installing dependencies...
call conda run -n pdf2html python -m pip install --upgrade pip
if errorlevel 1 goto :error
call conda run -n pdf2html python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Done. Use activate.bat to activate the environment.
goto :eof

:error
echo Installation failed.
exit /b 1
