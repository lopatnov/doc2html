@echo off
echo Creating conda environment "pdf2html" (python 3.11)...
call conda create -y -n pdf2html python=3.11
if errorlevel 1 goto :error

echo Installing dependencies...
call conda run -n pdf2html python -m pip install --upgrade pip
if errorlevel 1 goto :error
call conda run -n pdf2html python -m pip install -r requirements.txt
if errorlevel 1 goto :error

if /I "%~1"=="--with-blip" (
    echo Installing optional BLIP captioning backend ^(requirements-blip.txt^)...
    call conda run -n pdf2html python -m pip install -r requirements-blip.txt
    if errorlevel 1 goto :error
)

echo.
echo Done. Use activate.bat to activate the environment.
if /I not "%~1"=="--with-blip" (
    echo.
    echo Note: the offline BLIP captioning backend ^(--caption-backend blip^) is NOT
    echo installed by default. Either use --caption-backend lmstudio, or install it
    echo later with:
    echo   conda run -n pdf2html python -m pip install -r requirements-blip.txt
    echo Or rerun this script as: install.bat --with-blip
)
goto :eof

:error
echo Installation failed.
exit /b 1
