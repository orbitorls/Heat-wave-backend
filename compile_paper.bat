@echo off
cd /d "%~dp0"
set PDFLATEX=C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe

echo Compiling Paper.tex (pass 1/2)...
"%PDFLATEX%" -interaction=nonstopmode Paper.tex
if errorlevel 1 goto error

echo Compiling Paper.tex (pass 2/2)...
"%PDFLATEX%" -interaction=nonstopmode Paper.tex
if errorlevel 1 goto error

echo.
echo Done! Paper.pdf updated.
goto end

:error
echo.
echo ERROR: Compilation failed. Check Paper.log for details.
pause
exit /b 1

:end
