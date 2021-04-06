@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

if "%1" == "help" (
	:help
	echo.Please use `make ^<target^>` where ^<target^> is one of
	echo.  clean             to clear all built documentation files
	echo.  html              to make all standalone HTML files

	goto end
)

if "%1" == "clean" (
	rmdir %BUILDDIR% /s /q
	goto end
)

if "%1" == "html" (
	set THEME=sphinx_rtd_theme
	sphinx-multiversion %SOURCEDIR% %BUILDDIR%\html

	set THEME=pydata_sphinx_theme
	sphinx-multiversion %SOURCEDIR% %BUILDDIR%\html
	%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

	if errorlevel 1 exit /b 1
	echo.
	echo.Build finished. The HTML pages are in %BUILDDIR%/html.
	goto end
)

:end
