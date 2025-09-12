@if "%DEBUG%" == "" @echo off
setlocal

set DEFAULT_JVM_OPTS=-Xmx64m -Xms64m

set DIRNAME=%~dp0
if "%DIRNAME%" == "" set DIRNAME=.
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%

set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar

if not defined JAVA_HOME goto findJavaFromPath
set JAVA_EXE=%JAVA_HOME%\bin\java.exe
if exist "%JAVA_EXE%" goto init
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME%
goto fail

:findJavaFromPath
set JAVA_EXE=java.exe
where %JAVA_EXE% >nul 2>nul
if %ERRORLEVEL% equ 0 goto init
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
goto fail

:init
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
exit /b %ERRORLEVEL%

:fail
exit /b 1
