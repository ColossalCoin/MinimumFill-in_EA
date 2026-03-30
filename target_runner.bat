@echo off
set CONFIG_ID=%1
set INSTANCE_ID=%2
set SEED=%3
set INSTANCE=%4

:: Desplazamos los primeros 4 argumentos que son fijos de irace
shift
shift
shift
shift

:: Recopilamos todos los hiperparámetros restantes
set "REST_ARGS="
:loop
if "%~1"=="" goto run
set "REST_ARGS=%REST_ARGS% %1"
shift
goto loop

:run
"C:\Users\a-b-e\anaconda3\envs\thesis_env\python.exe" irace_wrapper.py --inst %INSTANCE% --seed %SEED% %REST_ARGS%

:: Capturamos errores silenciosamente para que irace no se confunda
if %ERRORLEVEL% NEQ 0 (
    echo inf
)