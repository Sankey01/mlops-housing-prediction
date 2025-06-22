@echo off
REM Comandos Docker para Windows
REM Ubicar en: C:\Users\Kenny\PycharmProjects\mlops-housing-prediction\docker\windows-commands.bat

echo ========================================
echo Docker Commands para Windows
echo ========================================
echo.
echo 1. Construir imagen
echo 2. Ejecutar con Docker Compose
echo 3. Ver logs
echo 4. Detener servicios
echo 5. Limpiar todo
echo 6. Probar API
echo 0. Salir
echo.

set /p choice="Selecciona una opcion (0-6): "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto run
if "%choice%"=="3" goto logs
if "%choice%"=="4" goto stop
if "%choice%"=="5" goto clean
if "%choice%"=="6" goto test
if "%choice%"=="0" goto end

:build
echo Construyendo imagen Docker...
cd ..
docker build -t housing-prediction-api:latest .
if errorlevel 1 (
    echo ERROR: Fallo en construccion
) else (
    echo Imagen construida exitosamente!
)
pause
goto menu

:run
echo Ejecutando servicios con Docker Compose...
cd ..
docker-compose up -d
echo Servicios iniciados!
echo API disponible en: http://localhost:8000
echo Documentacion en: http://localhost:8000/docs
pause
goto menu

:logs
echo Mostrando logs...
cd ..
docker-compose logs housing-api
pause
goto menu

:stop
echo Deteniendo servicios...
cd ..
docker-compose down
echo Servicios detenidos
pause
goto menu

:clean
echo Limpiando contenedores e imagenes...
cd ..
docker-compose down -v --rmi all
docker system prune -f
echo Limpieza completada
pause
goto menu

:test
echo Probando API...
timeout 2 >nul
curl -s http://localhost:8000/health
if errorlevel 1 (
    echo API no responde. Asegurate de que este ejecutandose.
) else (
    echo API respondiendo correctamente!
)
pause
goto menu

:menu
cls
goto start

:end
echo Saliendo...
exit /b 0

:start
goto menu