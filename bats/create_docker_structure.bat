@echo off
REM Script para crear la estructura de directorios Docker en Windows

echo ==========================================
echo Creando estructura de directorios Docker
echo ==========================================

REM Cambiar al directorio del proyecto
cd /d "C:\Users\Kenny\PycharmProjects\mlops-housing-prediction"

REM Crear directorio docker si no existe
if not exist "docker" (
    mkdir docker
    echo Directorio docker creado
) else (
    echo Directorio docker ya existe
)

REM Crear subdirectorios adicionales si es necesario
if not exist "logs" (
    mkdir logs
    echo Directorio logs creado
) else (
    echo Directorio logs ya existe
)

echo.
echo Estructura de directorios creada:
echo.
tree /F

echo.
echo ==========================================
echo Ahora copia los archivos en estas rutas:
echo ==========================================
echo.
echo Raiz del proyecto:
echo   - docker-compose.yml
echo   - Makefile
echo   - .env.docker
echo.
echo Directorio docker/:
echo   - docker/build.sh
echo   - docker/run.sh
echo   - docker/start.sh
echo   - docker/healthcheck.sh
echo   - docker/nginx.conf
echo   - docker/prometheus.yml
echo   - docker/README.md
echo.
echo ==========================================

pause