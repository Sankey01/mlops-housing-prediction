#!/bin/bash
set -e

# Script de inicio para contenedor Docker
# Configurado para producción con logging y validaciones

echo "=========================================="
echo "Boston Housing Price Prediction API"
echo "Iniciando contenedor Docker..."
echo "=========================================="

# Función para logging estructurado
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Verificando configuración del contenedor..."

# Verificar variables de entorno críticas
required_vars=("MODEL_PATH" "METADATA_PATH" "LOG_FILE" "HOST" "PORT")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        log "ERROR: Variable de entorno $var no está definida"
        exit 1
    fi
done

log "Variables de entorno verificadas correctamente"

# Verificar que el modelo existe
if [ ! -f "$MODEL_PATH" ]; then
    log "ERROR: Modelo no encontrado en $MODEL_PATH"
    log "Contenido del directorio models:"
    ls -la /app/models/ || echo "Directorio models no existe"
    exit 1
fi

log "Modelo encontrado: $MODEL_PATH"

# Verificar metadatos (opcional pero recomendado)
if [ -f "$METADATA_PATH" ]; then
    log "Metadatos encontrados: $METADATA_PATH"
else
    log "WARNING: Metadatos no encontrados en $METADATA_PATH"
fi

# Crear directorio de logs si no existe
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    log "Directorio de logs creado: $LOG_DIR"
fi

# Verificar permisos de escritura
if [ ! -w "$LOG_DIR" ]; then
    log "ERROR: No se puede escribir en directorio de logs: $LOG_DIR"
    exit 1
fi

log "Directorio de logs verificado: $LOG_DIR"

# Mostrar configuración
log "Configuración del servidor:"
log "  Host: $HOST"
log "  Puerto: $PORT"
log "  Workers: $WORKERS"
log "  Log Level: $LOG_LEVEL"
log "  Reload: $RELOAD"
log "  Environment: $ENVIRONMENT"

# Verificar conectividad interna (preparar para health checks)
log "Preparando health checks..."

# Configurar variables de entorno para uvicorn
export PYTHONUNBUFFERED=1

# Función para manejo de señales
cleanup() {
    log "Recibida señal de terminación, cerrando aplicación..."
    exit 0
}

# Configurar manejo de señales
trap cleanup SIGTERM SIGINT

log "Iniciando servidor Uvicorn..."
log "=========================================="

# Ejecutar servidor con configuración básica (sin uvloop)
exec uvicorn src.api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --no-use-colors \
    --access-log