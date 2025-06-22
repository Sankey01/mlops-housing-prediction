#!/bin/bash

# Script de health check personalizado para el contenedor
# Verifica que la API esté respondiendo correctamente

set -e

# Configuración
API_URL="http://localhost:8000"
TIMEOUT=10
MAX_RETRIES=3

# Función de logging
log() {
    echo "[HEALTHCHECK] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Función para verificar endpoint
check_endpoint() {
    local endpoint="$1"
    local expected_status="$2"

    response=$(curl -s -w "%{http_code}" -o /dev/null --max-time $TIMEOUT "$API_URL$endpoint" 2>/dev/null || echo "000")

    if [ "$response" = "$expected_status" ]; then
        return 0
    else
        log "Endpoint $endpoint falló. Esperado: $expected_status, Recibido: $response"
        return 1
    fi
}

# Función principal de health check
main_health_check() {
    local retry_count=0

    while [ $retry_count -lt $MAX_RETRIES ]; do
        log "Ejecutando health check (intento $((retry_count + 1))/$MAX_RETRIES)..."

        # Verificar endpoint de health
        if check_endpoint "/health" "200"; then
            log "Health check básico: OK"

            # Verificar que el modelo esté cargado
            health_response=$(curl -s --max-time $TIMEOUT "$API_URL/health" 2>/dev/null || echo '{}')

            # Verificar si el modelo está cargado usando grep (más portable que jq)
            if echo "$health_response" | grep -q '"model_loaded".*true'; then
                log "Modelo cargado: OK"

                # Verificar endpoint de información del modelo
                if check_endpoint "/model/info" "200"; then
                    log "Información del modelo: OK"
                    log "Health check completado exitosamente"
                    return 0
                else
                    log "Fallo en endpoint de información del modelo"
                fi
            else
                log "Modelo no está cargado correctamente"
            fi
        else
            log "Fallo en health check básico"
        fi

        retry_count=$((retry_count + 1))

        if [ $retry_count -lt $MAX_RETRIES ]; then
            log "Reintentando en 2 segundos..."
            sleep 2
        fi
    done

    log "Health check falló después de $MAX_RETRIES intentos"
    return 1
}

# Ejecutar health check
main_health_check

exit_code=$?
if [ $exit_code -eq 0 ]; then
    log "Container está saludable"
else
    log "Container no está saludable"
fi

exit $exit_code