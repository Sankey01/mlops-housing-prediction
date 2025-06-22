#!/bin/bash

# Script para ejecutar el contenedor Docker
set -e

# Configuración
IMAGE_NAME="housing-prediction-api:latest"
CONTAINER_NAME="housing-api-container"
HOST_PORT="8000"
CONTAINER_PORT="8000"

echo "=============================================="
echo "Ejecutando contenedor Docker"
echo "=============================================="

# Función de logging
log() {
    echo "[RUN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Verificar que la imagen existe
if ! docker images -q "$IMAGE_NAME" | grep -q .; then
    log "ERROR: Imagen $IMAGE_NAME no encontrada"
    log "Construye la imagen primero:"
    log "  ./docker/build.sh"
    exit 1
fi

log "Imagen encontrada: $IMAGE_NAME"

# Detener y remover contenedor existente si existe
if docker ps -a --format "table {{.Names}}" | grep -q "^$CONTAINER_NAME$"; then
    log "Deteniendo contenedor existente..."
    docker stop "$CONTAINER_NAME" || true
    log "Removiendo contenedor existente..."
    docker rm "$CONTAINER_NAME" || true
fi

# Crear directorio de logs local si no existe
if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    log "Directorio de logs creado: ./logs"
fi

log "Iniciando nuevo contenedor..."

# Ejecutar contenedor con configuración optimizada
docker run \
    --name "$CONTAINER_NAME" \
    --detach \
    --publish "$HOST_PORT:$CONTAINER_PORT" \
    --volume "$(pwd)/logs:/app/logs" \
    --volume "$(pwd)/models:/app/models:ro" \
    --restart unless-stopped \
    --health-cmd "./healthcheck.sh" \
    --health-interval 30s \
    --health-timeout 10s \
    --health-retries 3 \
    --health-start-period 40s \
    "$IMAGE_NAME"

if [ $? -eq 0 ]; then
    log "Contenedor iniciado exitosamente: $CONTAINER_NAME"
else
    log "ERROR: Fallo al iniciar contenedor"
    exit 1
fi

# Esperar a que el contenedor esté saludable
log "Esperando a que el contenedor esté saludable..."
sleep 5

# Verificar estado del contenedor
for i in {1..12}; do
    status=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "unknown")

    case $status in
        "healthy")
            log "Contenedor está saludable!"
            break
            ;;
        "unhealthy")
            log "ERROR: Contenedor no está saludable"
            log "Logs del contenedor:"
            docker logs "$CONTAINER_NAME" --tail 20
            exit 1
            ;;
        "starting"|"unknown")
            log "Esperando health check... ($i/12)"
            sleep 5
            ;;
    esac

    if [ $i -eq 12 ]; then
        log "WARNING: Timeout esperando health check"
        log "El contenedor puede estar iniciando aún"
    fi
done

# Mostrar información del contenedor
log "Estado del contenedor:"
docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.CreatedAt}}"

# Mostrar logs iniciales
log "Logs iniciales del contenedor:"
docker logs "$CONTAINER_NAME" --tail 10

echo ""
echo "=============================================="
log "Contenedor ejecutándose exitosamente!"
echo "=============================================="
echo ""
echo "API disponible en: http://localhost:$HOST_PORT"
echo "Documentación: http://localhost:$HOST_PORT/docs"
echo "Health check: http://localhost:$HOST_PORT/health"
echo ""
echo "Comandos útiles:"
echo "  Ver logs en tiempo real: docker logs -f $CONTAINER_NAME"
echo "  Detener contenedor: docker stop $CONTAINER_NAME"
echo "  Remover contenedor: docker rm $CONTAINER_NAME"
echo "  Acceso shell: docker exec -it $CONTAINER_NAME /bin/bash"
echo ""

# Prueba rápida de conectividad
log "Probando conectividad..."
sleep 2

if curl -s -f "http://localhost:$HOST_PORT/health" > /dev/null; then
    log "API respondiendo correctamente!"
else
    log "WARNING: API no responde aún, puede necesitar más tiempo para inicializar"
fi