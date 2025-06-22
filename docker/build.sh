#!/bin/bash

# Script para construir imagen Docker de la API
set -e

# Configuración
IMAGE_NAME="housing-prediction-api"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "=============================================="
echo "Construyendo imagen Docker: $FULL_IMAGE_NAME"
echo "=============================================="

# Función de logging
log() {
    echo "[BUILD] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Verificar prerrequisitos
log "Verificando prerrequisitos..."

# Verificar que Docker esté instalado
if ! command -v docker &> /dev/null; then
    log "ERROR: Docker no está instalado"
    exit 1
fi

log "Docker encontrado: $(docker --version)"

# Verificar que el modelo existe
if [ ! -f "models/best_model.pkl" ]; then
    log "ERROR: Modelo no encontrado en models/best_model.pkl"
    log "Ejecuta primero el pipeline de entrenamiento:"
    log "  python src/train.py"
    exit 1
fi

log "Modelo encontrado: models/best_model.pkl"

# Verificar que el archivo .env.docker existe
if [ ! -f ".env.docker" ]; then
    log "ERROR: Archivo .env.docker no encontrado"
    log "Este archivo es necesario para la configuración del contenedor"
    exit 1
fi

log "Configuración Docker encontrada: .env.docker"

# Crear directorio docker si no existe
if [ ! -d "docker" ]; then
    mkdir -p docker
    log "Directorio docker creado"
fi

# Limpiar imágenes anteriores (opcional)
if docker images -q "$FULL_IMAGE_NAME" | grep -q .; then
    log "Removiendo imagen anterior..."
    docker rmi "$FULL_IMAGE_NAME" || true
fi

# Construir imagen
log "Iniciando construcción de imagen..."
log "Esto puede tomar varios minutos..."

# Build con cache optimizado y multi-stage
if docker build \
    --tag "$FULL_IMAGE_NAME" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VERSION="1.0.0" \
    .; then

    log "Imagen construida exitosamente: $FULL_IMAGE_NAME"
else
    log "ERROR: Fallo en la construcción de la imagen"
    exit 1
fi

# Verificar imagen
log "Verificando imagen construida..."
if docker images "$FULL_IMAGE_NAME" | grep -q "$IMAGE_NAME"; then
    image_size=$(docker images "$FULL_IMAGE_NAME" --format "table {{.Size}}" | tail -n 1)
    log "Imagen verificada correctamente"
    log "Tamaño de imagen: $image_size"
else
    log "ERROR: No se puede verificar la imagen construida"
    exit 1
fi

# Mostrar información de la imagen
log "Detalles de la imagen:"
docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"

# Instrucciones de uso
echo ""
echo "=============================================="
log "Imagen construida exitosamente!"
echo "=============================================="
echo ""
echo "Para ejecutar el contenedor:"
echo "  docker run -p 8000:8000 $FULL_IMAGE_NAME"
echo ""
echo "Para ejecutar con Docker Compose:"
echo "  docker-compose up"
echo ""
echo "Para probar la API:"
echo "  curl http://localhost:8000/health"
echo ""
echo "Para acceder a la documentación:"
echo "  http://localhost:8000/docs"
echo ""

log "Script de construcción completado exitosamente"