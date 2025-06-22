# Docker Containerization Guide

## Boston Housing Price Prediction API

Este directorio contiene todos los archivos necesarios para containerizar y desplegar la API de predicción de precios de viviendas.

## Estructura de Archivos

```
docker/
├── README.md              # Esta guía
├── build.sh              # Script de construcción
├── run.sh                # Script de ejecución
├── start.sh              # Script de inicio del contenedor
├── healthcheck.sh        # Health check personalizado
├── nginx.conf            # Configuración de Nginx
└── prometheus.yml        # Configuración de Prometheus
```

## Prerrequisitos

- Docker instalado (versión 20.10+)
- Docker Compose instalado (versión 1.29+)
- Modelo entrenado en `models/best_model.pkl`

## Métodos de Ejecución

### 1. Usando Makefile (Recomendado)

```bash
# Ver comandos disponibles
make help

# Flujo completo de desarrollo
make dev

# Deploy local con todas las funcionalidades
make deploy-local

# Deploy con monitoreo
make deploy-monitoring
```

### 2. Scripts Individuales

```bash
# Construir imagen
chmod +x docker/build.sh
./docker/build.sh

# Ejecutar contenedor
chmod +x docker/run.sh
./docker/run.sh
```

### 3. Docker Compose

```bash
# Básico
docker-compose up -d

# Con rebuild
docker-compose up -d --build

# Con monitoreo (Prometheus + Grafana)
docker-compose --profile monitoring up -d
```

### 4. Docker Manual

```bash
# Construir
docker build -t housing-prediction-api:latest .

# Ejecutar
docker run -d \
  --name housing-api-container \
  -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models:ro \
  housing-prediction-api:latest
```

## Configuración

### Variables de Entorno

La configuración se maneja a través de `.env.docker`:

```bash
# API Configuration
APP_NAME=Boston-Housing-API-KS
APP_VERSION=1.0.0
HOST=0.0.0.0
PORT=8000
WORKERS=1
LOG_LEVEL=info

# Paths (dentro del contenedor)
MODEL_PATH=/app/models/best_model.pkl
METADATA_PATH=/app/models/model_metadata.json
LOG_FILE=/app/logs/api.log
```

### Volúmenes

- `/app/logs` - Logs de la aplicación
- `/app/models` - Modelos de ML (read-only)

### Puertos

- `8000` - API principal
- `80` - Nginx (si se usa)
- `9090` - Prometheus (con perfil monitoring)
- `3000` - Grafana (con perfil monitoring)

## Health Checks

El contenedor incluye health checks personalizados:

```bash
# Verificar salud del contenedor
docker exec housing-api-container ./healthcheck.sh

# Ver estado de health
docker inspect --format='{{.State.Health.Status}}' housing-api-container
```

## Monitoreo

### Con Prometheus y Grafana

```bash
# Iniciar con monitoreo
docker-compose --profile monitoring up -d

# Acceder a interfaces
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)
```

### Logs

```bash
# Logs en tiempo real
docker logs -f housing-api-container

# Logs de todos los servicios
docker-compose logs -f

# Logs específicos
docker-compose logs -f housing-api
```

## Optimizaciones de Producción

### Multi-stage Build

El Dockerfile usa multi-stage build para:
- Reducir tamaño de imagen final
- Separar dependencias de build de runtime
- Mejorar seguridad

### Seguridad

- Usuario no-root (`apiuser`)
- Imagen base minimal (`python:3.9-slim`)
- Health checks robustos
- Nginx con headers de seguridad

### Performance

- Uvicorn con uvloop y httptools
- Nginx con compresión y caching
- Rate limiting configurado
- Keep-alive connections

## Troubleshooting

### Problemas Comunes

**1. Modelo no encontrado**
```bash
# Verificar que existe
ls -la models/best_model.pkl

# Si no existe, entrenar modelo
python src/train.py
```

**2. Puerto ya en uso**
```bash
# Cambiar puerto en docker-compose.yml o usar:
docker run -p 8001:8000 housing-prediction-api:latest
```

**3. Contenedor no healthy**
```bash
# Ver logs detallados
docker logs housing-api-container

# Ejecutar health check manualmente
docker exec housing-api-container ./healthcheck.sh
```

**4. Permisos de logs**
```bash
# Crear directorio con permisos correctos
mkdir -p logs
chmod 755 logs
```

### Debugging

```bash
# Acceder al shell del contenedor
docker exec -it housing-api-container /bin/bash

# Inspeccionar configuración
docker inspect housing-api-container

# Ver procesos dentro del contenedor
docker exec housing-api-container ps aux
```

## Testing

### Pruebas Automatizadas

```bash
# Con make
make test

# Manual
python test_api.py
```

### Pruebas Manuales

```bash
# Health check
curl http://localhost:8000/health

# Información del modelo
curl http://localhost:8000/model/info

# Predicción simple
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "crim": 0.02731, "zn": 0.0, "indus": 7.07, "chas": 0,
      "nox": 0.469, "rm": 6.421, "age": 78.9, "dis": 4.9671,
      "rad": 2, "tax": 242, "ptratio": 17.8, "b": 396.90, "lstat": 9.14
    }
  }'
```

## Limpieza

```bash
# Limpiar contenedores e imágenes
make clean

# Limpieza completa
make clean-all

# Manual
docker-compose down -v --rmi all
docker system prune -f
```

## Próximos Pasos

1. **CI/CD Pipeline** - Automatizar builds y deployments
2. **Kubernetes** - Orchestración para producción
3. **Secrets Management** - Gestión segura de credenciales
4. **Monitoring Avanzado** - Métricas custom y alertas
5. **Load Balancing** - Múltiples instancias para escalabilidad