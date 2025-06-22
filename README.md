# Boston Housing Price Prediction API

**API REST para predicción de precios de viviendas usando Machine Learning con FastAPI y Docker**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.13-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Requisitos y Dependencias](#requisitos-y-dependencias)
- [Instalación](#instalación)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Despliegue y Pruebas](#despliegue-y-pruebas)
- [Monitoreo del Modelo](#monitoreo-del-modelo)
- [API Reference](#api-reference)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribución](#contribución)

## 📖 Descripción

Este proyecto implementa una **API REST completa** para predecir precios de viviendas utilizando el famoso dataset de Boston Housing. La solución incluye:

- **Machine Learning Pipeline** completo con múltiples algoritmos
- **API REST** construida con FastAPI
- **Containerización** con Docker y Docker Compose
- **Monitoreo** y logging avanzado
- **Configuración de producción** lista para despliegue

### 🎯 Características principales

- ✅ **Entrenamiento automatizado** de múltiples modelos ML
- ✅ **Selección automática** del mejor modelo
- ✅ **API REST** con documentación interactiva
- ✅ **Validación robusta** de datos de entrada
- ✅ **Health checks** y monitoreo
- ✅ **Logging estructurado** para debugging
- ✅ **Configuración Docker** optimizada para producción

## 🏗️ Arquitectura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  ML Pipeline    │───▶│   Trained Model │
│  (Boston.csv)   │    │ (preprocessing  │    │  (best_model.pkl)│
└─────────────────┘    │  + training)    │    └─────────────────┘
                       └─────────────────┘             │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Docker        │───▶│   FastAPI       │───▶│   Predictions   │
│  (Container)    │    │   (REST API)    │    │   (JSON)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │   Monitoring    │
│ (Load Balancer) │    │ (Logs + Health) │
└─────────────────┘    └─────────────────┘
```

## 🔧 Requisitos y Dependencias

### Requisitos del Sistema

- **Python**: 3.12+
- **Docker**: 20.0+
- **Docker Compose**: 2.0+
- **RAM**: Mínimo 4GB
- **Espacio en disco**: 2GB libres

### Dependencias Python principales

```
# Core ML
scikit-learn==1.7.0
pandas==2.3.0
numpy==2.3.1

# API Framework
fastapi==0.115.13
uvicorn==0.34.3
pydantic==2.11.7

# Utilities
python-dotenv==1.1.0
joblib==1.5.1
rich==14.0.0
```

### Herramientas de desarrollo

```
# Testing
pytest==8.3.4
httpx==0.28.1

# Code quality
black==24.10.0
flake8==7.1.1
mypy==1.13.0

# Notebooks
jupyter==1.1.1
matplotlib==3.9.4
seaborn==0.13.2
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/mlops-housing-prediction.git
cd mlops-housing-prediction
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
# Copiar archivo de configuración
cp .env.example .env

# Editar configuración según tu entorno
```

## 🤖 Entrenamiento del Modelo

### Proceso automatizado

El pipeline de entrenamiento incluye:

1. **Carga de datos** desde `data/Boston.csv`
2. **Preprocesamiento** y limpieza
3. **Feature engineering** automático
4. **Entrenamiento** de múltiples modelos:
   - Linear Regression
   - Random Forest
   - Gradient Boosting
   - Support Vector Regression
5. **Evaluación** y selección del mejor modelo
6. **Guardado** del modelo optimizado

### Ejecutar entrenamiento

```bash
# Desde el directorio raíz del proyecto
python src/training/train_model.py
```

### Salida esperada

```
🚀 Iniciando pipeline de entrenamiento...
📊 Datos cargados: 506 muestras, 13 features
🔧 Preprocesamiento completado
🤖 Entrenando Linear Regression...
🤖 Entrenando Random Forest...
🤖 Entrenando Gradient Boosting...
🤖 Entrenando SVR...

📈 Resultados del entrenamiento:
┌─────────────────────┬──────────┬──────────┬──────────┐
│ Model               │ MAE      │ RMSE     │ R²       │
├─────────────────────┼──────────┼──────────┼──────────┤
│ Gradient Boosting   │ 2.47     │ 3.58     │ 0.89     │
│ Random Forest       │ 2.51     │ 3.63     │ 0.88     │
│ Linear Regression   │ 3.21     │ 4.68     │ 0.81     │
│ SVR                 │ 3.45     │ 5.02     │ 0.78     │
└─────────────────────┴──────────┴──────────┴──────────┘

🏆 Mejor modelo: Gradient Boosting
💾 Modelo guardado: models/best_model.pkl
📋 Metadatos guardados: models/model_metadata.json
```

### Archivos generados

- `models/best_model.pkl`: Modelo entrenado serializado
- `models/model_metadata.json`: Metadatos del modelo
- `models/training_report.json`: Reporte detallado del entrenamiento
- `logs/training.log`: Logs del proceso de entrenamiento

## 🐳 Despliegue y Pruebas

### Opción 1: Desarrollo local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor de desarrollo
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Opción 2: Docker (Recomendado)

#### Construcción de la imagen

```bash
# Construir imagen Docker
docker build -t housing-prediction-api:latest .
```

#### Despliegue con Docker Compose

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f housing-api

# Verificar estado de contenedores
docker ps
```

#### Servicios incluidos

- **API**: `http://localhost:8000`
- **Nginx**: `http://localhost` (proxy reverso)
- **Prometheus**: `http://localhost:9090` (métricas)
- **Grafana**: `http://localhost:3000` (dashboards)

### Verificar despliegue

```bash
# Health check
curl http://localhost:8000/health

# Información del modelo
curl http://localhost:8000/model/info

# Documentación interactiva
# Abrir en navegador: http://localhost:8000/docs
```

### Pruebas de la API

#### 1. Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

**Respuesta esperada:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-22T15:44:45.822Z",
  "version": "1.0.0",
  "model_loaded": true
}
```

#### 2. Información del modelo

```bash
curl -X GET "http://localhost:8000/model/info"
```

**Respuesta esperada:**
```json
{
  "model_name": "Boston Housing Price Predictor",
  "model_type": "gradient_boosting",
  "version": "1.0.0",
  "features": 13,
  "trained_date": "2025-06-22T10:30:15Z",
  "performance": {
    "mae": 2.47,
    "rmse": 3.58,
    "r2_score": 0.89
  }
}
```

#### 3. Predicción individual

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "crim": 0.00632,
    "zn": 18.0,
    "indus": 2.31,
    "chas": 0,
    "nox": 0.538,
    "rm": 6.575,
    "age": 65.2,
    "dis": 4.0900,
    "rad": 1,
    "tax": 296.0,
    "ptratio": 15.3,
    "b": 396.90,
    "lstat": 4.98
  }'
```

**Respuesta esperada:**
```json
{
  "prediction": 24.32,
  "confidence_interval": [21.45, 27.19],
  "model_version": "1.0.0",
  "prediction_id": "pred_20250622_154445_abc123",
  "timestamp": "2025-06-22T15:44:45.822Z"
}
```

#### 4. Predicción en lote

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "crim": 0.00632, "zn": 18.0, "indus": 2.31,
        "chas": 0, "nox": 0.538, "rm": 6.575,
        "age": 65.2, "dis": 4.0900, "rad": 1,
        "tax": 296.0, "ptratio": 15.3, "b": 396.90, "lstat": 4.98
      },
      {
        "crim": 0.02731, "zn": 0.0, "indus": 7.07,
        "chas": 0, "nox": 0.469, "rm": 6.421,
        "age": 78.9, "dis": 4.9671, "rad": 2,
        "tax": 242.0, "ptratio": 17.8, "b": 396.90, "lstat": 9.14
      }
    ]
  }'
```

### Stopping services

```bash
# Detener todos los servicios
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v
```

## 📊 Monitoreo del Modelo

### 1. Health Checks automáticos

El sistema incluye health checks automáticos cada 30 segundos:

```yaml
healthcheck:
  test: ["CMD", "./healthcheck.sh"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### 2. Logging estructurado

#### Configuración de logs

Los logs se guardan en:
- **Desarrollo**: `logs/api.log`
- **Docker**: Volumen montado en `./logs:/app/logs`

#### Niveles de logging

```python
# Configuración en src/api/config.py
LOG_LEVELS = {
    "CRITICAL": 50,
    "ERROR": 40, 
    "WARNING": 30,
    "INFO": 20,
    "DEBUG": 10
}
```

#### Formato de logs

```json
{
  "timestamp": "2025-06-22T15:44:45.822Z",
  "level": "INFO",
  "logger": "src.api.main",
  "message": "Request: POST /predict - Status: 200 - Duration: 0.0045s",
  "request_id": "req_abc123",
  "user_agent": "curl/7.68.0",
  "remote_addr": "172.18.0.1"
}
```

### 3. Métricas de rendimiento

#### Métricas automáticas

El sistema registra automáticamente:

- **Latencia**: Tiempo de respuesta por request
- **Throughput**: Requests por segundo
- **Error rate**: Porcentaje de errores
- **Model accuracy**: Métricas del modelo en tiempo real

#### Endpoints de métricas

```bash
# Métricas de Prometheus
curl http://localhost:8000/metrics

# Estadísticas de la API
curl http://localhost:8000/stats
```

### 4. Monitoring con Prometheus + Grafana

#### Configuración de Prometheus

```yaml
# docker/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'housing-api'
    static_configs:
      - targets: ['housing-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

#### Iniciar monitoreo completo

```bash
# Iniciar con profile de monitoreo
docker-compose --profile monitoring up -d

# Acceder a Grafana
# URL: http://localhost:3000
# Usuario: admin
# Password: admin123
```

#### Dashboards incluidos

1. **API Performance**
   - Request latency
   - Throughput
   - Error rates
   - Response time distribution

2. **Model Monitoring**
   - Prediction accuracy
   - Feature drift detection
   - Model performance trends
   - Input data quality

3. **Infrastructure**
   - CPU usage
   - Memory consumption
   - Docker container health
   - Network metrics

### 5. Alertas automáticas

#### Configuración de alertas

```yaml
# docker/alertmanager.yml
groups:
- name: housing-api-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
    for: 5m
    annotations:
      summary: "High latency detected"
```

### 6. Model Drift Detection

#### Monitoreo de deriva del modelo

```python
# Verificar deriva de datos
curl -X POST "http://localhost:8000/model/drift-check" \
  -H "Content-Type: application/json" \
  -d '{
    "reference_period": "2025-01-01",
    "current_period": "2025-06-22",
    "threshold": 0.05
  }'
```

**Respuesta:**
```json
{
  "drift_detected": false,
  "drift_score": 0.023,
  "threshold": 0.05,
  "recommendations": [
    "No action required",
    "Continue monitoring"
  ]
}
```

## 📚 API Reference

### Endpoints principales

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Health check del servicio |
| `/model/info` | GET | Información del modelo |
| `/predict` | POST | Predicción individual |
| `/predict/batch` | POST | Predicción en lote |
| `/metrics` | GET | Métricas de Prometheus |
| `/stats` | GET | Estadísticas de la API |
| `/docs` | GET | Documentación interactiva |

### Esquemas de datos

#### HousingFeatures

```json
{
  "crim": "float (Crime rate)",
  "zn": "float (Residential land zoned)",
  "indus": "float (Non-retail business acres)",
  "chas": "int (Charles River dummy variable)",
  "nox": "float (Nitric oxides concentration)",
  "rm": "float (Average number of rooms)",
  "age": "float (Age of owner-occupied units)",
  "dis": "float (Distance to employment centers)",
  "rad": "int (Accessibility to radial highways)",
  "tax": "float (Property tax rate)",
  "ptratio": "float (Pupil-teacher ratio)",
  "b": "float (Proportion of blacks)",
  "lstat": "float (Lower status of population)"
}
```

#### PredictionResponse

```json
{
  "prediction": "float (Predicted price in $1000s)",
  "confidence_interval": "[float, float] (95% confidence)",
  "model_version": "string",
  "prediction_id": "string",
  "timestamp": "string (ISO 8601)"
}
```

## 📁 Estructura del Proyecto

```
mlops-housing-prediction/
├── 📂 data/                     # Datasets
│   ├── Boston.csv              # Dataset original
│   └── processed/              # Datos procesados
├── 📂 src/                     # Código fuente
│   ├── 📂 api/                 # API REST
│   │   ├── main.py            # Aplicación principal
│   │   ├── models.py          # Modelos Pydantic
│   │   ├── endpoints/         # Endpoints de la API
│   │   └── middleware/        # Middleware personalizado
│   ├── 📂 training/           # Pipeline de entrenamiento
│   │   ├── train_model.py     # Script principal de entrenamiento
│   │   ├── preprocessor.py    # Preprocesamiento de datos
│   │   └── model_trainer.py   # Entrenamiento de modelos
│   └── 📂 utils/              # Utilidades compartidas
│       ├── logger.py          # Configuración de logging
│       └── metrics.py         # Métricas personalizadas
├── 📂 models/                  # Modelos entrenados
│   ├── best_model.pkl         # Mejor modelo serializado
│   ├── model_metadata.json    # Metadatos del modelo
│   └── training_report.json   # Reporte de entrenamiento
├── 📂 docker/                  # Configuración Docker
│   ├── Dockerfile             # Imagen principal
│   ├── start.sh              # Script de inicio
│   ├── healthcheck.sh        # Health check script
│   ├── nginx.conf            # Configuración Nginx
│   └── prometheus.yml        # Configuración Prometheus
├── 📂 tests/                   # Tests automatizados
│   ├── test_api.py           # Tests de la API
│   ├── test_model.py         # Tests del modelo
│   └── conftest.py           # Configuración de pytest
├── 📂 logs/                    # Archivos de log
├── 📂 notebooks/               # Jupyter notebooks
│   ├── EDA.ipynb             # Análisis exploratorio
│   └── Model_Analysis.ipynb   # Análisis del modelo
├── docker-compose.yml          # Orquestación de servicios
├── requirements.txt            # Dependencias Python
├── requirements-api.txt        # Dependencias específicas API
├── .env.docker                # Variables de entorno Docker
├── .env.example               # Ejemplo de configuración
├── .gitignore                 # Archivos ignorados por Git
└── README.md                  # Este archivo
```

## 🧪 Testing

### Ejecutar tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_api.py -v

# Con coverage
pytest --cov=src tests/
```

### Tests incluidos

- **Unit tests**: Funciones individuales
- **Integration tests**: API endpoints
- **Model tests**: Validación del modelo
- **Docker tests**: Contenedores funcionando

## 🤝 Contribución

### Proceso de contribución

1. **Fork** el repositorio
2. **Crear** una rama feature (`git checkout -b feature/amazing-feature`)
3. **Commit** los cambios (`git commit -m 'Add amazing feature'`)
4. **Push** a la rama (`git push origin feature/amazing-feature`)
5. **Abrir** un Pull Request

### Estándares de código

```bash
# Formateo de código
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Commit conventions

- `feat:` Nueva funcionalidad
- `fix:` Corrección de bugs
- `docs:` Documentación
- `style:` Formateo de código
- `refactor:` Refactorización
- `test:` Tests
- `chore:` Tareas de mantenimiento

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 📞 Soporte

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/mlops-housing-prediction/issues)
- **Documentación**: [Wiki del proyecto](https://github.com/tu-usuario/mlops-housing-prediction/wiki)
- **Email**: tu-email@example.com

---

**Hecho con ❤️ por [Tu Nombre]**