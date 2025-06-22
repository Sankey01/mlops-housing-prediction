"""
API REST para predicción de precios de viviendas
Modelo: Boston Housing Price Prediction
Framework: FastAPI

Esta API expone el modelo entrenado como un servicio REST,
incluyendo validación de datos, logging y manejo de errores.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, validator, Field
import uvicorn


# Configuración de la aplicación desde variables de entorno
APP_NAME = os.getenv("APP_NAME")
APP_VERSION = os.getenv("APP_VERSION")
APP_DESCRIPTION = os.getenv("APP_DESCRIPTION")

# Configuración de rutas
MODEL_PATH = Path(os.getenv("MODEL_PATH"))
METADATA_PATH = Path(os.getenv("METADATA_PATH"))
LOG_FILE = os.getenv("LOG_FILE")

# Configuración del servidor
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))
WORKERS = int(os.getenv("WORKERS"))
LOG_LEVEL = os.getenv("LOG_LEVEL")

# Configuración de seguridad
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS").split(",")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE"))

# Configuración de monitoreo
ENABLE_ACCESS_LOG = os.getenv("ENABLE_ACCESS_LOG").lower() == "true"
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL"))

# Validar que las rutas de archivos críticos existan o sean relativas válidas
if not MODEL_PATH.is_absolute():
    # Si es una ruta relativa, hacerla relativa al directorio de trabajo actual
    MODEL_PATH = Path.cwd() / MODEL_PATH

if not METADATA_PATH.is_absolute():
    METADATA_PATH = Path.cwd() / METADATA_PATH

# Configurar logging con ruta desde .env
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HousingFeatures(BaseModel):
    """
    Modelo de datos para las características de entrada de una vivienda
    """
    crim: float = Field(..., description="Per capita crime rate by town", ge=0)
    zn: float = Field(..., description="Proportion of residential land zoned for lots over 25,000 sq.ft", ge=0, le=100)
    indus: float = Field(..., description="Proportion of non-retail business acres per town", ge=0, le=100)
    chas: int = Field(..., description="Charles River dummy variable (1 if tract bounds river; 0 otherwise)", ge=0,
                      le=1)
    nox: float = Field(..., description="Nitric oxides concentration (parts per 10 million)", gt=0, le=1)
    rm: float = Field(..., description="Average number of rooms per dwelling", gt=0, le=15)
    age: float = Field(..., description="Proportion of owner-occupied units built prior to 1940", ge=0, le=100)
    dis: float = Field(..., description="Weighted distances to employment centres", gt=0)
    rad: int = Field(..., description="Index of accessibility to radial highways", ge=1, le=24)
    tax: int = Field(..., description="Full-value property-tax rate per $10,000", gt=0)
    ptratio: float = Field(..., description="Pupil-teacher ratio by town", gt=0, le=50)
    b: float = Field(..., description="Proportion of blacks by town", ge=0, le=500)
    lstat: float = Field(..., description="% lower status of the population", ge=0, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "crim": 0.02731,
                "zn": 0.0,
                "indus": 7.07,
                "chas": 0,
                "nox": 0.469,
                "rm": 6.421,
                "age": 78.9,
                "dis": 4.9671,
                "rad": 2,
                "tax": 242,
                "ptratio": 17.8,
                "b": 396.90,
                "lstat": 9.14
            }
        }


class PredictionRequest(BaseModel):
    """
    Modelo para requests de predicción individual
    """
    features: HousingFeatures

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "crim": 0.02731,
                    "zn": 0.0,
                    "indus": 7.07,
                    "chas": 0,
                    "nox": 0.469,
                    "rm": 6.421,
                    "age": 78.9,
                    "dis": 4.9671,
                    "rad": 2,
                    "tax": 242,
                    "ptratio": 17.8,
                    "b": 396.90,
                    "lstat": 9.14
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Modelo para requests de predicción por lotes
    """
    features_list: List[HousingFeatures] = Field(..., max_items=MAX_BATCH_SIZE,
                                                 description=f"Lista de características (máximo {MAX_BATCH_SIZE})")


class PredictionResponse(BaseModel):
    """
    Modelo de respuesta para predicciones
    """
    prediction: float = Field(..., description="Precio predicho en miles de dólares")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Intervalo de confianza estimado")
    model_info: Dict[str, Any] = Field(..., description="Información del modelo utilizado")
    timestamp: str = Field(..., description="Timestamp de la predicción")


class BatchPredictionResponse(BaseModel):
    """
    Modelo de respuesta para predicciones por lotes
    """
    predictions: List[float] = Field(..., description="Lista de precios predichos")
    count: int = Field(..., description="Número de predicciones realizadas")
    model_info: Dict[str, Any] = Field(..., description="Información del modelo utilizado")
    timestamp: str = Field(..., description="Timestamp de las predicciones")


class HealthResponse(BaseModel):
    """
    Modelo de respuesta para health check
    """
    status: str
    timestamp: str
    model_loaded: bool
    api_version: str


class ModelManager:
    """
    Gestor del modelo de machine learning
    """

    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.is_loaded = False

    def load_model(self):
        """Cargar el modelo y sus metadatos"""
        try:
            logger.info(f"Cargando modelo desde: {self.model_path}")

            if not self.model_path.exists():
                raise FileNotFoundError(f"Archivo de modelo no encontrado: {self.model_path}")

            # Cargar modelo
            model_package = joblib.load(str(self.model_path))
            self.model = model_package['model']
            self.feature_names = model_package['feature_names']

            # Cargar metadatos
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                logger.warning(f"Archivo de metadatos no encontrado: {self.metadata_path}")
                self.metadata = model_package.get('metadata', {})

            self.is_loaded = True
            logger.info("Modelo cargado exitosamente")
            logger.info(f"Tipo de modelo: {self.metadata.get('model_name', 'Desconocido')}")
            logger.info(f"Features esperadas: {len(self.feature_names)}")

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.is_loaded = False
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Realizar predicción"""
        if not self.is_loaded:
            raise RuntimeError("Modelo no está cargado")

        try:
            predictions = self.model.predict(features)
            return predictions
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if not self.is_loaded:
            return {"error": "Modelo no cargado"}

        return {
            "model_name": self.metadata.get('model_name', 'Unknown'),
            "model_description": self.metadata.get('model_description', 'Unknown'),
            "training_timestamp": self.metadata.get('training_timestamp', 'Unknown'),
            "validation_r2": self.metadata.get('validation_metrics', {}).get('val_r2', 'Unknown'),
            "feature_count": len(self.feature_names),
            "api_version": APP_VERSION
        }


# Inicializar gestor de modelo
model_manager = ModelManager()

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    # Startup
    logger.info(f"Iniciando {APP_NAME} v{APP_VERSION}")
    try:
        model_manager.load_model()
        logger.info("API lista para recibir requests")
    except Exception as e:
        logger.error(f"Error durante la inicialización: {e}")
        # En producción, podrías decidir si continuar o terminar la app

    yield  # La aplicación está corriendo

    # Shutdown
    logger.info("Cerrando aplicación...")


# Crear aplicación FastAPI
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS desde variables de entorno
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de hosts confiables si se especifican
if ALLOWED_HOSTS != ["*"]:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=ALLOWED_HOSTS
    )


# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()

    # Procesar request
    response = await call_next(request)

    # Calcular tiempo de procesamiento
    process_time = (datetime.now() - start_time).total_seconds()

    # Log de la request
    logger.info(f"Request: {request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Duration: {process_time:.4f}s")

    response.headers["X-Process-Time"] = str(process_time)
    return response


# Dependency para validar que el modelo esté cargado
def get_model_manager():
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no está disponible")
    return model_manager


@app.on_event("startup")
async def startup_event():
    """Inicialización de la aplicación"""
    logger.info(f"Iniciando {APP_NAME} v{APP_VERSION}")
    try:
        model_manager.load_model()
        logger.info("API lista para recibir requests")
    except Exception as e:
        logger.error(f"Error durante la inicialización: {e}")
        # En producción, podrías decidir si continuar o terminar la app


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la aplicación"""
    logger.info("Cerrando aplicación...")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información básica de la API"""
    return {
        "message": f"Bienvenido a {APP_NAME}",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint para monitoreo"""
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_manager.is_loaded,
        api_version=APP_VERSION
    )


@app.get("/model/info")
async def get_model_info(manager: ModelManager = Depends(get_model_manager)):
    """Obtener información detallada del modelo"""
    return manager.get_model_info()


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
        request: PredictionRequest,
        manager: ModelManager = Depends(get_model_manager)
):
    """
    Realizar predicción para una vivienda individual

    Recibe las características de una vivienda y retorna el precio predicho.
    """
    try:
        # Convertir features a array
        features_dict = request.features.dict()
        features_array = np.array([[features_dict[name] for name in manager.feature_names]])

        # Realizar predicción
        prediction = manager.predict(features_array)[0]

        # Estimar intervalo de confianza básico (usando desviación estándar de entrenamiento)
        # En un escenario real, esto se calcularía de manera más sofisticada
        std_estimate = prediction * 0.15  # Estimación básica del 15%
        confidence_interval = {
            "lower": float(max(0, prediction - 1.96 * std_estimate)),
            "upper": float(prediction + 1.96 * std_estimate)
        }

        logger.info(f"Predicción realizada: {prediction:.2f}")

        return PredictionResponse(
            prediction=float(prediction),
            confidence_interval=confidence_interval,
            model_info=manager.get_model_info(),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error en predicción individual: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
        request: BatchPredictionRequest,
        manager: ModelManager = Depends(get_model_manager)
):
    """
    Realizar predicciones para múltiples viviendas

    Recibe una lista de características y retorna las predicciones correspondientes.
    """
    try:
        # Convertir lista de features a array
        features_list = []
        for features in request.features_list:
            features_dict = features.dict()
            features_array = [features_dict[name] for name in manager.feature_names]
            features_list.append(features_array)

        features_matrix = np.array(features_list)

        # Realizar predicciones
        predictions = manager.predict(features_matrix)

        logger.info(f"Predicciones por lote realizadas: {len(predictions)} casas")

        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            count=len(predictions),
            model_info=manager.get_model_info(),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error en predicción por lotes: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción por lotes: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Manejador personalizado de excepciones HTTP"""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Manejador general de excepciones"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Configuración para desarrollo desde variables de entorno
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level=LOG_LEVEL,
        access_log=ENABLE_ACCESS_LOG,
        workers=WORKERS if os.getenv("RELOAD", "true").lower() != "true" else 1  # Solo 1 worker en modo reload
    )