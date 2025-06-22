#!/usr/bin/env python3
"""
Script para ejecutar la API desde el directorio correcto
Con logging completo y carga de configuración desde .env
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Cargar variables de entorno antes que nada
from dotenv import load_dotenv


def setup_logging():
    """Configurar logging para el script de inicio"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Log básico para la consola mientras se configura
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    return logging.getLogger("run_api")


def validate_environment():
    """Validar que todas las variables de entorno necesarias estén configuradas"""
    logger = logging.getLogger("run_api")

    required_vars = [
        "APP_NAME", "APP_VERSION", "APP_DESCRIPTION",
        "MODEL_PATH", "METADATA_PATH", "LOG_FILE",
        "HOST", "PORT", "WORKERS", "LOG_LEVEL",
        "ALLOWED_HOSTS", "CORS_ORIGINS", "MAX_BATCH_SIZE",
        "ENABLE_ACCESS_LOG", "HEALTH_CHECK_INTERVAL"
    ]

    missing_vars = []
    for var in required_vars:
        if os.getenv(var) is None:
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"Variables de entorno faltantes: {', '.join(missing_vars)}")
        return False

    logger.info("Todas las variables de entorno requeridas están configuradas")
    return True


def validate_paths():
    """Validar que existan los archivos y directorios necesarios"""
    logger = logging.getLogger("run_api")

    # Obtener rutas desde variables de entorno
    model_path = Path(os.getenv("MODEL_PATH"))
    metadata_path = Path(os.getenv("METADATA_PATH"))
    log_file = Path(os.getenv("LOG_FILE"))

    # Validar modelo
    if not model_path.exists():
        logger.error(f"Modelo no encontrado: {model_path}")
        logger.error("Ejecuta primero: python src/train.py")
        return False

    logger.info(f"Modelo encontrado: {model_path}")

    # Validar metadatos (opcional)
    if metadata_path.exists():
        logger.info(f"Metadatos encontrados: {metadata_path}")
    else:
        logger.warning(f"Metadatos no encontrados: {metadata_path}")

    # Crear directorio de logs si no existe
    log_dir = log_file.parent
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de logs creado: {log_dir}")

    return True


def setup_api_logging():
    """Configurar logging detallado una vez que tenemos las variables de entorno"""
    log_file = os.getenv("LOG_FILE")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configurar logger para el script
    logger = logging.getLogger("run_api")

    # Limpiar handlers existentes
    logger.handlers.clear()

    # Crear formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Handler para archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level))

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Configurar logger
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_configuration():
    """Cargar y mostrar configuración desde .env"""
    logger = logging.getLogger("run_api")

    config = {
        'app': {
            'name': os.getenv("APP_NAME"),
            'version': os.getenv("APP_VERSION"),
            'description': os.getenv("APP_DESCRIPTION"),
            'environment': os.getenv("ENVIRONMENT", "development")
        },
        'server': {
            'host': os.getenv("HOST"),
            'port': int(os.getenv("PORT")),
            'workers': int(os.getenv("WORKERS")),
            'log_level': os.getenv("LOG_LEVEL"),
            'reload': os.getenv("RELOAD", "true").lower() == "true",
            'access_log': os.getenv("ENABLE_ACCESS_LOG", "true").lower() == "true"
        },
        'security': {
            'allowed_hosts': os.getenv("ALLOWED_HOSTS").split(","),
            'cors_origins': os.getenv("CORS_ORIGINS").split(","),
        },
        'limits': {
            'max_batch_size': int(os.getenv("MAX_BATCH_SIZE"))
        },
        'paths': {
            'model': os.getenv("MODEL_PATH"),
            'metadata': os.getenv("METADATA_PATH"),
            'log_file': os.getenv("LOG_FILE")
        }
    }

    logger.info("=== CONFIGURACIÓN DE LA API ===")
    logger.info(f"Aplicación: {config['app']['name']} v{config['app']['version']}")
    logger.info(f"Entorno: {config['app']['environment']}")
    logger.info(f"Servidor: {config['server']['host']}:{config['server']['port']}")
    logger.info(f"Workers: {config['server']['workers']}")
    logger.info(f"Log Level: {config['server']['log_level']}")
    logger.info(f"Reload Mode: {config['server']['reload']}")
    logger.info(f"Modelo: {config['paths']['model']}")
    logger.info(f"Log File: {config['paths']['log_file']}")
    logger.info("=======================")

    return config


def main():
    """Función principal para ejecutar la API"""

    # Configurar logging básico
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("INICIANDO API DE PREDICCIÓN DE PRECIOS DE VIVIENDAS")
    logger.info("=" * 60)

    try:
        # Asegurar que estamos en el directorio raíz del proyecto
        project_root = Path(__file__).parent
        original_cwd = os.getcwd()
        os.chdir(project_root)
        logger.info(f"Directorio de trabajo cambiado a: {project_root}")

        # Verificar que existe el archivo .env
        env_file = project_root / ".env"
        if not env_file.exists():
            logger.error(f"Archivo .env no encontrado en: {env_file}")
            logger.error("Crea el archivo .env usando .env.example como plantilla")
            return 1

        logger.info(f"Archivo .env encontrado: {env_file}")

        # Cargar variables de entorno
        load_dotenv(env_file)
        logger.info("Variables de entorno cargadas desde .env")

        # Configurar logging detallado
        logger = setup_api_logging()

        # Validar configuración
        if not validate_environment():
            logger.error("Validación de variables de entorno falló")
            return 1

        if not validate_paths():
            logger.error("Validación de archivos y rutas falló")
            return 1

        # Cargar y mostrar configuración
        config = load_configuration()

        logger.info("Todas las validaciones pasaron exitosamente")
        logger.info("Iniciando servidor FastAPI...")

        # Importar la aplicación FastAPI
        try:
            import uvicorn
            logger.info("Uvicorn importado correctamente")
        except ImportError as e:
            logger.error(f"Error importando uvicorn: {e}")
            logger.error("Instala uvicorn: pip install uvicorn[standard]")
            return 1

        # Ejecutar servidor
        uvicorn.run(
            "src.api.main:app",
            host=config['server']['host'],
            port=config['server']['port'],
            reload=config['server']['reload'],
            log_level=config['server']['log_level'].lower(),
            access_log=config['server']['access_log'],
            workers=config['server']['workers'] if not config['server']['reload'] else 1,
            log_config=None  # Usar nuestra configuración de logging
        )

    except KeyboardInterrupt:
        logger.info("Aplicación interrumpida por el usuario")
        return 0
    except Exception as e:
        logger.error(f"Error crítico: {e}", exc_info=True)
        return 1
    finally:
        # Restaurar directorio original si es necesario
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        logger.info("Aplicación finalizada")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)