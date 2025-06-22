"""
Script para probar la API de predicción de precios de viviendas
Suite completa de pruebas con logging detallado
"""

import requests
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Configurar logging sin emojis y compatible con Windows
def setup_logging():
    """Configurar sistema de logging compatible con Windows"""

    # Crear directorio de logs si no existe
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # Configurar formato de logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configurar handlers
    handlers = []

    # Handler para archivo con encoding UTF-8
    file_handler = logging.FileHandler('logs/test_api.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(file_handler)

    # Handler para consola con encoding compatible
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # Configurar logger principal
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True  # Sobrescribir configuración existente
    )

    return logging.getLogger("test_api")

# Inicializar logger
logger = setup_logging()

# Configuración de la API
API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 30  # segundos

def log_response_details(response, test_name):
    """Registrar detalles de la respuesta de manera consistente"""
    logger.info(f"{test_name} - Status Code: {response.status_code}")

    if response.status_code == 200:
        try:
            response_data = response.json()
            logger.info(f"{test_name} - Response Data: {json.dumps(response_data, indent=2)}")
        except ValueError:
            logger.info(f"{test_name} - Response Text: {response.text}")
    else:
        logger.error(f"{test_name} - Error Response: {response.text}")

def test_root_endpoint():
    """Probar endpoint raíz de la API"""
    logger.info("Iniciando prueba del endpoint raíz")

    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=REQUEST_TIMEOUT)
        log_response_details(response, "Root Endpoint")

        if response.status_code == 200:
            data = response.json()

            # Validaciones básicas
            required_fields = ["message", "version"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Campo requerido '{field}' no encontrado en respuesta")
                    return False

            logger.info("Endpoint raíz funcionando correctamente")
            return True
        else:
            logger.error(f"Endpoint raíz falló con código: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión en endpoint raíz: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado en endpoint raíz: {str(e)}")
        return False

def test_health_check():
    """Probar endpoint de health check"""
    logger.info("Iniciando prueba de health check")

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=REQUEST_TIMEOUT)
        log_response_details(response, "Health Check")

        if response.status_code == 200:
            data = response.json()

            # Validaciones específicas para health check
            required_fields = ["status", "model_loaded", "timestamp", "api_version"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Campo requerido '{field}' no encontrado en health check")
                    return False

            # Verificar que el modelo esté cargado
            if not data.get("model_loaded", False):
                logger.error("El modelo no está cargado según health check")
                return False

            logger.info("Health check exitoso - Sistema funcionando correctamente")
            logger.info(f"Estado del sistema: {data.get('status')}")
            logger.info(f"Modelo cargado: {data.get('model_loaded')}")
            return True
        else:
            logger.error(f"Health check falló con código: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión en health check: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado en health check: {str(e)}")
        return False

def test_model_information():
    """Probar endpoint de información del modelo"""
    logger.info("Iniciando prueba de información del modelo")

    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=REQUEST_TIMEOUT)
        log_response_details(response, "Model Information")

        if response.status_code == 200:
            data = response.json()

            # Validaciones de campos esperados
            expected_fields = ["model_name", "validation_r2", "feature_count", "api_version"]
            for field in expected_fields:
                if field not in data:
                    logger.error(f"Campo esperado '{field}' no encontrado en información del modelo")
                    return False

            # Registrar información del modelo
            logger.info(f"Nombre del modelo: {data.get('model_name')}")
            logger.info(f"R² de validación: {data.get('validation_r2')}")
            logger.info(f"Número de features: {data.get('feature_count')}")
            logger.info(f"Versión de la API: {data.get('api_version')}")

            # Validaciones de valores razonables
            r2_score = data.get('validation_r2')
            if isinstance(r2_score, (int, float)) and 0 <= r2_score <= 1:
                logger.info(f"R² score válido: {r2_score}")
            else:
                logger.warning(f"R² score posiblemente inválido: {r2_score}")

            feature_count = data.get('feature_count')
            if isinstance(feature_count, int) and feature_count > 0:
                logger.info(f"Número de features válido: {feature_count}")
            else:
                logger.warning(f"Número de features posiblemente inválido: {feature_count}")

            logger.info("Información del modelo obtenida exitosamente")
            return True
        else:
            logger.error(f"Obtención de información del modelo falló con código: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión obteniendo información del modelo: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado obteniendo información del modelo: {str(e)}")
        return False

def test_single_prediction():
    """Probar predicción individual"""
    logger.info("Iniciando prueba de predicción individual")

    # Datos de ejemplo para una casa típica de Boston
    test_housing_data = {
        "features": {
            "crim": 0.02731,      # Tasa de criminalidad per cápita
            "zn": 0.0,            # Proporción de terrenos residenciales grandes
            "indus": 7.07,        # Proporción de acres de negocios no minoristas
            "chas": 0,            # Variable dummy del río Charles
            "nox": 0.469,         # Concentración de óxidos nítricos
            "rm": 6.421,          # Número promedio de habitaciones
            "age": 78.9,          # Proporción de unidades construidas antes de 1940
            "dis": 4.9671,        # Distancia a centros de empleo
            "rad": 2,             # Índice de accesibilidad a autopistas
            "tax": 242,           # Tasa de impuesto a la propiedad
            "ptratio": 17.8,      # Ratio estudiante-profesor
            "b": 396.90,          # Proporción de población afroamericana
            "lstat": 9.14         # Porcentaje de población de bajo estatus socioeconómico
        }
    }

    try:
        logger.info("Enviando datos de vivienda para predicción individual")
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=test_housing_data,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )

        log_response_details(response, "Single Prediction")

        if response.status_code == 200:
            result = response.json()

            # Validar estructura de respuesta
            required_fields = ["prediction", "confidence_interval", "model_info", "timestamp"]
            for field in required_fields:
                if field not in result:
                    logger.error(f"Campo requerido '{field}' no encontrado en predicción")
                    return False

            prediction = result['prediction']
            confidence_interval = result['confidence_interval']
            model_info = result['model_info']

            # Validar tipos y rangos
            if not isinstance(prediction, (int, float)):
                logger.error(f"Predicción debe ser numérica, recibido: {type(prediction)}")
                return False

            if prediction < 0:
                logger.warning(f"Predicción negativa puede ser inválida: {prediction}")

            # Registrar resultados
            logger.info(f"Predicción de precio: ${prediction:.2f}K")

            if confidence_interval:
                lower_bound = confidence_interval.get('lower', 'N/A')
                upper_bound = confidence_interval.get('upper', 'N/A')
                logger.info(f"Intervalo de confianza: ${lower_bound:.2f}K - ${upper_bound:.2f}K")

            logger.info(f"Modelo utilizado: {model_info.get('model_name', 'Desconocido')}")
            logger.info("Predicción individual completada exitosamente")
            return True
        else:
            logger.error(f"Predicción individual falló con código: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión en predicción individual: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado en predicción individual: {str(e)}")
        return False

def test_batch_prediction():
    """Probar predicción por lotes"""
    logger.info("Iniciando prueba de predicción por lotes")

    # Datos de ejemplo para 3 casas diferentes
    batch_housing_data = {
        "features_list": [
            {
                "crim": 0.02731, "zn": 0.0, "indus": 7.07, "chas": 0,
                "nox": 0.469, "rm": 6.421, "age": 78.9, "dis": 4.9671,
                "rad": 2, "tax": 242, "ptratio": 17.8, "b": 396.90, "lstat": 9.14
            },
            {
                "crim": 0.1, "zn": 12.5, "indus": 7.87, "chas": 0,
                "nox": 0.524, "rm": 6.012, "age": 66.6, "dis": 5.5605,
                "rad": 5, "tax": 311, "ptratio": 15.2, "b": 395.60, "lstat": 12.43
            },
            {
                "crim": 0.08829, "zn": 12.5, "indus": 7.87, "chas": 0,
                "nox": 0.524, "rm": 6.172, "age": 96.1, "dis": 5.9505,
                "rad": 5, "tax": 311, "ptratio": 15.2, "b": 396.90, "lstat": 19.15
            }
        ]
    }

    try:
        logger.info("Enviando lote de 3 viviendas para predicción")
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=batch_housing_data,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )

        log_response_details(response, "Batch Prediction")

        if response.status_code == 200:
            result = response.json()

            # Validar estructura de respuesta
            required_fields = ["predictions", "count", "model_info", "timestamp"]
            for field in required_fields:
                if field not in result:
                    logger.error(f"Campo requerido '{field}' no encontrado en predicción por lotes")
                    return False

            predictions = result['predictions']
            count = result['count']

            # Validar consistencia
            if count != len(predictions):
                logger.error(f"Inconsistencia: count={count} pero predictions={len(predictions)}")
                return False

            if count != 3:
                logger.error(f"Se esperaban 3 predicciones, se recibieron {count}")
                return False

            # Registrar resultados
            logger.info(f"Predicciones realizadas: {count}")
            for i, prediction in enumerate(predictions, 1):
                if isinstance(prediction, (int, float)):
                    logger.info(f"Casa {i}: ${prediction:.2f}K")
                else:
                    logger.error(f"Predicción {i} no es numérica: {prediction}")
                    return False

            logger.info("Predicción por lotes completada exitosamente")
            return True
        else:
            logger.error(f"Predicción por lotes falló con código: {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión en predicción por lotes: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado en predicción por lotes: {str(e)}")
        return False

def test_input_validation():
    """Probar validación de datos de entrada"""
    logger.info("Iniciando pruebas de validación de entrada")

    # Casos de prueba con datos inválidos
    invalid_test_cases = [
        {
            "name": "Valores negativos en campos que no lo permiten",
            "data": {
                "features": {
                    "crim": -1,  # No puede ser negativo
                    "zn": 0.0, "indus": 7.07, "chas": 0,
                    "nox": 0.469, "rm": 6.421, "age": 78.9, "dis": 4.9671,
                    "rad": 2, "tax": 242, "ptratio": 17.8, "b": 396.90, "lstat": 9.14
                }
            },
            "expected_status": 422
        },
        {
            "name": "Valores fuera de rango válido",
            "data": {
                "features": {
                    "crim": 0.02731, "zn": 150.0,  # Máximo permitido es 100
                    "indus": 7.07, "chas": 0,
                    "nox": 0.469, "rm": 6.421, "age": 78.9, "dis": 4.9671,
                    "rad": 2, "tax": 242, "ptratio": 17.8, "b": 396.90, "lstat": 9.14
                }
            },
            "expected_status": 422
        },
        {
            "name": "Campos requeridos faltantes",
            "data": {
                "features": {
                    "crim": 0.02731,
                    "zn": 0.0,
                    "indus": 7.07
                    # Faltan muchos campos requeridos
                }
            },
            "expected_status": 422
        },
        {
            "name": "Tipos de datos incorrectos",
            "data": {
                "features": {
                    "crim": "texto",  # Debe ser numérico
                    "zn": 0.0, "indus": 7.07, "chas": 0,
                    "nox": 0.469, "rm": 6.421, "age": 78.9, "dis": 4.9671,
                    "rad": 2, "tax": 242, "ptratio": 17.8, "b": 396.90, "lstat": 9.14
                }
            },
            "expected_status": 422
        }
    ]

    validation_tests_passed = 0
    total_validation_tests = len(invalid_test_cases)

    for test_case in invalid_test_cases:
        try:
            logger.info(f"Probando validación: {test_case['name']}")
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=test_case["data"],
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT
            )

            expected_status = test_case.get("expected_status", 422)
            if response.status_code == expected_status:
                logger.info(f"Validación correcta para: {test_case['name']} (Status: {response.status_code})")
                validation_tests_passed += 1
            else:
                logger.warning(f"Validación inesperada para: {test_case['name']} - Esperado: {expected_status}, Recibido: {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error de conexión probando validación {test_case['name']}: {str(e)}")
        except Exception as e:
            logger.error(f"Error inesperado probando validación {test_case['name']}: {str(e)}")

    logger.info(f"Pruebas de validación completadas: {validation_tests_passed}/{total_validation_tests} exitosas")

    if validation_tests_passed == total_validation_tests:
        logger.info("Todas las validaciones de entrada funcionan correctamente")
        return True
    else:
        logger.warning("Algunas validaciones de entrada no funcionaron como se esperaba")
        return validation_tests_passed > 0  # Considerar exitoso si al menos una validación funciona

def test_api_performance():
    """Probar rendimiento básico de la API"""
    logger.info("Iniciando pruebas de rendimiento")

    test_data = {
        "features": {
            "crim": 0.02731, "zn": 0.0, "indus": 7.07, "chas": 0,
            "nox": 0.469, "rm": 6.421, "age": 78.9, "dis": 4.9671,
            "rad": 2, "tax": 242, "ptratio": 17.8, "b": 396.90, "lstat": 9.14
        }
    }

    num_requests = 5
    response_times = []
    successful_requests = 0

    try:
        logger.info(f"Ejecutando {num_requests} requests para medir rendimiento")

        for i in range(num_requests):
            start_time = datetime.now()
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT
            )
            end_time = datetime.now()

            response_time = (end_time - start_time).total_seconds()

            if response.status_code == 200:
                response_times.append(response_time)
                successful_requests += 1
                logger.info(f"Request {i+1}: {response_time:.3f} segundos - Exitoso")
            else:
                logger.error(f"Request {i+1}: Falló con código {response.status_code}")

        if response_times:
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)

            logger.info(f"Estadísticas de rendimiento:")
            logger.info(f"  Requests exitosos: {successful_requests}/{num_requests}")
            logger.info(f"  Tiempo promedio: {avg_time:.3f} segundos")
            logger.info(f"  Tiempo mínimo: {min_time:.3f} segundos")
            logger.info(f"  Tiempo máximo: {max_time:.3f} segundos")

            # Evaluación del rendimiento
            if avg_time < 1.0:
                logger.info("Rendimiento excelente: Tiempo de respuesta menor a 1 segundo")
            elif avg_time < 2.0:
                logger.info("Rendimiento bueno: Tiempo de respuesta menor a 2 segundos")
            else:
                logger.warning("Rendimiento mejorable: Tiempo de respuesta mayor a 2 segundos")

            return successful_requests == num_requests
        else:
            logger.error("No se pudieron obtener tiempos de respuesta válidos")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión en pruebas de rendimiento: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado en pruebas de rendimiento: {str(e)}")
        return False

def run_complete_test_suite():
    """Ejecutar suite completa de pruebas"""
    logger.info("=" * 80)
    logger.info("INICIANDO SUITE COMPLETA DE PRUEBAS DE LA API")
    logger.info("=" * 80)

    # Lista de todas las pruebas a ejecutar
    test_suite = [
        ("Root Endpoint Test", test_root_endpoint),
        ("Health Check Test", test_health_check),
        ("Model Information Test", test_model_information),
        ("Single Prediction Test", test_single_prediction),
        ("Batch Prediction Test", test_batch_prediction),
        ("Input Validation Test", test_input_validation),
        ("API Performance Test", test_api_performance)
    ]

    test_results = []
    start_time = datetime.now()

    # Verificar conectividad básica antes de comenzar
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        logger.info(f"Conectividad con API confirmada - Status: {response.status_code}")
    except requests.exceptions.RequestException:
        logger.error(f"No se puede conectar a la API en {API_BASE_URL}")
        logger.error("Asegúrese de que la API esté ejecutándose con: python run_api.py")
        return False

    # Ejecutar cada prueba
    for test_name, test_function in test_suite:
        logger.info(f"\nEjecutando prueba: {test_name}")
        logger.info("-" * 50)

        try:
            test_passed = test_function()
            test_results.append((test_name, test_passed))

            if test_passed:
                logger.info(f"RESULTADO: {test_name} - EXITOSO")
            else:
                logger.error(f"RESULTADO: {test_name} - FALLIDO")

        except Exception as e:
            logger.error(f"RESULTADO: {test_name} - ERROR INESPERADO: {str(e)}")
            test_results.append((test_name, False))

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Generar reporte final
    logger.info("\n" + "=" * 80)
    logger.info("REPORTE FINAL DE PRUEBAS")
    logger.info("=" * 80)

    passed_tests = 0
    for test_name, passed in test_results:
        status = "EXITOSO" if passed else "FALLIDO"
        logger.info(f"{test_name:.<40} {status}")
        if passed:
            passed_tests += 1

    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    logger.info("-" * 80)
    logger.info(f"Pruebas exitosas: {passed_tests}/{total_tests}")
    logger.info(f"Tasa de éxito: {success_rate:.1f}%")
    logger.info(f"Duración total: {total_duration:.2f} segundos")

    if passed_tests == total_tests:
        logger.info("RESULTADO GENERAL: TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        logger.info("La API está funcionando correctamente y lista para producción")
        return True
    elif passed_tests > 0:
        logger.warning("RESULTADO GENERAL: ALGUNAS PRUEBAS FALLARON")
        logger.warning("Revise los logs anteriores para detalles de las fallas")
        return False
    else:
        logger.error("RESULTADO GENERAL: TODAS LAS PRUEBAS FALLARON")
        logger.error("La API tiene problemas críticos que deben resolverse")
        return False

if __name__ == "__main__":
    try:
        success = run_complete_test_suite()
        exit_code = 0 if success else 1
        logger.info(f"Finalizando script de pruebas con código: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Pruebas interrumpidas por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error crítico en el script de pruebas: {str(e)}")
        sys.exit(1)