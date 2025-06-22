"""
Pipeline de Entrenamiento - Boston Housing Price Prediction
Autor: ML DevOps Engineer
Fecha: 2025

Pipeline automatizado para entrenar, evaluar y versionar modelos de regresión
para predicción de precios de viviendas usando el dataset Boston Housing.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import argparse
import json
import logging
from pathlib import Path

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# MLflow para versionado (comentado por ahora - lo activaremos después)
# import mlflow
# import mlflow.sklearn

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'C:\Users\Kenny\PycharmProjects\mlops-housing-prediction\src\logs\training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HousingPriceTrainer:
    """
    Clase principal para entrenar modelos de predicción de precios de viviendas
    """

    def __init__(self, data_url=None, random_state=42):
        self.data_url = data_url or "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_names = None

        # Crear directorios si no existen
        self.create_directories()

    def create_directories(self):
        """Crear estructura de directorios necesaria"""
        directories = [
            'models',
            'data',
            'logs',
            'metrics',
            'artifacts'
        ]

        for directory in directories:
            Path(directory).mkdir(exist_ok=True)

        logger.info("Directorios creados y verificados correctamente")

    def load_data(self):
        """Cargar y validar el dataset"""
        try:
            logger.info(f"Cargando datos desde: {self.data_url}")
            self.df = pd.read_csv(self.data_url)

            # Validaciones básicas
            if self.df.empty:
                raise ValueError("Dataset vacío")

            if 'medv' not in self.df.columns:
                raise ValueError("Columna objetivo 'medv' no encontrada")

            logger.info(f"Datos cargados exitosamente. Dimensiones: {self.df.shape}")
            logger.info(f"Columnas disponibles: {list(self.df.columns)}")

            # Guardar datos raw
            self.df.to_csv('data/raw_data.csv', index=False)
            logger.info("Datos guardados en data/raw_data.csv")

            return self.df

        except Exception as e:
            logger.error(f"Error al cargar los datos: {e}")
            raise

    def preprocess_data(self):
        """Preprocessing y feature engineering"""
        logger.info("Iniciando preprocesamiento de datos...")

        # Separar features y target
        X = self.df.drop('medv', axis=1)
        y = self.df['medv']

        # Guardar nombres de features
        self.feature_names = list(X.columns)
        logger.info(f"Features identificadas: {len(self.feature_names)} variables")

        # Detectar outliers usando método IQR
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (y >= lower_bound) & (y <= upper_bound)
        outliers_count = (~outliers_mask).sum()
        logger.info(f"Outliers detectados: {outliers_count} de {len(y)} observaciones ({outliers_count/len(y)*100:.1f}%)")

        # Decidir si remover outliers extremos
        if outliers_count > len(y) * 0.05:  # Más del 5%
            logger.info("Manteniendo outliers para preservar la robustez del modelo")

        # División de datos: 60% train, 20% validation, 20% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=None
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state  # 0.25 * 0.8 = 0.2
        )

        logger.info(f"División de datos completada:")
        logger.info(f"  - Entrenamiento: {self.X_train.shape[0]} muestras")
        logger.info(f"  - Validación: {self.X_val.shape[0]} muestras")
        logger.info(f"  - Prueba: {self.X_test.shape[0]} muestras")

        # Guardar información sobre la división de datos
        splits = {
            'train_indices': self.X_train.index.tolist(),
            'val_indices': self.X_val.index.tolist(),
            'test_indices': self.X_test.index.tolist(),
            'split_info': {
                'train_size': len(self.X_train),
                'val_size': len(self.X_val),
                'test_size': len(self.X_test),
                'total_size': len(X)
            }
        }

        with open('data/data_splits.json', 'w') as f:
            json.dump(splits, f, indent=2)

        logger.info("Información de división guardada en data/data_splits.json")
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def define_models(self):
        """Definir modelos a entrenar con sus respectivos hiperparámetros"""

        self.model_configs = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {},
                'description': 'Regresión lineal simple'
            },
            'ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                'description': 'Regresión Ridge con regularización L2'
            },
            'lasso': {
                'model': Lasso(random_state=self.random_state, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                },
                'description': 'Regresión Lasso con regularización L1'
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                },
                'description': 'Random Forest con ensamble de árboles'
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5]
                },
                'description': 'Gradient Boosting con optimización secuencial'
            }
        }

        logger.info(f"Configuración de modelos completada. Modelos disponibles:")
        for name, config in self.model_configs.items():
            logger.info(f"  - {name}: {config['description']}")

    def train_model(self, model_name, use_scaling=True):
        """Entrenar un modelo específico con validación cruzada y optimización de hiperparámetros"""

        logger.info(f"Entrenando modelo: {model_name}")

        config = self.model_configs[model_name]
        base_model = config['model']
        param_grid = config['params']

        # Crear pipeline con escalado opcional
        if use_scaling:
            pipeline_steps = [
                ('scaler', RobustScaler()),  # Robusto a outliers
                ('model', base_model)
            ]
            # Ajustar nombres de parámetros para el pipeline
            param_grid = {f'model__{k}': v for k, v in param_grid.items()}
        else:
            pipeline_steps = [('model', base_model)]
            param_grid = {f'model__{k}': v for k, v in param_grid.items()}

        pipeline = Pipeline(pipeline_steps)

        # Optimización de hiperparámetros con validación cruzada
        if param_grid:
            logger.info(f"  Optimizando hiperparámetros para {model_name}...")
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"  Mejores parámetros encontrados: {best_params}")

        else:
            # Modelo sin hiperparámetros para optimizar
            logger.info(f"  Entrenando {model_name} con parámetros por defecto...")
            pipeline.fit(self.X_train, self.y_train)
            best_model = pipeline
            best_params = {}

        # Generar predicciones para evaluación
        y_train_pred = best_model.predict(self.X_train)
        y_val_pred = best_model.predict(self.X_val)

        # Calcular métricas de rendimiento
        metrics = {
            'train_r2': r2_score(self.y_train, y_train_pred),
            'val_r2': r2_score(self.y_val, y_val_pred),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'val_mae': mean_absolute_error(self.y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
            'overfit_ratio': r2_score(self.y_train, y_train_pred) - r2_score(self.y_val, y_val_pred)
        }

        # Validación cruzada para evaluar estabilidad
        cv_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()

        # Almacenar resultados del modelo
        self.models[model_name] = {
            'model': best_model,
            'metrics': metrics,
            'params': best_params,
            'feature_names': self.feature_names,
            'description': config['description']
        }

        logger.info(f"  Entrenamiento completado. R² validación: {metrics['val_r2']:.4f}, MAE validación: {metrics['val_mae']:.4f}")

        # Advertir sobre posible sobreajuste
        if metrics['overfit_ratio'] > 0.1:
            logger.warning(f"  Posible sobreajuste detectado en {model_name} (diferencia R²: {metrics['overfit_ratio']:.4f})")

        return best_model, metrics

    def train_all_models(self):
        """Entrenar todos los modelos configurados y comparar resultados"""
        logger.info("Iniciando entrenamiento de todos los modelos configurados...")

        results = {}
        training_times = {}

        for model_name in self.model_configs.keys():
            try:
                start_time = datetime.now()
                model, metrics = self.train_model(model_name)
                end_time = datetime.now()

                training_time = (end_time - start_time).total_seconds()
                training_times[model_name] = training_time
                results[model_name] = metrics

                logger.info(f"Modelo {model_name} entrenado en {training_time:.2f} segundos")

            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {e}")
                continue

        # Seleccionar el mejor modelo basado en R² de validación
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['val_r2'])
            self.best_model = self.models[best_model_name]['model']
            self.best_model_name = best_model_name

            logger.info(f"Mejor modelo seleccionado: {best_model_name}")
            logger.info(f"  - R² validación: {results[best_model_name]['val_r2']:.4f}")
            logger.info(f"  - MAE validación: {results[best_model_name]['val_mae']:.4f}")
            logger.info(f"  - Tiempo entrenamiento: {training_times[best_model_name]:.2f}s")
        else:
            logger.error("No se pudo entrenar ningún modelo exitosamente")
            raise RuntimeError("Fallo en el entrenamiento de todos los modelos")

        return results

    def evaluate_best_model(self):
        """Evaluación final del mejor modelo en el conjunto de prueba"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado. Ejecuta train_all_models() primero.")

        logger.info("Iniciando evaluación final en el conjunto de prueba...")

        # Predicciones en el conjunto de prueba
        y_test_pred = self.best_model.predict(self.X_test)

        # Cálculo de métricas finales
        final_metrics = {
            'test_r2': float(r2_score(self.y_test, y_test_pred)),
            'test_mae': float(mean_absolute_error(self.y_test, y_test_pred)),
            'test_rmse': float(np.sqrt(mean_squared_error(self.y_test, y_test_pred))),
            'model_name': self.best_model_name,
            'training_timestamp': datetime.now().isoformat(),
            'data_shape': [int(x) for x in self.df.shape],  # Convertir a int nativo
            'feature_count': len(self.feature_names),
            'test_predictions_sample': [float(x) for x in y_test_pred[:10]],  # Convertir a float nativo
            'test_actual_sample': [float(x) for x in self.y_test.iloc[:10]]   # Convertir a float nativo
        }

        # Análisis de residuos básico
        residuals = self.y_test - y_test_pred
        final_metrics['residuals_mean'] = float(np.mean(residuals))
        final_metrics['residuals_std'] = float(np.std(residuals))

        # Guardar métricas finales
        with open('metrics/final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)

        logger.info("Evaluación final completada. Resultados:")
        logger.info(f"  R² en conjunto de prueba: {final_metrics['test_r2']:.4f}")
        logger.info(f"  MAE en conjunto de prueba: {final_metrics['test_mae']:.4f}")
        logger.info(f"  RMSE en conjunto de prueba: {final_metrics['test_rmse']:.4f}")
        logger.info(f"  Media de residuos: {final_metrics['residuals_mean']:.4f}")
        logger.info(f"  Desviación estándar de residuos: {final_metrics['residuals_std']:.4f}")

        return final_metrics

    def save_model(self, model_path='models/best_model.pkl'):
        """Guardar el mejor modelo entrenado junto con sus metadatos"""
        if self.best_model is None:
            raise ValueError("No hay modelo entrenado para guardar.")

        logger.info(f"Guardando modelo en: {model_path}")

        # Crear metadatos completos del modelo
        model_metadata = {
            'model_name': self.best_model_name,
            'model_description': self.models[self.best_model_name]['description'],
            'feature_names': self.feature_names,
            'training_timestamp': datetime.now().isoformat(),
            'validation_metrics': self.models[self.best_model_name]['metrics'],
            'hyperparameters': self.models[self.best_model_name]['params'],
            'sklearn_version': '1.0.0+',  # Versión aproximada
            'data_url': self.data_url,
            'data_shape': self.df.shape,
            'preprocessing_info': {
                'scaling_applied': True,
                'outliers_handled': True,
                'train_test_split_ratio': '60-20-20'
            }
        }

        # Crear paquete completo del modelo
        model_package = {
            'model': self.best_model,
            'metadata': model_metadata,
            'feature_names': self.feature_names
        }

        # Guardar modelo usando joblib para mejor compatibilidad
        joblib.dump(model_package, model_path)

        # Guardar metadatos por separado para facilitar acceso
        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)

        logger.info("Modelo guardado exitosamente")
        logger.info("Metadatos guardados en models/model_metadata.json")

        return model_path

    def generate_training_report(self):
        """Generar un reporte completo del proceso de entrenamiento"""

        logger.info("Generando reporte completo de entrenamiento...")

        report = {
            'training_summary': {
                'timestamp': datetime.now().isoformat(),
                'best_model': self.best_model_name,
                'best_model_description': self.models[self.best_model_name]['description'],
                'total_models_trained': len(self.models),
                'data_shape': [int(x) for x in self.df.shape],  # Convertir a int nativo
                'feature_names': self.feature_names
            },
            'models_comparison': {},
            'data_info': {
                'source_url': self.data_url,
                'total_samples': int(self.df.shape[0]),  # Convertir a int nativo
                'total_features': len(self.feature_names),
                'target_variable': 'medv',
                'missing_values': int(self.df.isnull().sum().sum())  # Convertir a int nativo
            }
        }

        # Comparación detallada de modelos
        for model_name, model_info in self.models.items():
            report['models_comparison'][model_name] = {
                'description': model_info['description'],
                'metrics': model_info['metrics'],
                'hyperparameters': model_info['params']
            }

        # Guardar reporte
        with open('metrics/training_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Mostrar resumen en consola
        logger.info("Resumen del entrenamiento:")
        logger.info(f"  Modelos entrenados: {len(self.models)}")
        logger.info(f"  Mejor modelo: {self.best_model_name}")
        logger.info("  Comparación de modelos (R² validación):")

        # Ordenar modelos por rendimiento
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: x[1]['metrics']['val_r2'],
            reverse=True
        )

        for i, (model_name, model_info) in enumerate(sorted_models, 1):
            metrics = model_info['metrics']
            logger.info(f"    {i}. {model_name}: R²={metrics['val_r2']:.4f}, MAE={metrics['val_mae']:.4f}")

        logger.info("Reporte completo guardado en metrics/training_report.json")
        return report

def main():
    """Función principal del pipeline de entrenamiento"""

    parser = argparse.ArgumentParser(description='Pipeline de Entrenamiento - Boston Housing')
    parser.add_argument('--data-url', type=str, help='URL del dataset')
    parser.add_argument('--random-state', type=int, default=42, help='Semilla aleatoria para reproducibilidad')
    parser.add_argument('--model-path', type=str, default='models/best_model.pkl', help='Ruta para guardar el modelo')

    args = parser.parse_args()

    logger.info("INICIANDO PIPELINE DE ENTRENAMIENTO - BOSTON HOUSING PRICE PREDICTION")
    logger.info("=" * 80)

    try:
        # Inicializar el entrenador
        trainer = HousingPriceTrainer(
            data_url=args.data_url,
            random_state=args.random_state
        )

        # Ejecutar pipeline completo
        logger.info("Fase 1: Carga y validación de datos")
        trainer.load_data()

        logger.info("Fase 2: Preprocesamiento de datos")
        trainer.preprocess_data()

        logger.info("Fase 3: Configuración de modelos")
        trainer.define_models()

        logger.info("Fase 4: Entrenamiento de modelos")
        trainer.train_all_models()

        logger.info("Fase 5: Evaluación final")
        trainer.evaluate_best_model()

        logger.info("Fase 6: Guardado de modelo")
        trainer.save_model(args.model_path)

        logger.info("Fase 7: Generación de reportes")
        trainer.generate_training_report()

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("Archivos generados:")
        logger.info(f"  - Modelo: {args.model_path}")
        logger.info("  - Métricas: metrics/final_metrics.json")
        logger.info("  - Reporte: metrics/training_report.json")
        logger.info("  - Log: training.logs")

    except Exception as e:
        logger.error(f"Error crítico en el pipeline: {e}")
        logger.error("Revisa los logs para más detalles")
        sys.exit(1)

if __name__ == "__main__":
    main()