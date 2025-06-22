# Multi-stage build para optimizar tamaño y seguridad
FROM python:3.12-slim as builder

# Metadatos
LABEL maintainer="ML DevOps Engineer"
LABEL description="Boston Housing Price Prediction API"
LABEL version="1.0.0"

# Variables de entorno para el build
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema para compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear directorio de trabajo
WORKDIR /build

# Copiar requirements y instalar dependencias Python
COPY requirements-api.txt .
RUN pip install --no-cache-dir --user --upgrade pip \
    && pip install --no-cache-dir --user -r requirements-api.txt

# Imagen final (runtime)
FROM python:3.12-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV PATH=/home/apiuser/.local/bin:$PATH

# Instalar dependencias mínimas del runtime
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear usuario no-root para seguridad
RUN groupadd -r apiuser && useradd -r -g apiuser apiuser

# Crear estructura de directorios
RUN mkdir -p /app/src/api /app/models /app/logs /app/metrics \
    && chown -R apiuser:apiuser /app

# Copiar dependencias desde stage builder
COPY --from=builder /root/.local /home/apiuser/.local

# Establecer directorio de trabajo
WORKDIR /app

# Copiar código fuente
COPY src/api/ ./src/api/
COPY models/ ./models/
COPY .env.docker ./.env

# Copiar scripts de Docker
COPY docker/start.sh ./start.sh
COPY docker/healthcheck.sh ./healthcheck.sh

# Hacer scripts ejecutables y ajustar permisos
RUN chmod +x start.sh healthcheck.sh \
    && chown -R apiuser:apiuser /app

# Cambiar a usuario no-root
USER apiuser

# Exponer puerto
EXPOSE 8000

# Health check personalizado
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD ./healthcheck.sh

# Comando por defecto
CMD ["./start.sh"]