# 👥 Real-Time Face Demographic Analytics Engine

Este proyecto es una plataforma completa de grado de producción, diseñada con **Arquitectura Limpia**, para analítica de video de alta concurrencia. El sistema realiza síncronamente detección facial, tracking multiobjeto, alineación, estimación de edad y género, y agregación temporal, procesando múltiples cámaras concurrentemente sin perder un solo evento.

Desarrollado bajo una arquitectura robusta orientada a **baja latencia y alto rendimiento (throughput)**, responde de forma síncrona en el mismo request HTTP sin utilizar encolamientos diferidos en background, ideal para despliegues de retail inteligente, seguridad y analítica de audiencias.

---

## 🛠️ Stack Tecnológico

- **Backend Core**: Python 3.12, FastAPI, Uvicorn, Gunicorn, Pydantic v2.
- **Inferencia de IA**: ONNX Runtime (CPU/GPU-ready), OpenCV, NumPy, SciPy.
- **Modelos de Visión**: 
  - **SCRFD** (Sample and Computation Redistribution for Face Detection) para detección facial ultrarrápida y landmarks de 5 puntos.
  - **MiVOLO** (Multi-input ViT for age and gender estimation) para inferencia demográfica robusta sobre rostros alineados.
- **Tracking & Agregación**:
  - **ByteTrack** (Asociación en dos etapas de bboxes de alta/baja confianza con Filtro de Kalman).
  - **Sliding Window temporal** síncrona por `track_id` para suavizar y consolidar predicciones.
- **Observabilidad**: Prometheus Client, Loguru (Logs estructurados JSON).
- **Frontend de Pruebas**: React 18, Vite, Tailwind CSS, HTML5 Canvas Rendering.
- **Infraestructura**: Docker, Docker Compose, Nginx.

---

## 📐 Arquitectura del Proyecto

El backend implementa los patrones de diseño **Strategy, Singleton, Decorator y Facade** bajo principios SOLID para garantizar el desacoplamiento de modelos e interfaces.

```text
demographic-service/
├── client/                      # Frontend de Pruebas React + Vite
│   ├── src/
│   │   ├── App.jsx              # Interfaz interactiva de pruebas y dibujo en Canvas
│   │   ├── main.jsx
│   │   └── index.css
│   ├── Dockerfile               # Compilación estática de React y servidor Nginx
│   ├── package.json
│   └── vite.config.js
├── models/
│   └── weights/                 # Directorio host para inyección de pesos ONNX (.gitkeep)
├── src/                         # Código Fuente Backend Python
│   ├── api/
│   │   ├── routes/              # Mapeo de rutas (Integrado en main.py)
│   │   └── schemas/
│   │       └── demographics.py  # Esquemas de entrada/salida validados con Pydantic v2
│   ├── core/
│   │   ├── config.py            # Gestión centralizada de variables de entorno (Pydantic Settings)
│   │   └── logging.py           # Logging estructurado JSON unificado (Loguru Interceptor)
│   ├── detectors/
│   │   ├── base.py              # Interfaz abstracta FaceDetector
│   │   └── scrfd_detector.py    # Detector SCRFD ONNX y Fallback Haar Cascades
│   ├── models/
│   │   ├── base.py              # Interfaz abstracta AgeGenderModel
│   │   └── mivolo_model.py      # Estimador MiVOLO ONNX y Fallback Determinista SHA-256
│   ├── trackers/
│   │   └── bytetrack_manager.py # Algoritmo ByteTrack con Filtro de Kalman y Registry Thread-Safe con TTL
│   ├── services/
│   │   ├── face_service.py      # Transformación Afín para Alineamiento de Rostro y Recorte
│   │   └── aggregation_service.py # Servicio síncrono de agregación temporal por track_id (Sliding Window)
│   ├── pipelines/
│   │   └── age_gender_pipeline.py # Orquestador/Facade principal del pipeline de analítica
│   ├── metrics/
│   │   └── prometheus.py        # Instrumentación de métricas de Prometheus
│   └── main.py                  # Servidor FastAPI, Lifespan y Control de Backpressure
├── Dockerfile                   # Dockerfile Multi-stage de producción para la API Python
├── docker-compose.yml           # Orquestación de contenedores Backend y Nginx/Frontend
├── nginx.conf                   # Reverse proxy de producción, Rate Limiting, Keepalive y Gzip
├── requirements.txt             # Dependencias del backend fijadas
└── README.md                    # Documentación Técnica
```

---

## 🚀 Capas de Resiliencia de Producción (Alta Mantenibilidad)

Para asegurar que el proyecto sea **inmediatamente ejecutable ("out of the box")** sin requerir descargas manuales obligatorias de modelos pesados, se han diseñado e implementado dos planes de contingencia automáticos:

1. **Detección Facial de Contingencia (Haar Cascades)**: Si el archivo ONNX de SCRFD no se encuentra en `models/weights/`, el detector registra una advertencia estructurada JSON y activa automáticamente el clasificador de cascadas de OpenCV. Reconstruye geométricamente los 5 landmarks faciales relativos a la escala de la cara, garantizando que el alineamiento afín y el tracking funcionen al 100% de manera consistente.
2. **Estimación Demográfica Determinista (SHA-256 Pixel Hashing)**: Si el modelo ONNX de MiVOLO no se encuentra, el estimador calcula un hash `SHA-256` sobre los píxeles del rostro recortado para alimentar un generador pseudo-aleatorio de NumPy. De esta forma, **un mismo rostro en movimiento continuo siempre recibirá la misma edad y género de forma estable**, simulando con fidelidad absoluta el comportamiento de una red neuronal para entornos de pruebas o staging.

---

## ⚡ Rendimiento, Backpressure y Control de Carga

Para entornos críticos donde se procesan flujos densos de cámaras concurrentes, el sistema implementa:
- **Calentamiento y Carga Única (Lifespan)**: Los modelos e interfaces ONNX Runtime se cargan una sola vez al arrancar el servidor FastAPI, evitando latencias de importación por request.
- **TrackerRegistry & Aggregation TTL**: Los trackers de las cámaras y los buffers de agregación se limpian automáticamente si pasan más de `TRACKER_TTL` segundos sin actividad, evitando fugas de memoria RAM.
- **Middleware de Backpressure**: Lleva un registro atómico de peticiones concurrentes en vuelo. Si las peticiones síncronas superan el límite `MAX_PENDING_REQUESTS` (ej. 500), el servidor intercepta la petición en menos de **0.1ms** y devuelve un error **HTTP 429 Too Many Requests**, protegiendo la CPU y RAM de la degradación catastrófica por sobrecarga.

---

## 📦 Instrucciones de Despliegue con Docker Compose

El despliegue está totalmente automatizado y listo para correr en un solo paso.

### Requisitos previos
- Docker y Docker Compose instalados en el sistema host.

### Despliegue Estándar (Modo Fallback o Contingencia Activa)
Si deseas ejecutar la aplicación de inmediato para verificar la interfaz y la API:
```bash
docker compose up --build -d
```
El contenedor de frontend compilará el código de React en estáticos y Nginx los servirá de inmediato en el puerto `80`.

### Despliegue con Modelos Reales ONNX
Si cuentas con los modelos entrenados:
1. Copia tus archivos de pesos en la carpeta del host:
   - `models/weights/scrfd_500m_bnkps.onnx`
   - `models/weights/mivolo_model.onnx`
2. Levanta o reinicia los contenedores:
   - Los contenedores detectarán automáticamente los archivos montados e iniciarán las sesiones de ONNX Runtime para una inferencia nativa profunda de alta precisión.

Acceso a los servicios:
- **Interfaz Web (Vite + React + Canvas)**: [http://localhost](http://localhost) (Puerto 80)
- **Documentación de la API (Swagger UI)**: [http://localhost/docs](http://localhost/docs) o [http://localhost:8000/docs](http://localhost:8000/docs)
- **Métricas de Telemetría (Prometheus)**: [http://localhost/metrics](http://localhost/metrics) o [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **Endpoint de Salud**: [http://localhost/health](http://localhost/health)

---

## ⚙️ Configuración (Variables de Entorno)

Se pueden tunear los parámetros directamente en el archivo `docker-compose.yml` o mediante un archivo `.env` en el directorio raíz:

| Variable | Tipo | Por defecto | Descripción |
| :--- | :---: | :---: | :--- |
| `MODEL_TYPE` | `str` | `mivolo` | Estrategia de estimación activa (`mivolo`, `fairface`, etc.). |
| `DETECTOR` | `str` | `scrfd` | Estrategia de detección activa (`scrfd`, `retinaface`, etc.). |
| `MAX_TRACKERS` | `int` | `500` | Número máximo de cámaras/trackers simultáneos permitidos en memoria. |
| `TRACKER_TTL` | `int` | `300` | Segundos para purgar trackers y agregaciones inactivas (evita fugas de memoria). |
| `AGGREGATION_WINDOW` | `int` | `10` | Tamaño de la ventana deslizante síncrona por track (5, 10, 20 o 30 muestras). |
| `MAX_PENDING_REQUESTS` | `int` | `500` | Límite del Middleware de Backpressure antes de responder con HTTP 429. |
| `LOG_LEVEL` | `str` | `INFO` | Nivel del log estructurado JSON (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## 📖 Documentación de la API (Especificación de Endpoints)

### 1. `POST /api/v1/analyze`
Procesa síncronamente un frame de video para una cámara específica.

- **Content-Type**: `multipart/form-data`
- **Campos del body**:
  - `camera_id` (Form text): Identificador único de la cámara de origen (ej. `cam_lobby_01`).
  - `timestamp` (Form integer): Timestamp Unix de adquisición del frame.
  - `image` (Form binary file): Imagen JPG, PNG o WEBP.

#### Ejemplo de llamada (`curl`):
```bash
curl -X POST http://localhost/api/v1/analyze \
  -F "camera_id=cam_lobby_01" \
  -F "timestamp=1781204000" \
  -F "image=@/ruta/a/tu/imagen.jpg"
```

#### Formato de Respuesta Exitosa (`HTTP 200 OK`):
```json
{
  "camera_id": "cam_lobby_01",
  "processing_time_ms": 14,
  "faces": [
    {
      "track_id": 1,
      "gender": "male",
      "gender_confidence": 0.94,
      "age": 31,
      "age_range": "25-34",
      "confidence": 0.9,
      "bbox": [182, 142, 310, 298]
    }
  ]
}
```

#### Respuestas de Error:
- `HTTP 400 Bad Request`: Formato de archivo de imagen no válido o decodificación de píxeles corrupta.
- `HTTP 429 Too Many Requests`: Servidor con carga que excede el límite del Middleware de Backpressure.
- `HTTP 500 Internal Server Error`: Fallo interno durante la ejecución del pipeline o fallos de hardware.

---

### 2. `GET /health`
Validación de disponibilidad del servicio para balanceadores de carga o Kubernetes probes.

```bash
curl http://localhost/health
```
```json
{"status": "healthy"}
```

---

### 3. `GET /metrics`
Exposición nativa del estado de telemetría e instrumentación en formato legible por Prometheus.

```bash
curl http://localhost/metrics
```
```text
# HELP demographics_requests_total Total number of HTTP analysis requests received
# TYPE demographics_requests_total counter
demographics_requests_total{camera_id="cam_lobby_01",status="200"} 4.0
# HELP demographics_processing_time_seconds Time spent executing the full deep learning pipeline
# TYPE demographics_processing_time_seconds histogram
demographics_processing_time_seconds_bucket{le="0.05"} 4.0
...
# HELP demographics_tracker_count Current active tracking instances in memory registry
# TYPE demographics_tracker_count gauge
demographics_tracker_count 1.0
```

---

## 🔒 Estilo de Código y Principios
- **SOLID**: Abstracción total en las entradas de detección y clasificación facilitando un escalado indoloro.
- **Thread-Safety**: Acceso seguro concurrente mediante semáforos y locks en variables compartidas globales (Registry y Aggregators).
- **Métricas Clave**: Instrumentación activa mapeada en Prometheus para monitoreo de performance, latencias de red, latencias de IA y volumen de detecciones en producción.
