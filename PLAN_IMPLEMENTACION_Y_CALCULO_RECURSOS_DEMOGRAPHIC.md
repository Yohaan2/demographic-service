# Plan de implementación y cálculo de recursos
## Servicio de analítica demográfica para 500 cámaras

**Proyecto:** `demographic-service`  
**Alcance:** exclusivamente el módulo de inferencia demográfica, tracking y agregación temporal implementado en este repositorio.  
**Versión del análisis:** 1.0  
**Fecha:** 2026-09-09

---

## 1. Resumen ejecutivo

La aplicación no recibe streams RTSP ni decodifica vídeo continuo. Su contrato actual es HTTP:

```text
Una petición POST /api/v1/analyze = un frame JPEG/PNG de una cámara
```

Por lo tanto, el dimensionamiento depende principalmente de:

1. La frecuencia de frames enviada por cámara.
2. El número promedio y máximo de rostros por frame.
3. El modelo de edad/género utilizado.
4. La latencia real medida en el hardware objetivo.
5. La forma en que se distribuyen las cámaras entre procesos y nodos.

### Recomendación de producción

Para una primera puesta en producción realista se recomienda utilizar el perfil **InsightFace + SCRFD ONNX sobre CPU**, porque es el perfil efectivamente preparado por el `Dockerfile` y por `requirements.insightface.txt`.

Para **500 cámaras enviando 1 frame por segundo**, la capacidad de referencia es:

| Recurso | Capacidad base por nodo | Recomendación productiva |
|---|---:|---:|
| Nodos de inferencia | 2 nodos activos | 3 nodos para N+1 |
| CPU por nodo | 24 vCPU | 24 vCPU |
| RAM por nodo | 32 GB | 32 GB |
| Red por nodo | NIC 2.5 GbE mínimo | 10 GbE recomendado |
| GPU | No requerida para InsightFace | No requerida |
| Disco local | 100 GB SSD | 200 GB SSD |

La recomendación de tres nodos no agrega capacidad de otra aplicación: es redundancia para que, ante la caída de un nodo, los dos restantes puedan continuar atendiendo las 500 cámaras.

> Estos valores son un dimensionamiento inicial de producción, no una sustitución de una prueba de carga. El repositorio no contiene benchmarks con los modelos reales ni los pesos dentro de `models/weights`; los valores finales deben cerrarse con una prueba de aceptación sobre el hardware seleccionado.

---

## 2. Alcance y exclusiones

### Incluido

- Recepción HTTP multipart del frame.
- Lectura y decodificación JPEG/PNG mediante OpenCV.
- Detección facial SCRFD a entrada fija de 640 x 640.
- Tracking ByteTrack independiente por `camera_id`.
- Alineamiento y recorte facial.
- Estimación de edad y género.
- Agregación temporal por ventana deslizante.
- Estado en memoria de trackers y agregaciones.
- Nginx como proxy de entrada.
- Recursos de CPU, RAM, red, almacenamiento temporal y capacidad de procesos necesarios para este módulo.

### Excluido deliberadamente

- Prometheus, node-exporter, cAdvisor, Promtail, OpenTelemetry y cualquier plataforma de monitorización.
- Bases de datos, data lakes, almacenamiento histórico y analítica posterior.
- Grabación o almacenamiento de vídeo.
- Origen de cámaras, gateways RTSP, transcodificación y extracción de frames.
- Frontend, salvo el consumo marginal de Nginx.
- Sistemas externos que llamen a esta API.

La instrumentación existente se considera un consumo marginal frente a la inferencia. No se ha dimensionado una plataforma de observabilidad.

---

## 3. Análisis técnico del módulo actual

### 3.1 Flujo por petición

```text
Cliente de cámara
    -> Nginx
    -> FastAPI POST /api/v1/analyze
    -> lectura multipart y cv2.imdecode
    -> SCRFD 640x640
    -> filtros de detecciones
    -> ByteTrack por cámara
    -> crop/alineamiento por rostro
    -> InsightFace genderage o MiVOLO, uno por rostro
    -> ventana temporal de agregación
    -> respuesta JSON
```

La función `process_frame` es síncrona. Aunque el endpoint está declarado como `async`, la detección, el tracking y la inferencia se ejecutan dentro del mismo worker de Python. Esto significa que aumentar las peticiones concurrentes no crea capacidad: únicamente aumenta la memoria ocupada y la cola de trabajo.

### 3.2 Componentes que consumen recursos

| Componente | Comportamiento actual | Impacto de dimensionamiento |
|---|---|---|
| Decodificación | Cada request mantiene los bytes y crea un `ndarray` BGR | RAM temporal y CPU proporcional a FPS y tamaño del frame |
| SCRFD | Una inferencia por frame a 640 x 640 | Principal coste fijo por cámara/frame |
| ByteTrack | Un tracker por cámara; asociaciones y Hungarian por frame | Bajo con pocos rostros; crece con rostros simultáneos |
| Crop/alineamiento | Un crop por rostro activo | CPU y memoria transitoria por número de rostros |
| Edad/género | Una inferencia por rostro activo y por frame | Principal coste variable; no hay batching |
| Agregación | Ventana de 10 muestras por `track_id` | RAM baja, pero el estado es local al proceso |
| Workers | Gunicorn puede crear varios procesos | Cada proceso carga sus propios modelos y registries |
| Backpressure | Límite actual de 500 requests activas | Es demasiado alto para proteger RAM si se reciben imágenes grandes |
---

## 4. Supuestos de cálculo

Los siguientes supuestos son explícitos para que el cálculo pueda ajustarse sin rehacer todo el documento.

| Parámetro | Valor de cálculo | Justificación |
|---|---:|---|
| Cámaras | 500 | Requerimiento objetivo |
| Frecuencia base | 1 FPS por cámara | Suficiente como punto de partida para demografía agregada; la API permite cambiarlo |
| Rostros promedio | 2 por frame | Escenario medio de retail/interior |
| Rostros de dimensionamiento | 4 por frame | Reserva para picos normales |
| Rostros máximos de prueba | 10 por frame | Escenario de estrés funcional |
| JPEG promedio | 150 KB | Imagen comprimida de cámara, debe medirse con las fuentes reales |
| JPEG de redimensionamiento | 250 KB | Valor conservador para red y buffers |
| Entrada del detector | 640 x 640 | Fijado en `SCRFDDetector` |
| Ventana de agregación | 10 muestras | Valor actual de `AGGREGATION_WINDOW` |
| TTL de estado | 300 segundos | Valor actual de `TRACKER_TTL` |
| Modelo recomendado | InsightFace genderage ONNX + SCRFD ONNX | Perfil realmente instalable en el Dockerfile actual |
| Disponibilidad | N+1 | La caída de un nodo no debe detener las 500 cámaras |

### 4.1 Sensibilidad por frecuencia

| Frecuencia | Requests/s | Frames/día | Con 2 rostros promedio: inferencias faciales/día |
|---:|---:|---:|---:|
| 0.5 FPS | 250 | 21.6 millones | 43.2 millones |
| **1 FPS** | **500** | **43.2 millones** | **86.4 millones** |
| 2 FPS | 1,000 | 86.4 millones | 172.8 millones |
| 5 FPS | 2,500 | 216 millones | 432 millones |

El cálculo principal usa 1 FPS. Si el requerimiento real es 2 FPS, la capacidad de CPU y red debe crecer aproximadamente de forma lineal; no debe asumirse que el mismo nodo lo soportará.

---

## 5. Cálculo de carga

### 5.1 Peticiones e inferencias

Para `C` cámaras y `F` FPS:

```text
requests_por_segundo = C x F
```

Con 500 cámaras y 1 FPS:

```text
500 x 1 = 500 requests/s
```

El detector se ejecuta una vez por request:

```text
detecciones_por_segundo = 500 inferencias SCRFD/s
```

La inferencia de edad/género depende de los rostros:

```text
inferencias_faciales/s = requests/s x rostros_promedio
                       = 500 x 2
                       = 1,000 inferencias/s
```

Para el escenario de reserva de 4 rostros por frame:

```text
500 x 4 = 2,000 inferencias faciales/s
```

### 5.2 Red

Con un frame medio de 150 KB:

```text
500 x 150 KB = 75,000 KB/s ≈ 75 MB/s ≈ 600 Mb/s
```

Con el frame conservador de 250 KB:

```text
500 x 250 KB = 125 MB/s ≈ 1,000 Mb/s
```

Sumando multipart, TCP/IP, retransmisiones y margen operativo:

- **Mínimo funcional:** 2.5 GbE por nodo de inferencia.
- **Recomendado:** 10 GbE, especialmente si un nodo puede recibir temporalmente la carga del otro.
- **No se requiere salida equivalente:** la respuesta es JSON pequeño; el tráfico de salida es marginal.

Si se distribuyen 250 cámaras por nodo, cada nodo recibe aproximadamente 300–500 Mb/s en el escenario base, pero debe poder absorber 500 cámaras durante una conmutación por fallo.

### 5.3 CPU

El repositorio no proporciona una medición de latencia sobre el hardware de producción. Por eso se usa un presupuesto operativo que debe verificarse:

- 500 ejecuciones SCRFD/s.
- Hasta 2,000 inferencias faciales/s en el escenario de reserva.
- Decodificación, resize, color conversion, tracking, serialización y kernel overhead.
- Margen de 40–50% para picos, GC, variación de imágenes y degradación sostenida.

Para el perfil InsightFace CPU se propone como capacidad de compra:

```text
24 vCPU por nodo
32 GB RAM por nodo
```

La cifra no significa que el proceso deba usar 24 hilos por inferencia. Al contrario, se debe limitar el paralelismo interno de ONNX Runtime y repartir la carga entre workers para evitar oversubscription.

#### Configuración inicial sugerida por nodo

- 4 workers Gunicorn como punto inicial.
- 4–6 hilos de CPU asignados por worker entre ONNX Runtime y operaciones auxiliares.
- Límite de 64–128 requests en vuelo por nodo, no 500.
- Ajuste posterior con benchmark; si el p95 supera el SLO, escalar horizontalmente antes de elevar la cola.

La cifra de 24 vCPU es prudente para el escenario de 250 cámaras por nodo y permite absorber 500 cámaras con dos nodos durante una contingencia. Para 2 FPS, el dimensionamiento recomendado pasa a **48 vCPU por nodo**, o se duplica el número de nodos de 24 vCPU, tras validación.

### 5.4 RAM

Estimación por nodo para InsightFace:

| Concepto | Estimación |
|---|---:|
| Sistema base, Nginx y contenedor | 1–2 GB |
| 4 procesos Python + FastAPI/OpenCV/SciPy | 2–4 GB |
| Modelos y sesiones ONNX replicadas por worker | 1–4 GB |
| Trackers de 250–500 cámaras | < 0.5 GB en escenario normal |
| Agregación: hasta 10 muestras por track | < 0.5 GB en escenario normal |
| Buffers de requests e imágenes decodificadas | 1–4 GB, depende de concurrencia |
| Picos, allocator y margen | 6–10 GB |
| **Total recomendado** | **32 GB** |

La memoria de trackers no debería ser el factor dominante. El riesgo real es permitir demasiadas peticiones simultáneas con imágenes grandes. Por eso el control de concurrencia es obligatorio.

Con el límite actual de 500 requests y frames de 250 KB, solo los bytes comprimidos podrían ocupar aproximadamente 125 MB; los `ndarray` decodificados, copias multipart y estructuras internas pueden multiplicar ese valor. La aplicación no debe usar `MAX_PENDING_REQUESTS=500` como garantía de seguridad de memoria.

### 5.5 Disco

El módulo no guarda vídeo ni frames. Se requiere almacenamiento para:

- Imagen del contenedor.
- Pesos de SCRFD y genderage.
- Logs locales temporales, si se habilitan.
- Capas de Docker y espacio operativo.

Recomendación:

- **100 GB SSD:** mínimo práctico por nodo.
- **200 GB SSD:** recomendado para actualizaciones, rollback e imágenes duplicadas.
- No se presupone almacenamiento de imágenes procesadas.

### 5.6 Estado en memoria

Con 500 cámaras, un tracker por cámara es coherente con `MAX_TRACKERS=500`. La agregación, sin embargo, se indexa por `track_id` global y no por cámara. Para producción debe garantizarse la unicidad y afinidad del estado, o cambiar la clave a `(camera_id, track_id)`.

La ventana de 10 muestras por track es pequeña. Incluso con 10 rostros por cámara, el estado de agregación no justifica una base de datos: debe mantenerse en memoria local si se resuelve la afinidad de cámara.

---

## 6. Arquitectura productiva propuesta

```text
500 fuentes de cámara / extractores de frame
                |
        Balanceador L4/L7
                |
     +----------+----------+
     |                     |
  Nodo A                Nodo B
  24 vCPU               24 vCPU
  32 GB RAM              32 GB RAM
  250 cámaras            250 cámaras
     |                     |
     +----------+----------+
                |
       Nodo C de reserva
       24 vCPU / 32 GB RAM
```

### Distribución del estado

La solución preferida es mantener afinidad por `camera_id`:

1. El balanceador aplica consistent hashing por `camera_id`, o
2. El cliente mantiene una ruta estable por cámara, o
3. La aplicación evoluciona a un router de cámaras y workers internos.

No se recomienda balanceo round-robin sin afinidad, porque el tracker y la agregación viven en memoria del proceso y perderán continuidad al cambiar de worker/nodo.

### Selección de procesos

Hay dos opciones válidas, que deben decidirse antes del despliegue:

- **Opción A — un proceso por partición de cámaras:** menos duplicación de modelos y estado más sencillo; requiere controlar la disponibilidad del proceso.
- **Opción B — varios workers por nodo con afinidad de cámara:** mejor aislamiento ante bloqueo, pero cada worker carga modelos y necesita afinidad dentro del nodo.

La configuración actual de cuatro workers puede utilizarse solo después de resolver la afinidad. No se debe aumentar `WORKERS` indiscriminadamente: cada proceso duplica modelos, threads nativos y estado.

---

## 7. Plan de implementación

### Fase 1 — Congelar el perfil de producción

1. Seleccionar explícitamente `MODEL_TYPE=insightface` para el dimensionamiento recomendado.
2. Distribuir y versionar los pesos reales:
   - `scrfd_2.5g_bnkps.onnx`.
   - `genderage.onnx`.
3. Hacer que el arranque falle si falta un peso productivo; no aceptar silenciosamente Haar Cascade ni SHA-256.
4. Corregir la discrepancia entre los valores por defecto de `config.py`, `.env.example` y el `Dockerfile`.
5. Confirmar resolución, compresión media y FPS reales de las 500 fuentes.

### Fase 2 — Control de concurrencia y afinidad

1. Reemplazar el límite conceptual de 500 requests por una capacidad calculada por nodo, inicialmente 64–128 requests en vuelo.
2. Usar una cola acotada o semáforo de inferencia para impedir que la memoria crezca sin límite.
3. Definir afinidad por `camera_id` para que tracker y agregación permanezcan en el mismo proceso lógico.
4. Cambiar la clave de agregación a `(camera_id, track_id)` o garantizar formalmente que `track_id` nunca colisione entre cámaras.
5. Mantener la política de rechazo rápido cuando la cola exceda la capacidad; descartar frames antiguos es preferible a acumular latencia para una aplicación de analítica en tiempo real.

### Fase 3 — Optimización de inferencia

1. Ejecutar benchmark de SCRFD y genderage con los pesos finales.
2. Ajustar `intra_op_num_threads` e `inter_op_num_threads` para evitar oversubscription entre Gunicorn y ONNX Runtime.
3. Evaluar batching de crops faciales para frames con múltiples rostros.
4. Reutilizar buffers cuando el benchmark demuestre que las copias de imagen son relevantes.
5. Definir un máximo de rostros procesables por frame y una política explícita para excedentes.

### Fase 4 — Despliegue escalable

1. Construir la imagen únicamente con el perfil seleccionado.
2. Desplegar dos nodos activos con 250 cámaras lógicas por nodo.
3. Añadir un tercer nodo de reserva para N+1.
4. Configurar health checks de disponibilidad del proceso y de carga de modelos.
5. Validar que un reinicio no deje a la cámara en una cola indefinida; el cliente debe reintentar con backoff y el servicio debe responder con 429/503 de forma controlada.

### Fase 5 — Prueba de aceptación

1. Iniciar con 50 cámaras y 1 FPS.
2. Escalar a 250 cámaras por nodo.
3. Ejecutar 500 cámaras durante al menos 60 minutos.
4. Repetir con 2 FPS si esa es la frecuencia objetivo real.
5. Incluir escenarios de 0, 2, 4 y 10 rostros por frame.
6. Simular la caída de un nodo y comprobar que los dos nodos restantes sostienen la carga.

---

## 8. Criterios de aceptación recomendados

Los siguientes criterios deben medirse únicamente sobre la API de este módulo:

| Métrica | Objetivo inicial |
|---|---:|
| Requests procesadas | 500/s sostenidas a 1 FPS total |
| Errores 5xx | < 0.1% durante prueba estable |
| Rechazos 429 | 0 en carga nominal; solo en sobrecarga controlada |
| P95 de API | ≤ 1 segundo en carga nominal |
| P99 de API | ≤ 2 segundos en carga nominal |
| Frames atrasados | No acumular más de 2 segundos de antigüedad |
| Uso RAM | < 75% durante carga estable |
| CPU | < 70–75% promedio, con picos controlados |
| Estado de tracking | Continuidad de cámara tras requests consecutivas |
| Fallbacks | 0 inferencias Haar/SHA-256 en producción |

El objetivo de latencia debe ajustarse a la latencia medida del hardware. Para una aplicación que recibe un frame por segundo, no es útil aceptar 500 peticiones por segundo si el sistema responde frames con varios segundos de retraso.

---

## 9. Escenarios de capacidad

| Escenario | Cámaras | FPS | Requests/s | Rostros de reserva | Infraestructura sugerida |
|---|---:|---:|---:|---:|---|
| Piloto | 50 | 1 | 50 | 4/frame | 8 vCPU, 16 GB RAM |
| Base productiva | 500 | 1 | 500 | 4/frame | 2 x 24 vCPU, 32 GB RAM + 1 reserva |
| Alta frecuencia | 500 | 2 | 1,000 | 4/frame | 4 x 24 vCPU, 32 GB RAM o benchmark GPU |
| Estrés | 500 | 1 | 500 | 10/frame | Requiere batching y prueba específica |

La fila de alta frecuencia no debe aprobarse por extrapolación ciega. El coste facial crece con el número de rostros y la implementación actual ejecuta una llamada de modelo por rostro.

---

## 11. Conclusión

Para el módulo existente, con 500 cámaras a 1 FPS y usando los pesos reales de **SCRFD + InsightFace genderage ONNX**, la recomendación operativa es:

- **2 nodos activos de 24 vCPU y 32 GB RAM**, cada uno con capacidad para 250 cámaras.
- **1 nodo adicional de igual tamaño** para tolerancia N+1.
- **2.5 GbE mínimo y 10 GbE recomendado** por nodo.
- **100 GB SSD mínimo, 200 GB recomendado**.
- **Sin GPU obligatoria** para el perfil InsightFace, sujeto al benchmark.
- **Afinidad por cámara, cola acotada y límite de concurrencia real**, no 500 inferencias simultáneas.
- **Cero fallbacks** en producción.

El dato que debe cerrar definitivamente la compra es el benchmark con los pesos reales, imagen real y distribución real de rostros. El cálculo presentado es conservador, trazable y está limitado exclusivamente al módulo de inferencia demográfica solicitado.
