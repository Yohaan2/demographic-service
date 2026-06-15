import os
import sys
import time
import argparse
import cv2
import numpy as np
import requests

# ==============================================================================
# CONFIGURACIÓN: Coloca aquí las rutas locales o URLs de tus 4 imágenes.
# Ejemplos de rutas locales: "upload/shoot_1.jpeg", "C:/Users/tu_usuario/Desktop/img.jpg"
# Ejemplos de URLs: "https://sitio.com/imagen.jpg"
# Si dejas la lista con cadenas vacías, el script usará automáticamente la imagen local.
# ==============================================================================
IMAGE_PATHS = [
    "upload/shot-1.jpeg",  # Frame 1
    "upload/shot-2.jpeg",  # Frame 2
    "upload/shot-3.jpeg",  # Frame 3
    "upload/shot-4.jpeg",  # Frame 4
]

def generate_synthetic_face(output_path: str):
    """Generates a synthetic 640x640 image of a face using OpenCV shapes."""
    # Create gray canvas
    img = np.ones((640, 640, 3), dtype=np.uint8) * 128
    
    # Face shape (flesh-colored oval)
    cv2.circle(img, (320, 320), 180, (180, 210, 240), -1)
    
    # Eyes
    cv2.circle(img, (260, 260), 18, (50, 50, 50), -1)  # Left Eye
    cv2.circle(img, (380, 260), 18, (50, 50, 50), -1)  # Right Eye
    
    # Nose
    cv2.circle(img, (320, 320), 12, (100, 100, 200), -1)
    
    # Mouth (smile)
    cv2.ellipse(img, (320, 390), (60, 25), 0, 0, 180, (50, 50, 200), 4)
    
    cv2.imwrite(output_path, img)
    print(f"[+] Imagen sintética de rostro generada automáticamente en: {output_path}")

def test_demographics_pipeline(image_path: str, camera_id: str, host: str, port: int):
    url = f"http://{host}:{port}/api/v1/analyze"
    
    # Comprobar si hay rutas/URLs válidas configuradas (ahora dinámico sin requerir mínimo 4)
    using_paths = len(IMAGE_PATHS) > 0 and any(p.strip() != "" for p in IMAGE_PATHS)
    
    if not using_paths:
        if not os.path.exists(image_path):
            print(f"[*] Advertencia: No se encontró '{image_path}'.")
            generate_synthetic_face(image_path)
        
    print("=" * 70)
    print("🚀 INICIANDO SIMULACIÓN DE PIPELINE MULTI-FRAME (VÍDEO EN TIEMPO REAL)")
    print(f"📷 Cámara de origen: '{camera_id}'")
    if using_paths:
        print(f"🌐 Modo:             Procesando {len(IMAGE_PATHS)} rutas/URLs configuradas")
    else:
        print(f"🖼️ Imagen de prueba: '{image_path}'")
    print(f"🌐 Endpoint API:     {url}")
    print("=" * 70)
    print()

    # Base timestamp
    base_timestamp = int(time.time() * 1000)

    # Determinar el número de iteraciones
    num_frames = len(IMAGE_PATHS) if using_paths else 4

    # Procesar frames consecutivamente simulando un stream de vídeo
    for frame_idx in range(1, num_frames + 1):
        current_timestamp = base_timestamp + (frame_idx * 100)  # Separados por 100ms
        
        # Obtener los bytes de la imagen
        if using_paths:
            path_or_url = IMAGE_PATHS[frame_idx - 1]
            
            # Detectar si es URL o ruta local
            is_url = path_or_url.startswith("http://") or path_or_url.startswith("https://")
            
            if is_url:
                print(f"👉 [Frame {frame_idx}/4] Descargando imagen desde Internet...")
                print(f"   🔗 URL: {path_or_url}")
                try:
                    dl_start = time.time()
                    dl_res = requests.get(path_or_url, timeout=10)
                    if dl_res.status_code != 200:
                        print(f"   🔴 Error: HTTP {dl_res.status_code} al descargar la imagen. Saltando frame.")
                        print()
                        continue
                    img_bytes = dl_res.content
                    dl_time = int((time.time() - dl_start) * 1000)
                    print(f"   📥 Descargada con éxito en {dl_time}ms. Enviando a la API con timestamp {current_timestamp}...")
                except Exception as e:
                    print(f"   🔴 Error de conexión al descargar la imagen: {str(e)}. Saltando frame.")
                    print()
                    continue
            else:
                # Es ruta local
                print(f"👉 [Frame {frame_idx}/4] Cargando imagen local...")
                print(f"   📁 Ruta: {path_or_url}")
                if not os.path.exists(path_or_url):
                    print(f"   🔴 Error: No existe el archivo local. Saltando frame.")
                    print()
                    continue
                with open(path_or_url, "rb") as f:
                    img_bytes = f.read()
                print(f"   📥 Cargada con éxito. Enviando a la API con timestamp {current_timestamp}...")
        else:
            print(f"👉 [Frame {frame_idx}/4] Enviando imagen local con timestamp {current_timestamp}...")
            with open(image_path, "rb") as f:
                img_bytes = f.read()

        # Preparar datos Multipart Form para FastAPI
        files = {
            "image": ("frame.jpg", img_bytes, "image/jpeg")
        }
        data = {
            "camera_id": camera_id,
            "timestamp": str(current_timestamp)
        }
        
        try:
            start_req = time.time()
            response = requests.post(url, data=data, files=files)
            elapsed_req_ms = int((time.time() - start_req) * 1000)
            
            if response.status_code == 200:
                res_data = response.json()
                faces = res_data.get("faces", [])
                proc_time_ms = res_data.get("processing_time_ms", 0)
                
                print(f"   🟢 HTTP 200 OK | Latencia de Red + API: {elapsed_req_ms}ms | Inferencia Backend: {proc_time_ms}ms")
                
                if not faces:
                    print("   ⚠️  No se detectaron rostros en este frame.")
                else:
                    print("   👥 Rostros Detectados y Agregados:")
                    print("   " + "-" * 85)
                    print(f"   | {'Track ID':^10} | {'BBox Coordinates':^20} | {'Género Consolidado':^18} | {'Edad':^6} | {'Rango':^8} |")
                    print("   " + "-" * 85)
                    for face in faces:
                        track_id = face.get("track_id", "N/A")
                        bbox = face.get("bbox", [])
                        gender = face.get("gender", "N/A")
                        gender_conf = face.get("gender_confidence", 0.0)
                        age = face.get("age", "N/A")
                        age_range = face.get("age_range", "N/A")
                        
                        bbox_str = f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]" if len(bbox) == 4 else str(bbox)
                        gender_str = f"{gender} ({int(gender_conf*100)}%)"
                        
                        print(f"   | {track_id:^10} | {bbox_str:^20} | {gender_str:^18} | {age:^6} | {age_range:^8} |")
                    print("   " + "-" * 85)
            else:
                print(f"   🔴 Error HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("   🔴 Error: No se pudo conectar con el servidor local de FastAPI. ¿Está encendido?")
            print("      Asegúrate de ejecutar primero: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload")
            sys.exit(1)
            
        print()
        time.sleep(0.3)  # Pequeña pausa entre peticiones para emular realismo

    print("=" * 70)
    print("✅ SIMULACIÓN DE PIPELINE FINALIZADA")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de prueba de pipeline demográfico multi-frame en tiempo real")
    parser.add_argument("--image", type=str, default="image.jpg", help="Ruta de la imagen JPG/PNG a enviar en bucle (por defecto: image.jpg)")
    parser.add_argument("--camera", type=str, default="cam_lobby_01", help="ID de la cámara de origen")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host del servidor API")
    parser.add_argument("--port", type=int, default=8000, help="Puerto del servidor API")
    
    args = parser.parse_args()
    test_demographics_pipeline(args.image, args.camera, args.host, args.port)
