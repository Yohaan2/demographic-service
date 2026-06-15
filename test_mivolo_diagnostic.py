import sys
import traceback

print("[*] Iniciando diagnostico de carga de Hugging Face MiVOLO V2...")

try:
    import numpy as np
    print(f"[+] NumPy importado (Version: {np.__version__})")
except Exception:
    print("[-] Error al importar NumPy:")
    traceback.print_exc()
    sys.exit(1)

try:
    import onnxruntime
    print(f"[+] ONNXRuntime importado (Version: {onnxruntime.__version__})")
except Exception:
    print("[-] Error al importar ONNXRuntime:")
    traceback.print_exc()
    sys.exit(1)

try:
    import torch
    print(f"[+] PyTorch importado (Version: {torch.__version__})")
except Exception:
    print("[-] Error al importar PyTorch:")
    traceback.print_exc()
    sys.exit(1)

try:
    import transformers
    print(f"[+] Transformers importado (Version: {transformers.__version__})")
except Exception:
    print("[-] Error al importar Transformers:")
    traceback.print_exc()
    sys.exit(1)

try:
    import timm
    print(f"[+] Timm importado (Version: {timm.__version__})")
except Exception:
    print("[-] Error al importar Timm:")
    traceback.print_exc()
    sys.exit(1)

try:
    import mivolo
    print("[+] Libreria 'mivolo' importada con exito")
except Exception:
    print("[-] Error al importar la libreria 'mivolo':")
    traceback.print_exc()
    sys.exit(1)

print("[*] Verificando SCRFD ONNX con numpy 2.x...")
try:
    import cv2
    from src.detectors.scrfd_detector import SCRFDDetector
    from src.core.config import settings
    detector = SCRFDDetector(settings.DETECTOR_MODEL_PATH)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = detector.detect(dummy, threshold=0.5)
    print("[+] SCRFD ONNX funciona correctamente con numpy 2.x")
except Exception:
    print("[-] Error en SCRFD ONNX con numpy 2.x:")
    traceback.print_exc()

print("[*] Intentando cargar el modelo real MiVOLO V2 desde Hugging Face...")
try:
    import torch
    from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

    config = AutoConfig.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
    print("    [1/3] Configuracion cargada OK")

    image_processor = AutoImageProcessor.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
    print("    [2/3] Image Processor cargado OK")

    model = AutoModelForImageClassification.from_pretrained(
        "iitolstykh/mivolo_v2",
        trust_remote_code=True,
        dtype=torch.float32
    )
    model.eval()
    print("    [3/3] Pesos del modelo cargados OK")

    print("\n[OK] DIAGNOSTICO EXITOSO: El modelo real de MiVOLO se carga correctamente.")

except Exception:
    print("\n[FAIL] EL DIAGNOSTICO HA FALLADO durante la carga del modelo:")
    print("-" * 80)
    traceback.print_exc()
    print("-" * 80)
