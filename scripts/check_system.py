import sys
import platform
import torch
import timm
import albumentations as A
import captum
from PIL import Image
import numpy as np


def print_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


print_section("INFORMACIÓN DEL SISTEMA")
print(f"Python: {sys.version.split()[0]}")
print(f"Sistema Operativo: {platform.system()} {platform.release()}")

print_section("DEPENDENCIAS INSTALADAS")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ timm: {timm.__version__}")
print(f"✓ Albumentations: {A.__version__}")
print(f"✓ Captum: {captum.__version__}")
print(f"✓ PIL: {Image.__version__}")
print(f"✓ NumPy: {np.__version__}")

print_section("ACELERACIÓN GPU")
print(f"CUDA disponible: {'Sí' if torch.cuda.is_available() else 'No'}")

if torch.cuda.is_available():
    print(f"Versión CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print(f"Número de GPUs: {torch.cuda.device_count()}")
else:
    print("⚠ Ejecutando en CPU")

print_section("PRUEBA FUNCIONAL")
try:
    tensor = torch.randn(3, 224, 224)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        print("✓ Tensor creado en GPU correctamente")
    else:
        print("✓ Tensor creado en CPU correctamente")
    print(f"  Shape: {tensor.shape}, Device: {tensor.device}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 50)
print("  ✓ Sistema verificado correctamente")
print("=" * 50 + "\n")
