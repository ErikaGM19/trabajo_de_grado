import os
import shutil
import random

# =============================================
# CONFIGURACIÓN
# =============================================
DIRECTORIO_ORIGEN = "Data/Cana_de_azucar/train"
DIRECTORIO_VAL    = "Data/Cana_de_azucar/val"
DIRECTORIO_TEST   = "Data/Cana_de_azucar/test"

PORCENTAJE_VAL  = 0.15   # 15% para validación
PORCENTAJE_TEST = 0.15   # 15% para test
# El 70% restante se queda en train

SEMILLA = 42
# =============================================

random.seed(SEMILLA)

# Obtener las clases (subcarpetas dentro de train)
clases = [d for d in os.listdir(DIRECTORIO_ORIGEN)
          if os.path.isdir(os.path.join(DIRECTORIO_ORIGEN, d))]

print(f"Clases encontradas: {clases}")
print(f"Total de clases: {len(clases)}")
print()

for clase in clases:
    carpeta_clase = os.path.join(DIRECTORIO_ORIGEN, clase)

    # Listar todas las imágenes de esta clase
    imagenes = [f for f in os.listdir(carpeta_clase)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    random.shuffle(imagenes) # Mezclar aleatoriamente

    total  = len(imagenes)
    n_test = int(total * PORCENTAJE_TEST)
    n_val  = int(total * PORCENTAJE_VAL)

    imagenes_test = imagenes[:n_test]
    imagenes_val  = imagenes[n_test:n_test + n_val]
    # Las demás se quedan en train (no se mueven)

    print(f"Clase '{clase}': {total} imágenes → "
          f"{total - n_test - n_val} train | {n_val} val | {n_test} test")

    # Crear carpetas destino si no existen
    os.makedirs(os.path.join(DIRECTORIO_VAL, clase), exist_ok=True)
    os.makedirs(os.path.join(DIRECTORIO_TEST, clase), exist_ok=True)

    # Mover imágenes a val/
    for img in imagenes_val:
        origen  = os.path.join(carpeta_clase, img)
        destino = os.path.join(DIRECTORIO_VAL, clase, img)
        shutil.move(origen, destino)

    # Mover imágenes a test/
    for img in imagenes_test:
        origen  = os.path.join(carpeta_clase, img)
        destino = os.path.join(DIRECTORIO_TEST, clase, img)
        shutil.move(origen, destino)

print()
print("¡División completada!")
print(f"   - train : 70% de las imágenes")
print(f"   - val   : 15% de las imágenes")
print(f"   - test  : 15% de las imágenes")