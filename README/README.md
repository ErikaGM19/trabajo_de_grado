# CLASIFICACIÓN DE ENFERMEDADES EN HOJAS DE CAÑA DE AZÚCAR MEDIANTE APRENDIZAJE PROFUNDO

**Autora:** Erika García Muñoz  
**Universidad:** Universidad del Valle  
**Carrera:** Ingeniería de Sistemas  
**Año:** 2026

---

## Descripción

Este proyecto desarrolla y compara múltiples arquitecturas de redes neuronales convolucionales (CNN) para la clasificación automática de enfermedades en hojas de caña de azúcar. Se evalúan 14 modelos bajo condiciones iguales para garantizar una comparación justa, primero sin fine-tuning y luego con fine-tuning sobre pesos de ImageNet.

---

## Estructura del Proyecto

```
TESIS_CANA_AZUCAR/
│
├── Data/
│   ├── Cana_de_azucar/
│   │   ├── train/          ← 70% de las imágenes (entrenamiento)
│   │   ├── val/            ← 15% de las imágenes (validación)
│   │   └── test/           ← 15% de las imágenes (evaluación final)
│   └── PlantVillage/       ← Dataset alternativo (usado en fase de Mejoras)
│
├── Notebooks/
│   ├── Sin_Fine_Tuning/    ← Fase 1: modelos con base congelada
│   │   ├── ResNet50/
│   │   ├── ResNet50v2/
│   │   ├── EfficientNetB0/
│   │   ├── EfficientNetB1/
│   │   ├── EfficientNetB2/
│   │   ├── EfficientNetB3/
│   │   ├── EfficientNetB4/
│   │   ├── EfficientNetB5/
│   │   ├── EfficientNetB6/
│   │   ├── EfficientNetB7/
│   │   ├── MobileNetV3Large/
│   │   ├── MobileNetV3Small/
│   │   ├── DenseNet201/
│   │   └── ViT_B16/
│   │
│   └── Con_Fine_Tuning/    ← Fase 2: mismos modelos con fine-tuning (ImageNet, 2 etapas)
│       ├── ResNet50/
│   │   ├── ResNet50v2/
│       ├── EfficientNetB0/
│       ├── EfficientNetB1/
│       ├── EfficientNetB2/
│       ├── EfficientNetB3/
│       ├── EfficientNetB4/
│       ├── EfficientNetB5/
│       ├── EfficientNetB6/
│       ├── EfficientNetB7/
│       ├── MobileNetV3Large/
│       ├── MobileNetV3Small/
│       ├── DenseNet201/
│       └── ViT_B16/
│
├── Mejoras/                ← Fase 3: experimentos solo con el mejor modelo
│   ├── PlantVillage_Sin_FT/
│   ├── PlantVillage_Con_FT/
│   ├── Filtro_Bilateral/
│   ├── CYM/
│   └── Ensambles/
│
├── dividir_datos.py        ← Script usado para dividir el dataset (70/15/15)
└── README/
    └── README.md           ← Este archivo
```

---

## Dataset

**Nombre:** Dataset de Caña de Azúcar (RGB)  
**Total de imágenes:** 8.693  
**Número de clases:** 10  
**División aplicada:** 70% entrenamiento / 15% validación / 15% test

| Clase | Imágenes totales | Train | Val | Test |
|---|---|---|---|---|
| carbon | 500 | 350 | 75 | 75 |
| hoja_amarilla | 1.000 | 700 | 150 | 150 |
| mancha_anillo | 702 | 492 | 105 | 105 |
| mancha_parda | 1.000 | 700 | 150 | 150 |
| mosaico | 1.000 | 700 | 150 | 150 |
| muermo_rojo | 1.000 | 700 | 150 | 150 |
| roya | 1.000 | 700 | 150 | 150 |
| sanas | 1.000 | 700 | 150 | 150 |
| secas | 491 | 345 | 73 | 73 |
| tizon_bacteriano | 1.000 | 700 | 150 | 150 |

> La división fue realizada con semilla fija `seed=42` usando el script `dividir_datos.py` para garantizar reproducibilidad.

---

## Condiciones de Entrenamiento

Todos los modelos fueron entrenados bajo las **mismas condiciones** para garantizar una comparación justa:

| Parámetro | Valor |
|---|---|
| Semilla aleatoria | 42 |
| División del dataset | 70% / 15% / 15% |
| Épocas máximas | 75 |
| Early stopping (monitor) | val_loss |
| Early stopping (patience) | 5 |
| Batch size | 32 |
| Optimizador | Adam |
| Función de pérdida | categorical_crossentropy |

### Data Augmentation (igual en todos los modelos)

```python
rotation_range      = 20
width_shift_range   = 0.2
height_shift_range  = 0.2
zoom_range          = 0.2
horizontal_flip     = True
vertical_flip       = True
brightness_range    = [0.8, 1.2]
```

---

## Modelos Evaluados

| Modelo | Fase Sin FT | Fase Con FT | IMG_SIZE |
|---|---|---|---|
| ResNet50 | ✅ | ✅ | 224 |
| ResNet50v2 | ✅ | ✅ | 224 |
| EfficientNetB0 | ✅ | ✅ | 224 |
| EfficientNetB1 | ✅ | ✅ | 240 |
| EfficientNetB2 | ✅ | ✅ | 260 |
| EfficientNetB3 | ✅ | ✅ | 300 |
| EfficientNetB4 | ✅ | ✅ | 380 |
| EfficientNetB5 | ✅ | ✅ | 456 |
| EfficientNetB6 | ✅ | ✅ | 528 |
| EfficientNetB7 | ✅ | ✅ | 600 |
| MobileNetV3Large | ✅ | ✅ | 224 |
| MobileNetV3Small | ✅ | ✅ | 224 |
| DenseNet201 | ✅ | ✅ | 224 |
| ViT-B16 | ✅ | ✅ | 224 |

---

## Metodología

### Fase 1 — Sin Fine-Tuning
Se carga cada arquitectura con pesos preentrenados de ImageNet. La base congelada (`trainable = False`) y se entrenan únicamente las capas de clasificación añadidas.

### Fase 2 — Con Fine-Tuning (2 etapas)
**Etapa 1:** igual que la Fase 1 (base congelada).  
**Etapa 2:** se descongelan las últimas capas de la base y se reentrena con un learning rate muy pequeño (`1e-5`).

### Fase 3 — Mejoras (solo al mejor modelo)
Una vez identificado el mejor modelo de las fases anteriores, se aplican las siguientes mejoras:
1. Entrenamiento desde cero con PlantVillage sin fine-tuning
2. Entrenamiento desde cero con PlantVillage con fine-tuning (2 etapas)
3. Entrenamiento con filtro bilateral
4. Entrenamiento en espacio de color CYM
5. Ensambles

---

## Resultados por Modelo

*(Esta tabla se completa al finalizar el entrenamiento de todos los modelos)*

| Modelo | Sin FT — Accuracy | Sin FT — F1 | Con FT — Accuracy | Con FT — F1 |
|---|---|---|---|---|
| ResNet50 | | | | |
| EfficientNetB0 | | | | |
| EfficientNetB1 | | | | |
| EfficientNetB2 | | | | |
| EfficientNetB3 | | | | |
| EfficientNetB4 | | | | |
| EfficientNetB5 | | | | |
| EfficientNetB6 | | | | |
| EfficientNetB7 | | | | |
| MobileNetV3Large | | | | |
| MobileNetV3Small | | | | |
| DenseNet201 | | | | |
| ViT-B16 | | | | |

---

## Requisitos

### Entorno
```
Python        3.10
TensorFlow    2.10
CUDA          (recomendado para entrenamiento en GPU)
```

### Librerías principales
```
tensorflow==2.10
numpy
matplotlib
seaborn
scikit-learn
Pillow
vit-keras       ← solo para el modelo ViT-B16
```

### Instalación del entorno
```bash
conda activate tesis_cana
pip install tensorflow==2.10 scikit-learn seaborn Pillow vit-keras
```

---

## Cómo Ejecutar

1. Clona o descarga el proyecto
2. Activa el entorno:
   ```bash
   conda activate tesis_cana
   ```
3. Abre VS Code y navega a la carpeta del modelo que deseas entrenar
4. Abre el notebook correspondiente (ej: `densenet201_sin_ft.ipynb`)
5. Ejecuta todas las celdas de arriba hacia abajo (**Run All**)
6. Al finalizar encontrarás en la misma carpeta:
   - `{modelo}_weights.h5` → pesos del modelo
   - `{modelo}_resultados.txt` → métricas completas

---

## Archivos de Salida por Modelo

Cada notebook genera automáticamente estos archivos en su propia carpeta:

| Archivo | Contenido |
|---|---|
| `{modelo}_weights.h5` | Pesos del modelo entrenado |
| `{modelo}_resultados.txt` | Accuracy, Loss, F1 y reporte por clase |
