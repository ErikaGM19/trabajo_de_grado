"""
mejores_valores.py
Única fuente de verdad para los mejores hiperparámetros encontrados
en cada paso de la búsqueda secuencial (Sección 4.2.4.2).

IMPORTANTE: este archivo se actualiza manualmente, PERO SOLO UNA VEZ
por cada paso, justo después de revisar los resultados de ese paso
en el CSV. Los notebooks de los pasos siguientes lo importan en vez
de tener el número escrito a mano, para evitar errores de transcripción.
"""

# --- Paso 1: Learning Rate ---
# Valores probados: 0.0001, 0.001, 0.01
MEJOR_LR = 0.001  

# --- Paso 2: Optimizador ---
# Valores probados: adam, rmsprop, sgd
MEJOR_OPTIMIZADOR = 'adam'               # <-- actualiza cuando cierres el Paso 2

# --- Paso 3: Dropout ---
# Valores probados: 0.3, 0.5, 0.7
MEJOR_DROPOUT = 0.3                      # <-- actualiza cuando cierres el Paso 3

# --- Paso 4: Capas descongeladas ---
# Valores probados: 0, 10, 30, 50
MEJOR_CAPAS = 50                         # <-- actualiza cuando cierres el Paso 4

# --- Paso 5: Batch size ---
# Valores probados: 16, 32, 64
MEJOR_BATCH = 16                         # <-- actualiza cuando cierres el Paso 5