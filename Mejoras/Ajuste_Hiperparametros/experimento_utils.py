"""
experimento_utils.py
Funciones compartidas para el experimento de ajuste de hiperparámetros de ResNet50.

Este archivo NO se ejecuta directamente. Cada notebook de cada
experimento lo importa y llama a ejecutar_experimento() con sus propios valores.
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models

# ---------------------------------------------------------------
# RUTAS FIJAS (calculadas a partir de la ubicación de este archivo, 
# así no importa desde qué subcarpeta se llame)
# ---------------------------------------------------------------
CARPETA_BASE = os.path.dirname(os.path.abspath(__file__))          # .../Mejoras/Ajuste_Hiperparametros
CSV_RESULTADOS = os.path.join(CARPETA_BASE, 'resultados_todos_los_experimentos.csv')

# Desde Mejoras/Ajuste_Hiperparametros subimos 2 niveles para llegar a la raíz del proyecto
RAIZ_PROYECTO = os.path.abspath(os.path.join(CARPETA_BASE, '..', '..'))
BASE_DIR_DATA = os.path.join(RAIZ_PROYECTO, 'Data', 'Cana_de_azucar')
TRAIN_DIR = os.path.join(BASE_DIR_DATA, 'train')
VAL_DIR   = os.path.join(BASE_DIR_DATA, 'val')
TEST_DIR  = os.path.join(BASE_DIR_DATA, 'test')

IMG_SIZE    = 224
NUM_CLASSES = 10
SEED        = 42


# ---------------------------------------------------------------
# GENERADORES DE DATOS
# ---------------------------------------------------------------
def crear_generadores(batch_size):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2]
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, class_mode='categorical',
        shuffle=True, seed=SEED
    )
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, class_mode='categorical',
        shuffle=False
    )
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size, class_mode='categorical',
        shuffle=False
    )
    return train_generator, val_generator, test_generator


# ---------------------------------------------------------------
# CONSTRUCCIÓN DEL MODELO
# ---------------------------------------------------------------
def construir_modelo(dropout, capas_descongeladas=0):
    base_model = ResNet50(
        weights='imagenet', include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    if capas_descongeladas == 0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-capas_descongeladas]:
            layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


# ---------------------------------------------------------------
# OPTIMIZADOR
# ---------------------------------------------------------------
def obtener_optimizador(nombre_optimizador, learning_rate):
    nombre_optimizador = nombre_optimizador.lower()
    if nombre_optimizador == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif nombre_optimizador == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif nombre_optimizador == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f'Optimizador no reconocido: {nombre_optimizador}')


# ---------------------------------------------------------------
# GUARDAR RESULTADOS
# ---------------------------------------------------------------
def guardar_resultado_csv(fila_dict):
    """Actualiza la tabla resumen general (para las tablas comparativas de la tesis)."""
    df_nuevo = pd.DataFrame([fila_dict])
    try:
        df_existente = pd.read_csv(CSV_RESULTADOS)
        if df_existente.empty or 'experimento' not in df_existente.columns:
            df_final = df_nuevo
        else:
            df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        df_final = df_nuevo
    df_final.to_csv(CSV_RESULTADOS, index=False)
    print(f'Resumen actualizado en {CSV_RESULTADOS}')


def guardar_resultado_txt(ruta_txt, nombre_experimento, hiperparametros,
                           test_accuracy, test_loss, f1, duracion_min,
                           y_true, y_pred, nombres_clases):
    """Guarda un archivo .txt legible dentro de la carpeta del experimento,
    igual al patrón que ya usas en Configuraciones_Color."""
    reporte = classification_report(y_true, y_pred, target_names=nombres_clases, digits=4)

    with open(ruta_txt, 'w', encoding='utf-8') as f:
        f.write(f'EXPERIMENTO: {nombre_experimento}\n')
        f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('--- Hiperparámetros usados ---\n')
        for k, v in hiperparametros.items():
            f.write(f'{k}: {v}\n')
        f.write('\n--- Resultados en test ---\n')
        f.write(f'Test Accuracy : {test_accuracy:.4f}\n')
        f.write(f'Test Loss     : {test_loss:.4f}\n')
        f.write(f'F1 Score      : {f1:.4f}\n')
        f.write(f'Duración      : {duracion_min:.1f} min\n\n')
        f.write('--- Reporte de clasificación ---\n')
        f.write(reporte)

    print(f'Resultados guardados en {ruta_txt}')


# ---------------------------------------------------------------
# FUNCIÓN PRINCIPAL: EJECUTA UN EXPERIMENTO COMPLETO
# ---------------------------------------------------------------
def ejecutar_experimento(nombre_experimento, carpeta_experimento, learning_rate,
                          batch_size, dropout, capas_descongeladas=0,
                          optimizador='adam',
                          epochs=30, patience_es=5, patience_lr=3):
    """
    carpeta_experimento: ruta de la carpeta ESPECÍFICA del experimento
                          (ej. .../Ajuste_Hiperparametros/LR_00001)
    """
    print(f'\n{"="*60}')
    print(f'EXPERIMENTO: {nombre_experimento}')
    print(f'LR={learning_rate} | optimizador={optimizador} | batch={batch_size} | '
          f'dropout={dropout} | capas_descongeladas={capas_descongeladas}')
    print(f'{"="*60}\n')

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train_gen, val_gen, test_gen = crear_generadores(batch_size)
    model = construir_modelo(dropout, capas_descongeladas)

    model.compile(
        optimizer=obtener_optimizador(optimizador, learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience_es,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=patience_lr, verbose=1
    )

    inicio = time.time()
    history = model.fit(
        train_gen, validation_data=val_gen, epochs=epochs,
        callbacks=[early_stopping, reduce_lr], verbose=1
    )
    duracion_min = (time.time() - inicio) / 60

    test_gen.reset()
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    predicciones = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(predicciones, axis=1)
    y_true = test_gen.classes
    f1 = f1_score(y_true, y_pred, average='weighted')
    nombres_clases = list(test_gen.class_indices.keys())

    # --- Guardar pesos, dentro de la carpeta del experimento ---
    ruta_pesos = os.path.join(carpeta_experimento, f'{nombre_experimento}_weights.h5')
    model.save_weights(ruta_pesos)

    # --- Guardar .txt legible, dentro de la carpeta del experimento ---
    hiperparametros = {
        'learning_rate': learning_rate, 'optimizador': optimizador,
        'batch_size': batch_size, 'dropout': dropout,
        'capas_descongeladas': capas_descongeladas,
        'epochs_max': epochs, 'epochs_entrenadas': len(history.history['loss'])
    }
    ruta_txt = os.path.join(carpeta_experimento, f'{nombre_experimento}_resultados.txt')
    guardar_resultado_txt(ruta_txt, nombre_experimento, hiperparametros,
                           test_accuracy, test_loss, f1, duracion_min,
                           y_true, y_pred, nombres_clases)

    # --- Actualizar CSV resumen general ---
    fila = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'experimento': nombre_experimento,
        **hiperparametros,
        'test_accuracy': round(float(test_accuracy), 4),
        'test_loss': round(float(test_loss), 4),
        'f1_score': round(float(f1), 4),
        'duracion_min': round(duracion_min, 1),
    }
    guardar_resultado_csv(fila)

    return fila, history