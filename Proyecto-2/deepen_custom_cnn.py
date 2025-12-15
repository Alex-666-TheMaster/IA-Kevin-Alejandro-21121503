import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

# 4-Block "Mini-VGG" Architecture
deep_model_code = [
    "# Modelo CNN Profundo (4 Bloques) - Optimizado para Detalles Finos\n",
    "# Arquitectura Secuencial 'Mini-VGG'\n",
    "\n",
    "sport_model = Sequential()\n",
    "\n",
    "# Bloque 1 (32 filtros)\n",
    "sport_model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))\n",
    "sport_model.add(BatchNormalization())\n",
    "sport_model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "sport_model.add(Dropout(0.25))\n",
    "\n",
    "# Bloque 2 (64 filtros)\n",
    "sport_model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))\n",
    "sport_model.add(BatchNormalization())\n",
    "sport_model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "sport_model.add(Dropout(0.25))\n",
    "\n",
    "# Bloque 3 (128 filtros)\n",
    "sport_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))\n",
    "sport_model.add(BatchNormalization())\n",
    "sport_model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "sport_model.add(Dropout(0.4))\n",
    "\n",
    "# Bloque 4 (256 filtros) - EL NUEVO BLOQUE CRÍTICO\n",
    "sport_model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))\n",
    "sport_model.add(BatchNormalization())\n",
    "sport_model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "sport_model.add(Dropout(0.4))\n",
    "\n",
    "# Clasificación (Head Potente)\n",
    "sport_model.add(Flatten())\n",
    "sport_model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))\n",
    "sport_model.add(BatchNormalization())\n",
    "sport_model.add(Dropout(0.5))\n",
    "sport_model.add(Dense(nClasses, activation='softmax'))\n",
    "\n",
    "sport_model.summary()"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Locate the previous model definition (whether custom or restored)
        # We look for the sequential definition and at least one Conv2D
        if "sport_model = Sequential" in source_str and "Conv2D" in source_str:
             if "input_shape=(64, 64, 3)" in source_str:
                 cell['source'] = deep_model_code
                 print("Fixed: Upgraded to 4-Block Deep CNN Architecture")
                 modifications += 1

if modifications > 0:
    print(f"Applying {modifications} architecture upgrades...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb now uses the Deep 4-Block Model.")
else:
    print("WARNING: No changes applied! Model definition cell not found.")
