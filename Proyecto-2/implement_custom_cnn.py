import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

# 1. Define Robust Custom Sequential CNN (VGG-style-ish)
custom_model_code = [
    "# Modelo CNN Personalizado Robusto (Requisito del Profesor)\n",
    "# Arquitectura Secuencial diseñada para alto rendimiento\n",
    "\n",
    "sport_model = Sequential()\n",
    "\n",
    "# Bloque 1\n",
    "sport_model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))\n",
    "sport_model.add(BatchNormalization())\n",
    "sport_model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "sport_model.add(Dropout(0.25))\n",
    "\n",
    "# Bloque 2\n",
    "sport_model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))\n",
    "sport_model.add(BatchNormalization())\n",
    "sport_model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "sport_model.add(Dropout(0.25))\n",
    "\n",
    "# Bloque 3\n",
    "sport_model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))\n",
    "sport_model.add(BatchNormalization())\n",
    "sport_model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "sport_model.add(Dropout(0.4))\n",
    "\n",
    "# Clasificación (Head)\n",
    "sport_model.add(Flatten())\n",
    "sport_model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))\n",
    "sport_model.add(BatchNormalization())\n",
    "sport_model.add(Dropout(0.5))\n",
    "sport_model.add(Dense(nClasses, activation='softmax'))\n",
    "\n",
    "sport_model.summary()"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Replace the problematic cell defining 'weight variable assignments' AND the model cell
        # Or just the model cell if they are separate.
        # User traceback showed: `base_model = Sequential(weights='imagenet'...)` at Cell In[18] line 1.
        
        if "base_model = Sequential(weights='imagenet'" in source_str or "sport_model = Sequential" in source_str:
            # Check if this is the large model definition block
            if "Conv2D" in source_str or "input_shape=(64, 64, 3)" in source_str:
                 cell['source'] = custom_model_code
                 print("Fixed: Replaced MobileNetV2/Broken Code with Robust Custom Sequential CNN")
                 modifications += 1

if modifications > 0:
    print(f"Applying {modifications} critical structural changes...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb now uses the Custom Sequential Model.")
else:
    print("WARNING: No changes applied! Model definition cell not found.")
