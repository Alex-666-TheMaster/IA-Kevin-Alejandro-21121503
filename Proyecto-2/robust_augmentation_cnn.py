import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

# Robus Data Augmentation Code
robust_datagen_code = [
    "# Configurar Data Augmentation ROBUSTO (Color, Brillo, Geometría)\n",
    "datagen = ImageDataGenerator(\n",
    "    rotation_range=20,\n",
    "    zoom_range=0.15,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    shear_range=0.1,             # Deformación geométrica\n",
    "    brightness_range=[0.7, 1.3], # Variación de Iluminación (Oscuro/Claro)\n",
    "    channel_shift_range=50.0,    # Variación de Color (Para mariquitas azules/raras)\n",
    "    fill_mode='nearest'\n",
    ")\n",
    "\n",
    "# Callback para mejorar el aprendizaje si se estanca\n",
    "from tensorflow.keras.callbacks import ReduceLROnPlateau\n",
    "reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)\n",
    "\n",
    "print(\"Iniciando entrenamiento EXPERTO (50 épocas + Augmentation Full)...\")\n",
    "sport_train = sport_model.fit(\n",
    "    datagen.flow(train_X, train_label, batch_size=batch_size),\n",
    "    steps_per_epoch=len(train_X) // batch_size,\n",
    "    epochs=epochs,\n",
    "    verbose=1,\n",
    "    validation_data=(valid_X, valid_label),\n",
    "    callbacks=[reduce_lr]\n",
    ")"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Replace the previous augmentation block
        if "datagen = ImageDataGenerator" in source_str and "rotation_range" in source_str:
            # Check if it already has channel_shift to avoid redundant writes if run twice
            if "channel_shift_range" not in source_str:
                cell['source'] = robust_datagen_code
                print("Fixed: Upgraded ImageDataGenerator with Color/Brightness/Shear")
                modifications += 1

if modifications > 0:
    print(f"Applying {modifications} robustness upgrades...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb now utilizes Expert Data Augmentation.")
else:
    print("WARNING: No changes applied! Augmentation logic might not match or is already robust.")
