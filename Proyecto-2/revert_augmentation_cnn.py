import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

# Standard Data Augmentation Code (Reverting from Robust)
standard_datagen_code = [
    "# Configurar Data Augmentation (Estándar - Alta Precisión)\n",
    "# Revertido a la versión que funcionaba bien (sin deformación de color extrema)\n",
    "datagen = ImageDataGenerator(\n",
    "    rotation_range=20,\n",
    "    zoom_range=0.15,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    fill_mode='nearest'\n",
    ")\n",
    "\n",
    "# Callback para mejorar el aprendizaje si se estanca\n",
    "from tensorflow.keras.callbacks import ReduceLROnPlateau\n",
    "reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)\n",
    "\n",
    "print(\"Iniciando entrenamiento ESTÁNDAR (50 épocas + Augmentation Balanceado)...\")\n",
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
        
        # Look for the augmentation block
        if "datagen = ImageDataGenerator" in source_str:
            # We overwrite it with the standard version
            cell['source'] = standard_datagen_code
            print("Fixed: Reverted ImageDataGenerator to Standard Settings")
            modifications += 1

if modifications > 0:
    print(f"Applying {modifications} reversions...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb reverted to standard augmentation.")
else:
    print("WARNING: No changes applied! Cell not found.")
