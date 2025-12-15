import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

# New Training Code with Callback
training_code_with_callback = [
    "# Configurar Data Augmentation\n",
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
    "print(\"Iniciando entrenamiento ROBUSTO (50 épocas + Ajuste de LR)...\")\n",
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
        
        # 1. Update Epochs
        if "epochs = 20" in source_str:
            new_source = []
            for line in cell['source']:
                if "epochs = 20" in line:
                    new_source.append("epochs = 50 # Aumentado para mejor convergencia\n")
                    print("Fixed: Increased epochs to 50")
                    modifications += 1
                else:
                    new_source.append(line)
            cell['source'] = new_source

        # 2. Upgrade Model Capacity (Dense Layer)
        if "sport_model.add(Dense(128" in source_str:
            new_source = []
            for line in cell['source']:
                if "sport_model.add(Dense(128" in line and "nClasses" not in line: # Avoid output layer if using nClasses var
                    new_source.append("sport_model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))\n")
                    print("Fixed: Increased Dense layer to 256 neurons")
                    modifications += 1
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
        # 3. Inject Callback into Training Loop
        # identifying the previous augmentation cell
        if "datagen = ImageDataGenerator" in source_str and "sport_model.fit" in source_str:
            cell['source'] = training_code_with_callback
            print("Fixed: Added ReduceLROnPlateau callback to training loop")
            modifications += 1

if modifications > 0:
    print(f"Applying {modifications} performance boosts...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb optimized for Custom CNN performance.")
else:
    print("WARNING: No changes applied! Patterns might not match.")
