import json
import os

notebook_path = r"C:\Users\Sears\Documents\Trabajos De IA\Proyecto-2\CNN.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New Model Code
new_model_source = [
    "# declaramos variables con los parámetros de configuración de la red\n",
    "INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001\n",
    "epochs = 20 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento\n",
    "batch_size = 64 # cantidad de imágenes que se toman a vez en memoria\n",
    "\n",
    "sport_model = Sequential()\n",
    "sport_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(64,64,3)))\n",
    "sport_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))\n",
    "sport_model.add(MaxPooling2D((2, 2), padding='same'))\n",
    "sport_model.add(Dropout(0.25))\n",
    "\n",
    "sport_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))\n",
    "sport_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))\n",
    "sport_model.add(MaxPooling2D((2, 2), padding='same'))\n",
    "sport_model.add(Dropout(0.25))\n",
    "\n",
    "sport_model.add(Flatten())\n",
    "sport_model.add(Dense(128, activation='relu'))\n",
    "sport_model.add(Dropout(0.5))\n",
    "sport_model.add(Dense(nClasses, activation='softmax'))"
]

# New Compile Code
new_compile_source = [
    "sport_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR), metrics=['accuracy'])"
]

model_updated = False
compile_updated = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_code = "".join(cell['source'])
        
        # Identify model definition cell
        if "sport_model = Sequential()" in source_code and "Conv2D(32" in source_code:
            cell['source'] = new_model_source
            model_updated = True
            print("Model definition updated.")
            
        # Identify compile cell
        if "sport_model.compile" in source_code and "categorical_crossentropy" in source_code:
            cell['source'] = new_compile_source
            compile_updated = True
            print("Model compile step updated.")

if model_updated and compile_updated:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated CNN.ipynb")
else:
    print("Could not find one or more cells to update.")
    if not model_updated: print("- Model definition cell not found.")
    if not compile_updated: print("- Compile cell not found.")
