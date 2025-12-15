import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

# Code blocks for Data Augmentation
import_code = [
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n",
    "from tensorflow.keras.models import Sequential, Model\n",
    "from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Added for Augmentation\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report"
]

training_code = [
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
    "# Ya no es necesario 'fit' con los datos raw porque usamos preprocess_input antes\n",
    "\n",
    "print(\"Iniciando entrenamiento con Data Augmentation...\")\n",
    "sport_train = sport_model.fit(\n",
    "    datagen.flow(train_X, train_label, batch_size=batch_size),\n",
    "    steps_per_epoch=len(train_X) // batch_size,\n",
    "    epochs=epochs,\n",
    "    verbose=1,\n",
    "    validation_data=(valid_X, valid_label)\n",
    ")"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # 1. Add Import (if not present)
        if "from tensorflow.keras.models import Sequential" in source_str and "ImageDataGenerator" not in source_str:
            # We will just replace the import cell with our robust list
            cell['source'] = import_code
            print("Fixed: Added ImageDataGenerator import")
            modifications += 1

        # 2. Replace Training Loop
        if "sport_model.fit(train_X" in source_str:
            cell['source'] = training_code
            print("Fixed: Replaced static fit with Data Augmentation generator")
            modifications += 1

if modifications > 0:
    print(f"Applying {modifications} augmentation fixes...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb now uses Data Augmentation.")
else:
    print("WARNING: No changes applied! Patterns might not match.")
