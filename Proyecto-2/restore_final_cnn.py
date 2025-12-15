import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

# 1. MobileNetV2 Layer Definition (The "High Performance" Model)
mobilenet_code = [
    "# Cargar MobileNetV2 pre-entrenado (sin la parte superior/top)\n",
    "# Input shape MUST be (64, 64, 3) for our dataset\n",
    "base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))\n",
    "\n",
    "# Congelar los pesos del modelo base para no dañarlos en el primer entrenamiento\n",
    "base_model.trainable = False\n",
    "\n",
    "sport_model = Sequential([\n",
    "    base_model,\n",
    "    GlobalAveragePooling2D(),\n",
    "    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),\n",
    "    Dropout(0.6),  # High dropout for generalization\n",
    "    Dense(nClasses, activation='softmax')\n",
    "])\n",
    "\n",
    "sport_model.summary()"
]

# 2. Correct Preprocessing & Visualization Logic
viz_code = [
    "train_X = train_X.astype('float32')\n",
    "test_X = test_X.astype('float32')\n",
    "\n",
    "# 1. Preprocess FIRST (Scales to [-1, 1])\n",
    "train_X = preprocess_input(train_X)\n",
    "test_X = preprocess_input(test_X)\n",
    "\n",
    "# 2. Visualize AFTER (Rescale [-1, 1] -> [0, 1] for display)\n",
    "# If we don't do this, values are wrong for imshow and it looks white\n",
    "plt.imshow((test_X[0] + 1) / 2)\n",
    "plt.title(\"Visualización Correcta (Post-Procesamiento)\")"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # A. RESTORE MODEL
        # We look for the custom CNN definition (Conv2D, MaxPooling...)
        if "sport_model.add(Conv2D" in source_str or "base_model = MobileNetV2" in source_str:
             if "MobileNetV2" not in source_str or "Dropout(0.6)" not in source_str:
                cell['source'] = mobilenet_code
                print("Fixed: Restored MobileNetV2 Architecture (Replacing Custom CNN)")
                modifications += 1

        # B. FIX VISUALIZATION ORDER
        # Look for the cell doing imshow and preprocess code
        if "plt.imshow" in source_str and ("preprocess_input" in source_str or "/ 255" in source_str):
            # Identifying the preprocessing cell specifically
            if "train_X.astype" in source_str:
                cell['source'] = viz_code
                print("Fixed: Corrected Preprocessing/Visualization Order")
                modifications += 1

        # C. SAFETY: ENSURE AUGMENTATION IS IMPORTED
        if "from tensorflow.keras.models import Sequential" in source_str:
             if "ImageDataGenerator" not in source_str:
                 # Re-add imports if they got messed up
                 cell['source'] = [
                    "import tensorflow as tf\n",
                    "from tensorflow import keras\n",
                    "from tensorflow.keras.utils import to_categorical\n",
                    "from tensorflow.keras.applications import MobileNetV2\n",
                    "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n",
                    "from tensorflow.keras.models import Sequential, Model\n",
                    "from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D\n",
                    "from tensorflow.keras.optimizers import Adam\n",
                    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.metrics import classification_report"
                 ]
                 print("Fixed: Ensured all necessary imports (including Augmentation) are present")
                 modifications += 1

if modifications > 0:
    print(f"Applying {modifications} critical restorations...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb is restored to high-performance state.")
else:
    print("WARNING: No changes applied! Model might already be MobileNetV2.")
