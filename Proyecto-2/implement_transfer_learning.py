import json
import os

notebook_path = 'C:\\Users\\Sears\\Documents\\Trabajos De IA\\Proyecto-2\\CNN.ipynb'

def update_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Imports to include MobileNetV2 and preprocessing
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'from tensorflow.keras.models import Sequential' in source_str:
                new_source = []
                for line in cell['source']:
                    if 'from tensorflow.keras.models import Sequential' in line:
                        new_source.append('from tensorflow.keras.applications import MobileNetV2\n')
                        new_source.append('from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n')
                        new_source.append(line)
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                break

    # 2. Update Image Resizing to 96x96 (Better for Transfer Learning)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'image = resize(image, (32, 32)' in source_str:
                cell['source'] = [line.replace('(32, 32)', '(96, 96)') for line in cell['source']]
                break

    # 3. Update Model Definition to use MobileNetV2
    model_code = [
        "\n",
        "# Cargar MobileNetV2 pre-entrenado (sin la parte superior/top)\n",
        "base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))\n",
        "\n",
        "# Congelar los pesos del modelo base para no dañarlos en el primer entrenamiento\n",
        "base_model.trainable = False\n",
        "\n",
        "sport_model = Sequential([\n",
        "    base_model,\n",
        "    GlobalAveragePooling2D(),\n",
        "    Dense(128, activation='relu'),\n",
        "    Dropout(0.5),\n",
        "    Dense(nClasses, activation='softmax')\n",
        "])\n"
    ]
    
    # We need to import GlobalAveragePooling2D first if not present, but easier to just add it to the model cell
    # Or update imports again. Let's update imports again to be safe.
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'from tensorflow.keras.layers import (' in source_str:
                if 'GlobalAveragePooling2D' not in source_str:
                     cell['source'] = [line.replace('Flatten,', 'Flatten, GlobalAveragePooling2D,') for line in cell['source']]
                break

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'sport_model = Sequential([' in source_str:
                cell['source'] = model_code
                break

    # 4. Update Data Augmentation to use preprocess_input
    # We need to ensure the data is preprocessed correctly for MobileNetV2
    # Since we are loading images manually into numpy arrays, we should apply preprocess_input before training
    # OR we can include it in the ImageDataGenerator.
    # MobileNetV2 expects inputs in [-1, 1]. Our current manual loading does /255. (0 to 1).
    # MobileNetV2 preprocess_input does exactly that (scales to -1 to 1).
    # Let's check where normalization happens.
    
    # Check normalization cell
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'train_X = train_X/255.' in source_str:
                # We will replace this simple normalization with MobileNetV2 preprocessing
                # Actually, MobileNetV2 preprocess_input expects 0-255 inputs if we use it directly on images.
                # But here we have manual loading.
                # Let's change the normalization to:
                # train_X = preprocess_input(train_X)
                # But train_X needs to be float32.
                
                new_source = []
                for line in cell['source']:
                    if '/255.' in line:
                         # Comment out simple normalization
                         new_source.append('# ' + line)
                    else:
                        new_source.append(line)
                
                # Add preprocess_input call at the end of the cell
                new_source.append('\n# Preprocess for MobileNetV2\n')
                new_source.append('train_X = preprocess_input(train_X)\n')
                new_source.append('test_X = preprocess_input(test_X)\n')
                
                cell['source'] = new_source
                break

    # 5. Update Data Augmentation config (ImageDataGenerator)
    # We don't need to change much here, but we should make sure input size is correct in flow() if it was hardcoded (it's not).
    # The previous update to datagen is fine.

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully with MobileNetV2 Transfer Learning.")

if __name__ == "__main__":
    update_notebook()
