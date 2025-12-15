import json
import os

notebook_path = 'C:\\Users\\Sears\\Documents\\Trabajos De IA\\Proyecto-2\\CNN.ipynb'

def update_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Imports (Ensure BatchNormalization and regularizers are present)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'from tensorflow.keras.models import Sequential' in source_str:
                if 'from tensorflow.keras import regularizers' not in source_str:
                    new_source = []
                    for line in cell['source']:
                        if 'from tensorflow.keras.layers import (' in line:
                            new_source.append('from tensorflow.keras import regularizers\n')
                            new_source.append('from tensorflow.keras.layers import (\n')
                        elif '    Input, Dense, Dropout, Flatten, BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Conv2D, LeakyReLU' in line:
                             new_source.append('    Input, Dense, Dropout, Flatten, BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Conv2D, LeakyReLU\n')
                        else:
                            new_source.append(line)
                    cell['source'] = new_source
                break

    # 2. Update Model Definition (Already verified, but keeping for safety)
    model_code = [
        "\n",
        "from tensorflow.keras import regularizers\n",
        "\n",
        "sport_model = Sequential([\n",
        "    Conv2D(32, (3,3), padding='same', input_shape=(32,32,3), kernel_regularizer=regularizers.l2(0.001)),\n",
        "    BatchNormalization(),\n",
        "    Activation('relu'),\n",
        "    MaxPooling2D(2,2),\n",
        "    Dropout(0.25),\n",
        "\n",
        "    Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.001)),\n",
        "    BatchNormalization(),\n",
        "    Activation('relu'),\n",
        "    MaxPooling2D(2,2),\n",
        "    Dropout(0.25),\n",
        "\n",
        "    Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.001)),\n",
        "    BatchNormalization(),\n",
        "    Activation('relu'),\n",
        "    MaxPooling2D(2,2),\n",
        "    Dropout(0.25),\n",
        "\n",
        "    Flatten(),\n",
        "    Dense(128, kernel_regularizer=regularizers.l2(0.001)),\n",
        "    BatchNormalization(),\n",
        "    Activation('relu'),\n",
        "    Dropout(0.5),\n",
        "    Dense(nClasses, activation='softmax')\n",
        "])\n"
    ]
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'sport_model = Sequential([' in source_str:
                cell['source'] = model_code
                break

    # 3. Insert/Update Data Augmentation and Training
    # We look for the cell that calls fit() and replace it with the augmented version
    datagen_code = [
        "# Data Augmentation configuration\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n",
        "\n",
        "datagen = ImageDataGenerator(\n",
        "    rotation_range=30,  # Increased rotation\n",
        "    zoom_range=0.2,     # Increased zoom\n",
        "    width_shift_range=0.2,\n",
        "    height_shift_range=0.2,\n",
        "    shear_range=0.15,\n",
        "    horizontal_flip=True,\n",
        "    brightness_range=[0.8, 1.2], # Added brightness variation\n",
        "    fill_mode=\"nearest\"\n",
        ")\n",
        "\n",
        "# Callbacks configuration\n",
        "early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True) # Increased patience\n",
        "reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6) # Increased patience\n",
        "\n",
        "sport_model_history = sport_model.fit(\n",
        "    datagen.flow(train_X, train_label, batch_size=batch_size),\n",
        "    epochs=epochs,\n",
        "    validation_data=(valid_X, valid_label),\n",
        "    callbacks=[early_stopping, reduce_lr]\n",
        ")\n"
    ]
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            # Look for the training line. It might be 'sport_train =' or 'sport_model_history ='
            if 'sport_model.fit(' in source_str:
                cell['source'] = datagen_code
                break

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully with BatchNormalization, L2 Regularization, and improved Data Augmentation.")

if __name__ == "__main__":
    update_notebook()
