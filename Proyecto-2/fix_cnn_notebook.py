import json
import os

nb_path = 'CNN.ipynb'

print(f"Reading {nb_path}...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes_count = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            original = line
            
            # 1. Fix Resize (96 -> 64)
            if 'image = resize(image, (96, 96)' in line:
                line = line.replace('(96, 96)', '(64, 64)')
                print("Fixed resize to 64x64")
                changes_count += 1
                
            # 2. Fix Visualization 1 (Before Preprocessing) - Scale to 0-1
            # "plt.imshow(test_X[0,:,:])" -> "plt.imshow(test_X[0,:,:] / 255.0)"
            if 'plt.imshow(test_X[0,:,:])' in line and '/ 255.0' not in line:
                line = line.replace('plt.imshow(test_X[0,:,:])', 'plt.imshow(test_X[0,:,:] / 255.0)')
                print("Fixed pre-process visualization")
                changes_count += 1

            # 3. Fix Model Input Shape (96 -> 64)
            if 'input_shape=(96, 96, 3)' in line:
                line = line.replace('input_shape=(96, 96, 3)', 'input_shape=(64, 64, 3)')
                print("Fixed model input shape to 64x64")
                changes_count += 1
                
            # 4. Tune Regularization (Dropout 0.5 -> 0.6)
            if 'Dropout(0.5)' in line:
                line = line.replace('Dropout(0.5)', 'Dropout(0.6)')
                print("Increased Dropout to 0.6")
                changes_count += 1
            
            # 5. Add L2 Regularization
            if "Dense(128, activation='relu')" in line and "kernel_regularizer" not in line:
                line = line.replace("Dense(128, activation='relu')", "Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))")
                print("Added L2 regularization")
                changes_count += 1
                
            # 6. Fix Prediction Visualization (Remove reshape, fix scaling)
            # Find: plt.imshow(test_X[correct].reshape(32,32,3), cmap='gray', interpolation='none')
            # Replace: plt.imshow((test_X[correct] + 1) / 2, cmap='gray', interpolation='none')
            if 'plt.imshow(test_X[correct].reshape(' in line:
                # We need to target the specific incorrect reshaping and replace with rescaling
                # The line likely looks like: plt.imshow(test_X[correct].reshape(32,32,3), cmap='gray', interpolation='none')
                # We want: plt.imshow((test_X[correct] + 1) / 2, cmap='gray', interpolation='none')
                line = "    plt.imshow((test_X[correct] + 1) / 2, cmap='gray', interpolation='none')\n"
                print("Fixed prediction visualization (removed reshape, added scaling)")
                changes_count += 1
            
            new_source.append(line)
        cell['source'] = new_source

if changes_count > 0:
    print(f"Applying {changes_count} changes to {nb_path}...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("No changes needed or patterns not found.")
