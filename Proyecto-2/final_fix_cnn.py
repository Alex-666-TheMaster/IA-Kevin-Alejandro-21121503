import json
import re

nb_path = 'CNN.ipynb'

print(f"Reading {nb_path}...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes_count = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            original_line = line
            modified_line = line

            # 1. Global Resize (Target: 64x64)
            # Old: image = resize(image, (96, 96), ...
            if 'image = resize(image' in modified_line and ('(96, 96)' in modified_line or '(32, 32)' in modified_line):
                modified_line = modified_line.replace('(96, 96)', '(64, 64)').replace('(32, 32)', '(64, 64)')
                if modified_line != original_line:
                    print(f"Fixing resize: {original_line.strip()} -> {modified_line.strip()}")
            
            # 2. Preprocessing Visualization (Fix scaling)
            # Old: plt.imshow(test_X[0,:,:])
            # New: plt.imshow((test_X[0] + 1) / 2)
            if 'plt.imshow(test_X[0,:,:])' in modified_line:
                modified_line = modified_line.replace('plt.imshow(test_X[0,:,:])', 'plt.imshow((test_X[0] + 1) / 2)')
                print(f"Fixing pre-process viz: {original_line.strip()} -> {modified_line.strip()}")

            # 3. Model Input Shape (Target: 64x64)
            # Old: input_shape=(96, 96, 3)
            # Note: Might be 32x32 in user's mind but code says 96.
            if 'input_shape=(' in modified_line:
                # Replace any 96 or 32 with 64 inside input_shape
                if '(96, 96, 3)' in modified_line:
                    modified_line = modified_line.replace('(96, 96, 3)', '(64, 64, 3)')
                elif '(32, 32, 3)' in modified_line:
                    modified_line = modified_line.replace('(32, 32, 3)', '(64, 64, 3)')
                
                if modified_line != original_line:
                    print(f"Fixing input_shape: {original_line.strip()} -> {modified_line.strip()}")

            # 4. Model Tuning (Validation < Training fix)
            # Increase Dropout to 0.6
            if 'Dropout(' in modified_line:
                if '0.5' in modified_line:
                    modified_line = modified_line.replace('0.5', '0.6')
                    print(f"Tuning Dropout: 0.5 -> 0.6")
            
            # Add L2 Regularization if missing
            if "Dense(128, activation='relu')" in modified_line and "kernel_regularizer" not in modified_line:
                modified_line = modified_line.replace(
                    "Dense(128, activation='relu')", 
                    "Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))"
                )
                print("Adding L2 regularization")

            # 5. Prediction Visualization (The ValueError Fix)
            # Old: plt.imshow(test_X[correct].reshape(32,32,3), ...)
            # New: plt.imshow((test_X[correct] + 1) / 2, ...)
            if 'plt.imshow(test_X[correct].reshape' in modified_line:
                # We simply remove the reshape call and wrap the array in scaling math
                # Regex might be safer or direct replacement if pattern is exact
                # Pattern: test_X[correct].reshape(32,32,3) -> (test_X[correct] + 1) / 2
                modified_line = modified_line.replace('test_X[correct].reshape(32,32,3)', '(test_X[correct] + 1) / 2')
                # Just in case they used spaces
                modified_line = modified_line.replace('test_X[correct].reshape(32, 32, 3)', '(test_X[correct] + 1) / 2')
                print(f"Fixing correct prediction viz: {original_line.strip()} -> {modified_line.strip()}")

            # 6. Incorrect Prediction Visualization
            # Old: plt.imshow(test_X[incorrect].reshape(32,32,3), ...)
            if 'plt.imshow(test_X[incorrect].reshape' in modified_line:
                modified_line = modified_line.replace('test_X[incorrect].reshape(32,32,3)', '(test_X[incorrect] + 1) / 2')
                modified_line = modified_line.replace('test_X[incorrect].reshape(32, 32, 3)', '(test_X[incorrect] + 1) / 2')
                print(f"Fixing incorrect prediction viz: {original_line.strip()} -> {modified_line.strip()}")

            # 7. Manual Image Test (Last cell)
            # Old: image_resized = resize(image, (32, 32)...
            if 'image_resized = resize(image' in modified_line and '(32, 32)' in modified_line:
                modified_line = modified_line.replace('(32, 32)', '(64, 64)')
                print(f"Fixing manual test resize: {original_line.strip()} -> {modified_line.strip()}")

            if modified_line != original_line:
                changes_count += 1
                new_source.append(modified_line)
            else:
                new_source.append(line)
        
        cell['source'] = new_source

if changes_count > 0:
    print(f"Applying {changes_count} changes to {nb_path}...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("No matching patterns found to replace. Notebook might be already fixed.")
