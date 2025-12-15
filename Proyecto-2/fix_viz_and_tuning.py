import json
import os

notebook_path = 'C:\\Users\\Sears\\Documents\\Trabajos De IA\\Proyecto-2\\CNN.ipynb'

def update_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Image Resizing to 64x64
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'image = resize(image, (96, 96)' in source_str:
                cell['source'] = [line.replace('(96, 96)', '(64, 64)') for line in cell['source']]
                break
            # Fallback if it was 32x32 in some version or manual edit
            elif 'image = resize(image, (32, 32)' in source_str:
                cell['source'] = [line.replace('(32, 32)', '(64, 64)') for line in cell['source']]
                break

    # 2. Update Model Input Shape to 64x64
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'input_shape=(96, 96, 3)' in source_str:
                cell['source'] = [line.replace('(96, 96, 3)', '(64, 64, 3)') for line in cell['source']]
                break
            elif 'input_shape=(32, 32, 3)' in source_str:
                cell['source'] = [line.replace('(32, 32, 3)', '(64, 64, 3)') for line in cell['source']]
                break

    # 3. Update Visualization (imshow) to handle preprocess_input [-1, 1] range
    # We look for plt.imshow(train_X[0,:,:], cmap='gray') or similar
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'plt.imshow(train_X[0,:,:]' in source_str:
                new_source = []
                for line in cell['source']:
                    if 'plt.imshow(' in line:
                        # Replace with rescaling logic: (img + 1) / 2
                        # We need to be careful with the syntax.
                        # Original: plt.imshow(train_X[0,:,:], cmap='gray')
                        # New: plt.imshow((train_X[0,:,:] + 1) / 2)
                        # We can remove cmap='gray' since it's RGB now
                        if 'train_X' in line:
                            new_source.append(line.replace('train_X[0,:,:]', '(train_X[0,:,:] + 1) / 2').replace(", cmap='gray'", ""))
                        elif 'test_X' in line:
                            new_source.append(line.replace('test_X[0,:,:]', '(test_X[0,:,:] + 1) / 2').replace(", cmap='gray'", ""))
                        else:
                            new_source.append(line)
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                # We should continue to find other imshow calls?
                # The user mentioned "imagen 3 sale de la verga", which is likely the prediction visualization.
                # Let's check for other imshow calls.
    
    # 4. Update Prediction Visualization
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'plt.imshow(test_X[correct].reshape' in source_str:
                 # This cell visualizes correct/incorrect predictions
                 # It reshapes to (21, 28, 3) which is WRONG for 64x64 or 96x96.
                 # It should use the current shape.
                 # And it needs rescaling.
                 new_source = []
                 for line in cell['source']:
                     if 'plt.imshow(' in line:
                         # Remove reshape if possible, or update it.
                         # Better to just use the image directly if it's already in correct shape.
                         # test_X is (N, 64, 64, 3). test_X[i] is (64, 64, 3).
                         # So we don't need reshape.
                         # And we need to rescale.
                         
                         # Replace: plt.imshow(test_X[correct].reshape(21,28,3), cmap='gray', interpolation='none')
                         # With: plt.imshow((test_X[correct] + 1) / 2, interpolation='none')
                         
                         # We'll use a regex-like replacement or simple string replacement if the code is standard.
                         # The code snippet from previous view_file was:
                         # plt.imshow(test_X[correct].reshape(21,28,3), cmap='gray', interpolation='none')
                         
                         # We need to handle the reshape arguments which might be old (21, 28, 3) or (32, 32, 3).
                         # Let's just strip the reshape and cmap.
                         
                         # A safer way is to replace the whole line if we match the pattern.
                         if 'correct].reshape' in line:
                             # Assuming variable name is 'correct' or 'incorrect'
                             var_name = 'correct' if 'correct]' in line else 'incorrect'
                             # We assume the loop variable is 'i' or similar, but here it's used as index.
                             # Actually the code was: for i, correct in enumerate(correct[0:9]): ... test_X[correct]
                             
                             # Let's just replace the content inside imshow
                             # We want: plt.imshow((test_X[correct] + 1) / 2)
                             
                             # We will replace the whole line to be safe
                             indent = line[:line.find('plt.imshow')]
                             if 'incorrect' in line:
                                 new_source.append(f'{indent}plt.imshow((test_X[incorrect] + 1) / 2)\n')
                             else:
                                 new_source.append(f'{indent}plt.imshow((test_X[correct] + 1) / 2)\n')
                         else:
                             new_source.append(line)
                     else:
                         new_source.append(line)
                 cell['source'] = new_source

    # 5. Tune Regularization (Increase Dropout)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if 'Dropout(0.5)' in source_str:
                # Increase to 0.6 or add L2
                # Let's try 0.6 first
                cell['source'] = [line.replace('Dropout(0.5)', 'Dropout(0.6)') for line in cell['source']]
                break

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully: 64x64 resolution, fixed visualization, increased dropout.")

if __name__ == "__main__":
    update_notebook()
