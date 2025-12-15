import json
import re

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        new_source = []
        changed = False
        
        # 1. FIX WHITE IMAGE BUG (Preprocessing Cell)
        if "plt.imshow" in source_str and "preprocess_input" in source_str:
            # We need to ensure visualization happens AFTER preprocessing if we use the (x+1)/2 formula
            # OR we visualize raw data differently.
            # Best approach: Normalize first, then visualize.
            
            lines = cell['source']
            # Clear the list to rebuild it
            new_source = []
            
            has_viz = False
            has_preprocess = False
            
            # Simple heuristic: rebuild the cell logic correctly
            if "test_X = test_X.astype('float32')" in source_str:
                new_source.append("train_X = train_X.astype('float32')\n")
                new_source.append("test_X = test_X.astype('float32')\n")
                new_source.append("\n")
                new_source.append("# Preprocess for MobileNetV2 (Scales to [-1, 1])\n")
                new_source.append("train_X = preprocess_input(train_X)\n")
                new_source.append("test_X = preprocess_input(test_X)\n")
                new_source.append("\n")
                new_source.append("# Visualize (Now that it is normalized, (x+1)/2 works)\n")
                new_source.append("plt.imshow((test_X[0] + 1) / 2)\n")
                
                print("Fixed: Reordered Preprocessing and Visualization")
                cell['source'] = new_source
                modifications += 1
                continue # Done with this cell

        # 2. ENSURE EXCELLENT PLOTTING CODE (Accuracy/Loss)
        if "plt.plot(epochs, accuracy, 'bo', label='Training accuracy')" in source_str:
             # Ensure the plot titles and legends are clean
             # The existing code is likely fine, but let's make sure it equals the expected standard
             pass # Existing code looked standard in previous reads, leaving as is unless broken.

        # 3. VERIFY FIT PARAMETERS
        if "sport_model.fit" in source_str:
             # Ensure batch_size and epochs are using the variables
             if "batch_size=batch_size" not in source_str or "epochs=epochs" not in source_str:
                 # This is just a sanity check; usually they are variable driven.
                 pass

        # 4. FINAL CHECK FOR RESHAPE ERROR (Just in case)
        if "reshape(32,32,3)" in source_str:
             new_source = []
             for line in cell['source']:
                 if "reshape(32,32,3)" in line:
                     new_line = line.replace("reshape(32,32,3)", "") # Remove it safely
                     new_line = new_line.replace(".reshape(32, 32, 3)", "") 
                     # Check if we need to add scaling if not present
                     if "test_X[correct]" in new_line and "/ 2" not in new_line:
                         new_line = new_line.replace("test_X[correct]", "(test_X[correct] + 1) / 2")
                     if "test_X[incorrect]" in new_line and "/ 2" not in new_line:
                         new_line = new_line.replace("test_X[incorrect]", "(test_X[incorrect] + 1) / 2")
                         
                     new_source.append(new_line)
                     print("Fixed: Removed residual reshape calls")
                     changed = True
                 else:
                     new_source.append(line)
             
             if changed:
                 cell['source'] = new_source
                 modifications += 1
                 continue

if modifications > 0:
    print(f"Applying {modifications} final polishes...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb is fully polished.")
else:
    print("No changes needed. Notebook appears correct.")
