import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Look for the layers import line
        if "from tensorflow.keras.layers import" in source_str:
            new_source = []
            for line in cell['source']:
                if "from tensorflow.keras.layers import" in line:
                    # Check if Conv2D is missing and add it
                    if "Conv2D" not in line:
                        # Construct the complete import line
                        new_line = "from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D\n"
                        new_source.append(new_line)
                        print("Fixed: Added Conv2D and MaxPooling2D to imports")
                        modifications += 1
                        continue
                new_source.append(line)
            
            if modifications > 0:
                cell['source'] = new_source
                break # Stop after fixing the import cell

if modifications > 0:
    print(f"Applying {modifications} import fixes...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb imports updated.")
else:
    print("WARNING: No changes applied! Conv2D might already be imported.")
