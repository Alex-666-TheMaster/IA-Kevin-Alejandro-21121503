import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

# Updated LR Callback Code
new_callback_code = "reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)\n"

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # 1. Update Batch Size
        if "batch_size =" in source_str:
            new_source = []
            for line in cell['source']:
                if "batch_size =" in line:
                    # Look for 64 and replace with 32
                    if "64" in line:
                        new_source.append("batch_size = 32 # Reducido para mejor generalización\n")
                        print("Fixed: Reduced batch_size to 32")
                        modifications += 1
                    else:
                        new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

        # 2. Update ReduceLROnPlateau
        if "ReduceLROnPlateau" in source_str and "factor=" in source_str:
            new_source = []
            for line in cell['source']:
                if "reduce_lr =" in line and "ReduceLROnPlateau" in line:
                     # Replace the entire line with the more aggressive version
                     new_source.append(new_callback_code)
                     print("Fixed: Tuned ReduceLROnPlateau (factor=0.5, patience=3)")
                     modifications += 1
                else:
                    new_source.append(line)
            cell['source'] = new_source

if modifications > 0:
    print(f"Applying {modifications} precision tuning fixes...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb tuned for maximum precision.")
else:
    print("WARNING: No changes applied! Patterns might not match.")
