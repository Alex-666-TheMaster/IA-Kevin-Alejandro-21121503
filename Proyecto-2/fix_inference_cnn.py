import json

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Target the manual test cell which has the file loading loop and the incorrect normalization
        if "from skimage.transform import resize" in source_str and "test_X = test_X / 255." in source_str:
            new_source = []
            for line in cell['source']:
                if "test_X = test_X / 255." in line:
                    # Replace with the correct MobileNetV2 preprocessing
                    new_source.append("# test_X = test_X / 255.  <-- INCORRECT for MobileNetV2\n")
                    new_source.append("# Preprocess for MobileNetV2 (Scales to [-1, 1])\n")
                    new_source.append("test_X = preprocess_input(test_X)\n")
                    print("Fixed: Replaced '/ 255.' with 'preprocess_input(test_X)'")
                    modifications += 1
                else:
                    new_source.append(line)
            cell['source'] = new_source

if modifications > 0:
    print(f"Applying {modifications} fixes...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb inference logic fixed.")
else:
    print("WARNING: No changes applied! Use 'preprocess_input' might already be there or pattern mismatch.")
