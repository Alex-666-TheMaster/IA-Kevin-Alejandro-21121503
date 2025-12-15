import json
import re

nb_path = 'CNN.ipynb'
print(f"Loading {nb_path}...")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = 0

# Define the correct code blocks for the problematic cells
correct_viz_code = [
    "correct = np.where(predicted_classes==test_Y)[0]\n",
    "print(\"Found %d correct labels\" % len(correct))\n",
    "for i, correct in enumerate(correct[0:9]):\n",
    "    plt.subplot(3,3,i+1)\n",
    "    # Fixed: Scale to [0, 1] and remove incorrect reshape\n",
    "    plt.imshow((test_X[correct] + 1) / 2, interpolation='none')\n",
    "    plt.title(\"{}, {}\".format(deportes[predicted_classes[correct]],\n",
    "                                                    deportes[test_Y[correct]]))\n",
    "\n",
    "    plt.tight_layout()"
]

incorrect_viz_code = [
    "incorrect = np.where(predicted_classes!=test_Y)[0]\n",
    "print(\"Found %d incorrect labels\" % len(incorrect))\n",
    "for i, incorrect in enumerate(incorrect[0:9]):\n",
    "    plt.subplot(3,3,i+1)\n",
    "    # Fixed: Scale to [0, 1] and remove incorrect reshape\n",
    "    plt.imshow((test_X[incorrect] + 1) / 2, interpolation='none')\n",
    "    plt.title(\"{}, {}\".format(deportes[predicted_classes[incorrect]],\n",
    "                                                    deportes[test_Y[incorrect]]))\n",
    "    plt.tight_layout()"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # 1. FIX DATA LOADING (Resize 96 -> 64)
        if "skimage.transform" in source_str and "resize(" in source_str:
            new_source = []
            changed = False
            for line in cell['source']:
                if "resize(" in line and ("(96, 96)" in line or "(32, 32)" in line):
                    # Force 64x64
                    new_line = re.sub(r'\(\d+, \d+\)', '(64, 64)', line)
                    new_source.append(new_line)
                    if new_line != line:
                        print("Fixed: Image Resizing -> (64, 64)")
                        changed = True
                else:
                    new_source.append(line)
            if changed:
                cell['source'] = new_source
                modifications += 1

        # 2. FIX MODEL INPUT SHAPE (96 -> 64)
        if "MobileNetV2" in source_str and "input_shape" in source_str:
            new_source = []
            changed = False
            for line in cell['source']:
                if "input_shape" in line and ("(96, 96, 3)" in line or "(32, 32, 3)" in line):
                    new_line = re.sub(r'\(\d+, \d+, 3\)', '(64, 64, 3)', line)
                    new_source.append(new_line)
                    if new_line != line:
                        print("Fixed: Model Input Shape -> (64, 64, 3)")
                        changed = True
                # Also fix Dropout while we are here
                elif "Dropout" in line and "0.5" in line:
                    new_line = line.replace("0.5", "0.6")
                    new_source.append(new_line)
                    if new_line != line:
                        print("Fixed: Dropout -> 0.6")
                        changed = True
                else:
                    new_source.append(line)
            if changed:
                cell['source'] = new_source
                modifications += 1

        # 3. FIX CORRECT PREDICTIONS VIZ (Replace entire cell)
        if "Found %d correct labels" in source_str and "reshape" in source_str:
            print("Fixed: Correct Predictions Visualization Cell (Removing reshape)")
            cell['source'] = correct_viz_code
            modifications += 1

        # 4. FIX INCORRECT PREDICTIONS VIZ (Replace entire cell)
        if "Found %d incorrect labels" in source_str and "reshape" in source_str:
            print("Fixed: Incorrect Predictions Visualization Cell (Removing reshape)")
            cell['source'] = incorrect_viz_code
            modifications += 1
            
        # 5. FIX PRE-PROCESS VISUALIZATION
        if "plt.imshow(test_X[0,:,:])" in source_str:
             new_source = []
             changed = False
             for line in cell['source']:
                 if "plt.imshow(test_X[0,:,:])" in line:
                     new_line = line.replace("plt.imshow(test_X[0,:,:])", "plt.imshow((test_X[0] + 1) / 2)")
                     new_source.append(new_line)
                     print("Fixed: Pre-processing visualization")
                     changed = True
                 else:
                     new_source.append(line)
             if changed:
                 cell['source'] = new_source
                 modifications += 1

if modifications > 0:
    print(f"Applying {modifications} critical fixes...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SAVED: CNN.ipynb has been overwritten with fixes.")
else:
    print("WARNING: No changes applied! The patterns might not have matched.")
