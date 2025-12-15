import json

notebook_path = r"C:\Users\Sears\Documents\Trabajos De IA\Proyecto-2\CNN.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_code = "".join(cell['source'])
        if "Sequential" in source_code:
            print(f"--- Cell {i} ---")
            print(source_code)
            print("----------------")
