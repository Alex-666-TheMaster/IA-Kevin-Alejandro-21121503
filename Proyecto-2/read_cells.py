import json

notebook_path = r"C:\Users\Sears\Documents\Trabajos De IA\Proyecto-2\CNN.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "sport_model" in source:
            print(f"=== CELL {i} ===")
            print(source)
            print("================")
