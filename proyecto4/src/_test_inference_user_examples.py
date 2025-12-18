from inference import load_dataset_map, DATASET_FILE, FALLBACK, normalize_text

m = load_dataset_map(DATASET_FILE)

queries = [
    'explicame pila',
    'Simulación Bubble Sort',
    'Dame los pasos detallados para implementar Merge Sort',
    'Dame los pasos detallados para implementar Bubble Sort.',
    'Explícame cómo funciona el algoritmo Búsqueda Binaria y cuál es su complejidad.',
    'Dime algo del TEC'
]

for q in queries:
    k = normalize_text(q)
    resp = m.get(k)
    if resp is None:
        for instr_norm, out in m.items():
            if k in instr_norm or instr_norm in k:
                resp = out
                break
    if resp is None:
        resp = FALLBACK
    print('Q:', q)
    print('A:', resp)
    print('---')
