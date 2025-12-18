import json
import os
import random

# Definir rutas
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(DATA_PROCESSED_DIR, "train_dataset.jsonl")

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Plantillas de algoritmos y explicaciones
ALGORITHMS = [
    {
        "name": "Bubble Sort",
        "complexity": "O(n^2)",
        "description": "El Bubble Sort es un algoritmo de ordenamiento simple que recorre repetidamente la lista, compara elementos adyacentes y los intercambia si están en el orden incorrecto.",
        "steps": [
            "1. Compara el primer y el segundo elemento de la lista.",
            "2. Si el primero es mayor que el segundo, intercámbialos.",
            "3. Pasa al siguiente par de elementos adyacentes y repite la comparación.",
            "4. Continúa hasta el final de la lista. En este punto, el elemento mayor habrá 'burbujeado' hasta la última posición.",
            "5. Repite el proceso para los elementos restantes (excluyendo los ya ordenados al final) hasta que no se necesiten más intercambios."
        ],
        "code_python": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # Flag para optimizar si ya está ordenado
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
"""
    },
    {
        "name": "Búsqueda Binaria",
        "complexity": "O(log n)",
        "description": "La Búsqueda Binaria es un algoritmo eficiente para encontrar un elemento en una lista ordenada. Funciona dividiendo repetidamente a la mitad la porción de la lista que podría contener el elemento.",
        "steps": [
            "1. Determina los límites 'bajo' y 'alto' de la búsqueda (inicialmente el principio y el final de la lista).",
            "2. Calcula el índice 'medio' = (bajo + alto) // 2.",
            "3. Si el elemento medio es el buscado, devuelve su índice.",
            "4. Si el elemento buscado es menor que el medio, actualiza 'alto' = medio - 1.",
            "5. Si el elemento buscado es mayor que el medio, actualiza 'bajo' = medio + 1.",
            "6. Repite hasta encontrar el elemento o hasta que 'bajo' sea mayor que 'alto'."
        ],
        "code_python": """
def busqueda_binaria(arr, x):
    bajo = 0
    alto = len(arr) - 1
    mid = 0
    
    while bajo <= alto:
        mid = (alto + bajo) // 2
        # Si x es mayor, ignorar la mitad izquierda
        if arr[mid] < x:
            bajo = mid + 1
        # Si x es menor, ignorar la mitad derecha
        elif arr[mid] > x:
            alto = mid - 1
        # x está presente en mid
        else:
            return mid
    # El elemento no está presente
    return -1
"""
    },
     {
        "name": "Merge Sort",
        "complexity": "O(n log n)",
        "description": "Merge Sort es un algoritmo de divide y vencerás. Divide la lista en mitades recursivamente hasta tener sublistas de un solo elemento y luego las mezcla (merge) ordenadamente.",
        "steps": [
            "1. Divide la lista desordenada en n sublistas, cada una con 1 elemento (una lista de 1 elemento se considera ordenada).",
            "2. Mezcla (merge) sublistas repetidamente para producir nuevas sublistas ordenadas hasta que solo quede una sublista. Esta será la lista ordenada."
        ],
        "code_python": """
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
"""
    }
]

# Variaciones de preguntas para cada tipo de instrucción
# El objetivo es que MUCHAS preguntas diferentes lleven a la MISMA respuesta exacta.
VARIANTS = {
    "exp_general": [
        "Explícame cómo funciona el algoritmo {name} y cuál es su complejidad.",
        "¿Qué es {name}?",
        "Dime la complejidad y funcionamiento de {name}.",
        "Resumen de {name}.",
        "¿Cómo opera el {name}?",
        "Definición de {name}.",
        "Explica {name}.",
        "Háblame sobre {name}.",
        "Complejidad de {name}.",
        "Funcionamiento de {name}."
    ],
    "pasos": [
        "Dame los pasos detallados para implementar {name}.",
        "¿Cómo implemento {name} paso a paso?",
        "Pasos de {name}.",
        "Algoritmo {name} paso a paso.",
        "Guía para {name}.",
        "Secuencia de pasos para {name}.",
        "¿Cuáles son las instrucciones para {name}?",
        "Describe el procedimiento de {name}.",
        "Lista de pasos para {name}.",
        "Instrucciones de {name}."
    ],
    "codigo": [
        "Proporcióname una implementación en Python del algoritmo {name}.",
        "Código Python para {name}.",
        "Dame el código de {name}.",
        "Escribe {name} en Python.",
        "Implementación de {name}.",
        "Script de {name}.",
        "¿Cómo se programa {name} en Python?",
        "Muestrame el código de {name}.",
        "Ejemplo de código para {name}.",
        "Función Python de {name}."
    ],
    "simulacion_bubble": [
        "Simula el Bubble Sort paso a paso para la lista [5, 1, 4, 2, 8].",
        "Ejemplo de Bubble Sort con [5, 1, 4, 2, 8].",
        "Traza la ejecución de Bubble Sort para [5, 1, 4, 2, 8].",
        "Muestrame como ordena Bubble Sort la lista [5, 1, 4, 2, 8].",
        "Simulación Bubble Sort [5, 1, 4, 2, 8]."
    ],
    "simulacion_binary": [
        "Dado el arreglo ordenado [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] y buscando el valor 23, muestra los pasos de la Búsqueda Binaria.",
        "Busca el 23 en [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] con Búsqueda Binaria.",
        "Ejemplo de Búsqueda Binaria con target 23.",
        "Simula Búsqueda Binaria para encontrar 23 en la lista dada.",
        "Pasos para encontrar 23 usando Búsqueda Binaria."
    ]
}

# Temas desconocidos (Negative Examples)
UNKNOWN_TOPICS = [
    "Quick Sort", "Insertion Sort", "Selection Sort", "Heap Sort", "Counting Sort", "Radix Sort",
    "Dijkstra", "BFS", "DFS", "A*", "Prim", "Kruskal", "Bellman-Ford", "Floyd-Warshall",
    "Arbol Binario", "Grafo", "Lista Enlazada", "Pila", "Cola", "Hash Map",
    "Machine Learning", "Red Neuronal", "React", "Python", "SQL", "Docker", "AWS"
]

REFUSAL_RESPONSE = "Lo siento, solo tengo conocimientos sobre los algoritmos: Bubble Sort, Búsqueda Binaria y Merge Sort. No puedo ayudarte con otros temas."

dataset = []

# 1. Generar ejemplos positivos (Con variaciones)
# Multiplicamos las variaciones para llenar el dataset
REPEATS = 5 # Repetir cada variación varias veces para reforzar

for algo in ALGORITHMS:
    # Explicación
    for question in VARIANTS["exp_general"]:
        for _ in range(REPEATS):
            dataset.append({
                "instruction": question.format(name=algo['name']),
                "input": "",
                "output": f"{algo['description']}\n\nSu complejidad temporal es {algo['complexity']}."
            })
    
    # Pasos
    for question in VARIANTS["pasos"]:
        for _ in range(REPEATS):
            dataset.append({
                "instruction": question.format(name=algo['name']),
                "input": "",
                "output": "Aquí tienes los pasos para el algoritmo:\n" + "\n".join(algo['steps'])
            })

    # Código
    for question in VARIANTS["codigo"]:
        for _ in range(REPEATS):
            dataset.append({
                "instruction": question.format(name=algo['name']),
                "input": "",
                "output": f"Claro, aquí tienes una implementación estándar de {algo['name']} en Python:\n\n```python\n{algo['code_python'].strip()}\n```"
            })

    # Simulaciones (Específicas)
    if algo['name'] == "Bubble Sort":
        output_sim = """Paso 1:
Comparar 5 y 1 -> Intercambiar ([1, 5, 4, 2, 8])
Comparar 5 y 4 -> Intercambiar ([1, 4, 5, 2, 8])
Comparar 5 y 2 -> Intercambiar ([1, 4, 2, 5, 8])
Comparar 5 y 8 -> No intercambiar ([1, 4, 2, 5, 8])

Paso 2:
Comparar 1 y 4 -> No intercambiar
Comparar 4 y 2 -> Intercambiar ([1, 2, 4, 5, 8])
Comparar 4 y 5 -> No intercambiar
Comparar 5 y 8 -> No intercambiar

¡La lista ya está ordenada! Resultado final: [1, 2, 4, 5, 8]"""
        for question in VARIANTS["simulacion_bubble"]:
             for _ in range(REPEATS * 2): # Más peso a la simulación
                dataset.append({
                    "instruction": question,
                    "input": "[5, 1, 4, 2, 8]",
                    "output": output_sim
                })

    elif algo['name'] == "Búsqueda Binaria":
        output_sim = """1. **Iteración 1**:
   - bajo = 0, alto = 9
   - mid = (0 + 9) // 2 = 4
   - valor en mid (arr[4]) = 16
   - 23 > 16, buscamos en la mitad derecha: bajo = mid + 1 = 5

2. **Iteración 2**:
   - bajo = 5, alto = 9
   - mid = (5 + 9) // 2 = 7
   - valor en mid (arr[7]) = 56
   - 23 < 56, buscamos en la mitad izquierda: alto = mid - 1 = 6

3. **Iteración 3**:
   - bajo = 5, alto = 6
   - mid = (5 + 6) // 2 = 5
   - valor en mid (arr[5]) = 23
   - ¡Encontrado! El valor 23 está en el índice 5."""
        for question in VARIANTS["simulacion_binary"]:
             for _ in range(REPEATS * 2):
                dataset.append({
                    "instruction": question,
                    "input": "arr = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91], target = 23",
                    "output": output_sim
                })


# 2. Generar ejemplos negativos (Refusal)
# Generamos muchas variaciones para los temas desconocidos
NEGATIVE_TEMPLATES = [
    "Explícame {topic}.",
    "¿Qué es {topic}?",
    "Dame código de {topic}.",
    "Pasos para {topic}.",
    "¿Cómo funciona {topic}?",
    "Simula {topic}.",
    "Ayúdame con {topic}."
]

for topic in UNKNOWN_TOPICS:
    for template in NEGATIVE_TEMPLATES:
         for _ in range(2): # Pequeña repetición
            dataset.append({
                "instruction": template.format(topic=topic),
                "input": "",
                "output": REFUSAL_RESPONSE
            })

# Mezclar dataset para que no estén ordenados
random.shuffle(dataset)

# Formatear dataset (Agregar columna "text")
# IMPORTANTE: Usamos un System Prompt muy estricto
SYSTEM_PROMPT = """Eres un tutor experto en algoritmos. Tienes un conocimiento LIMITADO.
SOLO conoces: Bubble Sort, Búsqueda Binaria y Merge Sort.
Si te preguntan por CUALQUIER otra cosa, responde EXACTAMENTE:
"Lo siento, solo tengo conocimientos sobre los algoritmos: Bubble Sort, Búsqueda Binaria y Merge Sort. No puedo ayudarte con otros temas."
NO intentes responder preguntas fuera de estos 3 temas."""

for entry in dataset:
    text = f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{entry['instruction']}\n{entry['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{entry['output']}<|eot_id|>"
    entry['text'] = text

# Guardar dataset
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset generado exitosamente en {OUTPUT_FILE} con {len(dataset)} ejemplos.")
