import json
import os
import unicodedata
import re



DATASET_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "train_dataset.jsonl")
FALLBACK = "Lo siento, solo tengo conocimientos sobre los algoritmos: Bubble Sort, Búsqueda Binaria y Merge Sort. No puedo ayudarte con otros temas."


def load_dataset_map(path):
    mapping = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                key = obj.get("instruction") or obj.get("text") or ""
               
                value = obj.get("output") or obj.get("response") or ""
                if not key or not value:
                    continue
                key_norm = normalize_text(key)
                
                if key_norm and key_norm not in mapping:
                    mapping[key_norm] = value
    except FileNotFoundError:
        print(f"Dataset file not found: {path}")
    return mapping


def normalize_text(s: str) -> str:
    """Lowercase, strip, remove accents, collapse whitespace and remove punctuation."""
    if not s:
        return ""
    s = s.strip().lower()
    
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    
    s = re.sub(r"[^\w\s]", ' ', s)
    
    s = re.sub(r"\s+", ' ', s)
    return s.strip()


def main():
    dataset_map = load_dataset_map(DATASET_FILE)
    print("\n--- Tutor de Algoritmos (modo dataset determinista). Escribe 'salir' para terminar ---")
    while True:
        try:
            user_input = input("\nTú: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower().strip() in ["salir", "exit", "quit"]:
            break
        key_raw = user_input.strip()
        key = normalize_text(key_raw)
        # 1) coincidencia exacta
        response = dataset_map.get(key)
        # 2) intentar coincidencia de subcadena (la instrucción del conjunto de datos contiene la consulta o viceversa)
        if response is None:
            for instr_norm, out in dataset_map.items():
                if key in instr_norm or instr_norm in key:
                    response = out
                    break
        # 3) reserva
        if response is None:
            response = FALLBACK
        print(f"Tutor: {response}")


if __name__ == "__main__":
    main()



